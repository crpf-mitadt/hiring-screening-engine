#!/usr/bin/env python3

import argparse
import base64
import io
import json
import time
from pathlib import Path

import requests

from chandra.input import load_file
from chandra.model.util import scale_to_fit
from chandra.prompts import OCR_LAYOUT_PROMPT


def encode_png(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def extract_content(payload: dict) -> str:
    message = payload["choices"][0]["message"]
    content = message.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if not isinstance(item, dict):
                continue

            text = item.get("text")
            if text:
                parts.append(str(text))

        return "\n".join(parts)

    return str(content)


def safe_name(path: Path) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in path.stem
    ).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run page one of a few PDFs through Chandra OCR 2 Q8."
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:18083",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/smoke-test-q8"),
    )
    args = parser.parse_args()

    server = args.server.rstrip("/")
    args.output.mkdir(parents=True, exist_ok=True)

    health = requests.get(f"{server}/health", timeout=15)
    health.raise_for_status()

    print(f"Server health: {health.text}")

    summary = {
        "backend": "llama.cpp",
        "model": "chandra-ocr-2-q8",
        "test_scope": "first page only",
        "documents": [],
    }

    for input_path in args.files:
        input_path = input_path.expanduser().resolve()
        result = {
            "input": str(input_path),
            "status": "pending",
        }

        try:
            if not input_path.is_file():
                raise FileNotFoundError(input_path)

            name = safe_name(input_path)
            output_dir = args.output / name
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nLoading first page: {input_path.name}")

            # Chandra's page range is zero-based, so "0" means page one.
            images = load_file(
                str(input_path),
                {"page_range": "0"},
            )

            if not images:
                raise RuntimeError("No PDF page was rendered")

            image = scale_to_fit(images[0])

            preview_path = output_dir / "input-page-001.png"
            image.save(preview_path)

            request_body = {
                "model": "chandra-ocr-2-q8",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": encode_png(image),
                                },
                            },
                            {
                                "type": "text",
                                "text": OCR_LAYOUT_PROMPT,
                            },
                        ],
                    }
                ],
                "temperature": 0.0,
                "top_p": 0.1,
                "max_tokens": args.max_tokens,
                "stream": False,
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
            }

            print(
                f"Sending {image.width}x{image.height} image...",
                flush=True,
            )

            started = time.perf_counter()

            response = requests.post(
                f"{server}/v1/chat/completions",
                json=request_body,
                timeout=1800,
            )

            elapsed = time.perf_counter() - started

            if not response.ok:
                raise RuntimeError(
                    f"HTTP {response.status_code}: "
                    f"{response.text[:3000]}"
                )

            payload = response.json()
            content = extract_content(payload)

            html_path = output_dir / "page-001.html"
            raw_path = output_dir / "response.json"

            html_path.write_text(content, encoding="utf-8")
            raw_path.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )

            choice = payload.get("choices", [{}])[0]

            result.update(
                {
                    "status": "success",
                    "rendered_width": image.width,
                    "rendered_height": image.height,
                    "elapsed_seconds": round(elapsed, 3),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": payload.get("usage", {}),
                    "timings": payload.get("timings", {}),
                    "preview": str(preview_path),
                    "html": str(html_path),
                    "raw_response": str(raw_path),
                    "output_characters": len(content),
                }
            )

            print(f"Completed in {elapsed:.2f} seconds")
            print(f"Finish reason: {result['finish_reason']}")
            print(f"HTML: {html_path}")

        except Exception as error:
            result.update(
                {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"ERROR: {result['error']}")

        summary["documents"].append(result)

    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
