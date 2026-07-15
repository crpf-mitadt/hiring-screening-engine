#!/usr/bin/env python3

import argparse
import json
import math
import random
import textwrap
from collections import defaultdict
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont, ImageOps


def render_first_page(path: Path) -> Image.Image:
    document = pdfium.PdfDocument(str(path))

    if len(document) == 0:
        raise RuntimeError("PDF has no pages")

    page = document[0]
    bitmap = page.render(scale=0.8)
    image = bitmap.to_pil().convert("RGB").copy()

    bitmap.close()
    page.close()
    document.close()

    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("inputs/resumes"),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--per-folder",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/design-candidates"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)

    for path in root.rglob("*.pdf"):
        relative = path.relative_to(root)

        category = (
            relative.parts[0]
            if len(relative.parts) > 1
            else "Uncategorised"
        )

        grouped[category].append(path)

    rng = random.Random(args.seed)

    for files in grouped.values():
        rng.shuffle(files)

    candidates = []

    # Round-robin selection prevents one large folder
    # from dominating the candidate set.
    for position in range(args.per_folder):
        categories = list(grouped)
        rng.shuffle(categories)

        for category in categories:
            files = grouped[category]

            if position < len(files):
                candidates.append(
                    {
                        "category": category,
                        "path": files[position].resolve(),
                    }
                )

            if len(candidates) >= args.count:
                break

        if len(candidates) >= args.count:
            break

    rendered = []

    for candidate in candidates:
        path = candidate["path"]

        print(f"Rendering: {path}", flush=True)

        try:
            image = render_first_page(path)
            candidate["image"] = image
            rendered.append(candidate)

        except Exception as error:
            print(
                f"Skipped {path.name}: "
                f"{type(error).__name__}: {error}"
            )

    columns = 4
    cell_width = 350
    cell_height = 450
    rows = math.ceil(len(rendered) / columns)

    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        "white",
    )

    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    manifest = []

    for number, candidate in enumerate(rendered, start=1):
        index = number - 1
        column = index % columns
        row = index // columns

        x = column * cell_width
        y = row * cell_height

        thumbnail = ImageOps.contain(
            candidate["image"],
            (310, 340),
            Image.Resampling.LANCZOS,
        )

        thumb_x = x + (cell_width - thumbnail.width) // 2
        thumb_y = y + 8

        sheet.paste(thumbnail, (thumb_x, thumb_y))

        draw.rectangle(
            (
                x,
                y,
                x + cell_width - 1,
                y + cell_height - 1,
            ),
            outline="black",
            width=1,
        )

        filename_lines = textwrap.wrap(
            candidate["path"].name,
            width=43,
        )[:3]

        label = (
            f"{number:02d} | {candidate['category']}\n"
            + "\n".join(filename_lines)
        )

        draw.multiline_text(
            (x + 10, y + 355),
            label,
            fill="black",
            font=font,
            spacing=3,
        )

        manifest.append(
            {
                "number": number,
                "category": candidate["category"],
                "path": str(candidate["path"]),
            }
        )

    sheet_path = args.output / "contact-sheet.png"
    manifest_path = args.output / "candidates.json"

    sheet.save(sheet_path)

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Rendered candidates: {len(rendered)}")
    print(f"Contact sheet: {sheet_path}")
    print(f"Candidate list: {manifest_path}")


if __name__ == "__main__":
    main()
