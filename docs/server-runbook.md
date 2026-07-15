# Server Runbook

Local Mac development should run offline with synthetic fixtures:

```bash
uv sync --frozen
uv run python -m unittest discover -s tests -v
uv run python scripts/run_matching_pilot.py
```

Tyrone live validation should happen only on the server where local endpoints are reachable.

Verify Gemma:

```bash
curl -s \
  -H "Authorization: Bearer no-key" \
  http://127.0.0.1:3001/v1/models \
  | python -m json.tool
```

```bash
curl -s \
  http://127.0.0.1:3001/v1/chat/completions \
  -H "Authorization: Bearer no-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4-12b",
    "messages": [
      {
        "role": "user",
        "content": "Reply with only the word OK."
      }
    ],
    "temperature": 0,
    "max_tokens": 10
  }' \
  | python -m json.tool
```

The direct Python client sends `gemma4-12b`, not `openai:gemma4-12b`, unless `/v1/models` explicitly reports the provider-prefixed ID.

## One JD / One Resume Flow

Use environment variables or `.env` values, not hardcoded paths:

```bash
export CHANDRA_BASE_URL=http://127.0.0.1:18083
export CHANDRA_MODEL=chandra-ocr-2-q8
export GEMMA_BASE_URL=http://127.0.0.1:3001/v1
export GEMMA_MODEL=gemma4-12b
export GEMMA_API_KEY=no-key
```

Run the live pipeline:

```bash
uv run python scripts/run_full_resume_ocr.py resume.pdf \
  --output outputs/live/ocr/candidate-001
```

```bash
uv run python scripts/extract_jd.py job.docx \
  --job-id marketing-coordinator \
  --output outputs/live/jd.json
```

```bash
uv run python scripts/structure_resume.py \
  outputs/live/ocr/candidate-001 \
  --candidate-id candidate-001 \
  --output outputs/live/candidate.json
```

```bash
uv run python scripts/evaluate_candidate.py \
  --jd outputs/live/jd.json \
  --resume outputs/live/candidate.json \
  --output outputs/live/evaluation.json
```

```bash
uv run python scripts/build_report.py outputs/live --json --csv --html
```

Generated OCR and report outputs are private and remain under ignored `outputs/` paths.
