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

Resume and JD structuring requests keep OpenAI-compatible `response_format={"type":"json_object"}`
enabled by default and send `chat_template_kwargs={"enable_thinking":false}`. The code removes
`response_format` only when the Gemma API explicitly rejects that request feature.

Resume extraction preserves the OCR output directory as the reusable handoff between Chandra OCR
and Gemma structuring. If structuring fails, rerun `scripts/structure_resume.py` against the existing
OCR directory; another live OCR run is not required.

The resume schema allows `null` for optional facts such as unknown degree fields, institutions,
titles, organizations, domains, and employment type. Prompts instruct Gemma to use `null` rather
than inventing values or writing empty strings. Required IDs, enum values, evidence shapes, and list
item types still fail validation.

When a Gemma response parses as JSON but fails Pydantic validation, the retry is a targeted repair:
it sends the parsed JSON and concise validation paths back to Gemma, preserves unrelated fields, and
does not resend OCR text. Empty assistant content is retried with `response_format` still enabled.
`finish_reason=length` is reported as a truncation-specific failure instead of being folded into
generic malformed JSON repair.

`GEMMA_RESUME_MAX_TOKENS` defaults to `8192`. Keep that default unless a final response is proven to
be truncated; the July 2026 live failure mode was nullable-field validation plus thinking/schema
fallback behavior, not an 8192-token budget failure.

Safe failure metadata is always written to `resume-structuring-error-metadata.json` beside the
requested candidate output. It includes model, finish reason, usage, message keys, content and
reasoning content types and lengths, response-format status, requested max tokens, elapsed seconds,
validation paths, and retry classification. Private raw model content is written only when:

```bash
export PIPELINE_DEBUG_RAW_RESPONSES=true
```

With that opt-in, failed resume structuring also writes
`structured-resume-gemma-response.json` in the same ignored output directory.

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

## Unified Match Command

The preferred live workflow is:

```bash
uv run chandra-eval match
```

or directly:

```bash
uv run chandra-eval match \
  --jd "inputs/jds/Job description - Marketing Coordinator.docx" \
  --resume "inputs/resumes/Marketing/Maheshwari Thakur Resume Marketing.pdf" \
  --candidate-id candidate-marketing-001 \
  --job-id marketing-coordinator
```

The command performs service preflight, JD ingestion, resume OCR when needed, structuring,
evaluation, report generation, run-summary generation, and a concise terminal summary. It reuses
cached artifacts only when source content hashes and relevant model/prompt settings match.

Use `--force` to regenerate all stages, `--no-cache` to disable reuse, `--list-jds` or
`--list-resumes` to inspect configured input directories, and `--search` to filter list output.
For resume PDFs, omit `--ocr-pages` to process every page. Use `--ocr-pages N` only as an explicit
cap. A cap that processes fewer than all source pages is marked partial and stops before scoring
unless `--allow-partial-evaluation` is passed.

Output layout:

```text
outputs/matches/<job-id>/<candidate-id>/<timestamp>/
  run-manifest.json
  run-summary.json
  run-summary.md
  structured-jd.json
  structured-resume.json
  evaluation.json
  evaluation.csv
  evaluation.html
  ocr/
```

The terminal and summary outputs do not include raw prompts, raw model responses, email addresses,
phone numbers, or home addresses. They retain the decision-support disclaimer and
`human_review_required=true`.

For the July 29, 2026 Tyrone retest of the Research Assistant AI CoE pair, avoid the incomplete
one-page cache:

```bash
PIPELINE_DEBUG_RAW_RESPONSES=false \
uv run chandra-eval match \
  --jd "inputs/jds/Job description - Research Assistant AI CoE.docx" \
  --resume "inputs/resumes/VijaySamuelHeHimResume.pdf" \
  --output-root outputs/matches \
  --force
```

Do not add `--ocr-pages` for that retest; all resume pages must be processed.
