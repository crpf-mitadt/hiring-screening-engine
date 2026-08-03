# Architecture

This repository now keeps the original Chandra OCR smoke-test scripts and adds a typed `src/chandra_eval` package for resume-to-job-description evaluation.

The offline flow is:

1. Validate or create a structured candidate profile.
2. Validate or create a structured job description.
3. Build a scoring-safe candidate profile that excludes contact details and protected or non-job-related personal attributes.
4. Score each criterion with deterministic Python arithmetic.
5. Evaluate mandatory eligibility gates.
6. Write JSON, CSV, and HTML reports with `human_review_required=true`.

The Gemma and Chandra clients are configurable wrappers for Tyrone pilots. Unit and integration tests do not call live services.

Gemma structured requests retain the full OpenAI-compatible response until assistant content has
been extracted and validated. Safe error metadata is available for failed responses without logging
private content by default. Raw failed Gemma responses are persisted only when
`PIPELINE_DEBUG_RAW_RESPONSES=true`.

Resume structuring uses nullable optional facts instead of empty strings for genuinely missing
values. When JSON parses but validation fails, the pipeline performs a targeted schema repair using
the parsed JSON and validation paths rather than resending OCR text. `response_format` remains
enabled unless the API explicitly rejects it, and structured requests disable Gemma thinking with
`chat_template_kwargs={"enable_thinking":false}`.

## Pipeline Contract

The stable live contract is:

```bash
python scripts/run_full_resume_ocr.py resume.pdf \
  --output outputs/live/ocr/candidate-001

python scripts/extract_jd.py job.docx \
  --job-id marketing-coordinator \
  --output outputs/live/jd.json

python scripts/structure_resume.py \
  outputs/live/ocr/candidate-001 \
  --candidate-id candidate-001 \
  --output outputs/live/candidate.json

python scripts/evaluate_candidate.py \
  --jd outputs/live/jd.json \
  --resume outputs/live/candidate.json \
  --output outputs/live/evaluation.json

python scripts/build_report.py outputs/live --json --csv --html
```

`--mock-llm` is available only for offline tests and demos. Live commands use Gemma by default.

`run_full_resume_ocr.py` writes:

- `page-001.html`, `page-001-metadata.json`, and subsequent pages
- `document.json`
- `source-map.json`
- `metadata.json`

Raw Chandra response bodies are written only when `PIPELINE_DEBUG_RAW_RESPONSES=true`. Metadata
records `source_page_count`, `processed_page_count`, `processed_page_numbers`,
`completed_page_numbers`, `truncated_page_numbers`, `failed_page_numbers`, `partial`,
`partial_reason`, page-level warnings, retry attempts, and whether raw responses were persisted.
Any page ending with `finish_reason=length` is truncated. The OCR caller retries that page once with
the bounded `CHANDRA_OCR_RETRY_MAX_TOKENS` ceiling; if the retry still truncates, ordinary scoring
stops before resume structuring.

The primary end-to-end command is:

```bash
uv run chandra-eval match
```

It discovers configured JD/resume inputs interactively or accepts `--jd` and `--resume` for
automation. Each run writes a unique match directory containing structured artifacts, reports, and
both JSON and Markdown summaries for later comparison.

For PDF resumes, omitted `--ocr-pages` means all source pages. An explicit page cap is recorded as
partial OCR when fewer than all source pages are processed. A document is complete only when every
requested page completed without truncation or failure; `source_page_count == processed_page_count`
alone is not enough. Partial OCR cannot proceed to ordinary scoring unless explicitly allowed.
