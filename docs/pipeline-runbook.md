# Pipeline Runbook

This runbook uses versioned output filenames where possible and avoids raw model response
persistence by default.

## Services

Start or verify:

```bash
curl "$CHANDRA_BASE_URL/v1/models"
curl "$GEMMA_BASE_URL/models"
```

Useful environment variables:

```bash
export PIPELINE_DEBUG_RAW_RESPONSES=false
export GEMMA_RESUME_MAX_TOKENS=8192
export CHANDRA_OCR_RETRY_MAX_TOKENS=8192
```

## Full Match

```bash
uv run chandra-eval match \
  --jd "$JD_PATH" \
  --resume "$RESUME_PATH" \
  --candidate-id "$CANDIDATE_ID" \
  --job-id "$JOB_ID" \
  --output-root "$OUTPUT_ROOT"
```

## OCR Only

```bash
uv run python scripts/run_full_resume_ocr.py \
  "$RESUME_PDF" \
  --output "$RUN_DIR/ocr"
```

Inspect OCR:

```bash
uv run python -m json.tool "$RUN_DIR/ocr/metadata.json"
rg -n "partial|truncated|failed|raw_responses_persisted" "$RUN_DIR/ocr/metadata.json"
```

## JD-Only Regeneration

Use a versioned output file to preserve the previous structured JD:

```bash
PIPELINE_DEBUG_RAW_RESPONSES=false uv run python scripts/extract_jd.py \
  "$JD_SOURCE" \
  --job-id "$JOB_ID" \
  --output "$RUN_DIR/structured-jd-$(date +%Y%m%d-%H%M%S).json"
```

## Resume Structuring Only

Reuse an existing OCR directory:

```bash
PIPELINE_DEBUG_RAW_RESPONSES=false uv run python scripts/structure_resume.py \
  "$RUN_DIR/ocr" \
  --candidate-id "$CANDIDATE_ID" \
  --output "$RUN_DIR/structured-resume-$(date +%Y%m%d-%H%M%S).json"
```

## Scoring Only

Reuse an existing structured resume and structured JD:

```bash
uv run python scripts/evaluate_candidate.py \
  --candidate "$STRUCTURED_RESUME" \
  --jd "$STRUCTURED_JD" \
  --output "$RUN_DIR/evaluation-$(date +%Y%m%d-%H%M%S).json"
```

## Report Only

```bash
uv run python scripts/build_report.py \
  "$EVALUATION_JSON" \
  --output "$REPORT_DIR" \
  --json --csv --html
```

## Cache Guidance

The workflow cache lives under `$OUTPUT_ROOT/.cache`.

- Use `--no-cache` to prove a clean run.
- Use `--force` to regenerate all stages.
- Reuse OCR and structured resume for scoring-only changes.
- Regenerate only `structured-jd.json` after JD prompt/post-processing changes.
- Regenerate OCR after OCR code, page count, token budget, or source PDF changes.

Do not manually copy cache artifacts into a run unless their source hash, model, prompt/schema
version, and stage parameters match.

## Artifact Inspection Order

1. `run-manifest.json` and `run-summary.json`.
2. OCR `metadata.json`.
3. OCR `source-map.json`.
4. `structured-resume.json`.
5. `structured-jd.json`.
6. `evaluation.json`.
7. `evaluation.csv` and `evaluation.html`.

## Common Errors

- `ScannedPdfRequiresOcrError`: JD PDF has too little embedded text; OCR it first through workflow.
- `finish_reason=length`: OCR or Gemma response was truncated.
- `OCR content is empty after safety filtering`: OCR contained no usable scoring-safe text.
- `scoring_rubric weights must total 100`: JD rubric needs regeneration or manual source correction.
- `weighted_score > 0 requires at least one valid evidence item`: scoring invariant blocked unsafe output.
- `source=preferred_skill_names implies class=preferred`: component class invariant blocked drift.

## Reading `component_assessment`

Start with:

```bash
uv run python -m json.tool "$EVALUATION_JSON" | rg -n "criterion_id|component_assessment|required_core_components|preferred_components|source_declared_adjacent_components|inferred_adjacent_components|evaluation_dimension_components|supported_|missing_"
```

Interpretation:

- `required_core_components`: mandatory/core denominator.
- `preferred_components`: lower-impact preferred denominator.
- `source_declared_adjacent_components`: adjacent skills from `StructuredJD`.
- `inferred_adjacent_components`: deterministic scorer-added adjacent evidence.
- `evaluation_dimension_components`: dimension-specific soft-skill checks.
- `components[].source`: which source field created the component.
- `components[].classification`: contribution class.
- `components[].evidence`: source-backed support.

## Eligibility Versus Score

Eligibility gates mandatory requirements. Score ranks evidence against weighted rubric criteria.
A candidate can have a positive score and still require review or fail a mandatory gate.

## Completeness

Completeness is not a score. It estimates how much source-backed evidence supports the evaluation,
after deduplication and OCR/date penalties.

## Privacy Checks

```bash
rg -n "raw_responses_persisted|raw-response|email|phone|address|personal_information" "$RUN_DIR"
```

Raw model content should not exist unless `PIPELINE_DEBUG_RAW_RESPONSES=true`.

## Compare Two Evaluations

```bash
uv run python -m json.tool "$OLD_EVALUATION" > /tmp/old-evaluation.pretty.json
uv run python -m json.tool "$NEW_EVALUATION" > /tmp/new-evaluation.pretty.json
diff -u /tmp/old-evaluation.pretty.json /tmp/new-evaluation.pretty.json
```

## Invariant Validation

```bash
uv run python scripts/validate_evaluation_invariants.py \
  --evaluation "$EVALUATION_JSON" \
  --jd "$STRUCTURED_JD" \
  --structured-resume "$STRUCTURED_RESUME" \
  --ocr-metadata "$OCR_METADATA" \
  --csv-report "$EVALUATION_CSV" \
  --html-report "$EVALUATION_HTML"
```

Every line should print `PASS`. Any `FAIL` should be investigated before interpreting the score.

## Safe New JD/Resume Test

1. Run a mock synthetic match first.
2. Run live OCR only if needed.
3. Structure resume with raw responses disabled.
4. Structure JD to a versioned file.
5. Score only from structured JSON artifacts.
6. Run invariant validation.
7. Review evidence and component classes manually.
