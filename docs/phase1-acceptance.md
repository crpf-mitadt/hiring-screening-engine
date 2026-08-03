# Phase 1 Acceptance

Phase 1 is the deterministic backend foundation for resume-to-JD evaluation. It does not build a
platform product, but it leaves stable Python service boundaries that a future platform can call.

## Scope

Phase 1 guarantees:

- scoring consumes validated structured JD and structured resume artifacts;
- LLMs extract structure but do not decide scores;
- score-producing rules are deterministic and evidence-backed;
- source JD provenance is preserved independently of candidate matches;
- reports are projections of `CandidateEvaluationReport`;
- shared-JD candidate comparison records structured-JD hashes;
- raw OCR and LLM response persistence is disabled by default;
- runtime and artifact validators enforce the same source-representation semantics.

Phase 1 deliberately does not guarantee:

- bug-free operation on every future JD shape;
- frontend workflows, accounts, billing, databases, queues, APIs, or deployment;
- automated hiring decisions;
- broad fuzzy matching outside documented controlled equivalences.

## Canonical Architecture

The architecture is:

1. JD/resume source documents are ingested.
2. OCR and LLM stages produce structured artifacts.
3. Pydantic schemas validate those artifacts.
4. Deterministic scoring builds canonical components.
5. Components are finalized so `matched=true` requires valid source-backed evidence.
6. `CandidateEvaluationReport` is persisted.
7. JSON, CSV, HTML, Markdown summaries, invariant validation, and matrix summaries are generated
   from the canonical evaluation artifact.

Stable Python entry points:

- structure JD: `chandra_eval.pipeline.jd_structuring`
- structure resume from OCR: `chandra_eval.pipeline.resume_structuring`
- evaluate candidate: `chandra_eval.scoring.engine.evaluate_candidate`
- evaluate files: `chandra_eval.pipeline.candidate_evaluation.evaluate_candidate_files`
- generate reports: `chandra_eval.reports`
- validate evaluation: `scripts/validate_evaluation_invariants.py`
- evaluate matrix: `scripts/validate_match_matrix.py`

## Artifact Versions

- `StructuredJD.schema_version`: `structured-jd-v1`
- `CandidateProfile.schema_version`: `candidate-profile-v1`
- `ScoringSafeProfile.schema_version`: `scoring-safe-profile-v1`
- `CandidateEvaluationReport.artifact_schema_version`: `candidate-evaluation-v1`
- matrix manifest: `match-matrix-manifest-v1`
- matrix comparison output: `match-matrix-comparison-v1`
- Phase 1 acceptance output: `phase1-acceptance-v1`

Unsupported explicit versions fail validation. Missing versions use current defaults for existing
repository and live artifacts.

## Invariants

The canonical source provenance rule is:

`represented(term, source_field)` is true when a canonical-equivalent component exists and
`source_field` is present in `component.source_fields`.

Representation does not depend on candidate evidence, `matched`, or primary component class. One
canonical component may represent required, preferred, adjacent, linked, and dimension source
fields, but it is counted once for score and completeness.

The evidence rule is:

`matched=true` requires non-empty valid evidence with source provenance. Criterion evidence must
equal the deduplicated union of matched component evidence. Evidence-less triggers, aliases, or
shared relationships remain unmatched.

## Controlled Equivalence

Controlled equivalences are narrow:

- React/React.js, Node/Node.js, and Express/Express.js for technical breadth provenance and the
  `Tech stack versatility` rule.
- MySQL/PostgreSQL/SQL only inside database-equivalent rules owned by a database component.
- Git, GitHub, and GitLab keep explicit boundaries.
- Java and JavaScript remain distinct.
- Generic AI/ML does not satisfy arbitrary specific academic or technical terms.

## Test Strategy

The maintained acceptance command is:

```bash
uv run python scripts/run_phase1_acceptance.py --output /tmp/chandra-phase1-acceptance --force
```

It runs synthetic strong, partial, weak, experience-only, ambiguous-date, partial-OCR, soft-skill,
evidence-less-trigger, Data Science provenance, report/privacy, mixed matrix, shared-JD hash, and
determinism checks. The full unit/integration suite remains the primary exhaustive regression
suite:

```bash
uv run python -m unittest discover -s tests -v
```

## Matrix Workflow

`scripts/validate_match_matrix.py` consumes a JSON manifest with:

- `schema_version`
- `pairs[].test_name`
- `pairs[].jd_group`
- `pairs[].job_id`
- `pairs[].candidate_id`
- `pairs[].structured_jd`
- `pairs[].structured_resume`
- optional `pairs[].ocr_metadata`

The runner copies each shared structured JD once per group, records hashes, evaluates every pair,
continues after failures, emits JSON/CSV/TSV comparison files, and exits non-zero if any row is not
`PASS`.

## Debugging Workflow

For a failed evaluation:

1. Re-run scoring from existing `structured-jd.json` and `structured-resume.json`.
2. Validate with `scripts/validate_evaluation_invariants.py` plus CSV, HTML, and OCR metadata.
3. Inspect invariant diagnostics. Source-representation failures include criterion ID, source field,
   original term, canonical term, and safe component metadata, not private evidence text.
4. Enable raw response persistence only with explicit debug settings and only for controlled local
   debugging.

## Known Limitations

The scorer is deterministic but intentionally conservative. It does not infer broad semantic
equivalence beyond documented rules. Human review is always required. Live model extraction can
still produce unusual structured rubrics, so validators fail closed before invalid evaluations are
accepted.

## Phase 2 Roadmap

Phase 2 may add persistent storage, organization/user accounts, role-based access, resumable
background jobs, an API layer, workers, artifact storage, evaluation history, comparison dashboards,
human-review workflows, comments/overrides, audit logs, configurable scoring policies,
model/provider configuration, deployment, and monitoring. Those are platform layers on top of the
Phase 1 Python service boundaries; they are not implemented here.

## Final Tyrone Acceptance

Phase 1 is complete when the repository checks and Phase 1 acceptance runner pass locally and the
final Tyrone command validates Chetan plus the shared-JD Sujay/Sunil matrix from existing structured
artifacts without OCR or Gemma regeneration.
