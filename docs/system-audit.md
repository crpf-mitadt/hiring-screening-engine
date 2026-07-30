# System Audit

This audit covers the resume-to-JD evaluation pipeline on `feature/resume-jd-scoring`
at the stabilization point after commit `20409cb`. It is based on the CLI entry
points, schemas, ingestion, OCR, LLM structuring, scoring, reports, cache logic,
tests, and the historical fixes in `a9e620a`, `a970626`, `d1af885`, `38e9345`,
`0d904f6`, `1619e91`, and `20409cb`.

## Pipeline Stages

| Stage | Entry Points | Input | Output | Source Of Truth | Model Dependency |
| --- | --- | --- | --- | --- | --- |
| JD ingestion | `scripts/extract_jd.py`, `run_match_workflow`, `ingest_jd_document` | `.txt`, `.md`, `.docx`, embedded-text `.pdf` | `IngestedDocument`, `jd-source.txt` | `IngestedDocument.text` | Deterministic |
| JD OCR fallback | `_ingest_jd_stage`, `ocr_pdf_pages` | scanned JD PDF | `jd-ocr/*`, `jd-source.txt` | OCR `document.json` and `source-map.json` | Chandra |
| JD structuring | `structure_jd_document_file`, `_extract_jd_with_gemma` | ingested JD text | `StructuredJD`, `structured-jd.json` | `StructuredJD` after post-processing | Gemma unless `--mock-llm` |
| JD post-processing | `_postprocess_jd_rubric` | `StructuredJD`, raw JD text | normalized `StructuredJD` plus warnings | deterministic rubric normalization | Deterministic |
| Resume OCR | `run_full_resume_ocr.py`, `_ocr_resume_pdf`, `write_ocr_artifacts` | resume PDF | page HTML, page metadata, `document.json`, `source-map.json`, `metadata.json` | OCR metadata contract v2 plus source map | Chandra |
| Resume structuring | `structure_resume_from_ocr`, `_structure_resume_with_gemma` | OCR directory/document/html | `CandidateProfile`, `structured-resume.json` | validated `CandidateProfile` plus `scoring_safe_profile` | Gemma unless `--mock-llm` |
| Safety filtering | `build_scoring_safe_profile` | `CandidateProfile` | `ScoringSafeProfile` | scoring-safe profile | Deterministic |
| Evaluation | `evaluate_candidate_files`, `evaluate_candidate` | `StructuredJD`, `ScoringSafeProfile` | `CandidateEvaluationReport`, `evaluation.json` | canonical report schema | Deterministic |
| Reports | `write_json_report`, `write_csv_report`, `write_html_report`, `_summary_markdown` | `CandidateEvaluationReport` | JSON, CSV, HTML, Markdown summaries | `CandidateEvaluationReport` | Deterministic |

## Canonical Data Flow

`run_match_workflow` orchestrates the full path:

1. Select or receive JD/resume paths.
2. Hash source file content with `_path_identity_hash`.
3. Ingest JD to text.
4. Structure JD and run deterministic JD rubric post-processing.
5. OCR resume PDF or reuse a structured candidate JSON/OCR directory.
6. Structure resume and rebuild the scoring-safe profile.
7. Evaluate from `StructuredJD + ScoringSafeProfile`.
8. Write JSON/CSV/HTML reports and run summaries.
9. Persist a run manifest with source hashes, model names, prompt versions, stage statuses, and artifact paths.

## Stage Inputs And Outputs

| Stage | Input Schema/Object | Output Schema/Object | Validation |
| --- | --- | --- | --- |
| JD structuring | `IngestedDocument.text` | `StructuredJD` | Pydantic strict schemas, weight total, duplicate criterion IDs |
| Resume structuring | `OcrDocument` | `CandidateProfile` | Pydantic strict schemas, null/list normalizers, repair attempts |
| Safety | `CandidateProfile` | `ScoringSafeProfile` | protected personal fields excluded by construction |
| Eligibility | `StructuredJD.mandatory_requirements`, `ScoringSafeProfile` | `EligibilityResult` | requirement status details, shared experience qualification |
| Criterion scoring | `StructuredJD.scoring_rubric`, `EligibilityResult`, evidence index | `CriterionEvaluation` | source-rubric, component, evidence, and arithmetic invariants |
| Completeness | validated `CriterionEvaluation` list | diagnostics dict and score | component totals reconcile by class |
| Reports | `CandidateEvaluationReport` | JSON/CSV/HTML/Markdown | report writers consume structured fields |

## Source Of Truth

- JD semantics: `StructuredJD` after `_postprocess_jd_rubric`.
- Resume scoring facts: `CandidateProfile.scoring_safe_profile`.
- OCR provenance: `source-map.json` and `EvidenceRef` page/block fields.
- Experience duration/relevance: `ExperienceQualification`, produced once by `_experience_qualifications`.
- Criterion semantics: `ScoringCriterion` fields and `component_assessment`.
- Reports: `CandidateEvaluationReport`; reports must not infer meaning from prose.

## Model-Dependent Behavior

- Chandra OCR returns HTML-like layout content per page. `write_ocr_artifacts` records completion/truncation/failure metadata.
- Gemma structures JD and resume JSON. `extract_assistant_text`, `parse_json_object`, and Pydantic validation retain safe metadata and reject empty/invalid content.
- Raw model responses are not persisted unless `PIPELINE_DEBUG_RAW_RESPONSES=true`.

## Deterministic Behavior

- DOCX/TXT/MD/PDF text ingestion.
- JD post-processing classification and warnings.
- Resume date normalization and date recovery.
- Safety filtering.
- Eligibility, matching, component construction, scoring, completeness, gates, reports, and cache keys.

## Caches And Artifact Reuse

`run_match_workflow` stores reusable stage artifacts under `output_root/.cache`.

- JD cache key: source content hash, Gemma model, JD prompt version, mock flag.
- OCR cache key: source content hash, Chandra model, source page count, processed pages, token budgets, retry budget, completion-aware contract, debug raw flag.
- Resume structuring cache key: resume source content hash, Gemma model, resume prompt version, mock flag.

Safe reuse:

- Scoring-code changes can reuse OCR, structured resume, and schema-compatible structured JD.
- JD-structuring changes should regenerate only `structured-jd.json`.
- OCR code or token-budget changes should invalidate OCR-dependent artifacts.
- Partial/truncated OCR cache artifacts are regenerated or blocked unless partial evaluation is explicitly allowed.

## Scoring Paths

The canonical path is additive:

1. Linked requirements.
2. Required/core skills.
3. Preferred skills.
4. Source-declared adjacent skills.
5. Deterministic inferred adjacent skills.
6. Evaluation dimensions.
7. Evidence matching.
8. Component validation.
9. Class-weight policy.
10. Discrete fraction.

Specialized matchers remain for education, qualitative experience, quantitative experience basis,
database controlled equivalents, and soft-skill dimensions. They produce components or validated
requirement details; they must not delete components created by another source.

## Component Classes

- `required/core`: mandatory linked requirements, `required_skill_names`, default required skills.
- `preferred`: `preferred_skill_names`, linked preferred requirements, default preferred skills.
- `adjacent/transferable`: `adjacent_skill_names` and deterministic adjacent inference.
- `evaluation_dimension`: `evaluation_dimensions`.
- Controlled equivalents are criterion-local evidence relationships on an owning component, not a global match.

## Evidence Sources

Evidence enters from:

- OCR page/block references via `evidence_from_block`.
- JD requirements via `EvidenceRef` in synthetic/fallback JD extraction.
- Structured resume fields and synthetic field evidence from `build_candidate_evidence_index`.
- Eligibility requirement details when a linked requirement supplies validated evidence.

Evidence identity is `(source_type, source_document_id, page_number, block_index, normalized_text or exact_text)`.

## Review Gates

- Mandatory eligibility can be `PASS`, `REVIEW`, or `FAIL`.
- Partial OCR forces `REVIEW` and caps completeness unless explicitly allowed.
- Any generated report remains decision support; `human_review_required` remains true.

## Failure Modes

- Unsupported JD/resume format.
- Scanned JD PDF without OCR.
- Chandra truncation, empty content, or invalid response shape.
- Empty OCR after privacy filtering.
- Gemma empty/invalid/truncated JSON.
- Pydantic schema validation errors.
- Duplicate rubric IDs or invalid weights.
- Missing evidence for positive scores.
- Component class/source mismatch.
- Eligibility/criterion contradiction for linked experience.
- Report inconsistency if a writer omits structured component fields.

## Privacy Controls

- `prompt_text_from_ocr` filters emails and phone numbers before resume structuring prompts.
- Resume prompt instructs Gemma not to include personal/protected fields.
- `build_scoring_safe_profile` excludes `personal_information`.
- Raw response persistence is disabled by default.
- `safe_response_metadata` excludes raw assistant content.
- Reports use candidate IDs and scoring-safe evidence only.

## Duplicate Or Legacy Logic

- Historical risk: `_score_soft_skill_criterion` could bypass skills/requirements when dimensions existed. It has been removed.
- `criterion_terms` remains for legacy fallback criteria with no structured fields.
- `_single_component_assessment` remains for education and pure experience fallback paths.
- Report summaries still compress structured fields for human readability, but use `component_assessment`.
- Cache keys are duplicated conceptually across stage helpers, but all live in `match_workflow.py`.

## Audit Table

| Subsystem | Current Source of Truth | Duplicate Logic | Risk | Required Fix | Test Coverage |
| --- | --- | --- | --- | --- | --- |
| CLI/workflow | `run_match_workflow` | scripts call stage functions directly | direct scripts may skip workflow manifest | document direct reruns | CLI help, file evaluation, workflow tests |
| JD ingestion | `IngestedDocument` | PDF OCR fallback in workflow only | direct `extract_jd.py` on scanned PDF fails by design | documented | JD ingestion tests |
| JD structuring | `StructuredJD` after post-processing | Gemma can vary | over-promoted required skills | deterministic post-processing and warnings | JD structuring tests |
| Resume OCR | OCR metadata/source map | raw response loader compatibility | stale/incomplete OCR cache | contract validation and cache key versions | OCR tests |
| Resume structuring | `CandidateProfile` | mock/live paths | model hallucination or null drift | schema validation and repair | resume tests |
| Safety | `ScoringSafeProfile` | model prompt also asks for privacy | scoring on personal info | deterministic sanitizer | safety tests |
| Experience | `ExperienceQualification` | duration summary plus qualification | contradictory month results | shared evaluation date and qualification reuse | experience/scoring tests |
| Eligibility | `EligibilityResult.requirement_details` | matching helpers also used in criteria | linked status disappears | additive component construction and invariants | scoring tests |
| Matching | `match_terms`, `match_requirement` | semantic fallback and aliases | broad false positives | hierarchy and boundary tests | matching/scoring tests |
| Component construction | `_criterion_components` | old soft-skill path removed | fields disappearing in mixed criteria | additive source representation | component matrix tests |
| Scoring | `_weighted_component_fraction` | discrete mapping helper | arbitrary fractions | approved fractions and arithmetic validation | scoring tests |
| Completeness | `evidence_completeness_diagnostics` | none | preferred counted as core | class-specific totals | completeness tests |
| Reports | `CandidateEvaluationReport` | summaries compress fields | prose reconstruction drift | structured summaries by component class | report consistency tests |
| Cache | `_cache_key` and stage helpers | separate file/dir helpers | stale artifact reuse | key includes content/model/prompt/stage | workflow cache tests |
| Privacy | settings + sanitizer + metadata | debug raw flag exists in OCR/resume paths | raw private content leakage | default false and safe metadata | privacy/raw-response tests |

## Historical Commit Findings

- `a9e620a`: made evaluation evidence-backed and tightened OCR/report safety.
- `a970626`: improved OCR completion, resume structuring repair, education and matching boundaries.
- `d1af885`: introduced `ExperienceQualification` and positive-score evidence enforcement.
- `38e9345`: preserved source rubric identity and added component assessments.
- `0d904f6`: protected source rubric names/categories/weights through file evaluation and reports.
- `1619e91`: introduced adjacent skills and criterion-local controlled equivalents.
- `20409cb`: separated preferred evidence, shared experience qualification, conservative soft skills, and Git/GitHub boundaries.

The repeated bugs occurred because those commits fixed local evaluator branches. The latest
stabilization removes the remaining mutually exclusive dimension path and validates additive
component representation.
