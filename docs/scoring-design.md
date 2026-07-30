# Scoring Design

The first scoring slice is deliberately deterministic. Gemma may be used later for structuring and criterion-level assessment, but final arithmetic is Python-owned.

The report preserves:

- criterion-level match fractions
- weighted criterion scores
- score before gates
- final score out of 100
- failed mandatory requirements
- missing or ambiguous information
- evidence completeness
- a recommendation label for human review only

Eligibility can be `PASS`, `REVIEW`, or `FAIL`. A `FAIL` does not delete the candidate, suppress the score, or make a final hiring decision.

Status semantics:

- `met`: affirmative resume evidence supports the requirement or criterion.
- `partially_met`: some required parts are supported and other parts are missing or unclear.
- `not_found`: the resume provides no evidence either way.
- `unclear`: evidence exists but is ambiguous, incomplete, or affected by extraction uncertainty.
- `not_met`: deterministic arithmetic or explicit evidence shows the requirement is below the minimum.
- `not_applicable`: the criterion has no applicable evidence terms.

Missing evidence is not a hard failure by itself. Mandatory eligibility uses:

- `PASS`: all mandatory requirements are supported.
- `REVIEW`: at least one mandatory requirement is unclear, partially supported, or not found, with no definitive failure.
- `FAIL`: at least one mandatory requirement is clearly contradicted or definitively below a required threshold.

Experience duration is overlap-adjusted. Confirmed months are calculated only from independently
parseable ranges. Malformed dates such as a truncated year are preserved as uncertain ranges and do
not erase duration from valid entries. Quantitative experience scoring must also attach source
evidence for the qualifying role, organization when available, dates, and role relevance. Aggregate
months alone cannot produce a positive score or a met mandatory requirement.

Criterion matching uses the scoring-safe profile only. It searches summary, education, experience
titles and responsibilities, skills used, top-level skills, projects, publications, teaching,
research, leadership, and achievements. Exact/normalized matches are combined with a narrow alias
map for auditable business terms such as MS Office, CRM, lead generation, and campaign management.
Term matching uses token or phrase boundaries. Short technical terms such as Git, R, C, Go, ML, AI,
AWS, GCP, and SQL do not match inside unrelated words. `Git` may match `GitHub` only through the
explicit alias map.

Database requirements use controlled alternatives. `MySQL or similar enterprise databases` can match
direct MySQL evidence or PostgreSQL evidence, but `MySQL only` remains strict. Generic terms such as
data, storage, backend, or database without a named controlled technology are conservative.

Multi-component technical criteria are scored by independently matching each component, then mapping
the supported component fraction to the approved discrete scale. For four required components, one,
two, three, and four supported components map to `0.25`, `0.50`, `0.75`, and `1.00`.
Rubric criterion identity is field-owned, not position-owned: each result is built from the
structured JD criterion's `criterion_id`, `name`, `category`, `weight`, `requirement_ids`,
`required_skill_names`, `preferred_skill_names`, and `evaluation_dimensions`. A linked requirement's
validated evidence may flow only to criteria that list that `requirement_id`.
File-based evaluation uses the loaded `StructuredJD` object directly. Existing non-empty
`scoring_rubric` entries are not renamed, regenerated, reordered, or replaced by compatibility
templates. Runtime validation compares each result with the source criterion, and component
assessment metadata includes a stable hash of the source criterion definition.
JD structuring post-processing checks rubric quality before writing `structured-jd.json`: mandatory
requirement links are preserved or warned, required/core skills must trace to mandatory wording or a
linked mandatory requirement, and contextual/helpful technologies are moved to `adjacent_skill_names`
rather than being accepted as required/core.

Technical-stack criteria distinguish:

- required/core components, such as PHP, Laravel, WordPress, and SQL/database management;
- controlled-equivalent evidence, such as PostgreSQL for a requirement that explicitly accepts
  MySQL-like enterprise databases;
- adjacent or transferable full-stack components, such as JavaScript, React.js, Node.js, Express.js,
  Vue.js, REST APIs, frontend development, and backend development.

Adjacent full-stack evidence can add at most one discrete scoring step and cannot replace a missing
mandatory component. Adjacent-only evidence is capped at `0.25`; direct core evidence plus adjacent
evidence may increase one step but remains capped at `0.75` while any core component is missing.
Reports label these classes separately.
The canonical structured field for this is `component_assessment`. It includes
`required_core_components`, `controlled_equivalent_components`, `adjacent_components_considered`,
`supported_core_components`, `supported_controlled_equivalents`, `supported_adjacent_components`,
`missing_core_components`, per-component source/origin, per-component evidence references, and the
source criterion hash. Skill and leadership criteria must populate this object.
Controlled-equivalent metadata is criterion-local: MySQL/PostgreSQL appears only when a criterion
owns a database component through required skills, a linked database requirement, or its own explicit
database definition. Criteria without database components have empty controlled-equivalent fields
even if the candidate has database evidence elsewhere.

Mixed criteria, such as experience plus SQL/MySQL, are split into independently evaluated
components. Experience duration/relevance, SQL, MySQL, and any controlled database equivalent each
carry their own source, classification, matched status, evidence, explanation, and missing state.
The criterion score is computed from those independent components before mapping to the approved
discrete fractions.

Soft-skill criteria use only their own `evaluation_dimensions`. Technical tools such as React.js,
JavaScript, HTML, CSS, and databases do not directly satisfy communication, ownership, or
proactivity. Additional technical-skill criteria use their own `preferred_skill_names`; soft-skill
terms do not migrate into them.

Degree requirements are source-constrained: they may match only structured education entries and
their evidence. Experience paragraphs, job titles, generic ML/AI mentions, leadership text, company
names, and technical operations text cannot become degree evidence.
Field-restricted degree requirements evaluate level and field separately. A master's degree in an
unrelated field, executive education, certificates, leadership programs, and institution names do
not satisfy a Computer Science, Artificial Intelligence, Data Science, or controlled closely related
technical-field requirement.

Qualitative experience requirements without a numeric duration are assessed from direct role,
internship, project, research, or publication evidence. Numeric duration logic runs only when the JD
specifies a minimum value. Generic operational references to AI/ML do not by themselves satisfy
hands-on or research experience requirements.

`accepted_evidence` describes later verification sources such as interviews, coding tests,
transcripts, resumes, portfolios, or publication lists. It is not a candidate qualification and is
not included in match terms, missing qualifications, gaps, or match-fraction denominators. These
items appear as `verification_needed`.

Final criterion `match_fraction` values are discrete: `0.00`, `0.25`, `0.50`, `0.75`, or `1.00`.
Weighted scores are `criterion.weight * match_fraction`; Python owns this arithmetic.
Any positive `weighted_score` must carry at least one valid evidence reference.

Evidence completeness is deterministic:

1. Compute the fraction of evaluated criteria with at least one supported or partially supported,
   criterion-relevant resume evidence item after criterion identity validation, linked-requirement
   propagation, required/adjacent classification, and evidence deduplication.
2. If that fraction is zero, completeness is zero.
3. Otherwise, compute `criteria_fraction * 0.70 + date_score * 0.15 + ocr_score * 0.15`.
4. `date_score` is `1.0` when experience dates are complete and `0.5` when date warnings exist.
5. `ocr_score` is `1.0` only when every requested OCR page completed without truncation or failure;
   it is `0.5` for explicitly allowed partial OCR.

Evidence is deduplicated by stable source identity: source type, source document ID, page number,
block index, and normalized or exact text. If one source block supports multiple facts, the evidence
object appears once and its notes retain the matched facts. The report also includes completeness
diagnostics for required/core component counts, adjacent component counts, deduplicated source count,
and penalties.
Adjacent components contribute at reduced weight in completeness diagnostics. Per-criterion
diagnostics show total and supported required/core components, adjacent components, controlled
equivalent components, supported controlled equivalents, and deduplicated source counts.
