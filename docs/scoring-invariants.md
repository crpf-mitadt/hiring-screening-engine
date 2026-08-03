# Scoring Invariants

This file defines the deterministic policy the scoring code must implement. It is the normative
reference for `src/chandra_eval/scoring`.

## Component Classes

| Class | Source | Weight | Can Satisfy Mandatory Core? |
| --- | --- | ---: | --- |
| `required/core` | `required_skill_names`, linked mandatory requirements, default required skills | 1.0 | Yes |
| `preferred` | `preferred_skill_names`, linked preferred requirements, default preferred skills | 0.5 | No |
| `adjacent/transferable` | `adjacent_skill_names`, deterministic adjacent rules | 0.25 when no required components; one-step bonus when required components exist | No |
| `evaluation_dimension` | `evaluation_dimensions` | 1.0 | Only for that dimension, not for skill gates |

Controlled equivalents are evidence relationships for one owning component. They are not global
aliases and are never exported to unrelated criteria.

## Source Provenance

Source representation and candidate matching are separate concepts. Every populated structured-JD
source field must remain represented even when the candidate has no matching evidence. Canonical
deduplication merges by deterministic source-term identity and keeps one primary scoring class,
but the component must retain every represented source field in `source_fields` and every original
source spelling in `source_terms`.

`represented(T, F)` is true when a canonical-equivalent component exists and `F` is present in that
component's `source_fields`. The check does not depend on `matched`, evidence count, or the
component's primary class. For example, a single `Data-Science` required/core component may carry
`source_fields=["required_skill_names", "adjacent_skill_names"]` and
`source_terms={"adjacent_skill_names": ["Data Science"]}`. That component represents both source
fields but is counted once for score and completeness.

## Matching Hierarchy

1. Exact normalized token/phrase match.
2. Exact alias match from the narrow alias table.
3. Explicit controlled equivalent for the owning component.
4. Source-declared adjacent skill.
5. Deterministic inferred adjacent skill.
6. No match.

Notes must identify the actual relationship. Levels 2-5 must not be described as level 1.

Boundary rules:

- Plain Git is not GitHub or GitLab.
- GitHub is not GitLab.
- GitHub may match `GitHub` or `github.com/...`.
- Broad Git requirements may be supported by GitHub/GitLab evidence, but the reverse is not true.
- Java and JavaScript are distinct.
- C and C++ are distinct.
- React and React Native are distinct unless the JD explicitly accepts that relationship.
- SQL/MySQL/PostgreSQL equivalence is controlled and criterion-local.
- Generic AI/ML does not satisfy a specific model/framework unless explicitly named or controlled.
- React/React.js, Node/Node.js, and Express/Express.js are controlled only for the technical
  breadth dimension and source-provenance canonicalization.
- MySQL/PostgreSQL are controlled only by database-equivalent rules owned by the database component.

## Linked Requirements

Every `requirement_id` listed by a criterion must be represented in `component_assessment`.

- Linked mandatory skill requirements produce `required/core` components unless explicitly gate-only.
- Linked preferred requirements produce `preferred` components.
- Linked experience requirements reuse the shared `ExperienceQualification`.
- A met linked requirement cannot disappear from a criterion.
- A missing linked requirement cannot become supported through adjacent evidence.
- Eligibility and criterion status cannot contradict the same threshold.

## Evidence Requirements

- Every matched component has evidence.
- Every positive criterion has evidence.
- Criterion evidence equals the deduplicated union of matched component evidence.
- A heuristic, alias, trigger, or shared relationship without source-backed evidence is not a match.
- Every evidence item must be in scope for that criterion/component.
- Duplicated source blocks are merged by stable source identity.
- One source block can support multiple facts only when each fact has its own component-level rationale.

## Evaluation Dimensions

Dimension labels are semantic, not category-only. The domain is resolved from explicit dimension
metadata when present, then linked requirement IDs/categories, then criterion name and known
dimension terms, and finally the criterion category fallback.

`Tech stack versatility` uses deterministic breadth sharing. It is supported when at least two
distinct supported technical components normalize to this set: `React.js`, `Node.js`, and
`Express`/`Express.js`. The dimension reuses the deduplicated evidence from those supported
components and records the component display terms in `shared_evidence_with`.

## Scoring Policy

`match_fraction` must be one of `0`, `0.25`, `0.5`, `0.75`, `1`.

When required/evaluation components exist:

- Required and evaluation components form the primary fraction.
- Preferred components contribute at lower impact.
- Adjacent evidence can add at most one discrete step.
- Missing required/core components cap the final fraction below a full match unless all required-like components are met.

When only preferred/adjacent components exist:

- Preferred evidence can earn criterion credit.
- Missing preferred evidence does not create a core gap.
- Adjacent-only evidence can earn limited credit but cannot satisfy a mandatory gate.

Weighted score is always:

```text
criterion.weight * criterion.match_fraction
```

## Mandatory Gates

Eligibility is independent of weighted score:

- `PASS`: all deterministic mandatory checks passed.
- `REVIEW`: mandatory facts are unclear, missing, partial, or affected by allowed partial OCR.
- `FAIL`: a mandatory threshold is definitively below requirement.

The final output remains decision support. Human review is always required.

## Evidence Completeness

Completeness is calculated after criterion validation and evidence deduplication.

Diagnostics must include:

- required/core total and supported;
- preferred total and supported;
- adjacent total and supported;
- controlled-equivalent total and supported;
- evaluation-dimension total and supported;
- deduplicated source count;
- date score;
- OCR score;
- penalties.

Preferred gaps do not increase missing-core penalties. Adjacent components contribute at reduced
weight. Duplicate evidence does not inflate completeness.

## Strengths And Gaps

Strengths come from matched components with evidence:

- `required/core: <criterion>: <component>`
- `preferred: <criterion>: <component>`
- `adjacent: <criterion>: <component>`

Gaps come from structured missing arrays:

- `mandatory/core gap: <criterion>: <component>`
- `preferred gap: <criterion>: <component>`
- `unverified soft-skill dimension: <criterion>: <dimension>`

Reports must not reconstruct these classes from explanation prose.

## Worked Synthetic Examples

### 1. Exact Strong Match

JD requires Python and SQL. Resume has Python and SQL evidence. Both components are `required/core`,
both match, fraction is `1.0`, eligibility can pass if mandatory requirements are also met.

### 2. Experience Met But Core Technologies Missing

JD requires 24 months and PHP/Laravel/WordPress. Resume has 30 relevant months but no PHP stack.
The experience component matches through the shared qualification. Missing stack components remain
core gaps.

### 3. Database Equivalent Plus Adjacent Full-Stack Skills

JD accepts MySQL-like enterprise databases. Resume has PostgreSQL plus JavaScript/React. PostgreSQL
is criterion-local controlled-equivalent evidence for the database component. JavaScript/React are
adjacent and cannot replace PHP/Laravel/WordPress.

### 4. Most Preferred Skills Met But One Missing

JD preferred skills are GitHub, SEO, Google Analytics. Resume has GitHub and SEO. Supported preferred
is 2/3; the missing preferred item is not a core gap.

### 5. Strong Core Skills With No Soft-Skill Evidence

Core technical criteria can score well. A separate dimension criterion for communication/ownership
does not match unless source evidence contains dimension-specific wording.

### 6. Partial OCR

If OCR processed fewer pages or any page is truncated/failed, ordinary scoring is blocked unless
partial evaluation is explicitly allowed. Allowed partial evaluation forces `REVIEW` and reduces
completeness.

### 7. Uncertain Dates

Malformed dates remain uncertain. Confirmed months are calculated from parseable ranges only.
Uncertain possible months do not silently satisfy a quantitative threshold.

### 8. Mandatory Requirement Not Found

A missing mandatory skill marks the requirement `not_found` or `unclear`, keeps human review
required, and appears as a core gap only for criteria that own that component.

### 9. Mixed Criterion With All Component Sources

A criterion with linked DevOps, required Python, preferred GitHub, adjacent JavaScript, and
Communication dimension must contain five separate components. Adding the dimension cannot remove
Python, GitHub, JavaScript, or the linked requirement.
