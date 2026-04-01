# src/eval/negative_gen/CLAUDE.md

3-file pipeline: generate → validate → repair. Produces intentionally mismatched synthetic jobs (negative examples for eval).

## Flow

`negatives_gen.py` → `negatives_validate.py` → `negatives_repair.py`
(No separate orchestrator; inline loop in `scripts/eval/generate_synthetic_negatives.py`)

## Mismatch Types

- **"seniority"**: wrong level (e.g., Senior job for Junior candidate)
  - `years_required` forcibly set to target seniority's bracket (not clamped to candidate years) — ensures unambiguous mismatch
- **"responsibility"**: right seniority/domain, but responsibilities describe unfamiliar sub-role
  - Skills/domain intentionally overlap (to test fine-grained ranking)

## Per-File Details

**negatives_gen.py**
- Returns `(skeleton, mismatch_context)` tuple — context must survive through repair
- Reuses `positives_gen` internals: `_generate_deterministic_fields()`, `_generate_skills()`, etc.

**negatives_validate.py**
- Reuses positives rule sets 1–2 (responsibilities, structural)
- Adds 3 new checks (only for mismatches):
  - `validate_seniority_mismatch()`: deterministic, checks gap ≥ min required
  - `validate_skill_domain_overlap()`: LLM, checks ≥2 shared skills + domain alignment (explicitly ignores seniority)
  - `validate_responsibility_mismatch()`: LLM, checks responsibilities describe work candidate doesn't do
- `_MIN_MISMATCH_GAP` table: asymmetric (Junior requires ≥2 gap; Senior requires ≥1)

**negatives_repair.py**
- Same structure as `positives_repair.py`
- Critical: `mismatch_context` passed through to prevent repair from "fixing" the intentional mismatch
- Backwards-compat aliases (`_format_fields_for_prompt`, `_merge_repaired_fields`) for test imports — do NOT remove

## Key Non-Obvious
- Intentional mismatch must be preserved through repair, not "corrected"
- LLM prompts explicitly tell model to ignore seniority when validating skill overlap
