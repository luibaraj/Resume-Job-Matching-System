# src/eval/positive_gen/CLAUDE.md

4-file pipeline: generate → validate → repair → orchestrate. Produces synthetic job listings matching a resume's profile.

## Flow

`positives_gen.py` → `positives_validate.py` → `positives_repair.py` → `positives_pipeline.py`

## Per-File Details

**positives_gen.py**
- Generates `JobSkeleton` TypedDict (7 fields: title, seniority, years_required, domain, primary_skills, secondary_skills, responsibilities)
- Deterministic fields (title, seniority, domain, years) via lookup tables — avoids LLM variance
- Responsibilities: one-at-a-time in retry loop (up to `TARGET_RESPONSIBILITY_COUNT * 4` attempts)
  - Already-generated responsibilities injected into prompt to suppress repetition

**positives_validate.py**
- 4 ordered rule sets; stops at first failure
- Rule set 1 (responsibilities): deterministic
- Rule sets 2–4: LLM-based
- Two cheap deterministic pre-guards (exact seniority match, years ceiling) before expensive LLM calls
- `_normalize_skeleton()`: maps alias strings (e.g., "Mid-level") → canonical enums

**positives_repair.py**
- 2-attempt loop with targeted prompts
- Only fields relevant to failed check are repaired; non-failing fields preserved
- Attempt 2: lower temperature (0.3), stricter format hints, more forceful instruction
- Failing check can shift between attempts (fix may expose different failure)

**positives_pipeline.py**
- Orchestrates gen → validate → repair loop
- Safety cap: `target_count * 10` total attempts (prevents infinite loops)
- Uses `print()` (not logging) for status — intentional for visibility
- Returns list of validated `JobSkeleton` dicts; optionally writes JSON

## Key Non-Obvious
- `JobSkeleton` is a plain dict TypedDict (not Pydantic) — shared via `eval/types.py`
- Repair only overwrites failing fields; `merge_repaired_fields()` is partial merge
