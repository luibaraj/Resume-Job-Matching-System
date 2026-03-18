"""NIAH evaluation pipeline for the retrieval layer."""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.database import DatabaseManager

if TYPE_CHECKING:
    from src.config import Config


GOLDEN_NEEDLE_ID = -1        # sentinel job_id for the golden needle
ADVERSARIAL_NEEDLE_ID = -2   # sentinel job_id for the adversarial needle
RELEVANCE_THRESHOLD = 2      # ≥2 counts as "relevant" for Precision@K
TARGET_RECALL_AT_K = 0.95
TARGET_MRR = 0.80
TARGET_NDCG_AT_K = 0.85
TARGET_PRECISION_AT_K = 0.70
PRECISION_AT_K = 20
JUDGE_MAX_WORKERS = 5
JUDGE_RETRY_ATTEMPTS = 3
JUDGE_RETRY_DELAY = 2.0      # seconds


@dataclass
class SyntheticNeedle:
    needle_id: int            # GOLDEN_NEEDLE_ID or ADVERSARIAL_NEEDLE_ID
    needle_type: str          # "golden" or "adversarial"
    title: str
    company: str
    description: str
    deal_breaker: Optional[str]   # None for golden; one-sentence description for adversarial
    true_relevance: int       # 5 for golden, 0 for adversarial


@dataclass
class EvalCase:
    resume_id: str            # slug, e.g. "user_profile"
    resume_text: str
    golden: SyntheticNeedle
    adversarial: SyntheticNeedle


@dataclass
class RetrievedItem:
    job_id: int
    rank: int                 # 1-based
    rrf_score: float
    rerank_score: float
    title: str
    description: str
    is_needle: bool
    needle_type: Optional[str]    # "golden" | "adversarial" | None


@dataclass
class JudgedItem:
    job_id: int
    rank: int
    relevance_score: int      # 0–5
    judge_reasoning: str
    is_needle: bool
    needle_type: Optional[str]


@dataclass
class EvalResult:
    resume_id: str
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    precision_at_k: float
    golden_rank: Optional[int]
    adversarial_rank: Optional[int]
    judged_items: list[JudgedItem]


@dataclass
class EvalReport:
    run_id: str               # ISO timestamp
    eval_top_k: int
    precision_at_k: int       # the K used for precision (default 20)
    n_cases: int
    mean_recall_at_k: float
    mean_mrr: float
    mean_ndcg_at_k: float
    mean_precision_at_k: float
    thresholds_met: dict[str, bool]
    per_case: list[dict]
    generator_model: str
    judge_model: str

    def overall_pass(self) -> bool:
        return all(self.thresholds_met.values())

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def save_needles_to_db(db: DatabaseManager, cases: list[EvalCase], generator_model_id: str) -> None:
    """INSERT OR REPLACE each EvalCase into eval_needles table."""
    with db.get_connection() as conn:
        cursor = conn.cursor()
        for case in cases:
            cursor.execute(
                """INSERT OR REPLACE INTO eval_needles
                (resume_id, resume_text, golden_title, golden_company, golden_description,
                 adversarial_title, adversarial_company, adversarial_description, deal_breaker,
                 generator_model_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    case.resume_id,
                    case.resume_text,
                    case.golden.title,
                    case.golden.company,
                    case.golden.description,
                    case.adversarial.title,
                    case.adversarial.company,
                    case.adversarial.description,
                    case.adversarial.deal_breaker,
                    generator_model_id,
                ),
            )
        conn.commit()


def load_needles_from_db(db: DatabaseManager) -> list[EvalCase]:
    """Load all rows from eval_needles and reconstruct EvalCase list."""
    cases = []
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT * FROM eval_needles")
        for row in cursor.fetchall():
            # Column order: id, resume_id, resume_text, golden_title, golden_company,
            # golden_description, adversarial_title, adversarial_company,
            # adversarial_description, deal_breaker, generator_model_id, created_at
            resume_id = row[1]
            resume_text = row[2]
            golden = SyntheticNeedle(
                needle_id=GOLDEN_NEEDLE_ID,
                needle_type="golden",
                title=row[3],
                company=row[4],
                description=row[5],
                deal_breaker=None,
                true_relevance=5,
            )
            adversarial = SyntheticNeedle(
                needle_id=ADVERSARIAL_NEEDLE_ID,
                needle_type="adversarial",
                title=row[6],
                company=row[7],
                description=row[8],
                deal_breaker=row[9],
                true_relevance=0,
            )
            cases.append(EvalCase(resume_id=resume_id, resume_text=resume_text, golden=golden, adversarial=adversarial))
    return cases


def save_needles_to_json(cases: list[EvalCase], path: str) -> None:
    """Serialize list[EvalCase] to a JSON file (pretty-printed, indent=2).
    Creates parent directories if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "resume_id": case.resume_id,
            "resume_text": case.resume_text,
            "golden": dataclasses.asdict(case.golden),
            "adversarial": dataclasses.asdict(case.adversarial),
        }
        for case in cases
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_needles_from_json(path: str) -> list[EvalCase]:
    """Deserialize JSON → list[EvalCase]. Returns [] if file does not exist."""
    p = Path(path)
    if not p.exists():
        return []

    with open(path, "r") as f:
        data = json.load(f)

    cases = []
    for item in data:
        golden_data = item["golden"]
        adversarial_data = item["adversarial"]

        golden = SyntheticNeedle(
            needle_id=golden_data["needle_id"],
            needle_type=golden_data["needle_type"],
            title=golden_data["title"],
            company=golden_data["company"],
            description=golden_data["description"],
            deal_breaker=golden_data["deal_breaker"],
            true_relevance=golden_data["true_relevance"],
        )
        adversarial = SyntheticNeedle(
            needle_id=adversarial_data["needle_id"],
            needle_type=adversarial_data["needle_type"],
            title=adversarial_data["title"],
            company=adversarial_data["company"],
            description=adversarial_data["description"],
            deal_breaker=adversarial_data["deal_breaker"],
            true_relevance=adversarial_data["true_relevance"],
        )

        case = EvalCase(
            resume_id=item["resume_id"],
            resume_text=item["resume_text"],
            golden=golden,
            adversarial=adversarial,
        )
        cases.append(case)

    return cases


def _build_golden_needle_prompt(resume_text: str) -> str:
    """
    Returns the user prompt for Gemini.
    Constraints to enforce (include these explicitly in the prompt):
    - Write a realistic job posting (title, company name, full description with responsibilities and qualifications)
    - DO NOT copy any sentence or phrase verbatim from the resume
    - Use professional synonyms and paraphrasing throughout (prevents lexical leakage)
    - Match the seniority level, domain, and key technical skills of the candidate
    Response JSON schema: {\"title\": str, \"company\": str, \"description\": str}
    """
    return f"""You are a recruitment expert. Based on the resume below, create a realistic job posting that the candidate would be a strong match for.

CRITICAL CONSTRAINTS:
1. Write a COMPLETE job posting with title, company name, and full description.
2. DO NOT copy any sentence or phrase verbatim from the resume.
3. Use professional synonyms and paraphrasing throughout to prevent lexical leakage.
4. Match the candidate's seniority level, domain, and key technical skills.
5. The job description should include both responsibilities and required qualifications.

Resume:
{resume_text}

Respond with ONLY valid JSON (no markdown, no explanation):
{{"title": "...", "company": "...", "description": "..."}}"""


def _build_adversarial_needle_prompt(resume_text: str, golden: SyntheticNeedle) -> str:
    """
    Returns the user prompt for Gemini.
    Constraints:
    - Clone the golden needle almost exactly
    - Inject exactly ONE objective deal-breaker (e.g., requires work auth in a country
      the candidate cannot work in, requires 10+ yrs experience when candidate has 3,
      requires a domain certification the candidate explicitly lacks)
    - All other content must remain semantically identical to the golden needle
    Response JSON schema: {\"title\": str, \"company\": str, \"description\": str, \"deal_breaker\": str}
    """
    return f"""You are a recruitment expert. Given a job posting and a resume, create an ADVERSARIAL version of that job posting by injecting exactly ONE objective deal-breaker that disqualifies the candidate.

CRITICAL CONSTRAINTS:
1. Clone the job posting almost exactly (title, company, description structure).
2. Inject exactly ONE objective, verifiable deal-breaker that disqualifies the candidate. Examples:
   - Requires work authorization in a country the candidate cannot work in
   - Requires 10+ years of experience when candidate has 3
   - Requires a specific certification the candidate explicitly lacks
   - Requires domain expertise in an area candidate has no background in
3. Keep all other content semantically identical to the original job posting.
4. The deal-breaker must be CLEAR and OBJECTIVE, not subjective preference.

Original Job Posting:
Title: {golden.title}
Company: {golden.company}
Description: {golden.description}

Resume:
{resume_text}

Respond with ONLY valid JSON (no markdown, no explanation):
{{"title": "...", "company": "...", "description": "...", "deal_breaker": "..."}}"""


def generate_needles(
    resume_text: str,
    resume_id: str,
    gemini_client,           # google.genai.Client
    model_id: str,
    logger: logging.Logger,
    max_retries: int = 3,
) -> EvalCase:
    """
    1. Call Gemini with _build_golden_needle_prompt; response_mime_type="application/json"; temperature=0.7
    2. Validate JSON has keys: title, company, description (raise ValueError if not)
    3. Construct golden SyntheticNeedle (needle_id=GOLDEN_NEEDLE_ID, true_relevance=5)
    4. Call Gemini with _build_adversarial_needle_prompt
    5. Validate JSON has keys: title, company, description, deal_breaker
    6. Construct adversarial SyntheticNeedle (needle_id=ADVERSARIAL_NEEDLE_ID, true_relevance=0)
    7. Return EvalCase
    Retry each step up to max_retries with exponential backoff on API error or validation failure.
    """
    # Generate golden needle
    golden = None
    for attempt in range(max_retries):
        try:
            logger.info(f"Generating golden needle for {resume_id} (attempt {attempt + 1}/{max_retries})")
            response = gemini_client.models.generate_content(
                model=model_id,
                contents=_build_golden_needle_prompt(resume_text),
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.7,
                },
            )
            golden_data = json.loads(response.text)

            # Validate required keys
            if not all(k in golden_data for k in ["title", "company", "description"]):
                raise ValueError(f"Missing required keys in golden needle response: {golden_data.keys()}")

            golden = SyntheticNeedle(
                needle_id=GOLDEN_NEEDLE_ID,
                needle_type="golden",
                title=golden_data["title"],
                company=golden_data["company"],
                description=golden_data["description"],
                deal_breaker=None,
                true_relevance=5,
            )
            logger.info(f"Golden needle generated successfully for {resume_id}")
            break
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Golden needle generation failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to generate golden needle after {max_retries} attempts: {e}")
            time.sleep(2 ** attempt)  # exponential backoff
        except Exception as e:
            logger.error(f"Unexpected error generating golden needle: {e}")
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to generate golden needle after {max_retries} attempts: {e}")
            time.sleep(2 ** attempt)

    # Generate adversarial needle
    adversarial = None
    for attempt in range(max_retries):
        try:
            logger.info(f"Generating adversarial needle for {resume_id} (attempt {attempt + 1}/{max_retries})")
            response = gemini_client.models.generate_content(
                model=model_id,
                contents=_build_adversarial_needle_prompt(resume_text, golden),
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.7,
                },
            )
            adversarial_data = json.loads(response.text)

            # Validate required keys
            if not all(k in adversarial_data for k in ["title", "company", "description", "deal_breaker"]):
                raise ValueError(f"Missing required keys in adversarial needle response: {adversarial_data.keys()}")

            adversarial = SyntheticNeedle(
                needle_id=ADVERSARIAL_NEEDLE_ID,
                needle_type="adversarial",
                title=adversarial_data["title"],
                company=adversarial_data["company"],
                description=adversarial_data["description"],
                deal_breaker=adversarial_data["deal_breaker"],
                true_relevance=0,
            )
            logger.info(f"Adversarial needle generated successfully for {resume_id}")
            break
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Adversarial needle generation failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to generate adversarial needle after {max_retries} attempts: {e}")
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"Unexpected error generating adversarial needle: {e}")
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to generate adversarial needle after {max_retries} attempts: {e}")
            time.sleep(2 ** attempt)

    return EvalCase(
        resume_id=resume_id,
        resume_text=resume_text,
        golden=golden,
        adversarial=adversarial,
    )


def _embed_text(embedding_model, text: str) -> "np.ndarray":
    """Embed a single text string using the loaded sentence-transformers model.

    Args:
        embedding_model: SentenceTransformer model instance
        text: Text to embed

    Returns:
        float32 unit vector of shape [dim]
    """
    from src.retrieval import embed_user_profile
    return embed_user_profile(embedding_model, text)


def run_retrieval_with_needles(
    eval_case: EvalCase,
    db: DatabaseManager,
    cfg: "Config",
    embedding_model,
    reranker_tokenizer,
    reranker_model,
    logger: logging.Logger,
) -> list[RetrievedItem]:
    """Run hybrid retrieval with in-memory needle augmentation.

    Never writes to DB. Steps:
    1. Load corpus embeddings
    2. Load job texts from DB
    3. Embed query, golden needle, and adversarial needle
    4. Augment in-memory: append needle vectors and texts
    5. Run dense + sparse retrieval on augmented corpus
    6. Fuse via RRF
    7. Rerank
    8. Build RetrievedItem list with needle flags
    """
    from src.retrieval import (
        load_corpus_embeddings,
        dense_top_k,
        build_bm25_index,
        sparse_top_k,
        reciprocal_rank_fusion,
    )
    from src.reranking import score_pairs_batched, build_job_text

    # Load corpus embeddings and job texts
    job_ids, corpus_matrix = load_corpus_embeddings(db, cfg.embedding_model_id)
    job_texts = db.get_all_cleaned_descriptions()

    if not job_ids:
        logger.warning("No corpus embeddings found for evaluation")
        return []

    # Build dict of job_id -> (title, description) for easy lookup and reranking
    job_lookup = {}
    with db.get_connection() as conn:
        cursor = conn.execute(
            "SELECT id, title, cleaned_description FROM jobs WHERE embedded=1 AND is_target_role=1"
        )
        for job_id, title, description in cursor.fetchall():
            job_lookup[job_id] = (title, description)

    # Embed query, golden, and adversarial
    query_vec = _embed_text(embedding_model, eval_case.resume_text)
    golden_vec = _embed_text(embedding_model, eval_case.golden.description)
    adversarial_vec = _embed_text(embedding_model, eval_case.adversarial.description)

    # Augment corpus in-memory
    augmented_ids = job_ids + [GOLDEN_NEEDLE_ID, ADVERSARIAL_NEEDLE_ID]
    augmented_matrix = np.vstack([corpus_matrix, golden_vec.reshape(1, -1), adversarial_vec.reshape(1, -1)])

    # Build augmented job texts list for BM25
    augmented_texts = job_texts + [
        (GOLDEN_NEEDLE_ID, eval_case.golden.description),
        (ADVERSARIAL_NEEDLE_ID, eval_case.adversarial.description),
    ]

    logger.info(f"Augmented corpus: {len(augmented_ids)} jobs (original: {len(job_ids)}, needles: 2)")

    # Dense retrieval
    candidate_k = min(cfg.eval_top_k * 2, len(augmented_ids))
    dense_results = dense_top_k(query_vec, augmented_matrix, augmented_ids, candidate_k)
    logger.info(f"Dense retrieval: {len(dense_results)} candidates")

    # Sparse retrieval
    bm25_ids, bm25 = build_bm25_index(augmented_texts)
    query_tokens = eval_case.resume_text.lower().split()
    sparse_results = sparse_top_k(bm25, query_tokens, bm25_ids, candidate_k)
    logger.info(f"Sparse retrieval: {len(sparse_results)} candidates")

    # RRF fusion
    fused = reciprocal_rank_fusion(
        dense_results, sparse_results, cfg.retrieval_rrf_k, cfg.eval_top_k
    )
    logger.info(f"RRF fusion: {len(fused)} results")

    # Prepare texts for reranking
    rerank_texts = []
    rerank_job_ids = []
    for job_id, _, _ in fused:
        if job_id == GOLDEN_NEEDLE_ID:
            rerank_texts.append(build_job_text(eval_case.golden.title, eval_case.golden.description))
        elif job_id == ADVERSARIAL_NEEDLE_ID:
            rerank_texts.append(build_job_text(eval_case.adversarial.title, eval_case.adversarial.description))
        else:
            title, description = job_lookup[job_id]
            rerank_texts.append(build_job_text(title, description))
        rerank_job_ids.append(job_id)

    # Rerank
    rerank_scores = score_pairs_batched(
        reranker_tokenizer,
        reranker_model,
        eval_case.resume_text,
        rerank_texts,
        batch_size=32,
    )
    logger.info(f"Reranking complete: {len(rerank_scores)} scores")

    # Build RetrievedItem list
    retrieved_items = []
    for rank, (job_id, rrf_score, _) in enumerate(fused):
        rerank_score = rerank_scores[rank]

        # Determine title and description
        if job_id == GOLDEN_NEEDLE_ID:
            title = eval_case.golden.title
            description = eval_case.golden.description
            is_needle = True
            needle_type = "golden"
        elif job_id == ADVERSARIAL_NEEDLE_ID:
            title = eval_case.adversarial.title
            description = eval_case.adversarial.description
            is_needle = True
            needle_type = "adversarial"
        else:
            title, description = job_lookup[job_id]
            is_needle = False
            needle_type = None

        retrieved_items.append(
            RetrievedItem(
                job_id=job_id,
                rank=rank + 1,
                rrf_score=rrf_score,
                rerank_score=rerank_score,
                title=title,
                description=description,
                is_needle=is_needle,
                needle_type=needle_type,
            )
        )

    return retrieved_items


def run_retrieval_phase(
    cases: list[EvalCase],
    db: DatabaseManager,
    cfg: "Config",
    embedding_model,
    reranker_tokenizer,
    reranker_model,
    logger: logging.Logger,
) -> dict[str, list[RetrievedItem]]:
    """Run retrieval for all eval cases.

    Args:
        cases: List of EvalCase objects
        db: DatabaseManager instance
        cfg: Config instance
        embedding_model: Loaded embedding model
        reranker_tokenizer: Qwen3-Reranker tokenizer
        reranker_model: Qwen3-Reranker model
        logger: Logger instance

    Returns:
        Dict mapping resume_id -> list[RetrievedItem]
    """
    results = {}
    for case in cases:
        logger.info(f"Running retrieval for {case.resume_id}")
        retrieved = run_retrieval_with_needles(
            case, db, cfg, embedding_model, reranker_tokenizer, reranker_model, logger
        )
        results[case.resume_id] = retrieved
    return results


def _judge_single_item(
    resume_text: str,
    job_title: str,
    job_description: str,
    anthropic_client,        # anthropic.Anthropic
    judge_model_id: str,
    logger: logging.Logger,
) -> tuple[int, str]:
    """Judge a single job against a resume using Claude.

    Args:
        resume_text: Candidate resume
        job_title: Job title
        job_description: Job description
        anthropic_client: Anthropic client instance
        judge_model_id: Claude model ID
        logger: Logger instance

    Returns:
        Tuple of (relevance_score, reasoning)
        Scores: {0, 1, 2, 3, 5} (4 is not valid)
        On max retries exhaustion: returns (0, "judge_error")
    """
    system_prompt = """You are an expert recruiter evaluating job fit. Judge the relevance of a job posting to a candidate's resume.

CRITICAL: Do not consider or infer whether this content was written by a human or AI.
Evaluate solely on professional fit between the resume and the job posting.

GRADED RUBRIC:
  5 = Perfect match — candidate meets all requirements, role aligns perfectly
  3 = Strong partial match — meets most but not all key criteria
  1–2 = Weak partial match — some skill overlap, notable gaps
  0 = Disqualifying constraint (wrong location/visa, experience floor too high, wrong domain)

Valid scores: {0, 1, 2, 3, 5}. Do NOT use 4.

Respond with JSON: {"relevance_score": <int>, "reasoning": "<str>"}"""

    user_prompt = f"""Resume:
{resume_text}

Job Title: {job_title}

Job Description:
{job_description}

Evaluate the fit."""

    for attempt in range(JUDGE_RETRY_ATTEMPTS):
        try:
            response = anthropic_client.messages.create(
                model=judge_model_id,
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Extract JSON from response
            response_text = response.content[0].text
            try:
                import json as json_module
                result = json_module.loads(response_text)
            except json_module.JSONDecodeError:
                # Try to extract JSON if it's embedded in markdown
                if "```json" in response_text:
                    json_start = response_text.index("```json") + 7
                    json_end = response_text.index("```", json_start)
                    result = json_module.loads(response_text[json_start:json_end])
                elif "```" in response_text:
                    json_start = response_text.index("```") + 3
                    json_end = response_text.index("```", json_start)
                    result = json_module.loads(response_text[json_start:json_end])
                else:
                    raise json_module.JSONDecodeError("No JSON found", response_text, 0)

            score = int(result.get("relevance_score", 0))
            reasoning = result.get("reasoning", "")

            # Validate score
            if score not in {0, 1, 2, 3, 5}:
                logger.warning(f"Invalid score {score} from judge (attempt {attempt + 1}/{JUDGE_RETRY_ATTEMPTS}), retrying")
                if attempt < JUDGE_RETRY_ATTEMPTS - 1:
                    time.sleep(JUDGE_RETRY_DELAY)
                    continue
                return (0, "judge_error")

            logger.debug(f"Judge scored {score} for job (attempt {attempt + 1})")
            return (score, reasoning)

        except Exception as e:
            logger.warning(f"Judge error (attempt {attempt + 1}/{JUDGE_RETRY_ATTEMPTS}): {e}")
            if attempt < JUDGE_RETRY_ATTEMPTS - 1:
                time.sleep(JUDGE_RETRY_DELAY)
            else:
                logger.error(f"Judge exhausted retries after {JUDGE_RETRY_ATTEMPTS} attempts")
                return (0, "judge_error")

    return (0, "judge_error")


def judge_retrieved_items(
    resume_text: str,
    items: list[RetrievedItem],
    anthropic_client,
    judge_model_id: str,
    logger: logging.Logger,
) -> list[JudgedItem]:
    """Judge retrieved items with deterministic scoring for needles.

    Args:
        resume_text: Candidate resume text
        items: List of RetrievedItem objects
        anthropic_client: Anthropic client instance
        judge_model_id: Claude model ID
        logger: Logger instance

    Returns:
        List of JudgedItem objects with relevance scores and reasoning
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    judged_items = []
    judged_dict = {}  # job_id -> JudgedItem for preserving order

    def judge_task(item: RetrievedItem) -> tuple[int, JudgedItem]:
        """Judge a single item. Returns (index, JudgedItem)."""
        if item.is_needle:
            if item.needle_type == "golden":
                score = 5
                reasoning = "golden needle (deterministic)"
            else:  # adversarial
                score = 0
                reasoning = "adversarial needle (deterministic)"
        else:
            score, reasoning = _judge_single_item(
                resume_text,
                item.title,
                item.description,
                anthropic_client,
                judge_model_id,
                logger,
            )

        judged = JudgedItem(
            job_id=item.job_id,
            rank=item.rank,
            relevance_score=score,
            judge_reasoning=reasoning,
            is_needle=item.is_needle,
            needle_type=item.needle_type,
        )
        return (item.rank - 1, judged)  # Use rank-1 as index for ordering

    # Judge items concurrently
    with ThreadPoolExecutor(max_workers=JUDGE_MAX_WORKERS) as executor:
        futures = [executor.submit(judge_task, item) for item in items]
        for future in as_completed(futures):
            idx, judged = future.result()
            judged_dict[idx] = judged

    # Reconstruct list in original order
    judged_items = [judged_dict[i] for i in sorted(judged_dict.keys())]

    return judged_items


def run_judge_phase(
    cases: list[EvalCase],
    retrieved_per_case: dict[str, list[RetrievedItem]],
    anthropic_client,
    judge_model_id: str,
    logger: logging.Logger,
) -> dict[str, list[JudgedItem]]:
    """Run judging phase for all eval cases.

    Args:
        cases: List of EvalCase objects
        retrieved_per_case: Dict mapping resume_id -> list[RetrievedItem]
        anthropic_client: Anthropic client instance
        judge_model_id: Claude model ID
        logger: Logger instance

    Returns:
        Dict mapping resume_id -> list[JudgedItem]
    """
    results = {}
    for case in cases:
        logger.info(f"Running judge phase for {case.resume_id}")
        retrieved = retrieved_per_case[case.resume_id]
        judged = judge_retrieved_items(
            case.resume_text,
            retrieved,
            anthropic_client,
            judge_model_id,
            logger,
        )
        results[case.resume_id] = judged
    return results


def compute_ndcg_at_k(judged_items: list[JudgedItem], k: int) -> float:
    """Compute NDCG@K using graded relevance scores.

    Args:
        judged_items: List of JudgedItem objects
        k: Cutoff position

    Returns:
        NDCG@K score (0.0 to 1.0)

    NDCG = DCG@K / IDCG@K
    where DCG@K = sum((2^rel_i - 1) / log2(i + 2)) for i in 0..min(k, len)-1
    """
    if not judged_items:
        return 0.0

    # Take only top k items
    top_k_items = judged_items[:k]

    # Compute DCG@K
    dcg = 0.0
    for i, item in enumerate(top_k_items):
        rel = item.relevance_score
        dcg += (2 ** rel - 1) / np.log2(i + 2)

    # Compute IDCG@K (ideal = sorted by score descending)
    sorted_items = sorted(top_k_items, key=lambda x: x.relevance_score, reverse=True)
    idcg = 0.0
    for i, item in enumerate(sorted_items):
        rel = item.relevance_score
        idcg += (2 ** rel - 1) / np.log2(i + 2)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def compute_metrics(
    eval_case: EvalCase,
    retrieved_items: list[RetrievedItem],
    judged_items: list[JudgedItem],
) -> EvalResult:
    """Compute all four metrics from judged items.

    Args:
        eval_case: The evaluation case
        retrieved_items: List of RetrievedItem (used to find needle ranks)
        judged_items: List of JudgedItem with relevance scores

    Returns:
        EvalResult with computed metrics

    Metrics:
    - Recall@K: 1.0 if golden needle is in top eval_top_k, else 0.0
    - MRR: 1.0 / rank of golden needle, or 0.0 if not found
    - NDCG@K: Normalized discounted cumulative gain
    - Precision@20: Fraction of top 20 items with score >= RELEVANCE_THRESHOLD
    """
    from src.config import Config

    cfg = Config()  # Get default config to access eval_top_k
    eval_top_k = cfg.eval_top_k

    # Find golden and adversarial ranks in retrieved_items
    golden_rank = None
    adversarial_rank = None
    for item in retrieved_items:
        if item.is_needle and item.needle_type == "golden":
            golden_rank = item.rank
        elif item.is_needle and item.needle_type == "adversarial":
            adversarial_rank = item.rank

    # Recall@K: golden needle in top K
    recall_at_k = 1.0 if golden_rank is not None and golden_rank <= eval_top_k else 0.0

    # MRR: reciprocal rank of golden needle
    mrr = (1.0 / golden_rank) if golden_rank is not None else 0.0

    # NDCG@K
    ndcg_at_k = compute_ndcg_at_k(judged_items, eval_top_k)

    # Precision@20: fraction of top 20 with score >= RELEVANCE_THRESHOLD
    top_20 = judged_items[:PRECISION_AT_K]
    if top_20:
        relevant_count = sum(1 for item in top_20 if item.relevance_score >= RELEVANCE_THRESHOLD)
        precision_at_k = relevant_count / len(top_20)
    else:
        precision_at_k = 0.0

    return EvalResult(
        resume_id=eval_case.resume_id,
        recall_at_k=recall_at_k,
        mrr=mrr,
        ndcg_at_k=ndcg_at_k,
        precision_at_k=precision_at_k,
        golden_rank=golden_rank,
        adversarial_rank=adversarial_rank,
        judged_items=judged_items,
    )


def aggregate_results(
    cases: list[EvalCase],
    results: list[EvalResult],
    cfg: "Config",
    generator_model_id: str,
    judge_model_id: str,
    run_id: str,
) -> EvalReport:
    """Aggregate per-case metrics and check thresholds.

    Args:
        cases: List of EvalCase objects
        results: List of EvalResult objects
        cfg: Config instance
        generator_model_id: Model ID used for needle generation
        judge_model_id: Model ID used for judging
        run_id: Unique run identifier (ISO timestamp)

    Returns:
        EvalReport with aggregated metrics and threshold checks
    """
    if not results:
        return EvalReport(
            run_id=run_id,
            eval_top_k=cfg.eval_top_k,
            precision_at_k=PRECISION_AT_K,
            n_cases=0,
            mean_recall_at_k=0.0,
            mean_mrr=0.0,
            mean_ndcg_at_k=0.0,
            mean_precision_at_k=0.0,
            thresholds_met={"recall_at_k": False, "mrr": False, "ndcg_at_k": False, "precision_at_k": False},
            per_case=[],
            generator_model=generator_model_id,
            judge_model=judge_model_id,
        )

    # Compute means
    mean_recall = sum(r.recall_at_k for r in results) / len(results)
    mean_mrr = sum(r.mrr for r in results) / len(results)
    mean_ndcg = sum(r.ndcg_at_k for r in results) / len(results)
    mean_precision = sum(r.precision_at_k for r in results) / len(results)

    # Check thresholds
    thresholds_met = {
        "recall_at_k": mean_recall >= TARGET_RECALL_AT_K,
        "mrr": mean_mrr >= TARGET_MRR,
        "ndcg_at_k": mean_ndcg >= TARGET_NDCG_AT_K,
        "precision_at_k": mean_precision >= TARGET_PRECISION_AT_K,
    }

    # Build per-case summaries
    per_case = [
        {
            "resume_id": r.resume_id,
            "recall_at_k": r.recall_at_k,
            "mrr": r.mrr,
            "ndcg_at_k": r.ndcg_at_k,
            "precision_at_k": r.precision_at_k,
            "golden_rank": r.golden_rank,
            "adversarial_rank": r.adversarial_rank,
        }
        for r in results
    ]

    return EvalReport(
        run_id=run_id,
        eval_top_k=cfg.eval_top_k,
        precision_at_k=PRECISION_AT_K,
        n_cases=len(results),
        mean_recall_at_k=mean_recall,
        mean_mrr=mean_mrr,
        mean_ndcg_at_k=mean_ndcg,
        mean_precision_at_k=mean_precision,
        thresholds_met=thresholds_met,
        per_case=per_case,
        generator_model=generator_model_id,
        judge_model=judge_model_id,
    )


def save_results_to_db(db: DatabaseManager, run_id: str, results: list[EvalResult]) -> None:
    """Insert metric rows into eval_results table.

    Args:
        db: DatabaseManager instance
        run_id: Unique run identifier
        results: List of EvalResult objects
    """
    with db.get_connection() as conn:
        cursor = conn.cursor()
        for result in results:
            # Insert one row per metric per case
            cursor.execute(
                "INSERT INTO eval_results (run_id, resume_id, metric, value) VALUES (?, ?, ?, ?)",
                (run_id, result.resume_id, "recall_at_k", result.recall_at_k),
            )
            cursor.execute(
                "INSERT INTO eval_results (run_id, resume_id, metric, value) VALUES (?, ?, ?, ?)",
                (run_id, result.resume_id, "mrr", result.mrr),
            )
            cursor.execute(
                "INSERT INTO eval_results (run_id, resume_id, metric, value) VALUES (?, ?, ?, ?)",
                (run_id, result.resume_id, "ndcg_at_k", result.ndcg_at_k),
            )
            cursor.execute(
                "INSERT INTO eval_results (run_id, resume_id, metric, value) VALUES (?, ?, ?, ?)",
                (run_id, result.resume_id, "precision_at_k", result.precision_at_k),
            )
        conn.commit()


def save_report(report: EvalReport, path: str) -> None:
    """Write JSON report to file.

    Args:
        report: EvalReport object
        path: Output file path (parent directories created if needed)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report.as_dict(), f, indent=2)


def print_report_summary(report: EvalReport) -> None:
    """Print formatted evaluation report summary to stdout.

    Args:
        report: EvalReport object
    """
    print("\n" + "=" * 60)
    print("=== NIAH Evaluation Report ===")
    print("=" * 60)
    print(f"Cases evaluated : {report.n_cases}")
    print(f"Generator model : {report.generator_model}")
    print(f"Judge model     : {report.judge_model}")
    print()
    print("Metric           Score     Target    Pass?")
    print("-" * 60)

    metrics = [
        ("Recall@50", report.mean_recall_at_k, TARGET_RECALL_AT_K, "recall_at_k"),
        ("MRR", report.mean_mrr, TARGET_MRR, "mrr"),
        ("NDCG@50", report.mean_ndcg_at_k, TARGET_NDCG_AT_K, "ndcg_at_k"),
        ("Precision@20", report.mean_precision_at_k, TARGET_PRECISION_AT_K, "precision_at_k"),
    ]

    for name, score, target, key in metrics:
        passed = "PASS" if report.thresholds_met[key] else "FAIL"
        print(f"{name:16} {score:.3f}     > {target:.2f}    {passed}")

    print("-" * 60)
    print(f"\nFull report saved to: {Path.cwd() / 'data' / 'eval_report.json'}")
    print("=" * 60 + "\n")


def setup_logging(log_level: str, name: str) -> logging.Logger:
    """Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        name: Logger name

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def main() -> None:
    """Main entry point for NIAH evaluation pipeline.

    Steps:
    1. Parse CLI arguments (--regen, --resume)
    2. Load config and validate anthropic_api_key
    3. Setup logging
    4. Load resume text
    5. Load/generate needles
    6. Open database
    7. Load models (embedding, reranker)
    8. Init clients (Gemini, Claude)
    9. Run retrieval → judge → metrics pipeline
    10. Aggregate results and save report
    11. Exit with status based on overall_pass()
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="NIAH evaluation for retrieval layer")
    parser.add_argument("--regen", action="store_true", help="Force needle regeneration")
    parser.add_argument("--resume", type=str, default=None, help="Override resume profile path")
    args = parser.parse_args()

    # Load config
    from src.config import load_config

    cfg = load_config()

    # Validate anthropic_api_key
    if not cfg.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable must be set")

    # Setup logging
    logger = setup_logging(cfg.log_level, "evaluation")
    logger.info("Starting NIAH evaluation pipeline")

    # Load resume text
    resume_path = args.resume or cfg.retrieval_user_profile_path
    logger.info(f"Loading resume from {resume_path}")
    try:
        with open(resume_path, "r") as f:
            resume_text = f.read()
    except FileNotFoundError:
        logger.error(f"Resume file not found: {resume_path}")
        sys.exit(1)

    # Load or generate needles
    logger.info(f"Loading needles from {cfg.eval_needles_path}")
    cases = load_needles_from_json(cfg.eval_needles_path)

    if not cases or args.regen:
        logger.info("Generating needles (no cache or --regen requested)")
        try:
            import google.genai

            gemini_client = google.genai.Client(api_key=cfg.google_api_key)
            case = generate_needles(
                resume_text=resume_text,
                resume_id="user_profile",
                gemini_client=gemini_client,
                model_id=cfg.eval_needle_gen_model_id,
                logger=logger,
            )
            cases = [case]
            logger.info(f"Generated {len(cases)} eval case(s)")
            save_needles_to_json(cases, cfg.eval_needles_path)
        except Exception as e:
            logger.error(f"Failed to generate needles: {e}")
            sys.exit(1)
    else:
        logger.info(f"Loaded {len(cases)} eval case(s) from cache")

    # Open database
    logger.info(f"Opening database: {cfg.db_path}")
    db = DatabaseManager(cfg.db_path)
    db.initialize_schema()

    # Load models
    logger.info(f"Loading embedding model: {cfg.embedding_model_id}")
    try:
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer(cfg.embedding_model_id)
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        sys.exit(1)

    logger.info(f"Loading reranker model: {cfg.reranking_model_id}")
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        reranker_tokenizer = AutoTokenizer.from_pretrained(cfg.reranking_model_id)
        reranker_model = AutoModelForSequenceClassification.from_pretrained(cfg.reranking_model_id)
    except Exception as e:
        logger.error(f"Failed to load reranker: {e}")
        sys.exit(1)

    # Init clients
    try:
        import google.genai

        gemini_client = google.genai.Client(api_key=cfg.google_api_key)
    except Exception as e:
        logger.error(f"Failed to init Gemini client: {e}")
        sys.exit(1)

    try:
        import anthropic

        anthropic_client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    except Exception as e:
        logger.error(f"Failed to init Anthropic client: {e}")
        sys.exit(1)

    # Run evaluation pipeline
    logger.info("Starting retrieval phase")
    retrieved_per_case = run_retrieval_phase(
        cases, db, cfg, embedding_model, reranker_tokenizer, reranker_model, logger
    )

    logger.info("Starting judge phase")
    judged_per_case = run_judge_phase(cases, retrieved_per_case, anthropic_client, cfg.eval_judge_model_id, logger)

    logger.info("Computing metrics")
    eval_results = []
    for case in cases:
        retrieved = retrieved_per_case[case.resume_id]
        judged = judged_per_case[case.resume_id]
        result = compute_metrics(case, retrieved, judged)
        eval_results.append(result)

    # Aggregate and report
    run_id = datetime.utcnow().isoformat() + "Z"
    logger.info("Aggregating results")
    report = aggregate_results(
        cases,
        eval_results,
        cfg,
        cfg.eval_needle_gen_model_id,
        cfg.eval_judge_model_id,
        run_id,
    )

    logger.info(f"Saving results to database")
    save_results_to_db(db, run_id, eval_results)

    logger.info(f"Saving report to {cfg.eval_report_path}")
    save_report(report, cfg.eval_report_path)

    print_report_summary(report)

    if report.overall_pass():
        logger.info("All thresholds met!")
        sys.exit(0)
    else:
        logger.warning("Some thresholds not met")
        sys.exit(1)


if __name__ == "__main__":
    main()
