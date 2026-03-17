"""
Unit tests for generation.py data structures and helper functions.

Tests cover:
- JobContext and GenerationResult dataclasses
- _build_job_context: JSON list deserialization
- _parse_citations: extraction and deduplication
- _validate_summary_structure: validation of summary format
- Integration tests with mocked Gemini client
"""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from src.generation import (
    JobContext,
    GenerationResult,
    _build_job_context,
    _parse_citations,
    _validate_summary_structure,
    _run_con_filter,
    _run_generation,
    _run_evaluator,
    _process_single_job,
    generate_summaries,
)


class TestBuildJobContext:
    """Tests for _build_job_context function."""

    def test_deserializes_json_lists(self):
        """Test that JSON list fields are properly deserialized."""
        row = (
            1,  # job_id
            1,  # rank
            0.95,  # score
            "Senior ML Engineer",  # title
            "Acme Corp",  # company
            "San Francisco, CA",  # location
            "https://example.com/job/1",  # absolute_url
            "Build ML systems",  # cleaned_description
            json.dumps(["Design pipelines", "Train models"]),  # responsibilities (JSON)
            json.dumps(["PyTorch", "Pandas"]),  # skills (JSON)
            json.dumps(["AWS", "Docker"]),  # tools_and_platforms (JSON)
            5,  # experience_min_years
        )

        context = _build_job_context(
            row,
            deserialize_list_fields=[
                "responsibilities",
                "skills",
                "tools_and_platforms",
            ],
        )

        assert context.job_id == 1
        assert context.rank == 1
        assert context.score == 0.95
        assert context.title == "Senior ML Engineer"
        assert context.company == "Acme Corp"
        assert context.location == "San Francisco, CA"
        assert context.absolute_url == "https://example.com/job/1"
        assert context.cleaned_description == "Build ML systems"
        assert context.responsibilities == ["Design pipelines", "Train models"]
        assert context.skills == ["PyTorch", "Pandas"]
        assert context.tools_and_platforms == ["AWS", "Docker"]
        assert context.experience_min_years == 5

    def test_handles_empty_json_lists(self):
        """Test deserialization of empty JSON lists."""
        row = (
            2,  # job_id
            2,  # rank
            0.85,  # score
            "Data Scientist",  # title
            "Beta Inc",  # company
            None,  # location
            None,  # absolute_url
            "Analyze data",  # cleaned_description
            json.dumps([]),  # responsibilities (empty)
            json.dumps([]),  # skills (empty)
            json.dumps([]),  # tools_and_platforms (empty)
            None,  # experience_min_years
        )

        context = _build_job_context(
            row,
            deserialize_list_fields=[
                "responsibilities",
                "skills",
                "tools_and_platforms",
            ],
        )

        assert context.responsibilities == []
        assert context.skills == []
        assert context.tools_and_platforms == []
        assert context.location is None
        assert context.absolute_url is None
        assert context.experience_min_years is None

    def test_handles_null_json_fields(self):
        """Test deserialization when JSON fields are None/null."""
        row = (
            3,  # job_id
            3,  # rank
            0.75,  # score
            "Analyst",  # title
            "Gamma LLC",  # company
            "New York, NY",  # location
            "https://example.com/job/3",  # absolute_url
            "Analyze trends",  # cleaned_description
            None,  # responsibilities (None, not JSON)
            None,  # skills (None, not JSON)
            None,  # tools_and_platforms (None, not JSON)
            3,  # experience_min_years
        )

        context = _build_job_context(
            row,
            deserialize_list_fields=[
                "responsibilities",
                "skills",
                "tools_and_platforms",
            ],
        )

        assert context.responsibilities is None
        assert context.skills is None
        assert context.tools_and_platforms is None


class TestParseCitations:
    """Tests for _parse_citations function."""

    def test_extracts_resume_citations(self):
        """Test extraction of [R:...] resume citations."""
        text = "Built PyTorch models [R:pytorch-models] for recommendation systems."
        citations = _parse_citations(text)

        assert len(citations) == 1
        assert citations[0] == {"source": "resume", "label": "pytorch-models"}

    def test_extracts_job_citations(self):
        """Test extraction of [J:...] job citations."""
        text = "Role requires ML infrastructure expertise [J:ml-infra] and mentorship [J:mentorship]."
        citations = _parse_citations(text)

        assert len(citations) == 2
        assert {"source": "job", "label": "ml-infra"} in citations
        assert {"source": "job", "label": "mentorship"} in citations

    def test_extracts_mixed_citations(self):
        """Test extraction of both R and J citations."""
        text = (
            "Candidate's PyTorch experience [R:pytorch] matches the role's ML infrastructure needs [J:ml-infra]. "
            "Leadership background [R:team-lead] aligns with mentorship requirements [J:mentorship]."
        )
        citations = _parse_citations(text)

        assert len(citations) == 4
        assert {"source": "resume", "label": "pytorch"} in citations
        assert {"source": "job", "label": "ml-infra"} in citations
        assert {"source": "resume", "label": "team-lead"} in citations
        assert {"source": "job", "label": "mentorship"} in citations

    def test_deduplicates_citations(self):
        """Test that duplicate citations are removed."""
        text = (
            "PyTorch expertise [R:pytorch] is crucial. [R:pytorch] mentioned again in second paragraph. "
            "PyTorch [R:pytorch] is mentioned three times total."
        )
        citations = _parse_citations(text)

        assert len(citations) == 1
        assert citations[0] == {"source": "resume", "label": "pytorch"}

    def test_allows_hyphens_and_underscores(self):
        """Test that labels with hyphens and underscores are parsed correctly."""
        text = "References [R:ml-infra_v2] and [R:team_lead-2] are valid."
        citations = _parse_citations(text)

        assert len(citations) == 2
        assert {"source": "resume", "label": "ml-infra_v2"} in citations
        assert {"source": "resume", "label": "team_lead-2"} in citations

    def test_ignores_malformed_citations(self):
        """Test that malformed citations are ignored."""
        text = (
            "Valid: [R:pytorch]. Invalid (space): [R: pytorch]. "
            "Invalid (special char): [R:pytorch$]. Invalid (no label): [R:]."
        )
        citations = _parse_citations(text)

        assert len(citations) == 1
        assert citations[0] == {"source": "resume", "label": "pytorch"}

    def test_empty_text_returns_empty_list(self):
        """Test parsing empty text."""
        citations = _parse_citations("")
        assert citations == []

    def test_no_citations_returns_empty_list(self):
        """Test text with no citations."""
        citations = _parse_citations("This is plain text without any citations.")
        assert citations == []


class TestValidateSummaryStructure:
    """Tests for _validate_summary_structure function."""

    def test_valid_summary_passes(self):
        """Test that a properly formatted summary passes validation."""
        text = """<thinking>
PyTorch aligns with ML infrastructure needs. Experience matches mentorship role.
</thinking>

**Similarity 1 — Deep Learning Engineering**
Candidate built a PyTorch autoencoder [R:pytorch], directly matching the role's ML infra needs [J:ml-infra].

**Similarity 2 — Technical Leadership**
Team leadership experience [R:team-lead] aligns with junior mentorship requirements [J:mentorship].

**Similarity 3 — Iterative Development**
Data analysis background [R:data-analysis] supports continuous improvement practices [J:iteration].
"""
        issues = _validate_summary_structure(text)
        assert issues == []

    def test_missing_thinking_block(self):
        """Test that missing <thinking> block is flagged."""
        text = """
**Similarity 1 — Test**
Some text [R:ref1] and [J:ref2].

**Similarity 2 — Test**
More text [R:ref3] and [J:ref4].

**Similarity 3 — Test**
Final text [R:ref5] and [J:ref6].
"""
        issues = _validate_summary_structure(text)
        assert "Missing <thinking> block" in issues

    def test_wrong_similarity_count_too_few(self):
        """Test that fewer than 3 similarities is flagged."""
        text = """<thinking>Thinking here.</thinking>

**Similarity 1 — Test**
Text [R:ref1] and [J:ref2].

**Similarity 2 — Test**
More text [R:ref3] and [J:ref4].
"""
        issues = _validate_summary_structure(text)
        assert any("Similarity" in issue for issue in issues)

    def test_wrong_similarity_count_too_many(self):
        """Test that more than 3 similarities is flagged."""
        text = """<thinking>Thinking here.</thinking>

**Similarity 1 — Test**
Text [R:ref1] and [J:ref2].

**Similarity 2 — Test**
More text [R:ref3] and [J:ref4].

**Similarity 3 — Test**
Final text [R:ref5] and [J:ref6].

**Similarity 4 — Test**
Extra text [R:ref7] and [J:ref8].
"""
        issues = _validate_summary_structure(text)
        assert any("Similarity" in issue for issue in issues)

    def test_missing_resume_citation(self):
        """Test that missing [R:...] citation is flagged."""
        text = """<thinking>Thinking here.</thinking>

**Similarity 1 — Test**
Text [J:ref1].

**Similarity 2 — Test**
More text [J:ref2].

**Similarity 3 — Test**
Final text [J:ref3].
"""
        issues = _validate_summary_structure(text)
        assert any("[R:" in issue for issue in issues)

    def test_missing_job_citation(self):
        """Test that missing [J:...] citation is flagged."""
        text = """<thinking>Thinking here.</thinking>

**Similarity 1 — Test**
Text [R:ref1].

**Similarity 2 — Test**
More text [R:ref2].

**Similarity 3 — Test**
Final text [R:ref3].
"""
        issues = _validate_summary_structure(text)
        assert any("[J:" in issue for issue in issues)

    def test_missing_both_citation_types(self):
        """Test that missing both citation types is caught."""
        text = """<thinking>Thinking here.</thinking>

**Similarity 1 — Test**
Text without citations.

**Similarity 2 — Test**
More text without citations.

**Similarity 3 — Test**
Final text without citations.
"""
        issues = _validate_summary_structure(text)
        assert any("[R:" in issue for issue in issues)
        assert any("[J:" in issue for issue in issues)

    def test_case_insensitive_similarity_matching(self):
        """Test that Similarity matching is case-insensitive."""
        text = """<thinking>Thinking here.</thinking>

**similarity 1 — Test**
Text [R:ref1] and [J:ref2].

**SIMILARITY 2 — Test**
More text [R:ref3] and [J:ref4].

**Similarity 3 — Test**
Final text [R:ref5] and [J:ref6].
"""
        issues = _validate_summary_structure(text)
        assert issues == []

    def test_multiline_thinking_block(self):
        """Test that multiline <thinking> blocks are recognized."""
        text = """<thinking>
This is a multiline
thinking block that spans
multiple paragraphs.
</thinking>

**Similarity 1 — Test**
Text [R:ref1] and [J:ref2].

**Similarity 2 — Test**
More text [R:ref3] and [J:ref4].

**Similarity 3 — Test**
Final text [R:ref5] and [J:ref6].
"""
        issues = _validate_summary_structure(text)
        assert issues == []


class TestRunConFilter:
    """Integration tests for _run_con_filter with mocked Gemini client."""

    def setup_method(self):
        """Set up test fixtures."""
        self.job = JobContext(
            job_id=1,
            rank=1,
            score=0.95,
            title="Senior ML Engineer",
            company="Acme Corp",
            location="San Francisco, CA",
            absolute_url="https://example.com/job/1",
            cleaned_description="Build ML systems for production.",
            responsibilities=["Design pipelines", "Train models"],
            skills=["PyTorch", "Pandas"],
            tools_and_platforms=["AWS", "Docker"],
            experience_min_years=5,
        )
        self.resume_text = "I built ML systems with PyTorch. Led teams. 7 years experience."
        self.logger = logging.getLogger("test")

    def test_con_filter_relevant(self):
        """Test CoN filter returning 'relevant' verdict."""
        client = MagicMock()
        response_mock = MagicMock()
        response_mock.text = json.dumps({
            "relevance_verdict": "relevant",
            "relevance_reasoning": "Good overlap in skills and experience.",
            "contradictions": [],
            "strong_alignments": ["PyTorch expertise", "Leadership background"]
        })
        client.models.generate_content.return_value = response_mock

        result = _run_con_filter(self.job, self.resume_text, client, "gemini-2.5-flash", self.logger)

        assert result is not None
        assert result["relevance_verdict"] == "relevant"
        assert len(result["strong_alignments"]) == 2

    def test_con_filter_irrelevant(self):
        """Test CoN filter returning 'irrelevant' verdict."""
        client = MagicMock()
        response_mock = MagicMock()
        response_mock.text = json.dumps({
            "relevance_verdict": "irrelevant",
            "relevance_reasoning": "No overlapping skills.",
            "contradictions": [],
            "strong_alignments": []
        })
        client.models.generate_content.return_value = response_mock

        result = _run_con_filter(self.job, self.resume_text, client, "gemini-2.5-flash", self.logger)

        assert result is not None
        assert result["relevance_verdict"] == "irrelevant"

    def test_con_filter_contradictory(self):
        """Test CoN filter returning 'contradictory' verdict."""
        client = MagicMock()
        response_mock = MagicMock()
        response_mock.text = json.dumps({
            "relevance_verdict": "contradictory",
            "relevance_reasoning": "Requires 5+ years, candidate has 2 years.",
            "contradictions": ["Experience level mismatch"],
            "strong_alignments": []
        })
        client.models.generate_content.return_value = response_mock

        result = _run_con_filter(self.job, self.resume_text, client, "gemini-2.5-flash", self.logger)

        assert result is not None
        assert result["relevance_verdict"] == "contradictory"

    def test_con_filter_json_error(self):
        """Test CoN filter handling invalid JSON response."""
        client = MagicMock()
        response_mock = MagicMock()
        response_mock.text = "Invalid JSON{{"
        client.models.generate_content.return_value = response_mock

        result = _run_con_filter(self.job, self.resume_text, client, "gemini-2.5-flash", self.logger)

        assert result is None

    def test_con_filter_schema_error(self):
        """Test CoN filter handling schema validation error."""
        client = MagicMock()
        response_mock = MagicMock()
        response_mock.text = json.dumps({
            "relevance_verdict": "invalid_verdict",
            "relevance_reasoning": "Wrong schema.",
            "contradictions": [],
            "strong_alignments": []
        })
        client.models.generate_content.return_value = response_mock

        result = _run_con_filter(self.job, self.resume_text, client, "gemini-2.5-flash", self.logger)

        assert result is None

    def test_con_filter_api_error(self):
        """Test CoN filter handling API exceptions."""
        client = MagicMock()
        client.models.generate_content.side_effect = Exception("API error")

        result = _run_con_filter(self.job, self.resume_text, client, "gemini-2.5-flash", self.logger)

        assert result is None


class TestRunGeneration:
    """Integration tests for _run_generation with mocked Gemini client."""

    def setup_method(self):
        """Set up test fixtures."""
        self.job = JobContext(
            job_id=1,
            rank=1,
            score=0.95,
            title="Senior ML Engineer",
            company="Acme Corp",
            location="San Francisco, CA",
            absolute_url="https://example.com/job/1",
            cleaned_description="Build ML systems.",
            responsibilities=["Design pipelines"],
            skills=["PyTorch"],
            tools_and_platforms=["AWS"],
            experience_min_years=5,
        )
        self.resume_text = "I built ML systems with PyTorch. 7 years experience."
        self.con_notes = {
            "strong_alignments": ["PyTorch expertise"],
            "relevance_verdict": "relevant"
        }
        self.logger = logging.getLogger("test")

    def test_generation_success(self):
        """Test generation returning valid summary."""
        client = MagicMock()
        response_mock = MagicMock()
        response_mock.text = """<thinking>PyTorch expertise aligns with ML infra needs.</thinking>

**Similarity 1 — Deep Learning Engineering**
Candidate built PyTorch systems [R:pytorch], matching role's ML infra [J:ml-infra].

**Similarity 2 — System Design**
Led pipeline design [R:pipeline], aligns with architecture work [J:architecture].

**Similarity 3 — Production Experience**
Built production ML systems [R:production], matching deployment needs [J:deployment].
"""
        client.models.generate_content.return_value = response_mock

        result = _run_generation(
            self.job, self.resume_text, self.con_notes,
            client, "gemini-2.5-flash", self.logger, max_retries=2
        )

        assert result is not None
        assert "<thinking>" in result
        assert "**Similarity 1" in result
        assert "[R:pytorch]" in result

    def test_generation_retries_on_structural_failure(self):
        """Test generation retrying on structural validation failure."""
        client = MagicMock()

        # First call: invalid (only 2 similarities), second call: valid
        invalid_response = MagicMock()
        invalid_response.text = """<thinking>Some thinking.</thinking>

**Similarity 1 — Test**
Text [R:ref1] and [J:ref2].

**Similarity 2 — Test**
More text [R:ref3] and [J:ref4].
"""

        valid_response = MagicMock()
        valid_response.text = """<thinking>Some thinking.</thinking>

**Similarity 1 — Test**
Text [R:ref1] and [J:ref2].

**Similarity 2 — Test**
More text [R:ref3] and [J:ref4].

**Similarity 3 — Test**
Final text [R:ref5] and [J:ref6].
"""

        client.models.generate_content.side_effect = [invalid_response, valid_response]

        result = _run_generation(
            self.job, self.resume_text, self.con_notes,
            client, "gemini-2.5-flash", self.logger, max_retries=2
        )

        assert result is not None
        assert "**Similarity 3" in result
        assert client.models.generate_content.call_count == 2

    def test_generation_fails_after_max_retries(self):
        """Test generation returning None after max retries exceeded."""
        client = MagicMock()
        response_mock = MagicMock()
        # Invalid response: missing thinking block
        response_mock.text = """**Similarity 1 — Test**
Text [R:ref1] and [J:ref2].

**Similarity 2 — Test**
More text [R:ref3] and [J:ref4].

**Similarity 3 — Test**
Final text [R:ref5] and [J:ref6].
"""
        client.models.generate_content.return_value = response_mock

        result = _run_generation(
            self.job, self.resume_text, self.con_notes,
            client, "gemini-2.5-flash", self.logger, max_retries=2
        )

        assert result is None
        assert client.models.generate_content.call_count == 2


class TestRunEvaluator:
    """Integration tests for _run_evaluator with mocked Gemini client."""

    def setup_method(self):
        """Set up test fixtures."""
        self.job = JobContext(
            job_id=1,
            rank=1,
            score=0.95,
            title="Senior ML Engineer",
            company="Acme Corp",
            location="San Francisco, CA",
            absolute_url="https://example.com/job/1",
            cleaned_description="Build ML systems.",
            responsibilities=["Design pipelines"],
            skills=["PyTorch"],
            tools_and_platforms=["AWS"],
            experience_min_years=5,
        )
        self.resume_text = "I built ML systems. 7 years experience."
        self.summary = """<thinking>Great alignment.</thinking>

**Similarity 1 — Test**
Text [R:ref1] and [J:ref2].

**Similarity 2 — Test**
More text [R:ref3] and [J:ref4].

**Similarity 3 — Test**
Final text [R:ref5] and [J:ref6].
"""
        self.logger = logging.getLogger("test")

    def test_evaluator_returns_dict(self):
        """Test evaluator returning valid evaluation dict."""
        client = MagicMock()
        response_mock = MagicMock()
        response_mock.text = json.dumps({
            "faithfulness": {
                "score": 9,
                "justification": "All claims are grounded.",
                "flags": []
            },
            "completeness": {
                "score": 10,
                "justification": "Three distinct similarities."
            },
            "structural_adherence": {
                "score": 10,
                "justification": "Perfect format.",
                "issues": []
            },
            "overall_pass": True
        })
        client.models.generate_content.return_value = response_mock

        result = _run_evaluator(
            self.job, self.resume_text, self.summary,
            client, "gemini-2.5-flash", self.logger
        )

        assert result is not None
        assert result["overall_pass"] is True
        assert result["faithfulness"]["score"] == 9

    def test_evaluator_json_error(self):
        """Test evaluator handling invalid JSON."""
        client = MagicMock()
        response_mock = MagicMock()
        response_mock.text = "Invalid JSON{{"
        client.models.generate_content.return_value = response_mock

        result = _run_evaluator(
            self.job, self.resume_text, self.summary,
            client, "gemini-2.5-flash", self.logger
        )

        assert result is None

    def test_evaluator_schema_error(self):
        """Test evaluator handling schema validation error."""
        client = MagicMock()
        response_mock = MagicMock()
        response_mock.text = json.dumps({
            "faithfulness": {"score": "invalid"},
            "completeness": {"score": 8},
            "structural_adherence": {"score": 9, "issues": []},
            "overall_pass": True
        })
        client.models.generate_content.return_value = response_mock

        result = _run_evaluator(
            self.job, self.resume_text, self.summary,
            client, "gemini-2.5-flash", self.logger
        )

        assert result is None


class TestProcessSingleJob:
    """Integration tests for _process_single_job orchestration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.job = JobContext(
            job_id=1,
            rank=1,
            score=0.95,
            title="Senior ML Engineer",
            company="Acme Corp",
            location="San Francisco, CA",
            absolute_url="https://example.com/job/1",
            cleaned_description="Build ML systems.",
            responsibilities=["Design pipelines"],
            skills=["PyTorch"],
            tools_and_platforms=["AWS"],
            experience_min_years=5,
        )
        self.resume_text = "I built ML systems. 7 years experience."
        self.logger = logging.getLogger("test")

    def test_process_single_job_success(self):
        """Test successful processing through all stages."""
        client = MagicMock()

        # CoN response (relevant)
        con_response = MagicMock()
        con_response.text = json.dumps({
            "relevance_verdict": "relevant",
            "relevance_reasoning": "Good match.",
            "contradictions": [],
            "strong_alignments": ["PyTorch expertise"]
        })

        # Generation response (valid)
        gen_response = MagicMock()
        gen_response.text = """<thinking>Great alignment.</thinking>

**Similarity 1 — Test**
Text [R:ref1] and [J:ref2].

**Similarity 2 — Test**
More text [R:ref3] and [J:ref4].

**Similarity 3 — Test**
Final text [R:ref5] and [J:ref6].
"""

        # Evaluation response (pass)
        eval_response = MagicMock()
        eval_response.text = json.dumps({
            "faithfulness": {"score": 9, "justification": "Good.", "flags": []},
            "completeness": {"score": 10, "justification": "Three similarities."},
            "structural_adherence": {"score": 10, "justification": "Valid.", "issues": []},
            "overall_pass": True
        })

        client.models.generate_content.side_effect = [con_response, gen_response, eval_response]

        result = _process_single_job(
            self.job, self.resume_text, client, "gemini-2.5-flash", 2, self.logger
        )

        assert result is not None
        assert result.job_id == 1
        assert result.rank == 1
        assert result.passed_eval is True
        assert len(result.citations) > 0
        assert len(result.evaluation) > 0

    def test_process_single_job_dropped_by_con_filter(self):
        """Test job dropped by CoN filter (irrelevant)."""
        client = MagicMock()
        con_response = MagicMock()
        con_response.text = json.dumps({
            "relevance_verdict": "irrelevant",
            "relevance_reasoning": "No overlap.",
            "contradictions": [],
            "strong_alignments": []
        })
        client.models.generate_content.return_value = con_response

        result = _process_single_job(
            self.job, self.resume_text, client, "gemini-2.5-flash", 2, self.logger
        )

        assert result is None

    def test_process_single_job_generation_fails(self):
        """Test job dropped when generation fails."""
        client = MagicMock()

        # CoN response (relevant)
        con_response = MagicMock()
        con_response.text = json.dumps({
            "relevance_verdict": "relevant",
            "relevance_reasoning": "Good match.",
            "contradictions": [],
            "strong_alignments": []
        })

        # Generation response (invalid, will retry and fail)
        gen_response = MagicMock()
        gen_response.text = "Invalid summary without structure"

        client.models.generate_content.side_effect = [con_response, gen_response, gen_response]

        result = _process_single_job(
            self.job, self.resume_text, client, "gemini-2.5-flash", 2, self.logger
        )

        assert result is None

    def test_process_single_job_evaluator_fails_still_writes(self):
        """Test job written despite evaluator failure."""
        client = MagicMock()

        # CoN response (relevant)
        con_response = MagicMock()
        con_response.text = json.dumps({
            "relevance_verdict": "relevant",
            "relevance_reasoning": "Good match.",
            "contradictions": [],
            "strong_alignments": []
        })

        # Generation response (valid)
        gen_response = MagicMock()
        gen_response.text = """<thinking>Great.</thinking>

**Similarity 1 — Test**
Text [R:ref1] and [J:ref2].

**Similarity 2 — Test**
More text [R:ref3] and [J:ref4].

**Similarity 3 — Test**
Final text [R:ref5] and [J:ref6].
"""

        # Evaluator raises exception
        client.models.generate_content.side_effect = [con_response, gen_response, Exception("API error")]

        result = _process_single_job(
            self.job, self.resume_text, client, "gemini-2.5-flash", 2, self.logger
        )

        assert result is not None
        assert result.passed_eval is False
        assert result.evaluation == {}


class TestGenerateSummaries:
    """Integration tests for generate_summaries main function."""

    def test_generate_summaries_end_to_end(self):
        """Test end-to-end generation with mocked DB and Gemini."""
        # Mock DB
        db = MagicMock()
        db.get_reranked_with_full_text.return_value = [
            (1, 1, 0.95, "Senior ML Engineer", "Acme", "CA", "https://example.com",
             "Build ML systems.", '["Design"]', '["PyTorch"]', '["AWS"]', 5),
        ]

        # Mock config
        config = MagicMock()
        config.google_api_key = "test_key"
        config.retrieval_user_profile_path = "test_profile.txt"
        config.generation_model_id = "gemini-2.5-flash"
        config.generation_top_k = 10
        config.generation_max_retries = 2

        # Mock resume file
        resume_text = "I have ML experience."

        # Mock Gemini client
        with patch("src.retrieval.load_user_profile", return_value=resume_text):
            with patch("src.generation.genai.Client") as mock_client_class:
                client = MagicMock()
                mock_client_class.return_value = client

                # Setup API responses
                con_response = MagicMock()
                con_response.text = json.dumps({
                    "relevance_verdict": "relevant",
                    "relevance_reasoning": "Good match.",
                    "contradictions": [],
                    "strong_alignments": ["ML expertise"]
                })

                gen_response = MagicMock()
                gen_response.text = """<thinking>Great alignment.</thinking>

**Similarity 1 — Test**
Text [R:ref1] and [J:ref2].

**Similarity 2 — Test**
More text [R:ref3] and [J:ref4].

**Similarity 3 — Test**
Final text [R:ref5] and [J:ref6].
"""

                eval_response = MagicMock()
                eval_response.text = json.dumps({
                    "faithfulness": {"score": 9, "justification": "Good.", "flags": []},
                    "completeness": {"score": 10, "justification": "Three."},
                    "structural_adherence": {"score": 10, "justification": "Valid.", "issues": []},
                    "overall_pass": True
                })

                client.models.generate_content.side_effect = [con_response, gen_response, eval_response]

                processed, dropped = generate_summaries(db, config)

                assert processed == 1
                assert dropped == 0
                db.insert_summaries.assert_called_once()

    def test_generate_summaries_empty_db(self):
        """Test generate_summaries with no reranked jobs."""
        db = MagicMock()
        db.get_reranked_with_full_text.return_value = []

        config = MagicMock()
        config.google_api_key = "test_key"
        config.retrieval_user_profile_path = "test_profile.txt"
        config.generation_model_id = "gemini-2.5-flash"
        config.generation_top_k = 10
        config.generation_max_retries = 2

        with patch("src.retrieval.load_user_profile", return_value="resume"):
            with patch("src.generation.genai.Client"):
                processed, dropped = generate_summaries(db, config)

                assert processed == 0
                assert dropped == 0
                db.insert_summaries.assert_not_called()

    def test_generate_summaries_all_dropped(self):
        """Test generate_summaries with all jobs dropped."""
        db = MagicMock()
        db.get_reranked_with_full_text.return_value = [
            (1, 1, 0.95, "Job 1", "Company 1", "CA", "https://1",
             "Description 1", '[]', '[]', '[]', None),
            (2, 2, 0.85, "Job 2", "Company 2", "NY", "https://2",
             "Description 2", '[]', '[]', '[]', None),
        ]

        config = MagicMock()
        config.google_api_key = "test_key"
        config.retrieval_user_profile_path = "test_profile.txt"
        config.generation_model_id = "gemini-2.5-flash"
        config.generation_top_k = 10
        config.generation_max_retries = 2

        with patch("src.retrieval.load_user_profile", return_value="resume"):
            with patch("src.generation.genai.Client") as mock_client_class:
                    client = MagicMock()
                    mock_client_class.return_value = client

                    # All jobs marked as irrelevant
                    con_response = MagicMock()
                    con_response.text = json.dumps({
                        "relevance_verdict": "irrelevant",
                        "relevance_reasoning": "No match.",
                        "contradictions": [],
                        "strong_alignments": []
                    })
                    client.models.generate_content.return_value = con_response

                    processed, dropped = generate_summaries(db, config)

                    assert processed == 0
                    assert dropped == 2
                    db.insert_summaries.assert_not_called()
