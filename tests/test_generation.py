"""Tests for the generation pipeline module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generation import (
    PairResult,
    RequirementMatch,
    extract_requirements,
    filter_pairs,
    find_resume_matches,
    generate_explanation,
    log_result,
    run_generation_pipeline,
    _normalize_whitespace,
    _parse_requirements,
    _parse_resume_match,
    _span_exists_in_text,
)


def make_ollama_response(content: str) -> dict:
    """Build a minimal mock ollama.chat return value."""
    return {"message": {"content": content}}


# ============================================================================
# TestSpanExistsInText
# ============================================================================


class TestSpanExistsInText:
    """Tests for _span_exists_in_text span validation."""

    def test_exact_match(self):
        """Span present verbatim in text returns True."""
        assert _span_exists_in_text("Python", "I know Python and JavaScript")

    def test_not_present(self):
        """Span absent from text returns False."""
        assert not _span_exists_in_text("Rust", "I know Python and JavaScript")

    def test_regex_special_chars_in_span(self):
        """Span containing regex metacharacters does not raise and validates correctly."""
        # C++ has metacharacters
        assert _span_exists_in_text("C++", "Experience with C++ and Java")
        assert not _span_exists_in_text("C++", "Experience with CX and Java")

        # .NET has a dot
        assert _span_exists_in_text(".NET", "Proficient in .NET and Python")
        assert not _span_exists_in_text(".NET", "Proficient in NET and Python")

    def test_empty_span(self):
        """Empty span returns True (documents intended behavior)."""
        assert _span_exists_in_text("", "any text")

    def test_case_sensitive(self):
        """Validation is case-sensitive."""
        assert not _span_exists_in_text("Python", "python")
        assert _span_exists_in_text("Python", "Python")


# ============================================================================
# TestParseRequirements
# ============================================================================


class TestParseRequirements:
    """Tests for _parse_requirements parsing."""

    def test_numbered_list(self):
        """Numbered list with 1., 2., etc. parses correctly."""
        response = "1. Python\n2. SQL\n3. Docker"
        assert _parse_requirements(response) == ["Python", "SQL", "Docker"]

    def test_bulleted_list(self):
        """Bulleted list with - or * parses correctly."""
        response = "- Python\n- SQL\n* Docker"
        assert _parse_requirements(response) == ["Python", "SQL", "Docker"]

    def test_strips_whitespace(self):
        """Extra whitespace around spans is stripped."""
        response = "1.   Python   \n2.  SQL  \n"
        assert _parse_requirements(response) == ["Python", "SQL"]

    def test_empty_response(self):
        """Empty string returns empty list."""
        assert _parse_requirements("") == []
        assert _parse_requirements("   \n\n  ") == []

    def test_single_item(self):
        """Single item without numbering returns one-element list."""
        response = "Python"
        assert _parse_requirements(response) == ["Python"]

    def test_skips_blank_lines(self):
        """Blank lines are not included as spans."""
        response = "1. Python\n\n\n2. SQL"
        assert _parse_requirements(response) == ["Python", "SQL"]

    def test_mixed_numbering_styles(self):
        """Mixed numbering (1. and 1)) handles correctly."""
        response = "1. Python\n2) SQL\n3. Docker"
        assert _parse_requirements(response) == ["Python", "SQL", "Docker"]


# ============================================================================
# TestParseResumeMatch
# ============================================================================


class TestParseResumeMatch:
    """Tests for _parse_resume_match parsing."""

    def test_valid_span(self):
        """Valid span is returned stripped."""
        assert _parse_resume_match("  5 years of Python  ") == "5 years of Python"

    def test_not_found_exact(self):
        """Exact 'NOT FOUND' returns None."""
        assert _parse_resume_match("NOT FOUND") is None

    def test_not_found_case_insensitive(self):
        """'not found' (lowercase) returns None."""
        assert _parse_resume_match("not found") is None
        assert _parse_resume_match("Not Found") is None

    def test_not_found_with_surrounding_whitespace(self):
        """'NOT FOUND' with whitespace returns None."""
        assert _parse_resume_match("  NOT FOUND  ") is None

    def test_empty_response(self):
        """Empty or whitespace-only response returns None."""
        assert _parse_resume_match("") is None
        assert _parse_resume_match("   ") is None


# ============================================================================
# TestNormalizeWhitespace
# ============================================================================


class TestNormalizeWhitespace:
    """Tests for _normalize_whitespace normalization."""

    def test_collapses_spaces(self):
        """Multiple spaces become one."""
        assert _normalize_whitespace("Python   and   Java") == "Python and Java"

    def test_collapses_newlines(self):
        """Newlines are collapsed to single space."""
        assert _normalize_whitespace("Python\nand\nJava") == "Python and Java"

    def test_strips_leading_trailing(self):
        """Leading and trailing whitespace is stripped."""
        assert _normalize_whitespace("  Python  ") == "Python"

    def test_tab_collapsed(self):
        """Tabs are normalized to single space."""
        assert _normalize_whitespace("Python\t\tand\tJava") == "Python and Java"

    def test_mixed_whitespace(self):
        """Mixed whitespace types are normalized."""
        assert _normalize_whitespace("  Python \n\t and  \n Java  ") == "Python and Java"


# ============================================================================
# TestExtractRequirements
# ============================================================================


class TestExtractRequirements:
    """Tests for extract_requirements function."""

    @patch("generation.ollama.chat")
    def test_happy_path_returns_validated_spans(self, mock_chat):
        """LLM returns spans present in job_posting; all are returned."""
        job_posting = "We need Python, SQL, and Docker experience"
        mock_chat.return_value = make_ollama_response("1. Python\n2. SQL\n3. Docker")

        result = extract_requirements(job_posting)

        assert result == ["Python", "SQL", "Docker"]
        mock_chat.assert_called_once()

    @patch("generation.ollama.chat")
    def test_hallucinated_spans_discarded(self, mock_chat):
        """LLM returns spans not in job_posting; none are returned."""
        job_posting = "We need Python experience"
        mock_chat.return_value = make_ollama_response("1. Python\n2. Rust\n3. Go")

        result = extract_requirements(job_posting)

        assert result == ["Python"]

    @patch("generation.ollama.chat")
    def test_partial_validation(self, mock_chat):
        """Mix of valid and invalid spans; only valid ones returned."""
        job_posting = "Python and Java developer role"
        mock_chat.return_value = make_ollama_response("1. Python\n2. Rust\n3. Java")

        result = extract_requirements(job_posting)

        assert result == ["Python", "Java"]

    @patch("generation.ollama.chat")
    def test_empty_job_posting(self, mock_chat):
        """Empty job posting returns empty list."""
        mock_chat.return_value = make_ollama_response("1. Python\n2. SQL")

        result = extract_requirements("")

        assert result == []

    @patch("generation.ollama.chat")
    def test_ollama_called_with_correct_options(self, mock_chat):
        """ollama.chat is called with correct temperature, top_p, num_predict."""
        mock_chat.return_value = make_ollama_response("1. Python")

        extract_requirements("Python job posting")

        call_args = mock_chat.call_args
        assert call_args[1]["options"]["temperature"] == 0.7
        assert call_args[1]["options"]["top_p"] == 0.9
        assert call_args[1]["options"]["num_predict"] == 150

    @patch("generation.ollama.chat")
    def test_custom_model_passed_to_ollama(self, mock_chat):
        """Custom model string is forwarded to ollama.chat."""
        mock_chat.return_value = make_ollama_response("1. Python")

        extract_requirements("Python job", model="custom-model:latest")

        assert mock_chat.call_args[1]["model"] == "custom-model:latest"


# ============================================================================
# TestFindResumeMatches
# ============================================================================


class TestFindResumeMatches:
    """Tests for find_resume_matches function."""

    @patch("generation.ollama.chat")
    def test_happy_path_all_match(self, mock_chat):
        """All LLM-returned spans exist in resume; returns full list, 0 hallucinations."""
        resume = "5 years of Python and SQL experience"
        requirements = ["Python", "SQL"]
        mock_chat.side_effect = [
            make_ollama_response("5 years of Python"),  # Python match
            make_ollama_response("SQL experience"),      # SQL match
        ]

        validated_pairs, hallucination_count = find_resume_matches(resume, requirements)

        assert len(validated_pairs) == 2
        assert validated_pairs[0]["requirement"] == "Python"
        assert validated_pairs[0]["resume_match"] == "5 years of Python"
        assert validated_pairs[1]["requirement"] == "SQL"
        assert hallucination_count == 0

    @patch("generation.ollama.chat")
    def test_not_found_excluded(self, mock_chat):
        """LLM returns 'NOT FOUND'; excluded from validated_pairs, NOT counted as hallucination."""
        resume = "5 years of Python"
        requirements = ["Python", "Rust"]
        mock_chat.side_effect = [
            make_ollama_response("5 years of Python"),
            make_ollama_response("NOT FOUND"),
        ]

        validated_pairs, hallucination_count = find_resume_matches(resume, requirements)

        assert len(validated_pairs) == 1
        assert validated_pairs[0]["requirement"] == "Python"
        assert hallucination_count == 0

    @patch("generation.ollama.chat")
    def test_hallucination_counted(self, mock_chat):
        """LLM returns span not in resume; counted as hallucination."""
        resume = "5 years of Python"
        requirements = ["Rust"]
        mock_chat.return_value = make_ollama_response("Experienced in Rust")

        validated_pairs, hallucination_count = find_resume_matches(resume, requirements)

        assert len(validated_pairs) == 0
        assert hallucination_count == 1

    @patch("generation.ollama.chat")
    def test_whitespace_normalization_applied(self, mock_chat):
        """Span with extra whitespace is normalized and matched."""
        resume = "5 years of Python"
        requirements = ["Python"]
        mock_chat.return_value = make_ollama_response("5   years   of   Python")

        validated_pairs, hallucination_count = find_resume_matches(resume, requirements)

        assert len(validated_pairs) == 1
        assert validated_pairs[0]["resume_match"] == "5 years of Python"
        assert hallucination_count == 0

    @patch("generation.ollama.chat")
    def test_empty_requirements(self, mock_chat):
        """Empty requirements list returns ([], 0)."""
        validated_pairs, hallucination_count = find_resume_matches("any resume", [])

        assert validated_pairs == []
        assert hallucination_count == 0
        mock_chat.assert_not_called()

    @patch("generation.ollama.chat")
    def test_multiple_requirements_some_hallucinated(self, mock_chat):
        """3 requirements: 2 valid, 1 hallucinated."""
        resume = "Python and SQL experience"
        requirements = ["Python and", "SQL", "Rust"]
        mock_chat.side_effect = [
            make_ollama_response("Python and"),
            make_ollama_response("SQL experience"),
            make_ollama_response("Rust skills"),  # Not in resume
        ]

        validated_pairs, hallucination_count = find_resume_matches(resume, requirements)

        assert len(validated_pairs) == 2
        assert hallucination_count == 1

    @patch("generation.ollama.chat")
    def test_ollama_called_once_per_requirement(self, mock_chat):
        """With 3 requirements, ollama.chat called exactly 3 times."""
        mock_chat.side_effect = [
            make_ollama_response("match1"),
            make_ollama_response("match2"),
            make_ollama_response("NOT FOUND"),
        ]

        find_resume_matches("resume text", ["req1", "req2", "req3"])

        assert mock_chat.call_count == 3


# ============================================================================
# TestFilterPairs
# ============================================================================


class TestFilterPairs:
    """Tests for filter_pairs function."""

    @patch("generation.ollama.chat")
    def test_pair_with_matches_retained(self, mock_chat):
        """Pair with validated matches is kept."""
        mock_chat.side_effect = [
            make_ollama_response("1. Python"),           # extract_requirements
            make_ollama_response("5 years of Python"),   # find_resume_matches
        ]

        pairs = [("5 years of Python", "Need Python")]
        retained, corpus_msg = filter_pairs(pairs)

        assert len(retained) == 1
        assert corpus_msg is None

    @patch("generation.ollama.chat")
    def test_pair_with_zero_matches_retained(self, mock_chat):
        """Pair with zero validated matches is retained with corpus warning."""
        mock_chat.side_effect = [
            make_ollama_response("1. Rust"),      # extract_requirements
            make_ollama_response("NOT FOUND"),     # find_resume_matches
        ]

        pairs = [("Python resume", "Need Rust")]
        retained, corpus_msg = filter_pairs(pairs)

        assert len(retained) == 1
        assert retained[0][2] == []  # validated_pairs is empty
        assert corpus_msg is not None
        assert "corpus limitations" in corpus_msg

    @patch("generation.ollama.chat")
    def test_all_pairs_have_zero_matches_returns_corpus_message(self, mock_chat):
        """All pairs have zero matches; still returns all pairs with corpus message."""
        mock_chat.side_effect = [
            make_ollama_response("1. Rust"),      # pair 1: extract
            make_ollama_response("NOT FOUND"),     # pair 1: find
            make_ollama_response("1. Go"),        # pair 2: extract
            make_ollama_response("NOT FOUND"),     # pair 2: find
        ]

        pairs = [("Python resume", "Need Rust"), ("Python resume", "Need Go")]
        retained, corpus_msg = filter_pairs(pairs)

        assert len(retained) == 2
        assert retained[0][2] == []  # both pairs have empty validated_pairs
        assert retained[1][2] == []
        assert corpus_msg is not None
        assert "corpus limitations" in corpus_msg

    @patch("generation.ollama.chat")
    def test_partial_scrapping_returns_corpus_message(self, mock_chat):
        """Some pairs have zero matches; returns corpus warning."""
        mock_chat.side_effect = [
            make_ollama_response("1. Python"),           # pair 1: extract
            make_ollama_response("5 years of Python"),   # pair 1: find
            make_ollama_response("1. Rust"),             # pair 2: extract
            make_ollama_response("NOT FOUND"),           # pair 2: find
        ]

        pairs = [("5 years of Python", "Need Python"), ("Python resume", "Need Rust")]
        retained, corpus_msg = filter_pairs(pairs)

        assert len(retained) == 2  # both pairs retained
        assert retained[0][2] != []  # pair 1 has matches
        assert retained[1][2] == []  # pair 2 has no matches
        assert corpus_msg is not None  # corpus warning set
        assert "corpus limitations" in corpus_msg


# ============================================================================
# TestGenerateExplanation
# ============================================================================


class TestGenerateExplanation:
    """Tests for generate_explanation function."""

    @patch("generation.ollama.chat")
    def test_returns_explanation_string(self, mock_chat):
        """LLM response is returned as stripped string."""
        pairs = [
            {"requirement": "Python", "resume_match": "5 years of Python"},
            {"requirement": "SQL", "resume_match": "SQL experience"},
        ]
        mock_chat.return_value = make_ollama_response("  Strong fit with Python and SQL.  ")

        result = generate_explanation(pairs)

        assert result == "Strong fit with Python and SQL."

    @patch("generation.ollama.chat")
    def test_whitespace_stripped(self, mock_chat):
        """Leading/trailing whitespace is stripped."""
        mock_chat.return_value = make_ollama_response("\n\nExplanation.\n\n")

        result = generate_explanation([])

        assert result == "Explanation."

    @patch("generation.ollama.chat")
    def test_ollama_called_with_validated_pairs(self, mock_chat):
        """Prompt contains the requirement and resume_match text."""
        pairs = [
            {"requirement": "Python", "resume_match": "5 years of Python"},
        ]
        mock_chat.return_value = make_ollama_response("Fit explanation")

        generate_explanation(pairs)

        prompt = mock_chat.call_args[1]["messages"][0]["content"]
        assert "Python" in prompt
        assert "5 years of Python" in prompt

    @patch("generation.ollama.chat")
    def test_empty_validated_pairs(self, mock_chat):
        """Empty list still triggers LLM call."""
        mock_chat.return_value = make_ollama_response("Some explanation")

        result = generate_explanation([])

        assert result == "Some explanation"
        mock_chat.assert_called_once()


# ============================================================================
# TestLogResult
# ============================================================================


class TestLogResult:
    """Tests for log_result logging."""

    def test_logs_explanation(self, caplog):
        """Explanation text appears in log output."""
        result: PairResult = {
            "explanation": "Strong fit",
            "validated_pairs": [],
            "num_validated_pairs": 2,
            "hallucination_count": 0,
            "flagged_for_review": False,
        }

        with caplog.at_level("INFO"):
            log_result(result)

        assert "Strong fit" in caplog.text

    def test_logs_num_validated_pairs(self, caplog):
        """num_validated_pairs appears in log output."""
        result: PairResult = {
            "explanation": "Test",
            "validated_pairs": [],
            "num_validated_pairs": 3,
            "hallucination_count": 0,
            "flagged_for_review": False,
        }

        with caplog.at_level("INFO"):
            log_result(result)

        assert "3" in caplog.text

    def test_logs_hallucination_count(self, caplog):
        """hallucination_count appears in log output."""
        result: PairResult = {
            "explanation": "Test",
            "validated_pairs": [],
            "num_validated_pairs": 1,
            "hallucination_count": 2,
            "flagged_for_review": False,
        }

        with caplog.at_level("INFO"):
            log_result(result)

        assert "2" in caplog.text

    def test_flagged_logs_warning(self, caplog):
        """flagged_for_review=True produces a WARNING-level log entry."""
        result: PairResult = {
            "explanation": "Test",
            "validated_pairs": [],
            "num_validated_pairs": 1,
            "hallucination_count": 1,
            "flagged_for_review": True,
        }

        log_result(result)

        assert "WARNING" in caplog.text
        assert "manual review" in caplog.text

    def test_not_flagged_no_warning(self, caplog):
        """flagged_for_review=False produces no WARNING-level entry."""
        result: PairResult = {
            "explanation": "Test",
            "validated_pairs": [],
            "num_validated_pairs": 1,
            "hallucination_count": 0,
            "flagged_for_review": False,
        }

        log_result(result)

        # Check that WARNING is not in caplog
        assert "WARNING" not in caplog.text or "manual review" not in caplog.text


# ============================================================================
# TestRunGenerationPipeline
# ============================================================================


class TestRunGenerationPipeline:
    """Integration tests for run_generation_pipeline."""

    @patch("generation.ollama.chat")
    def test_single_pair_happy_path(self, mock_chat):
        """Single pair returns tuple (list, corpus_msg) with one PairResult."""
        mock_chat.side_effect = [
            make_ollama_response("1. Python"),      # extract_requirements
            make_ollama_response("Python"),         # find_resume_matches
            make_ollama_response("Good Python."),   # generate_explanation
        ]

        pairs = [("Python developer", "Need Python")]
        result, corpus_msg = run_generation_pipeline(pairs)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["explanation"] == "Good Python."
        assert corpus_msg is None

    @patch("generation.ollama.chat")
    def test_returns_list_of_pair_results(self, mock_chat):
        """Happy path returns tuple (list, corpus_msg) of PairResults with generated explanations."""
        mock_chat.side_effect = [
            make_ollama_response("Python and Javascript"),
            make_ollama_response("Python and Javascript"),
            make_ollama_response("Strong web developer."),
        ]

        pairs = [("Python and Javascript expert", "Required: Python and Javascript")]
        result, corpus_msg = run_generation_pipeline(pairs)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["num_validated_pairs"] == 1
        assert result[0]["explanation"] == "Strong web developer."
        assert corpus_msg is None

    @patch("generation.ollama.chat")
    def test_returns_corpus_message_when_all_pairs_have_zero_matches(self, mock_chat):
        """All pairs have zero matches; returns corpus_message in tuple."""
        mock_chat.side_effect = [
            make_ollama_response("1. Rust"),
            make_ollama_response("NOT FOUND"),
        ]

        pairs = [("Python resume", "Need Rust")]
        result, corpus_msg = run_generation_pipeline(pairs)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["explanation"] is None  # no explanation generated
        assert result[0]["num_validated_pairs"] == 0
        assert corpus_msg is not None
        assert "corpus limitations" in corpus_msg

    @patch("generation.ollama.chat")
    def test_pair_result_has_required_keys(self, mock_chat):
        """Output PairResult contains all required keys."""
        mock_chat.side_effect = [
            make_ollama_response("1. Python"),
            make_ollama_response("Python"),
            make_ollama_response("Fit explanation"),
        ]

        pairs = [("Python", "Need Python")]
        result, _ = run_generation_pipeline(pairs)

        assert isinstance(result, list)
        pair_result = result[0]
        assert "explanation" in pair_result
        assert "num_validated_pairs" in pair_result
        assert "hallucination_count" in pair_result
        assert "flagged_for_review" in pair_result
        assert "validated_pairs" in pair_result

    def test_batch_size_validation_raises_value_error(self):
        """Batch with >10 pairs raises ValueError."""
        pairs = [("resume", "job")] * 11

        with pytest.raises(ValueError):
            run_generation_pipeline(pairs)

    @patch("generation.ollama.chat")
    def test_hallucinated_pairs_flagged(self, mock_chat):
        """Pair with hallucinations has flagged_for_review=True."""
        mock_chat.side_effect = [
            make_ollama_response("1. Python\n2. Rust"),  # extract_requirements - 2 skills
            make_ollama_response("Python"),              # find_resume_matches for Python - valid
            make_ollama_response("Hallucinated"),        # find_resume_matches for Rust - hallucination
            make_ollama_response("Explanation"),         # generate_explanation
        ]

        pairs = [("Python developer", "Need Python and Rust")]
        result, _ = run_generation_pipeline(pairs)

        assert isinstance(result, list)
        assert result[0]["flagged_for_review"] is True
        assert result[0]["hallucination_count"] == 1

    @patch("generation.ollama.chat")
    def test_zero_hallucinations_not_flagged(self, mock_chat):
        """Pair with no hallucinations has flagged_for_review=False."""
        mock_chat.side_effect = [
            make_ollama_response("1. Python"),
            make_ollama_response("Python"),
            make_ollama_response("Explanation"),
        ]

        pairs = [("Python resume", "Need Python")]
        result, _ = run_generation_pipeline(pairs)

        assert result[0]["flagged_for_review"] is False
        assert result[0]["hallucination_count"] == 0

    @patch("generation.ollama.chat")
    def test_custom_model_forwarded(self, mock_chat):
        """Custom model string flows through to ollama.chat."""
        mock_chat.side_effect = [
            make_ollama_response("1. Python"),
            make_ollama_response("Python"),
            make_ollama_response("Explanation"),
        ]

        pairs = [("Python", "Need Python")]
        _, _ = run_generation_pipeline(pairs, model="custom-model:latest")

        # Check that custom model was used in at least one call
        model_arg = None
        for call in mock_chat.call_args_list:
            if call[1].get("model") == "custom-model:latest":
                model_arg = "custom-model:latest"
                break

        assert model_arg == "custom-model:latest"
