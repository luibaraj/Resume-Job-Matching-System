"""
Unit tests for src/llm_extraction.py
"""

import pytest
from unittest.mock import patch, MagicMock

from src.llm_extraction import (
    extract_degree_with_llm,
    extract_seniority_with_llm,
    extract_years_with_llm,
    _parse_int_response,
)
from src.config import (
    DEGREE_UNKNOWN,
    DEGREE_BACHELOR,
    DEGREE_MASTER,
    DEGREE_PHD,
    SENIORITY_UNKNOWN,
    SENIORITY_ENTRY,
    SENIORITY_MID,
    SENIORITY_SENIOR,
    YEARS_UNKNOWN,
)


class TestParseIntResponse:
    """Test _parse_int_response helper."""

    def test_parse_valid_integer(self):
        """Should parse valid integer from response."""
        assert _parse_int_response("3", DEGREE_UNKNOWN) == 3
        assert _parse_int_response("The answer is 2", DEGREE_UNKNOWN) == 2
        assert _parse_int_response("Result: 1\nMore text", DEGREE_UNKNOWN) == 1

    def test_parse_empty_response(self):
        """Should return default for empty response."""
        assert _parse_int_response("", DEGREE_UNKNOWN) == DEGREE_UNKNOWN
        assert _parse_int_response("   ", DEGREE_UNKNOWN) == DEGREE_UNKNOWN

    def test_parse_non_integer_response(self):
        """Should return default for non-integer response."""
        assert _parse_int_response("No number here", DEGREE_UNKNOWN) == DEGREE_UNKNOWN
        assert _parse_int_response("abc", DEGREE_UNKNOWN) == DEGREE_UNKNOWN

    def test_parse_negative_integer(self):
        """Should parse negative integers."""
        assert _parse_int_response("-1", YEARS_UNKNOWN) == -1


class TestExtractDegreeWithLLM:
    """Test extract_degree_with_llm function."""

    @patch("src.llm_extraction._call_ollama")
    def test_extract_phd(self, mock_ollama):
        """Should extract PhD (3)."""
        mock_ollama.return_value = "3"
        result = extract_degree_with_llm("PhD resume text", "test-model")
        assert result == DEGREE_PHD

    @patch("src.llm_extraction._call_ollama")
    def test_extract_masters(self, mock_ollama):
        """Should extract Master's degree (2)."""
        mock_ollama.return_value = "2"
        result = extract_degree_with_llm("Master's resume text", "test-model")
        assert result == DEGREE_MASTER

    @patch("src.llm_extraction._call_ollama")
    def test_extract_bachelor(self, mock_ollama):
        """Should extract Bachelor's degree (1)."""
        mock_ollama.return_value = "1"
        result = extract_degree_with_llm("Bachelor's resume text", "test-model")
        assert result == DEGREE_BACHELOR

    @patch("src.llm_extraction._call_ollama")
    def test_extract_unknown_degree(self, mock_ollama):
        """Should return UNKNOWN (0) for no degree."""
        mock_ollama.return_value = "0"
        result = extract_degree_with_llm("No degree resume text", "test-model")
        assert result == DEGREE_UNKNOWN

    @patch("src.llm_extraction._call_ollama")
    def test_extract_invalid_degree_response(self, mock_ollama):
        """Should return UNKNOWN for invalid response."""
        mock_ollama.return_value = "Not a number"
        result = extract_degree_with_llm("Resume text", "test-model")
        assert result == DEGREE_UNKNOWN

    @patch("src.llm_extraction._call_ollama")
    def test_extract_degree_clamp_out_of_range(self, mock_ollama):
        """Should return UNKNOWN for out-of-range degree value."""
        mock_ollama.return_value = "99"
        result = extract_degree_with_llm("Resume text", "test-model")
        assert result == DEGREE_UNKNOWN

    @patch("src.llm_extraction._call_ollama")
    def test_extract_degree_ollama_exception(self, mock_ollama):
        """Should return UNKNOWN on Ollama error."""
        mock_ollama.side_effect = Exception("Ollama unavailable")
        result = extract_degree_with_llm("Resume text", "test-model")
        assert result == DEGREE_UNKNOWN


class TestExtractSeniorityWithLLM:
    """Test extract_seniority_with_llm function."""

    @patch("src.llm_extraction._call_ollama")
    def test_extract_senior(self, mock_ollama):
        """Should extract Senior (3)."""
        mock_ollama.return_value = "3"
        result = extract_seniority_with_llm("Senior resume text", "test-model")
        assert result == SENIORITY_SENIOR

    @patch("src.llm_extraction._call_ollama")
    def test_extract_mid_level(self, mock_ollama):
        """Should extract Mid-level (2)."""
        mock_ollama.return_value = "2"
        result = extract_seniority_with_llm("Mid-level resume text", "test-model")
        assert result == SENIORITY_MID

    @patch("src.llm_extraction._call_ollama")
    def test_extract_entry_level(self, mock_ollama):
        """Should extract Entry-level (1)."""
        mock_ollama.return_value = "1"
        result = extract_seniority_with_llm("Junior resume text", "test-model")
        assert result == SENIORITY_ENTRY

    @patch("src.llm_extraction._call_ollama")
    def test_extract_unknown_seniority(self, mock_ollama):
        """Should return UNKNOWN (0) for unclear seniority."""
        mock_ollama.return_value = "0"
        result = extract_seniority_with_llm("Unclear resume text", "test-model")
        assert result == SENIORITY_UNKNOWN

    @patch("src.llm_extraction._call_ollama")
    def test_extract_invalid_seniority_response(self, mock_ollama):
        """Should return UNKNOWN for invalid response."""
        mock_ollama.return_value = "Senior or Mid?"
        result = extract_seniority_with_llm("Resume text", "test-model")
        assert result == SENIORITY_UNKNOWN

    @patch("src.llm_extraction._call_ollama")
    def test_extract_seniority_clamp_out_of_range(self, mock_ollama):
        """Should return UNKNOWN for out-of-range seniority value."""
        mock_ollama.return_value = "99"
        result = extract_seniority_with_llm("Resume text", "test-model")
        assert result == SENIORITY_UNKNOWN

    @patch("src.llm_extraction._call_ollama")
    def test_extract_seniority_ollama_exception(self, mock_ollama):
        """Should return UNKNOWN on Ollama error."""
        mock_ollama.side_effect = Exception("Ollama unavailable")
        result = extract_seniority_with_llm("Resume text", "test-model")
        assert result == SENIORITY_UNKNOWN


class TestExtractYearsWithLLM:
    """Test extract_years_with_llm function."""

    @patch("src.llm_extraction._call_ollama")
    def test_extract_valid_years(self, mock_ollama):
        """Should extract valid years of experience."""
        mock_ollama.return_value = "5"
        result = extract_years_with_llm("5 years experience resume", "test-model")
        assert result == 5

    @patch("src.llm_extraction._call_ollama")
    def test_extract_high_years(self, mock_ollama):
        """Should extract high years of experience."""
        mock_ollama.return_value = "20"
        result = extract_years_with_llm("20 years experience resume", "test-model")
        assert result == 20

    @patch("src.llm_extraction._call_ollama")
    def test_extract_zero_years(self, mock_ollama):
        """Should extract 0 years (entry-level)."""
        mock_ollama.return_value = "0"
        result = extract_years_with_llm("No experience resume", "test-model")
        assert result == 0

    @patch("src.llm_extraction._call_ollama")
    def test_extract_unknown_years(self, mock_ollama):
        """Should return UNKNOWN (-1) for unclear years."""
        mock_ollama.return_value = "-1"
        result = extract_years_with_llm("Unclear resume", "test-model")
        assert result == YEARS_UNKNOWN

    @patch("src.llm_extraction._call_ollama")
    def test_extract_invalid_years_response(self, mock_ollama):
        """Should return UNKNOWN for invalid response."""
        mock_ollama.return_value = "Several years"
        result = extract_years_with_llm("Resume text", "test-model")
        assert result == YEARS_UNKNOWN

    @patch("src.llm_extraction._call_ollama")
    def test_extract_years_clamp_invalid_negative(self, mock_ollama):
        """Should return UNKNOWN for invalid negative years."""
        mock_ollama.return_value = "-5"
        result = extract_years_with_llm("Resume text", "test-model")
        assert result == YEARS_UNKNOWN

    @patch("src.llm_extraction._call_ollama")
    def test_extract_years_ollama_exception(self, mock_ollama):
        """Should return UNKNOWN on Ollama error."""
        mock_ollama.side_effect = Exception("Ollama unavailable")
        result = extract_years_with_llm("Resume text", "test-model")
        assert result == YEARS_UNKNOWN

    @patch("src.llm_extraction._call_ollama")
    def test_extract_years_with_text_noise(self, mock_ollama):
        """Should extract years from noisy response."""
        mock_ollama.return_value = "The candidate has 8 years of experience"
        result = extract_years_with_llm("Resume text", "test-model")
        assert result == 8
