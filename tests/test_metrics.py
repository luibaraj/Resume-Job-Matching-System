"""
Unit tests for the eval.metrics module.

Tests precision@k, recall@k, single-query and batch evaluation,
and all edge cases (empty inputs, k > retrieved length, empty ground truth).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.metrics import (
    MetricsAtK,
    BatchMetricsAtK,
    precision_at_k,
    recall_at_k,
    compute_metrics_at_k,
    batch_precision_at_k,
    batch_recall_at_k,
    batch_compute_metrics_at_k,
)


@pytest.fixture
def retrieved_ids() -> list[str]:
    """Ranked list of 10 retrieved IDs for standard test scenarios."""
    return ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "j8", "j9", "j10"]


@pytest.fixture
def relevant_ids() -> set[str]:
    """Ground-truth set: j1, j3, j5 are relevant (3 total)."""
    return {"j1", "j3", "j5"}


class TestPrecisionAtK:
    """Tests for precision_at_k function."""

    def test_perfect_precision(
        self, retrieved_ids: list[str], relevant_ids: set[str]
    ) -> None:
        """All top-k items are relevant."""
        # j1, j3, j5 in top-5 are all relevant (3 total)
        assert precision_at_k(retrieved_ids, relevant_ids, k=5) == 0.6

    def test_partial_precision(
        self, retrieved_ids: list[str], relevant_ids: set[str]
    ) -> None:
        """Some top-k items are relevant."""
        # j1, j3 in top-3 are relevant (2 of 3)
        assert precision_at_k(retrieved_ids, relevant_ids, k=3) == pytest.approx(2 / 3)

    def test_no_relevant_in_top_k(self, retrieved_ids: list[str]) -> None:
        """No relevant items in top-k."""
        relevant_ids = {"j11", "j12"}  # Not in retrieved list
        assert precision_at_k(retrieved_ids, relevant_ids, k=5) == 0.0

    def test_k_greater_than_retrieved(
        self, retrieved_ids: list[str], relevant_ids: set[str]
    ) -> None:
        """k exceeds the length of retrieved list."""
        # k=20 but only 10 retrieved; uses len(top_k)=10 as denominator
        # All 3 relevant in 10 items
        assert precision_at_k(retrieved_ids, relevant_ids, k=20) == 0.3

    def test_empty_retrieved(self, relevant_ids: set[str]) -> None:
        """Empty retrieved list."""
        assert precision_at_k([], relevant_ids, k=5) == 0.0

    def test_empty_relevant(self, retrieved_ids: list[str]) -> None:
        """Empty ground truth set."""
        assert precision_at_k(retrieved_ids, set(), k=5) == 0.0

    def test_k_equals_one_hit(
        self, retrieved_ids: list[str], relevant_ids: set[str]
    ) -> None:
        """k=1 and first item is relevant."""
        assert precision_at_k(retrieved_ids, relevant_ids, k=1) == 1.0

    def test_k_equals_one_miss(self, retrieved_ids: list[str]) -> None:
        """k=1 and first item is not relevant."""
        relevant_ids = {"j3"}  # j3 is at position 2
        assert precision_at_k(retrieved_ids, relevant_ids, k=1) == 0.0

    def test_invalid_k_zero(
        self, retrieved_ids: list[str], relevant_ids: set[str]
    ) -> None:
        """k=0 raises ValueError."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            precision_at_k(retrieved_ids, relevant_ids, k=0)

    def test_invalid_k_negative(
        self, retrieved_ids: list[str], relevant_ids: set[str]
    ) -> None:
        """k < 0 raises ValueError."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            precision_at_k(retrieved_ids, relevant_ids, k=-1)


class TestRecallAtK:
    """Tests for recall_at_k function."""

    def test_full_recall(
        self, retrieved_ids: list[str], relevant_ids: set[str]
    ) -> None:
        """All relevant items are in top-k."""
        # All 3 relevant (j1, j3, j5) are in top-5
        assert recall_at_k(retrieved_ids, relevant_ids, k=5) == 1.0

    def test_partial_recall(
        self, retrieved_ids: list[str], relevant_ids: set[str]
    ) -> None:
        """Some relevant items are in top-k."""
        # j1, j3 in top-3 (2 of 3 relevant)
        assert recall_at_k(retrieved_ids, relevant_ids, k=3) == pytest.approx(2 / 3)

    def test_zero_recall(self, retrieved_ids: list[str]) -> None:
        """No relevant items in top-k."""
        relevant_ids = {"j11", "j12"}
        assert recall_at_k(retrieved_ids, relevant_ids, k=5) == 0.0

    def test_k_greater_than_retrieved(
        self, retrieved_ids: list[str], relevant_ids: set[str]
    ) -> None:
        """k exceeds the length of retrieved list."""
        # k=20 but only 10 retrieved; all 3 relevant are found
        assert recall_at_k(retrieved_ids, relevant_ids, k=20) == 1.0

    def test_empty_retrieved(self, relevant_ids: set[str]) -> None:
        """Empty retrieved list."""
        assert recall_at_k([], relevant_ids, k=5) == 0.0

    def test_empty_relevant(self, retrieved_ids: list[str]) -> None:
        """Empty ground truth set."""
        assert recall_at_k(retrieved_ids, set(), k=5) == 0.0

    def test_single_relevant_found(self, retrieved_ids: list[str]) -> None:
        """Only one of multiple relevant items is found."""
        relevant_ids = {"j1", "j11", "j12"}  # j1 found, j11 and j12 not in list
        assert recall_at_k(retrieved_ids, relevant_ids, k=5) == pytest.approx(1 / 3)

    def test_invalid_k_zero(
        self, retrieved_ids: list[str], relevant_ids: set[str]
    ) -> None:
        """k=0 raises ValueError."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            recall_at_k(retrieved_ids, relevant_ids, k=0)

    def test_invalid_k_negative(
        self, retrieved_ids: list[str], relevant_ids: set[str]
    ) -> None:
        """k < 0 raises ValueError."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            recall_at_k(retrieved_ids, relevant_ids, k=-1)


class TestComputeMetricsAtK:
    """Tests for compute_metrics_at_k function."""

    def test_returns_metrics_at_k_typeddict(
        self, retrieved_ids: list[str], relevant_ids: set[str]
    ) -> None:
        """Return type has correct keys."""
        result = compute_metrics_at_k(retrieved_ids, relevant_ids, k_values=[1, 5, 10])
        assert "k_values" in result
        assert "precision" in result
        assert "recall" in result

    def test_k_values_preserved(
        self, retrieved_ids: list[str], relevant_ids: set[str]
    ) -> None:
        """k_values list is preserved in return value."""
        k_values = [1, 5, 10]
        result = compute_metrics_at_k(retrieved_ids, relevant_ids, k_values=k_values)
        assert result["k_values"] == k_values

    def test_precision_keys_match_k_values(
        self, retrieved_ids: list[str], relevant_ids: set[str]
    ) -> None:
        """Precision dict has keys for each k value."""
        k_values = [1, 5, 10]
        result = compute_metrics_at_k(retrieved_ids, relevant_ids, k_values=k_values)
        assert set(result["precision"].keys()) == {1, 5, 10}

    def test_recall_keys_match_k_values(
        self, retrieved_ids: list[str], relevant_ids: set[str]
    ) -> None:
        """Recall dict has keys for each k value."""
        k_values = [1, 5, 10]
        result = compute_metrics_at_k(retrieved_ids, relevant_ids, k_values=k_values)
        assert set(result["recall"].keys()) == {1, 5, 10}

    def test_values_match_individual_functions(
        self, retrieved_ids: list[str], relevant_ids: set[str]
    ) -> None:
        """Metrics match values from individual functions."""
        k = 5
        result = compute_metrics_at_k(retrieved_ids, relevant_ids, k_values=[k])
        expected_precision = precision_at_k(retrieved_ids, relevant_ids, k=k)
        expected_recall = recall_at_k(retrieved_ids, relevant_ids, k=k)
        assert result["precision"][k] == expected_precision
        assert result["recall"][k] == expected_recall

    def test_empty_k_values_raises(
        self, retrieved_ids: list[str], relevant_ids: set[str]
    ) -> None:
        """Empty k_values raises ValueError."""
        with pytest.raises(ValueError, match="k_values must not be empty"):
            compute_metrics_at_k(retrieved_ids, relevant_ids, k_values=[])

    def test_single_k_value(
        self, retrieved_ids: list[str], relevant_ids: set[str]
    ) -> None:
        """Single k value produces single-entry dicts."""
        result = compute_metrics_at_k(retrieved_ids, relevant_ids, k_values=[10])
        assert len(result["precision"]) == 1
        assert len(result["recall"]) == 1


class TestBatchPrecisionAtK:
    """Tests for batch_precision_at_k function."""

    @pytest.fixture
    def batch_data(self) -> tuple[list[list[str]], list[set[str]]]:
        """Two queries: one strong match, one poor match."""
        return (
            [["j1", "j2", "j3"], ["j4", "j5", "j6"]],
            [{"j1", "j3"}, {"j7", "j8"}],
        )

    def test_mean_across_queries(
        self, batch_data: tuple[list[list[str]], list[set[str]]]
    ) -> None:
        """Mean precision across queries is computed correctly."""
        batch_retrieved, batch_relevant = batch_data
        # Query 1: 2 of 3 relevant -> 2/3
        # Query 2: 0 of 3 -> 0
        # Mean: (2/3 + 0) / 2 = 1/3
        result = batch_precision_at_k(batch_retrieved, batch_relevant, k=3)
        assert result == pytest.approx(1 / 3)

    def test_empty_batch_returns_zero(self) -> None:
        """Empty batch returns 0.0."""
        assert batch_precision_at_k([], [], k=5) == 0.0

    def test_mismatched_batch_lengths_raises(self) -> None:
        """Unequal batch lengths raise ValueError."""
        with pytest.raises(ValueError, match="must have the same length"):
            batch_precision_at_k(
                [["j1", "j2"]], [{"j1"}, {"j2"}], k=5
            )

    def test_perfect_batch(self) -> None:
        """All retrieved items are relevant."""
        batch_retrieved = [["j1", "j2"], ["j3", "j4"]]
        batch_relevant = [{"j1", "j2"}, {"j3", "j4"}]
        assert batch_precision_at_k(batch_retrieved, batch_relevant, k=5) == 1.0

    def test_zero_batch(self) -> None:
        """No relevant items in any query."""
        batch_retrieved = [["j1", "j2"], ["j3", "j4"]]
        batch_relevant = [{"j5"}, {"j6"}]
        assert batch_precision_at_k(batch_retrieved, batch_relevant, k=5) == 0.0

    def test_invalid_k_raises(
        self, batch_data: tuple[list[list[str]], list[set[str]]]
    ) -> None:
        """k < 1 raises ValueError."""
        batch_retrieved, batch_relevant = batch_data
        with pytest.raises(ValueError, match="k must be >= 1"):
            batch_precision_at_k(batch_retrieved, batch_relevant, k=0)


class TestBatchRecallAtK:
    """Tests for batch_recall_at_k function."""

    @pytest.fixture
    def batch_data(self) -> tuple[list[list[str]], list[set[str]]]:
        """Two queries with different recall profiles."""
        return (
            [["j1", "j2", "j3"], ["j4", "j5", "j6"]],
            [{"j1", "j3"}, {"j7", "j8"}],
        )

    def test_mean_across_queries(
        self, batch_data: tuple[list[list[str]], list[set[str]]]
    ) -> None:
        """Mean recall across queries is computed correctly."""
        batch_retrieved, batch_relevant = batch_data
        # Query 1: 2 of 2 relevant found -> 1.0
        # Query 2: 0 of 2 relevant found -> 0.0
        # Mean: (1.0 + 0.0) / 2 = 0.5
        result = batch_recall_at_k(batch_retrieved, batch_relevant, k=3)
        assert result == pytest.approx(0.5)

    def test_empty_batch_returns_zero(self) -> None:
        """Empty batch returns 0.0."""
        assert batch_recall_at_k([], [], k=5) == 0.0

    def test_mismatched_batch_lengths_raises(self) -> None:
        """Unequal batch lengths raise ValueError."""
        with pytest.raises(ValueError, match="must have the same length"):
            batch_recall_at_k([["j1", "j2"]], [{"j1"}, {"j2"}], k=5)

    def test_perfect_batch(self) -> None:
        """All relevant items are retrieved in all queries."""
        batch_retrieved = [["j1", "j2"], ["j3", "j4"]]
        batch_relevant = [{"j1", "j2"}, {"j3", "j4"}]
        assert batch_recall_at_k(batch_retrieved, batch_relevant, k=5) == 1.0

    def test_zero_batch(self) -> None:
        """No relevant items found in any query."""
        batch_retrieved = [["j1", "j2"], ["j3", "j4"]]
        batch_relevant = [{"j5"}, {"j6"}]
        assert batch_recall_at_k(batch_retrieved, batch_relevant, k=5) == 0.0

    def test_invalid_k_raises(
        self, batch_data: tuple[list[list[str]], list[set[str]]]
    ) -> None:
        """k < 1 raises ValueError."""
        batch_retrieved, batch_relevant = batch_data
        with pytest.raises(ValueError, match="k must be >= 1"):
            batch_recall_at_k(batch_retrieved, batch_relevant, k=-1)


class TestBatchComputeMetricsAtK:
    """Tests for batch_compute_metrics_at_k function."""

    @pytest.fixture
    def batch_data(self) -> tuple[list[list[str]], list[set[str]]]:
        """Two queries for batch evaluation."""
        return (
            [["j1", "j2", "j3"], ["j4", "j5", "j6"]],
            [{"j1", "j3"}, {"j7", "j8"}],
        )

    def test_returns_batch_metrics_typeddict(
        self, batch_data: tuple[list[list[str]], list[set[str]]]
    ) -> None:
        """Return type has correct keys."""
        batch_retrieved, batch_relevant = batch_data
        result = batch_compute_metrics_at_k(batch_retrieved, batch_relevant, [5, 10])
        assert "k_values" in result
        assert "mean_precision" in result
        assert "mean_recall" in result
        assert "num_queries" in result

    def test_num_queries_correct(
        self, batch_data: tuple[list[list[str]], list[set[str]]]
    ) -> None:
        """num_queries reflects the batch size."""
        batch_retrieved, batch_relevant = batch_data
        result = batch_compute_metrics_at_k(batch_retrieved, batch_relevant, [5, 10])
        assert result["num_queries"] == 2

    def test_mean_values_match_individual_batch_functions(
        self, batch_data: tuple[list[list[str]], list[set[str]]]
    ) -> None:
        """Composed values match individual batch functions."""
        batch_retrieved, batch_relevant = batch_data
        k = 5
        result = batch_compute_metrics_at_k(batch_retrieved, batch_relevant, [k])
        expected_precision = batch_precision_at_k(batch_retrieved, batch_relevant, k)
        expected_recall = batch_recall_at_k(batch_retrieved, batch_relevant, k)
        assert result["mean_precision"][k] == pytest.approx(expected_precision)
        assert result["mean_recall"][k] == pytest.approx(expected_recall)

    def test_empty_k_values_raises(
        self, batch_data: tuple[list[list[str]], list[set[str]]]
    ) -> None:
        """Empty k_values raises ValueError."""
        batch_retrieved, batch_relevant = batch_data
        with pytest.raises(ValueError, match="k_values must not be empty"):
            batch_compute_metrics_at_k(batch_retrieved, batch_relevant, [])

    def test_mismatched_batch_lengths_raises(self) -> None:
        """Unequal batch lengths raise ValueError."""
        with pytest.raises(ValueError, match="must have the same length"):
            batch_compute_metrics_at_k(
                [["j1", "j2"]], [{"j1"}, {"j2"}], [5, 10]
            )

    def test_empty_batch_returns_zero_means(self) -> None:
        """Empty batch returns 0.0 for all metrics."""
        result = batch_compute_metrics_at_k([], [], [5, 10])
        assert all(v == 0.0 for v in result["mean_precision"].values())
        assert all(v == 0.0 for v in result["mean_recall"].values())
        assert result["num_queries"] == 0
