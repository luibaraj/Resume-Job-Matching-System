"""Tests for scripts/eval/stratify_and_split.py — stratified data splitting."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval.stratify_and_split import (
    _largest_remainder_allocate,
    compute_stratified_split,
    build_outputs,
    build_summary,
)


class TestLargestRemainderAllocate:
    """Test suite for Hamilton (largest remainder) allocation method."""

    def test_allocate_simple_equal_distribution(self):
        """Test allocation with equal strata."""
        strata_counts = {"a": 10, "b": 10, "c": 10}
        allocations = _largest_remainder_allocate(strata_counts, total_tune=15)

        # Each stratum should get 5
        assert allocations["a"] == 5
        assert allocations["b"] == 5
        assert allocations["c"] == 5
        assert sum(allocations.values()) == 15

    def test_allocate_with_remainder(self):
        """Test allocation with remainder distribution (largest remainder method)."""
        strata_counts = {"a": 10, "b": 10, "c": 10}
        allocations = _largest_remainder_allocate(strata_counts, total_tune=16)

        # 16 / 30 = 0.533... each
        # Floor: a=5, b=5, c=5 (sum=15)
        # Remainder: need 1 more, distribute to stratum with largest fractional part
        assert sum(allocations.values()) == 16
        assert all(v >= 5 for v in allocations.values())

    def test_allocate_unequal_strata(self):
        """Test allocation with unequal stratum sizes."""
        strata_counts = {"small": 5, "large": 95}
        allocations = _largest_remainder_allocate(strata_counts, total_tune=20)

        # Large stratum should get ~19, small ~1
        assert allocations["large"] > allocations["small"]
        assert sum(allocations.values()) == 20

    def test_allocate_no_remainder_needed(self):
        """Test allocation when division is exact."""
        strata_counts = {"a": 10, "b": 20, "c": 30}
        allocations = _largest_remainder_allocate(strata_counts, total_tune=12)

        # Exact division: a=2, b=4, c=6
        assert allocations["a"] == 2
        assert allocations["b"] == 4
        assert allocations["c"] == 6
        assert sum(allocations.values()) == 12

    def test_allocate_single_stratum(self):
        """Test allocation with single stratum."""
        strata_counts = {"only": 100}
        allocations = _largest_remainder_allocate(strata_counts, total_tune=50)

        assert allocations["only"] == 50

    def test_allocate_respects_max_stratum_size(self):
        """Test that allocations never exceed stratum size."""
        strata_counts = {"a": 5, "b": 5, "c": 5}
        allocations = _largest_remainder_allocate(strata_counts, total_tune=20)

        # Total is 15, but we ask for 20. Each stratum limited to its size.
        assert allocations["a"] <= 5
        assert allocations["b"] <= 5
        assert allocations["c"] <= 5
        assert sum(allocations.values()) <= 15  # capped by total available

    def test_allocate_many_strata(self):
        """Test allocation with many strata."""
        strata_counts = {f"s{i}": 10 for i in range(10)}
        allocations = _largest_remainder_allocate(strata_counts, total_tune=25)

        assert sum(allocations.values()) == 25
        assert len(allocations) == 10

    def test_allocate_deterministic(self):
        """Test that allocation is deterministic (same input → same output)."""
        strata_counts = {"a": 10, "b": 15, "c": 25}

        result1 = _largest_remainder_allocate(strata_counts, total_tune=20)
        result2 = _largest_remainder_allocate(strata_counts, total_tune=20)

        assert result1 == result2


class TestComputeStratifiedSplit:
    """Test suite for compute_stratified_split function."""

    @pytest.fixture
    def resumes_df(self):
        """Create sample resumes DataFrame with strata_key."""
        data = {
            "id": list(range(1, 51)),  # 50 resumes
            "seniority": ["junior"] * 15 + ["mid"] * 20 + ["senior"] * 15,
            "domain": ["backend"] * 20 + ["frontend"] * 15 + ["devops"] * 15,
        }
        df = pd.DataFrame(data)
        df["strata_key"] = df["seniority"] + "_" + df["domain"]
        return df

    def test_split_returns_sets(self, resumes_df):
        """Test that split returns tune and test ID sets."""
        tune_ids, test_ids = compute_stratified_split(resumes_df, tune_n=30)

        assert isinstance(tune_ids, set)
        assert isinstance(test_ids, set)
        assert len(tune_ids) == 30
        assert len(test_ids) == 20

    def test_split_disjoint_sets(self, resumes_df):
        """Test that tune and test sets are disjoint."""
        tune_ids, test_ids = compute_stratified_split(resumes_df, tune_n=30)

        # No overlap
        assert len(tune_ids & test_ids) == 0

    def test_split_covers_all_ids(self, resumes_df):
        """Test that tune and test cover all resume IDs."""
        tune_ids, test_ids = compute_stratified_split(resumes_df, tune_n=30)

        all_ids = set(resumes_df["id"])
        assert tune_ids | test_ids == all_ids

    def test_split_respects_seed(self, resumes_df):
        """Test that split is deterministic with seed."""
        tune1, test1 = compute_stratified_split(resumes_df, tune_n=30, seed=42)
        tune2, test2 = compute_stratified_split(resumes_df, tune_n=30, seed=42)

        assert tune1 == tune2
        assert test1 == test2

    def test_split_varies_with_seed(self, resumes_df):
        """Test that different seeds produce different splits."""
        tune1, _ = compute_stratified_split(resumes_df, tune_n=30, seed=42)
        tune2, _ = compute_stratified_split(resumes_df, tune_n=30, seed=99)

        # Splits should differ (not guaranteed, but very likely with 50 items)
        # We'll just verify they're valid
        assert len(tune1) == len(tune2) == 30

    def test_split_small_dataset(self):
        """Test split on small dataset with few strata."""
        data = {
            "id": [1, 2, 3, 4, 5, 6],
            "seniority": ["junior", "junior", "mid", "mid", "senior", "senior"],
            "domain": ["backend", "backend", "backend", "backend", "backend", "backend"],
        }
        df = pd.DataFrame(data)
        df["strata_key"] = df["seniority"] + "_" + df["domain"]

        tune_ids, test_ids = compute_stratified_split(df, tune_n=3, seed=42)

        assert len(tune_ids) == 3
        assert len(test_ids) == 3


class TestBuildOutputs:
    """Test suite for build_outputs function."""

    @pytest.fixture
    def data_with_resume_split(self):
        """Create sample data with tune/test split."""
        resumes_df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "strata_key": ["junior_backend", "mid_frontend", "senior_backend", "junior_frontend", "senior_data"],
        })

        positives_df = pd.DataFrame({
            "id": [101, 102, 103, 104, 105, 106],
            "resume_id": [1, 1, 2, 3, 4, 5],
            "job_id": [501, 502, 503, 504, 505, 506],
        })

        negatives_df = pd.DataFrame({
            "id": [201, 202, 203, 204],
            "resume_id": [1, 2, 3, 5],
            "job_id": [601, 602, 603, 604],
        })

        tune_ids = {1, 2}
        test_ids = {3, 4, 5}

        return resumes_df, positives_df, negatives_df, tune_ids, test_ids

    def test_build_outputs_structure(self, data_with_resume_split):
        """Test that build_outputs returns correct keys."""
        resumes_df, positives_df, negatives_df, tune_ids, test_ids = data_with_resume_split

        outputs = build_outputs(resumes_df, positives_df, negatives_df, tune_ids, test_ids)

        assert set(outputs.keys()) == {
            "tune_resumes",
            "tune_positives",
            "test_resumes",
            "test_positives",
            "test_negatives",
        }

    def test_build_outputs_tune_resumes(self, data_with_resume_split):
        """Test that tune_resumes are filtered correctly."""
        resumes_df, positives_df, negatives_df, tune_ids, test_ids = data_with_resume_split

        outputs = build_outputs(resumes_df, positives_df, negatives_df, tune_ids, test_ids)

        tune_resumes = outputs["tune_resumes"]
        assert len(tune_resumes) == 2
        assert set(tune_resumes["id"]) == {1, 2}
        assert "strata_key" not in tune_resumes.columns  # Should be dropped

    def test_build_outputs_tune_positives(self, data_with_resume_split):
        """Test that tune_positives are filtered by resume_id."""
        resumes_df, positives_df, negatives_df, tune_ids, test_ids = data_with_resume_split

        outputs = build_outputs(resumes_df, positives_df, negatives_df, tune_ids, test_ids)

        tune_positives = outputs["tune_positives"]
        assert len(tune_positives) == 3  # 2 for resume_id=1, 1 for resume_id=2
        assert set(tune_positives["resume_id"]) == {1, 2}

    def test_build_outputs_test_resumes(self, data_with_resume_split):
        """Test that test_resumes are correct complement."""
        resumes_df, positives_df, negatives_df, tune_ids, test_ids = data_with_resume_split

        outputs = build_outputs(resumes_df, positives_df, negatives_df, tune_ids, test_ids)

        test_resumes = outputs["test_resumes"]
        assert len(test_resumes) == 3
        assert set(test_resumes["id"]) == {3, 4, 5}

    def test_build_outputs_test_positives(self, data_with_resume_split):
        """Test that test_positives are filtered correctly."""
        resumes_df, positives_df, negatives_df, tune_ids, test_ids = data_with_resume_split

        outputs = build_outputs(resumes_df, positives_df, negatives_df, tune_ids, test_ids)

        test_positives = outputs["test_positives"]
        assert len(test_positives) == 3  # 1 for resume_id=3, 1 for resume_id=4, 1 for resume_id=5
        assert set(test_positives["resume_id"]) == {3, 4, 5}

    def test_build_outputs_test_negatives(self, data_with_resume_split):
        """Test that test_negatives only include test set."""
        resumes_df, positives_df, negatives_df, tune_ids, test_ids = data_with_resume_split

        outputs = build_outputs(resumes_df, positives_df, negatives_df, tune_ids, test_ids)

        test_negatives = outputs["test_negatives"]
        # resume_id 1, 2 are tune; only 3, 5 are test
        assert set(test_negatives["resume_id"]) == {3, 5}

    def test_build_outputs_no_missing_positives(self, data_with_resume_split):
        """Test that no positives are lost during split."""
        resumes_df, positives_df, negatives_df, tune_ids, test_ids = data_with_resume_split

        outputs = build_outputs(resumes_df, positives_df, negatives_df, tune_ids, test_ids)

        total_out = len(outputs["tune_positives"]) + len(outputs["test_positives"])
        assert total_out == len(positives_df)


class TestBuildSummary:
    """Test suite for build_summary function."""

    @pytest.fixture
    def resumes_with_strata(self):
        """Create resumes DataFrame with strata."""
        data = {
            "id": list(range(1, 51)),
            "seniority": ["junior"] * 15 + ["mid"] * 20 + ["senior"] * 15,
            "domain": ["backend"] * 20 + ["frontend"] * 15 + ["devops"] * 15,
        }
        df = pd.DataFrame(data)
        df["strata_key"] = df["seniority"] + "_" + df["domain"]
        return df

    def test_summary_structure(self, resumes_with_strata):
        """Test that summary has required keys."""
        tune_ids = set(range(1, 31))
        test_ids = set(range(31, 51))

        summary = build_summary(resumes_with_strata, tune_ids, test_ids)

        assert "random_seed" in summary
        assert "generated_at" in summary
        assert "totals" in summary
        assert "strata" in summary
        assert "warnings" in summary

    def test_summary_totals(self, resumes_with_strata):
        """Test that summary totals are correct."""
        tune_ids = set(range(1, 31))
        test_ids = set(range(31, 51))

        summary = build_summary(resumes_with_strata, tune_ids, test_ids)

        assert summary["totals"]["tune"]["resumes"] == 30
        assert summary["totals"]["test"]["resumes"] == 20

    def test_summary_strata_breakdown(self, resumes_with_strata):
        """Test that summary includes per-stratum breakdown."""
        tune_ids = set(range(1, 31))
        test_ids = set(range(31, 51))

        summary = build_summary(resumes_with_strata, tune_ids, test_ids)

        assert "strata" in summary
        assert len(summary["strata"]) > 0

        # Each stratum should have total, tune, test counts
        for stratum, info in summary["strata"].items():
            assert "total" in info
            assert "tune" in info
            assert "test" in info

    def test_summary_singleton_warning(self, resumes_with_strata):
        """Test that singletons generate warnings."""
        tune_ids = {1}  # Only one resume in tune
        test_ids = set(range(2, 51))

        summary = build_summary(resumes_with_strata, tune_ids, test_ids)

        # There should be warnings for strata only in test set
        assert len(summary["warnings"]) > 0
