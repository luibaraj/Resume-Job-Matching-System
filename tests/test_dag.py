"""Tests for Airflow DAG structure and configuration.

Requires apache-airflow to be installed. Tests are automatically skipped if not available.
"""

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

# Skip entire file if airflow is not installed
airflow = pytest.importorskip("airflow", reason="apache-airflow not installed")

from airflow.providers.standard.operators.python import PythonOperator


@pytest.fixture(scope="module")
def loaded_dag():
    """Load the DAG module with Variable.get mocked.

    This patches Variable.get before importing the DAG so that the module-level
    Variable.get() calls don't fail.
    """
    with patch("airflow.models.Variable.get", return_value=""):
        # Load the DAG module using importlib
        dag_path = Path(__file__).parent.parent / "airflow" / "dag.py"
        spec = importlib.util.spec_from_file_location(
            "airflow_dag", dag_path
        )
        dag_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dag_module)
        yield dag_module.dag


class TestDagStructure:
    """Tests for DAG structure and configuration."""

    def test_dag_id(self, loaded_dag):
        """DAG ID is correct."""
        assert loaded_dag.dag_id == "job_pipeline"

    def test_schedule_interval(self, loaded_dag):
        """Schedule interval is 0 6 * * * (daily at 6am UTC)."""
        assert loaded_dag.schedule == "0 6 * * *"

    def test_catchup_is_false(self, loaded_dag):
        """Catchup is disabled."""
        assert loaded_dag.catchup is False

    def test_max_active_runs_is_one(self, loaded_dag):
        """Max active runs is limited to 1."""
        assert loaded_dag.max_active_runs == 1

    def test_start_date(self, loaded_dag):
        """Start date is 2025-01-01."""
        assert loaded_dag.start_date.year == 2025
        assert loaded_dag.start_date.month == 1
        assert loaded_dag.start_date.day == 1

    def test_has_collect_jobs_task(self, loaded_dag):
        """collect_jobs task exists."""
        assert "collect_jobs" in loaded_dag.task_ids

    def test_has_preprocess_jobs_task(self, loaded_dag):
        """preprocess_jobs task exists."""
        assert "preprocess_jobs" in loaded_dag.task_ids

    def test_task_count_is_nine(self, loaded_dag):
        """DAG has exactly 9 tasks."""
        assert len(loaded_dag.tasks) == 9

    def test_has_filter_roles_task(self, loaded_dag):
        """filter_roles task exists."""
        assert "filter_roles" in loaded_dag.task_ids

    def test_has_extract_jobs_task(self, loaded_dag):
        """extract_jobs task exists."""
        assert "extract_jobs" in loaded_dag.task_ids

    def test_has_embed_jobs_task(self, loaded_dag):
        """embed_jobs task exists."""
        assert "embed_jobs" in loaded_dag.task_ids

    def test_has_retrieve_jobs_task(self, loaded_dag):
        """retrieve_jobs task exists."""
        assert "retrieve_jobs" in loaded_dag.task_ids

    def test_has_rerank_jobs_task(self, loaded_dag):
        """rerank_jobs task exists."""
        assert "rerank_jobs" in loaded_dag.task_ids

    def test_has_generate_summaries_task(self, loaded_dag):
        """generate_summaries task exists."""
        assert "generate_summaries" in loaded_dag.task_ids

    def test_has_evaluate_task(self, loaded_dag):
        """evaluate task exists."""
        assert "evaluate" in loaded_dag.task_ids


class TestDagTaskTypes:
    """Tests for task types."""

    def test_all_tasks_are_python_operators(self, loaded_dag):
        """All tasks are PythonOperator instances."""
        for task in loaded_dag.tasks:
            assert isinstance(task, PythonOperator), (
                f"Task '{task.task_id}' is {type(task).__name__}, expected PythonOperator"
            )

    def test_collect_jobs_is_python_operator(self, loaded_dag):
        """collect_jobs is a PythonOperator."""
        task = loaded_dag.get_task("collect_jobs")
        assert isinstance(task, PythonOperator)

    def test_preprocess_jobs_is_python_operator(self, loaded_dag):
        """preprocess_jobs is a PythonOperator."""
        task = loaded_dag.get_task("preprocess_jobs")
        assert isinstance(task, PythonOperator)


class TestDagTaskDependency:
    """Tests for task dependencies."""

    def test_preprocess_depends_on_collect(self, loaded_dag):
        """preprocess_jobs is downstream of collect_jobs."""
        collect_task = loaded_dag.get_task("collect_jobs")
        assert "preprocess_jobs" in collect_task.downstream_task_ids

    def test_collect_has_no_upstream(self, loaded_dag):
        """collect_jobs has no upstream tasks."""
        collect_task = loaded_dag.get_task("collect_jobs")
        assert len(collect_task.upstream_task_ids) == 0

    def test_preprocess_downstream_is_filter_roles(self, loaded_dag):
        """preprocess_jobs is upstream of filter_roles."""
        preprocess_task = loaded_dag.get_task("preprocess_jobs")
        assert "filter_roles" in preprocess_task.downstream_task_ids

    def test_filter_roles_downstream_is_extract(self, loaded_dag):
        """filter_roles is upstream of extract_jobs."""
        filter_task = loaded_dag.get_task("filter_roles")
        assert "extract_jobs" in filter_task.downstream_task_ids

    def test_extract_downstream_is_embed(self, loaded_dag):
        """extract_jobs is upstream of embed_jobs."""
        extract_task = loaded_dag.get_task("extract_jobs")
        assert "embed_jobs" in extract_task.downstream_task_ids

    def test_embed_downstream_is_retrieve(self, loaded_dag):
        """embed_jobs is upstream of retrieve_jobs."""
        embed_task = loaded_dag.get_task("embed_jobs")
        assert "retrieve_jobs" in embed_task.downstream_task_ids

    def test_retrieve_downstream_is_rerank(self, loaded_dag):
        """retrieve_jobs is upstream of rerank_jobs."""
        retrieve_task = loaded_dag.get_task("retrieve_jobs")
        assert "rerank_jobs" in retrieve_task.downstream_task_ids

    def test_rerank_downstream_is_generate(self, loaded_dag):
        """rerank_jobs is upstream of generate_summaries."""
        rerank_task = loaded_dag.get_task("rerank_jobs")
        assert "generate_summaries" in rerank_task.downstream_task_ids

    def test_generate_downstream_is_evaluate(self, loaded_dag):
        """generate_summaries is upstream of evaluate."""
        generate_task = loaded_dag.get_task("generate_summaries")
        assert "evaluate" in generate_task.downstream_task_ids

    def test_evaluate_has_no_downstream(self, loaded_dag):
        """evaluate has no downstream tasks."""
        evaluate_task = loaded_dag.get_task("evaluate")
        assert len(evaluate_task.downstream_task_ids) == 0


class TestDagDefaultArgs:
    """Tests for DAG default arguments."""

    def test_retries_is_one(self, loaded_dag):
        """Default retries is 1."""
        assert loaded_dag.default_args["retries"] == 1

    def test_retry_delay_is_five_minutes(self, loaded_dag):
        """Default retry_delay is 5 minutes."""
        expected = timedelta(minutes=5)
        assert loaded_dag.default_args["retry_delay"] == expected

    def test_email_on_failure_is_false(self, loaded_dag):
        """email_on_failure is False."""
        assert loaded_dag.default_args["email_on_failure"] is False

    def test_owner_is_pipeline(self, loaded_dag):
        """Owner is 'pipeline'."""
        assert loaded_dag.default_args["owner"] == "pipeline"
