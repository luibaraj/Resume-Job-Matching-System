"""Tests for Airflow DAG structure and configuration.

Requires apache-airflow to be installed. Tests are automatically skipped if not available.
"""

import importlib.util
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

# Skip entire file if airflow is not installed
airflow = pytest.importorskip("airflow", reason="apache-airflow not installed")

from airflow.providers.docker.operators.docker import DockerOperator


@pytest.fixture(scope="module")
def loaded_dag():
    """Load the DAG module with Variable.get mocked.

    This patches Variable.get before importing the DAG so that the module-level
    Variable.get() calls don't fail.
    """
    with patch("airflow.models.Variable.get", return_value=""):
        # Load the DAG module using importlib
        spec = importlib.util.spec_from_file_location(
            "airflow_dag", "/Users/luisbarajas/Desktop/Projects/Resume-Job-Match/Resume-Job-Matching-System/airflow/dag.py"
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

    def test_task_count_is_three(self, loaded_dag):
        """DAG has exactly 3 tasks."""
        assert len(loaded_dag.tasks) == 3

    def test_has_extract_jobs_task(self, loaded_dag):
        """extract_jobs task exists."""
        assert "extract_jobs" in loaded_dag.task_ids


class TestDagTaskTypes:
    """Tests for task types."""

    def test_collect_jobs_is_docker_operator(self, loaded_dag):
        """collect_jobs is a DockerOperator."""
        task = loaded_dag.get_task("collect_jobs")
        assert isinstance(task, DockerOperator)

    def test_preprocess_jobs_is_docker_operator(self, loaded_dag):
        """preprocess_jobs is a DockerOperator."""
        task = loaded_dag.get_task("preprocess_jobs")
        assert isinstance(task, DockerOperator)


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

    def test_preprocess_downstream_is_extract(self, loaded_dag):
        """preprocess_jobs is upstream of extract_jobs."""
        preprocess_task = loaded_dag.get_task("preprocess_jobs")
        assert "extract_jobs" in preprocess_task.downstream_task_ids

    def test_extract_has_no_downstream(self, loaded_dag):
        """extract_jobs has no downstream tasks."""
        extract_task = loaded_dag.get_task("extract_jobs")
        assert len(extract_task.downstream_task_ids) == 0


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


class TestDagTaskConfig:
    """Tests for task-level configuration."""

    def test_collect_jobs_auto_remove_success(self, loaded_dag):
        """collect_jobs auto_remove is 'success'."""
        task = loaded_dag.get_task("collect_jobs")
        assert task.auto_remove == "success"

    def test_collect_jobs_network_mode_bridge(self, loaded_dag):
        """collect_jobs network_mode is 'bridge'."""
        task = loaded_dag.get_task("collect_jobs")
        assert task.network_mode == "bridge"

    def test_collect_jobs_docker_url(self, loaded_dag):
        """collect_jobs docker_url points to Docker socket."""
        task = loaded_dag.get_task("collect_jobs")
        assert task.docker_url == "unix://var/run/docker.sock"

    def test_preprocess_jobs_auto_remove_success(self, loaded_dag):
        """preprocess_jobs auto_remove is 'success'."""
        task = loaded_dag.get_task("preprocess_jobs")
        assert task.auto_remove == "success"

    def test_preprocess_jobs_network_mode_bridge(self, loaded_dag):
        """preprocess_jobs network_mode is 'bridge'."""
        task = loaded_dag.get_task("preprocess_jobs")
        assert task.network_mode == "bridge"

    def test_preprocess_jobs_docker_url(self, loaded_dag):
        """preprocess_jobs docker_url points to Docker socket."""
        task = loaded_dag.get_task("preprocess_jobs")
        assert task.docker_url == "unix://var/run/docker.sock"
