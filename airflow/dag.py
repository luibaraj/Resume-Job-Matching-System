import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.providers.standard.operators.python import PythonOperator

from src.collection import main as collection_main
from src.preprocessing import main as preprocessing_main
from src.role_filter import main as role_filter_main
from src.extraction import main as extraction_main
from src.embedding import main as embedding_main
from src.retrieval import main as retrieval_main
from src.reranking import main as reranking_main
from src.generation import main as generation_main
from src.evaluation import main as evaluation_main


# --- Helper to load .env file and fallback from Airflow Variables ---
def _load_dotenv(path):
    """Load environment variables from .env file into a dict."""
    env = {}
    try:
        with open(os.path.abspath(path)) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, _, val = line.partition("=")
                env[key.strip()] = val.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return env


_ENV_PATH = "/project/.env"
_dotenv = _load_dotenv(_ENV_PATH)


def _get(var_name, default=""):
    """Get config value from Airflow Variable, fallback to .env file."""
    airflow_val = Variable.get(var_name, default_var="")
    if airflow_val:
        return airflow_val
    return _dotenv.get(var_name, default)


# --- Config from Airflow Variables (set in UI: Admin > Variables) or .env fallback ---
LOG_LEVEL = _get("LOG_LEVEL", "INFO")
GOOGLE_API_KEY = _get("GOOGLE_API_KEY")


def _wrap(fn):
    """Wrap a pipeline main() so SystemExit becomes RuntimeError (safe for PythonOperator)."""
    def _inner(**context):
        try:
            fn()
        except SystemExit as e:
            raise RuntimeError(f"Pipeline step failed with exit code {e.code}") from e
    return _inner


def _wrap_evaluation(**context):
    """Wrap evaluation.main() clearing sys.argv to avoid argparse collision with Airflow CLI."""
    original_argv = sys.argv
    sys.argv = ["evaluation"]
    try:
        evaluation_main()
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(f"Evaluation step failed with exit code {e.code}") from e
    finally:
        sys.argv = original_argv


default_args = {
    "owner": "pipeline",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="job_pipeline",
    default_args=default_args,
    description="Collect jobs from Greenhouse and run full matching pipeline",
    schedule="0 6 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["pipeline", "jobs"],
) as dag:

    collect_jobs = PythonOperator(
        task_id="collect_jobs",
        python_callable=_wrap(collection_main),
    )

    preprocess_jobs = PythonOperator(
        task_id="preprocess_jobs",
        python_callable=_wrap(preprocessing_main),
    )

    filter_roles_task = PythonOperator(
        task_id="filter_roles",
        python_callable=_wrap(role_filter_main),
    )

    extract_jobs_task = PythonOperator(
        task_id="extract_jobs",
        python_callable=_wrap(extraction_main),
    )

    embed_jobs_task = PythonOperator(
        task_id="embed_jobs",
        python_callable=_wrap(embedding_main),
    )

    retrieve_jobs_task = PythonOperator(
        task_id="retrieve_jobs",
        python_callable=_wrap(retrieval_main),
    )

    rerank_jobs_task = PythonOperator(
        task_id="rerank_jobs",
        python_callable=_wrap(reranking_main),
    )

    generate_summaries_task = PythonOperator(
        task_id="generate_summaries",
        python_callable=_wrap(generation_main),
    )

    evaluate_task = PythonOperator(
        task_id="evaluate",
        python_callable=_wrap_evaluation,
    )

    collect_jobs >> preprocess_jobs >> filter_roles_task >> extract_jobs_task >> embed_jobs_task >> retrieve_jobs_task >> rerank_jobs_task >> generate_summaries_task >> evaluate_task
