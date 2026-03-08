import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


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
GREENHOUSE_BOARD_TOKENS = _get("GREENHOUSE_BOARD_TOKENS")
REQUEST_TIMEOUT = _get("REQUEST_TIMEOUT", "30")
MAX_RETRIES = _get("MAX_RETRIES", "3")
RETRY_BACKOFF = _get("RETRY_BACKOFF", "2.0")
LOG_LEVEL = _get("LOG_LEVEL", "INFO")
DATA_HOST_PATH = _get("DATA_HOST_PATH", "/Users/luisbarajas/Desktop/Projects/Resume-Job-Match/Resume-Job-Matching-System/data")
COLLECTION_IMAGE = _get("COLLECTION_IMAGE", "job-collection:latest")
PREPROCESSING_IMAGE = _get("PREPROCESSING_IMAGE", "job-preprocessing:latest")

shared_data_mount = Mount(
    target="/data",
    source=DATA_HOST_PATH,
    type="bind",
)

default_args = {
    "owner": "pipeline",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="job_pipeline",
    default_args=default_args,
    description="Collect jobs from Greenhouse and preprocess descriptions",
    schedule="0 6 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["pipeline", "jobs"],
) as dag:

    collect_jobs = DockerOperator(
        task_id="collect_jobs",
        image=COLLECTION_IMAGE,
        environment={
            "DB_PATH": "/data/jobs.db",
            "GREENHOUSE_BOARD_TOKENS": GREENHOUSE_BOARD_TOKENS,
            "REQUEST_TIMEOUT": REQUEST_TIMEOUT,
            "MAX_RETRIES": MAX_RETRIES,
            "RETRY_BACKOFF": RETRY_BACKOFF,
            "LOG_LEVEL": LOG_LEVEL,
        },
        mounts=[shared_data_mount],
        mount_tmp_dir=False,
        network_mode="bridge",
        auto_remove="success",
        docker_url="unix://var/run/docker.sock",
        retrieve_output=False,
        do_xcom_push=False,
    )

    preprocess_jobs = DockerOperator(
        task_id="preprocess_jobs",
        image=PREPROCESSING_IMAGE,
        environment={
            "DB_PATH": "/data/jobs.db",
            "LOG_LEVEL": LOG_LEVEL,
        },
        mounts=[shared_data_mount],
        mount_tmp_dir=False,
        network_mode="bridge",
        auto_remove="success",
        docker_url="unix://var/run/docker.sock",
        retrieve_output=False,
        do_xcom_push=False,
    )

    collect_jobs >> preprocess_jobs
