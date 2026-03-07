from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# --- Config from Airflow Variables (set in UI: Admin > Variables) ---
GREENHOUSE_BOARD_TOKENS = Variable.get("GREENHOUSE_BOARD_TOKENS", default_var="")
REQUEST_TIMEOUT = Variable.get("REQUEST_TIMEOUT", default_var="30")
MAX_RETRIES = Variable.get("MAX_RETRIES", default_var="3")
RETRY_BACKOFF = Variable.get("RETRY_BACKOFF", default_var="2.0")
LOG_LEVEL = Variable.get("LOG_LEVEL", default_var="INFO")
DATA_HOST_PATH = Variable.get("DATA_HOST_PATH", default_var="/Users/luisbarajas/Desktop/Projects/Resume-Job-Match/Resume-Job-Matching-System/data")
COLLECTION_IMAGE = Variable.get("COLLECTION_IMAGE", default_var="job-collection:latest")
PREPROCESSING_IMAGE = Variable.get("PREPROCESSING_IMAGE", default_var="job-preprocessing:latest")

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
