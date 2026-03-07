"""Tests for infrastructure: Dockerfiles, docker-compose files, and project structure."""

from pathlib import Path

import pytest
import yaml

# Root directory of the project
PROJECT_ROOT = Path(__file__).parent.parent


class TestDockerfileCollection:
    """Tests for docker/Dockerfile.collection."""

    def test_file_exists(self):
        """Dockerfile.collection exists."""
        dockerfile = PROJECT_ROOT / "docker" / "Dockerfile.collection"
        assert dockerfile.is_file()

    def test_starts_with_from_instruction(self):
        """Dockerfile starts with FROM instruction."""
        dockerfile = PROJECT_ROOT / "docker" / "Dockerfile.collection"
        content = dockerfile.read_text()
        first_line = next(
            (line.strip() for line in content.split("\n")
             if line.strip() and not line.strip().startswith("#")),
            None
        )
        assert first_line.startswith("FROM")

    def test_uses_python_311_slim(self):
        """Dockerfile uses python:3.11-slim base image."""
        dockerfile = PROJECT_ROOT / "docker" / "Dockerfile.collection"
        content = dockerfile.read_text()
        assert "FROM python:3.11-slim" in content

    def test_has_copy_instruction(self):
        """Dockerfile has at least one COPY instruction."""
        dockerfile = PROJECT_ROOT / "docker" / "Dockerfile.collection"
        content = dockerfile.read_text()
        assert any(line.strip().startswith("COPY") for line in content.split("\n"))

    def test_has_entrypoint_instruction(self):
        """Dockerfile has ENTRYPOINT instruction."""
        dockerfile = PROJECT_ROOT / "docker" / "Dockerfile.collection"
        content = dockerfile.read_text()
        assert any(line.strip().startswith("ENTRYPOINT") for line in content.split("\n"))

    def test_entrypoint_runs_collection_module(self):
        """ENTRYPOINT runs src.collection module."""
        dockerfile = PROJECT_ROOT / "docker" / "Dockerfile.collection"
        content = dockerfile.read_text()
        entrypoint_line = next(
            (line for line in content.split("\n") if line.strip().startswith("ENTRYPOINT")),
            ""
        )
        assert "src.collection" in entrypoint_line

    def test_no_volume_instruction(self):
        """Dockerfile does not declare VOLUME (uses bind mounts instead)."""
        dockerfile = PROJECT_ROOT / "docker" / "Dockerfile.collection"
        content = dockerfile.read_text()
        assert "VOLUME" not in content

    def test_copies_required_source_files(self):
        """Dockerfile copies all required source files."""
        dockerfile = PROJECT_ROOT / "docker" / "Dockerfile.collection"
        content = dockerfile.read_text()
        required_files = ["src/collection.py", "src/config.py", "src/database.py", "src/utils.py"]
        for required_file in required_files:
            assert required_file in content


class TestDockerfilePreprocessing:
    """Tests for docker/Dockerfile.preprocessing."""

    def test_file_exists(self):
        """Dockerfile.preprocessing exists."""
        dockerfile = PROJECT_ROOT / "docker" / "Dockerfile.preprocessing"
        assert dockerfile.is_file()

    def test_uses_python_311_slim(self):
        """Dockerfile uses python:3.11-slim base image."""
        dockerfile = PROJECT_ROOT / "docker" / "Dockerfile.preprocessing"
        content = dockerfile.read_text()
        assert "FROM python:3.11-slim" in content

    def test_entrypoint_runs_preprocessing_module(self):
        """ENTRYPOINT runs src.preprocessing module."""
        dockerfile = PROJECT_ROOT / "docker" / "Dockerfile.preprocessing"
        content = dockerfile.read_text()
        entrypoint_line = next(
            (line for line in content.split("\n") if line.strip().startswith("ENTRYPOINT")),
            ""
        )
        assert "src.preprocessing" in entrypoint_line

    def test_no_pip_install_line(self):
        """Dockerfile does not contain pip install (pure stdlib)."""
        dockerfile = PROJECT_ROOT / "docker" / "Dockerfile.preprocessing"
        content = dockerfile.read_text()
        assert "pip install" not in content

    def test_no_volume_instruction(self):
        """Dockerfile does not declare VOLUME (uses bind mounts instead)."""
        dockerfile = PROJECT_ROOT / "docker" / "Dockerfile.preprocessing"
        content = dockerfile.read_text()
        assert "VOLUME" not in content

    def test_copies_required_source_files(self):
        """Dockerfile copies preprocessing source files."""
        dockerfile = PROJECT_ROOT / "docker" / "Dockerfile.preprocessing"
        content = dockerfile.read_text()
        assert "src/preprocessing.py" in content


class TestDockerComposeMain:
    """Tests for docker/docker-compose.yml."""

    def test_file_exists(self):
        """docker-compose.yml exists."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.yml"
        assert compose_file.is_file()

    def test_parses_as_valid_yaml(self):
        """File parses as valid YAML."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.yml"
        content = compose_file.read_text()
        yaml.safe_load(content)  # Will raise if invalid

    def test_collection_service_exists(self):
        """collection service is defined."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.yml"
        content = yaml.safe_load(compose_file.read_text())
        assert "collection" in content["services"]

    def test_preprocessing_service_exists(self):
        """preprocessing service is defined."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.yml"
        content = yaml.safe_load(compose_file.read_text())
        assert "preprocessing" in content["services"]

    def test_no_named_volumes(self):
        """No named volumes defined (uses bind mounts instead)."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.yml"
        content = yaml.safe_load(compose_file.read_text())
        assert content.get("volumes") is None

    def test_preprocessing_depends_on_collection(self):
        """preprocessing service depends_on collection."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.yml"
        content = yaml.safe_load(compose_file.read_text())
        preprocessing = content["services"]["preprocessing"]
        assert "collection" in preprocessing.get("depends_on", [])

    def test_collection_mounts_data_bind(self):
        """collection service mounts ../data bind mount at /data."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.yml"
        content = yaml.safe_load(compose_file.read_text())
        collection = content["services"]["collection"]
        volumes = collection.get("volumes", [])
        # Look for ../data:/data bind mount
        assert any("../data" in str(v) and "/data" in str(v) for v in volumes)

    def test_preprocessing_mounts_data_bind(self):
        """preprocessing service mounts ../data bind mount at /data."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.yml"
        content = yaml.safe_load(compose_file.read_text())
        preprocessing = content["services"]["preprocessing"]
        volumes = preprocessing.get("volumes", [])
        # Look for ../data:/data bind mount
        assert any("../data" in str(v) and "/data" in str(v) for v in volumes)

    def test_collection_has_greenhouse_tokens_env(self):
        """collection service has GREENHOUSE_BOARD_TOKENS env var."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.yml"
        content = yaml.safe_load(compose_file.read_text())
        collection = content["services"]["collection"]
        env = collection.get("environment", {})
        assert "GREENHOUSE_BOARD_TOKENS" in str(env)

    def test_preprocessing_has_db_path_env(self):
        """preprocessing service has DB_PATH env var."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.yml"
        content = yaml.safe_load(compose_file.read_text())
        preprocessing = content["services"]["preprocessing"]
        env = preprocessing.get("environment", {})
        assert "DB_PATH" in str(env)

    def test_both_services_restart_no(self):
        """Both services have restart: 'no'."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.yml"
        content = yaml.safe_load(compose_file.read_text())
        for service_name in ["collection", "preprocessing"]:
            service = content["services"][service_name]
            assert service.get("restart") == "no"


class TestDockerComposeAirflow:
    """Tests for docker/docker-compose.airflow.yml."""

    def test_file_exists(self):
        """docker-compose.airflow.yml exists."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.airflow.yml"
        assert compose_file.is_file()

    def test_parses_as_valid_yaml(self):
        """File parses as valid YAML."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.airflow.yml"
        content = compose_file.read_text()
        yaml.safe_load(content)  # Will raise if invalid

    def test_airflow_service_exists(self):
        """airflow service is defined."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.airflow.yml"
        content = yaml.safe_load(compose_file.read_text())
        assert "airflow" in content["services"]

    def test_uses_apache_airflow_image(self):
        """airflow service uses apache/airflow image."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.airflow.yml"
        content = yaml.safe_load(compose_file.read_text())
        airflow = content["services"]["airflow"]
        image = airflow.get("image", "")
        assert image.startswith("apache/airflow:")

    def test_docker_sock_volume_mounted(self):
        """Docker socket is mounted for DinD."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.airflow.yml"
        content = yaml.safe_load(compose_file.read_text())
        airflow = content["services"]["airflow"]
        volumes = airflow.get("volumes", [])
        assert any("/var/run/docker.sock" in str(v) for v in volumes)

    def test_port_8080_exposed(self):
        """Port 8080 is exposed."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.airflow.yml"
        content = yaml.safe_load(compose_file.read_text())
        airflow = content["services"]["airflow"]
        ports = airflow.get("ports", [])
        assert any("8080" in str(p) for p in ports)

    def test_dags_folder_volume_mounted(self):
        """DAGs folder is mounted into container."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.airflow.yml"
        content = yaml.safe_load(compose_file.read_text())
        airflow = content["services"]["airflow"]
        volumes = airflow.get("volumes", [])
        # Look for ../airflow:/opt/airflow/dags or similar
        assert any("airflow" in str(v) and "/opt/airflow/dags" in str(v) for v in volumes)

    def test_airflow_home_volume_defined(self):
        """airflow_home volume is defined at top level."""
        compose_file = PROJECT_ROOT / "docker" / "docker-compose.airflow.yml"
        content = yaml.safe_load(compose_file.read_text())
        assert "airflow_home" in content.get("volumes", {})


class TestProjectFiles:
    """Tests for required project files."""

    def test_dag_file_exists(self):
        """airflow/dag.py exists."""
        dag_file = PROJECT_ROOT / "airflow" / "dag.py"
        assert dag_file.is_file()

    def test_requirements_txt_exists(self):
        """requirements.txt exists."""
        req_file = PROJECT_ROOT / "requirements.txt"
        assert req_file.is_file()

    def test_requirements_airflow_txt_exists(self):
        """requirements-airflow.txt exists."""
        req_file = PROJECT_ROOT / "requirements-airflow.txt"
        assert req_file.is_file()

    def test_requirements_contains_requests(self):
        """requirements.txt includes requests."""
        req_file = PROJECT_ROOT / "requirements.txt"
        content = req_file.read_text()
        assert "requests" in content

    def test_requirements_contains_pytest(self):
        """requirements.txt includes pytest."""
        req_file = PROJECT_ROOT / "requirements.txt"
        content = req_file.read_text()
        assert "pytest" in content

    def test_requirements_contains_pytest_cov(self):
        """requirements.txt includes pytest-cov."""
        req_file = PROJECT_ROOT / "requirements.txt"
        content = req_file.read_text()
        assert "pytest-cov" in content

    def test_requirements_contains_pytest_mock(self):
        """requirements.txt includes pytest-mock (added by plan)."""
        req_file = PROJECT_ROOT / "requirements.txt"
        content = req_file.read_text()
        assert "pytest-mock" in content

    def test_requirements_airflow_contains_apache_airflow(self):
        """requirements-airflow.txt includes apache-airflow."""
        req_file = PROJECT_ROOT / "requirements-airflow.txt"
        content = req_file.read_text()
        assert "apache-airflow" in content

    def test_requirements_airflow_contains_docker_provider(self):
        """requirements-airflow.txt includes docker provider."""
        req_file = PROJECT_ROOT / "requirements-airflow.txt"
        content = req_file.read_text()
        assert "apache-airflow-providers-docker" in content

    def test_env_example_exists(self):
        """.env.example exists."""
        env_file = PROJECT_ROOT / ".env.example"
        assert env_file.is_file()
