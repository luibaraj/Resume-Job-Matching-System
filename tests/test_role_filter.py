import pytest

from src.role_filter import classify_is_target_role, filter_roles


class TestClassifyIsTargetRole:
    """Unit tests for classify_is_target_role() classification logic."""

    # Edge cases
    def test_none_returns_0(self):
        assert classify_is_target_role(None) == 0

    def test_empty_string_returns_0(self):
        assert classify_is_target_role("") == 0

    # Target patterns: single-word or multi-word phrases
    def test_machine_learning(self):
        assert classify_is_target_role("Machine Learning Engineer") == 1

    def test_ml_abbreviation(self):
        assert classify_is_target_role("ML Platform Engineer") == 1

    def test_data_scientist(self):
        assert classify_is_target_role("Senior Data Scientist") == 1

    def test_data_science(self):
        assert classify_is_target_role("Data Science Lead") == 1

    def test_research_scientist(self):
        assert classify_is_target_role("Research Scientist, NLP") == 1

    def test_applied_scientist(self):
        assert classify_is_target_role("Applied Scientist II") == 1

    def test_ai(self):
        assert classify_is_target_role("AI Engineer") == 1

    def test_ai_architect(self):
        assert classify_is_target_role("AI Architect") == 1

    def test_ai_specialist(self):
        assert classify_is_target_role("AI Specialist") == 1

    def test_ai_research(self):
        assert classify_is_target_role("AI Research Engineer") == 1

    def test_research_engineer(self):
        assert classify_is_target_role("Research Engineer, Vision") == 1

    def test_computational(self):
        assert classify_is_target_role("Computational Biologist") == 1

    def test_nlp(self):
        assert classify_is_target_role("NLP Engineer") == 1

    def test_natural_language_processing(self):
        assert classify_is_target_role("Natural Language Processing Researcher") == 1

    def test_computer_vision(self):
        assert classify_is_target_role("Computer Vision Engineer") == 1

    # Case insensitivity
    def test_case_insensitive(self):
        assert classify_is_target_role("machine learning engineer") == 1
        assert classify_is_target_role("MACHINE LEARNING ENGINEER") == 1
        assert classify_is_target_role("Data SCIENTIST") == 1

    # Non-target roles
    def test_non_target_software_engineer(self):
        assert classify_is_target_role("Software Engineer, Backend") == 0

    def test_non_target_product_manager(self):
        assert classify_is_target_role("Product Manager") == 0

    def test_non_target_designer(self):
        assert classify_is_target_role("Senior UX Designer") == 0

    # Word boundary tests (ensure patterns don't match substrings)
    def test_word_boundary_email_not_ml(self):
        # "ML" should not match inside "Email"
        assert classify_is_target_role("Email Marketing Manager") == 0

    def test_word_boundary_normal_not_nlp(self):
        # "nlp" should not match inside "normalizer"
        assert classify_is_target_role("Operations Normalizer") == 0

    def test_word_boundary_learning_not_machine_learning(self):
        # "learning" alone should not match (needs "machine learning")
        assert classify_is_target_role("Continuous Learning Manager") == 0

    def test_word_boundary_research_without_scientist(self):
        # "research" alone should not match (needs "research scientist/engineer")
        assert classify_is_target_role("Research Manager") == 0


class TestFilterRoles:
    """Integration tests for filter_roles() pipeline function."""

    def _insert_preprocessed_job(self, db_manager, greenhouse_id: int, title: str) -> int:
        """Helper: insert a preprocessed job (preprocessed=1, is_target_role=NULL).

        Returns the job ID.
        """
        job_dict = {
            "greenhouse_id": greenhouse_id,
            "board_token": "test-board",
            "title": title,
            "company": "Test Company",
            "location": "San Francisco, CA",
            "raw_description": "This is a test job description.",
            "absolute_url": "https://example.com",
            "updated_at_source": "2026-03-16T00:00:00Z",
            "departments": '["Engineering"]',
            "offices": '["San Francisco"]',
            "collected_at": "2026-03-16T00:00:00Z",
        }
        # Insert and immediately mark as preprocessed
        db_manager.insert_job(job_dict)
        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT id FROM jobs WHERE greenhouse_id = ?", (greenhouse_id,))
            job_id = cursor.fetchone()[0]
            conn.execute("UPDATE jobs SET preprocessed = 1 WHERE id = ?", (job_id,))
        return job_id

    def test_empty_db_returns_zero(self, db_manager):
        """Empty database should return (0, 0)."""
        classified, errors = filter_roles(db_manager, run_id=1, chunk_size=100)
        assert classified == 0
        assert errors == 0

    def test_target_job_gets_flagged(self, db_manager):
        """Target role (ML) should get is_target_role=1."""
        job_id = self._insert_preprocessed_job(db_manager, 10001, "Machine Learning Engineer")

        filter_roles(db_manager, run_id=1, chunk_size=100)

        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT is_target_role FROM jobs WHERE id = ?", (job_id,))
            is_target = cursor.fetchone()[0]
        assert is_target == 1

    def test_non_target_job_gets_zero(self, db_manager):
        """Non-target role (SWE) should get is_target_role=0."""
        job_id = self._insert_preprocessed_job(db_manager, 10002, "Software Engineer, Backend")

        filter_roles(db_manager, run_id=1, chunk_size=100)

        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT is_target_role FROM jobs WHERE id = ?", (job_id,))
            is_target = cursor.fetchone()[0]
        assert is_target == 0

    def test_returns_classified_count(self, db_manager):
        """filter_roles should return count of classified jobs."""
        self._insert_preprocessed_job(db_manager, 10003, "Machine Learning Engineer")  # target
        self._insert_preprocessed_job(db_manager, 10004, "Data Science Lead")  # target
        self._insert_preprocessed_job(db_manager, 10005, "Product Manager")  # non-target

        classified, errors = filter_roles(db_manager, run_id=1, chunk_size=100)

        assert classified == 3
        assert errors == 0

    def test_skips_unpreprocessed_jobs(self, db_manager):
        """Jobs with preprocessed=0 should be skipped."""
        # Insert job but DON'T mark as preprocessed
        job_dict = {
            "greenhouse_id": 10006,
            "board_token": "test-board",
            "title": "Machine Learning Engineer",
            "company": "Test Company",
            "location": "San Francisco, CA",
            "raw_description": "A test job.",
            "absolute_url": "https://example.com",
            "updated_at_source": "2026-03-16T00:00:00Z",
            "departments": '["Engineering"]',
            "offices": '["San Francisco"]',
            "collected_at": "2026-03-16T00:00:00Z",
        }
        db_manager.insert_job(job_dict)

        classified, errors = filter_roles(db_manager, run_id=1, chunk_size=100)

        assert classified == 0
        assert errors == 0

        # Verify is_target_role is still NULL
        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT is_target_role FROM jobs WHERE greenhouse_id = 10006")
            is_target = cursor.fetchone()[0]
        assert is_target is None

    def test_idempotent(self, db_manager):
        """Running filter_roles twice should process 0 jobs the second time."""
        self._insert_preprocessed_job(db_manager, 10007, "Machine Learning Engineer")
        self._insert_preprocessed_job(db_manager, 10008, "Product Manager")

        # First run
        classified1, _ = filter_roles(db_manager, run_id=1, chunk_size=100)
        assert classified1 == 2

        # Second run should find no unclassified jobs
        classified2, _ = filter_roles(db_manager, run_id=2, chunk_size=100)
        assert classified2 == 0

    def test_chunked_processing(self, db_manager):
        """Test that chunking works correctly (all jobs processed, no duplicates)."""
        for i in range(5):
            self._insert_preprocessed_job(
                db_manager, 10009 + i, "Machine Learning Engineer" if i % 2 == 0 else "SWE"
            )

        # Process with chunk_size=2 to force multiple iterations
        classified, errors = filter_roles(db_manager, run_id=1, chunk_size=2)

        assert classified == 5
        assert errors == 0

        # Verify all jobs have is_target_role set (not NULL)
        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM jobs WHERE is_target_role IS NULL")
            unclassified = cursor.fetchone()[0]
        assert unclassified == 0

    def test_mixed_batch_counts(self, db_manager):
        """Verify correct counts of target vs non-target in a batch."""
        # 3 target roles
        self._insert_preprocessed_job(db_manager, 11001, "Machine Learning Engineer")
        self._insert_preprocessed_job(db_manager, 11002, "Data Scientist")
        self._insert_preprocessed_job(db_manager, 11003, "Research Engineer")
        # 2 non-target roles
        self._insert_preprocessed_job(db_manager, 11004, "Product Manager")
        self._insert_preprocessed_job(db_manager, 11005, "Software Engineer")

        filter_roles(db_manager, run_id=1, chunk_size=100)

        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM jobs WHERE is_target_role = 1")
            target_count = cursor.fetchone()[0]
            cursor = conn.execute("SELECT COUNT(*) FROM jobs WHERE is_target_role = 0")
            non_target_count = cursor.fetchone()[0]

        assert target_count == 3
        assert non_target_count == 2
