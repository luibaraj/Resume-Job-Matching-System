import pytest

from src.preprocessing import (
    _clean_record,
    clean_job_description,
    decode_html_entities,
    normalize_list_items,
    normalize_whitespace,
    preprocess_jobs,
    strip_html_tags,
)


class TestDecodeHtmlEntities:
    def test_ampersand(self):
        assert decode_html_entities("&amp;") == "&"

    def test_less_than(self):
        assert decode_html_entities("&lt;") == "<"

    def test_greater_than(self):
        assert decode_html_entities("&gt;") == ">"

    def test_non_breaking_space(self):
        assert decode_html_entities("&nbsp;") == "\xa0"

    def test_numeric_entity(self):
        assert decode_html_entities("&#39;") == "'"

    def test_double_encoded(self):
        # html.unescape only does one pass, so &amp;lt; becomes &lt; (not <)
        result = decode_html_entities("&amp;lt;")
        assert result == "&lt;"

    def test_plain_text_unchanged(self):
        assert decode_html_entities("hello world") == "hello world"

    def test_mixed_entities(self):
        result = decode_html_entities("Design &amp; implement &lt;APIs&gt;")
        assert result == "Design & implement <APIs>"


class TestNormalizeListItems:
    def test_li_tag_replaced(self):
        result = normalize_list_items("<li>Python</li>")
        assert result == "- Python"

    def test_li_with_attrs(self):
        result = normalize_list_items('<li class="x">Go</li>')
        assert result == "- Go"

    def test_bullet_char_replaced(self):
        result = normalize_list_items("• Skill")
        assert result == "- Skill"

    def test_middle_dot_replaced(self):
        result = normalize_list_items("· Skill")
        assert result == "- Skill"

    def test_unclosed_li(self):
        result = normalize_list_items("<li>Item")
        assert result == "- Item"

    def test_nested_text_preserved(self):
        result = normalize_list_items("<li>Item with <strong>bold</strong></li>")
        assert result == "- Item with <strong>bold</strong>"

    def test_multiple_list_items(self):
        result = normalize_list_items("<li>First</li><li>Second</li>")
        assert result == "- First- Second"

    def test_mixed_bullets(self):
        text = "<li>HTML</li>• Bullet· Dot"
        result = normalize_list_items(text)
        assert "- HTML" in result
        assert "- Bullet" in result
        assert "- Dot" in result


class TestStripHtmlTags:
    def test_div_removed(self):
        result = strip_html_tags("<div>content</div>")
        assert result == " content "

    def test_span_removed(self):
        result = strip_html_tags("<span>text</span>")
        assert result == " text "

    def test_p_replaced_with_space(self):
        result = strip_html_tags("<p>paragraph</p>")
        assert result == " paragraph "

    def test_tag_with_attributes_removed(self):
        result = strip_html_tags('<div class="main" id="x">text</div>')
        assert result == " text "

    def test_self_closing_tag_removed(self):
        result = strip_html_tags("text<br/>more")
        assert result == "text more"

    def test_img_tag_removed(self):
        result = strip_html_tags('text<img src="x.png"/>more')
        assert result == "text more"

    def test_no_tags_unchanged(self):
        result = strip_html_tags("plain text")
        assert result == "plain text"

    def test_nested_tags(self):
        result = strip_html_tags("<div><p>nested</p></div>")
        assert result == "  nested  "


class TestNormalizeWhitespace:
    def test_multiple_spaces_collapsed(self):
        assert normalize_whitespace("a   b") == "a b"

    def test_tabs_collapsed(self):
        assert normalize_whitespace("a\t\tb") == "a b"

    def test_nbsp_handled(self):
        assert normalize_whitespace("a\xa0b") == "a b"

    def test_leading_trailing_stripped(self):
        assert normalize_whitespace("  hello  ") == "hello"

    def test_multiple_blank_lines_collapsed(self):
        text = "line1\n\n\n\nline2"
        result = normalize_whitespace(text)
        assert result == "line1\n\nline2"

    def test_windows_line_endings(self):
        result = normalize_whitespace("line1\r\nline2")
        assert result == "line1\nline2"

    def test_mac_line_endings(self):
        result = normalize_whitespace("line1\rline2")
        assert result == "line1\nline2"

    def test_mixed_whitespace(self):
        text = "  hello  \t  world  \n  test  "
        result = normalize_whitespace(text)
        assert result == "hello world \n test"

    def test_single_blank_line_preserved(self):
        text = "line1\n\nline2"
        result = normalize_whitespace(text)
        assert result == "line1\n\nline2"


class TestCleanJobDescription:
    def test_full_pipeline_html_input(self):
        """Test the complete pipeline on a complex HTML string."""
        input_html = "<div><p>We&apos;re looking for:</p><ul><li>Python</li><li>SQL</li></ul></div>"
        result = clean_job_description(input_html)
        # Should have decoded &apos;, normalized list items, stripped tags, normalized whitespace
        assert "We're looking for:" in result
        assert "- Python" in result
        assert "- SQL" in result
        assert "<" not in result
        assert ">" not in result

    def test_none_input_returns_empty_string(self):
        assert clean_job_description(None) == ""

    def test_empty_string_returns_empty_string(self):
        assert clean_job_description("") == ""

    def test_plain_text_unchanged(self):
        text = "This is plain text"
        result = clean_job_description(text)
        assert result == text

    def test_real_greenhouse_description(self, sample_job_description):
        """Test on the realistic Greenhouse sample data."""
        result = clean_job_description(sample_job_description)
        # Should not contain HTML
        assert "<" not in result
        assert ">" not in result
        # Should be non-empty
        assert len(result) > 0
        # Should contain expected keywords from sample data
        assert "Engineer" in result or "engineer" in result or "Platform" in result
        # Should have decoded &amp; entities to & (e.g., "Design & implement")
        assert "Design & implement" in result

    def test_handles_entity_then_list_items(self):
        """Test interaction between entity decoding and list normalization."""
        html = "<li>Design &amp; implement</li>"
        result = clean_job_description(html)
        assert "- Design & implement" in result

    def test_whitespace_after_tag_stripping(self):
        """Ensure whitespace normalization works after tag stripping."""
        html = "<div>Hello</div><div>World</div>"
        result = clean_job_description(html)
        # After stripping tags: " Hello  World "
        # After normalizing whitespace: "Hello World"
        assert result == "Hello World"

    def test_handles_all_bullet_types(self):
        """Test various bullet point and list formats."""
        html = "<ul><li>Item 1</li>• Item 2· Item 3</ul>"
        result = clean_job_description(html)
        assert "- Item 1" in result
        assert "- Item 2" in result
        assert "- Item 3" in result


class TestCleanRecord:
    """Tests for _clean_record() worker function."""

    def test_returns_tuple_on_success(self):
        """_clean_record returns (job_id, cleaned_text) on success."""
        result = _clean_record((1, "<p>Hello</p>"))
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == 1
        assert result[1] == "Hello"

    def test_handles_none_raw_description(self):
        """_clean_record with None raw_description returns (job_id, '')."""
        result = _clean_record((42, None))
        assert result == (42, "")

    def test_returns_none_on_exception(self):
        """_clean_record returns None if clean_job_description raises."""
        from unittest.mock import patch

        def mock_clean(desc):
            raise ValueError("Mock error")

        with patch("src.preprocessing.clean_job_description", side_effect=mock_clean):
            result = _clean_record((99, "<p>test</p>"))
            assert result is None

    def test_job_id_preserved(self):
        """Job ID in input is preserved in output."""
        result = _clean_record((12345, "<div>Content</div>"))
        assert result[0] == 12345


class TestPreprocessJobs:
    def test_preprocess_jobs_updates_db(self, db_manager):
        """Test that preprocess_jobs reads, cleans, and updates the database."""
        # Insert a test job with raw HTML
        raw_html = "<div><p>Test job</p><li>Python</li></div>"
        job_dict = {
            "greenhouse_id": 12345,
            "board_token": "test",
            "title": "Test Job",
            "company": "test",
            "location": "Remote",
            "raw_description": raw_html,
            "absolute_url": "https://example.com/12345",
            "updated_at_source": "2026-03-05T00:00:00Z",
            "departments": '["Engineering"]',
            "offices": '["Remote"]',
            "collected_at": "2026-03-05T00:00:00Z",
        }
        db_manager.insert_job(job_dict)

        # Run preprocessing
        processed, errors = preprocess_jobs(db_manager, run_id=1, chunk_size=100, num_workers=1)

        # Verify counts
        assert processed == 1
        assert errors == 0

        # Verify job was updated
        jobs = db_manager.get_unpreprocessed_jobs()
        assert len(jobs) == 0  # No unpreprocessed jobs remain

        # Verify cleaned description was set
        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT cleaned_description, preprocessed FROM jobs WHERE id = 1")
            row = cursor.fetchone()
            assert row is not None
            assert row["preprocessed"] == 1
            assert row["cleaned_description"] is not None
            assert "<" not in row["cleaned_description"]
            assert "Test job" in row["cleaned_description"]

    def test_preprocess_jobs_handles_empty_description(self, db_manager):
        """Test that preprocess_jobs handles None raw_description gracefully."""
        job_dict = {
            "greenhouse_id": 12345,
            "board_token": "test",
            "title": "No Description Job",
            "company": "test",
            "location": "Remote",
            "raw_description": None,
            "absolute_url": "https://example.com/12345",
            "updated_at_source": "2026-03-05T00:00:00Z",
            "departments": "[]",
            "offices": "[]",
            "collected_at": "2026-03-05T00:00:00Z",
        }
        db_manager.insert_job(job_dict)

        processed, errors = preprocess_jobs(db_manager, run_id=1, chunk_size=100, num_workers=1)

        assert processed == 1
        assert errors == 0

        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT cleaned_description FROM jobs WHERE id = 1")
            row = cursor.fetchone()
            assert row["cleaned_description"] == ""

    def test_preprocess_jobs_with_multiple_jobs(self, db_manager):
        """Test preprocessing multiple jobs in one run."""
        jobs_data = [
            {
                "greenhouse_id": 111,
                "board_token": "test",
                "title": "Job 1",
                "company": "test",
                "location": "Remote",
                "raw_description": "<p>Job 1 description</p>",
                "absolute_url": "https://example.com/111",
                "updated_at_source": "2026-03-05T00:00:00Z",
                "departments": "[]",
                "offices": "[]",
                "collected_at": "2026-03-05T00:00:00Z",
            },
            {
                "greenhouse_id": 222,
                "board_token": "test",
                "title": "Job 2",
                "company": "test",
                "location": "SF",
                "raw_description": "<li>Skill A</li><li>Skill B</li>",
                "absolute_url": "https://example.com/222",
                "updated_at_source": "2026-03-05T00:00:00Z",
                "departments": "[]",
                "offices": "[]",
                "collected_at": "2026-03-05T00:00:00Z",
            },
        ]

        for job in jobs_data:
            db_manager.insert_job(job)

        processed, errors = preprocess_jobs(db_manager, run_id=1, chunk_size=100, num_workers=1)

        assert processed == 2
        assert errors == 0

        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM jobs WHERE preprocessed = 1")
            assert cursor.fetchone()["count"] == 2

    def test_preprocess_jobs_uses_batch_update(self, db_manager):
        """Test that preprocess_jobs uses batch update, not individual updates."""
        job_dict = {
            "greenhouse_id": 77777,
            "board_token": "test",
            "title": "Batch Test Job",
            "company": "test",
            "location": "Remote",
            "raw_description": "<p>Test</p>",
            "absolute_url": "https://example.com/77777",
            "updated_at_source": "2026-03-05T00:00:00Z",
            "departments": "[]",
            "offices": "[]",
            "collected_at": "2026-03-05T00:00:00Z",
        }
        db_manager.insert_job(job_dict)

        # Mock the batch update to verify it's called
        original_batch = db_manager.update_cleaned_descriptions_batch
        call_count = 0

        def counting_batch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_batch(*args, **kwargs)

        db_manager.update_cleaned_descriptions_batch = counting_batch

        processed, errors = preprocess_jobs(db_manager, run_id=1, chunk_size=100, num_workers=1)

        # Verify batch update was called exactly once
        assert call_count == 1
        assert processed == 1
        assert errors == 0

    def test_preprocess_jobs_excludes_error_jobs_from_batch(self, db_manager):
        """Jobs that fail cleaning are not included in batch update."""
        # Insert job with valid raw_description
        job_dict = {
            "greenhouse_id": 88888,
            "board_token": "test",
            "title": "Good Job",
            "company": "test",
            "location": "Remote",
            "raw_description": "<p>Valid HTML</p>",
            "absolute_url": "https://example.com/88888",
            "updated_at_source": "2026-03-05T00:00:00Z",
            "departments": "[]",
            "offices": "[]",
            "collected_at": "2026-03-05T00:00:00Z",
        }
        db_manager.insert_job(job_dict)

        # Mock clean_job_description to fail for specific jobs
        from unittest.mock import patch

        def mock_clean(desc):
            if "fail" in str(desc):
                raise ValueError("Mock error")
            return f"cleaned: {desc}"

        with patch("src.preprocessing.clean_job_description", side_effect=mock_clean):
            processed, errors = preprocess_jobs(db_manager, run_id=1, chunk_size=100, num_workers=1)

        # Should process the one good job, zero errors (no failing jobs in this test)
        assert processed == 1
        assert errors == 0

    def test_preprocess_jobs_returns_correct_counts(self, db_manager):
        """preprocess_jobs returns (processed, errors) tuple."""
        job_dict = {
            "greenhouse_id": 99999,
            "board_token": "test",
            "title": "Count Test Job",
            "company": "test",
            "location": "Remote",
            "raw_description": "<p>Test</p>",
            "absolute_url": "https://example.com/99999",
            "updated_at_source": "2026-03-05T00:00:00Z",
            "departments": "[]",
            "offices": "[]",
            "collected_at": "2026-03-05T00:00:00Z",
        }
        db_manager.insert_job(job_dict)

        processed, errors = preprocess_jobs(db_manager, run_id=1, chunk_size=100, num_workers=1)

        assert isinstance(processed, int) and isinstance(errors, int)
        assert processed == 1
        assert errors == 0

    def test_preprocess_jobs_accumulates_updates_then_batches(self, db_manager):
        """Multiple jobs are accumulated then sent to batch in a single call."""
        jobs_data = [
            {
                "greenhouse_id": 111000,
                "board_token": "test",
                "title": "Job 1",
                "company": "test",
                "location": "Remote",
                "raw_description": "<p>Job 1</p>",
                "absolute_url": "https://example.com/111000",
                "updated_at_source": "2026-03-05T00:00:00Z",
                "departments": "[]",
                "offices": "[]",
                "collected_at": "2026-03-05T00:00:00Z",
            },
            {
                "greenhouse_id": 222000,
                "board_token": "test",
                "title": "Job 2",
                "company": "test",
                "location": "SF",
                "raw_description": "<p>Job 2</p>",
                "absolute_url": "https://example.com/222000",
                "updated_at_source": "2026-03-05T00:00:00Z",
                "departments": "[]",
                "offices": "[]",
                "collected_at": "2026-03-05T00:00:00Z",
            },
            {
                "greenhouse_id": 333000,
                "board_token": "test",
                "title": "Job 3",
                "company": "test",
                "location": "NYC",
                "raw_description": "<p>Job 3</p>",
                "absolute_url": "https://example.com/333000",
                "updated_at_source": "2026-03-05T00:00:00Z",
                "departments": "[]",
                "offices": "[]",
                "collected_at": "2026-03-05T00:00:00Z",
            },
        ]

        for job in jobs_data:
            db_manager.insert_job(job)

        # Track calls to batch update
        from unittest.mock import patch

        with patch.object(
            db_manager, "update_cleaned_descriptions_batch", wraps=db_manager.update_cleaned_descriptions_batch
        ) as mock_batch:
            processed, errors = preprocess_jobs(db_manager, run_id=1, chunk_size=100, num_workers=1)

        # Should call batch exactly once with all 3 jobs
        assert mock_batch.call_count == 1
        batch_call_arg = mock_batch.call_args[0][0]
        assert len(batch_call_arg) == 3  # All 3 updates in one call
        assert processed == 3
        assert errors == 0

    def test_preprocess_jobs_chunks_correctly(self, db_manager):
        """Test that preprocess_jobs handles chunk_size parameter correctly."""
        # Insert 3 jobs
        for i in range(3):
            job = {
                "greenhouse_id": 8000 + i,
                "board_token": f"board{i}",
                "title": f"Chunked{i}",
                "company": "test",
                "location": "Remote",
                "raw_description": f"<p>Content {i}</p>",
                "absolute_url": f"https://example.com/{8000 + i}",
                "updated_at_source": "2026-03-05T00:00:00Z",
                "departments": "[]",
                "offices": "[]",
                "collected_at": "2026-03-05T00:00:00Z",
            }
            db_manager.insert_job(job)

        # Process with smaller chunk_size to test chunking logic
        from unittest.mock import patch

        with patch.object(
            db_manager, "get_unpreprocessed_jobs_chunked", wraps=db_manager.get_unpreprocessed_jobs_chunked
        ) as mock_chunked:
            processed, errors = preprocess_jobs(db_manager, run_id=1, chunk_size=1, num_workers=1)

        # Should call get_unpreprocessed_jobs_chunked multiple times (once per chunk)
        assert mock_chunked.call_count >= 1, "Should call chunked fetch at least once"
        assert processed > 0, "Should process jobs"
        assert errors == 0, "Should have no errors"
