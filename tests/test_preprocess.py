"""Test suite for the preprocessing module."""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocess import (
    _step1_unescape,
    _step2_strip_iframes_and_images,
    _step3_extract_text,
    _step4_normalize_whitespace,
    _step5_normalize_unicode_punctuation,
    preprocess_description,
    run_preprocessing,
    _add_column_if_missing,
)


@pytest.fixture
def tmp_db():
    """Create a temporary database with the jobs schema."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id TEXT NOT NULL,
            board_token TEXT NOT NULL,
            title TEXT,
            location TEXT,
            description TEXT,
            source TEXT,
            source_url TEXT,
            company_name TEXT,
            department TEXT,
            job_type TEXT,
            scraped_at TEXT,
            updated_at TEXT,
            UNIQUE(external_id, board_token)
        )
    """)
    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    os.unlink(db_path)


class TestStep1Unescape:
    """Test HTML entity unescaping (Step 1)."""

    def test_single_encoded(self):
        """Single-encoded HTML entities are decoded."""
        assert _step1_unescape("&lt;p&gt;Hello&lt;/p&gt;") == "<p>Hello</p>"

    def test_double_encoded(self):
        """Double-encoded entities are decoded to single characters."""
        assert _step1_unescape("&amp;lt;p&amp;gt;") == "<p>"

    def test_triple_encoded(self):
        """Triple-encoded entities are decoded correctly."""
        assert _step1_unescape("&amp;amp;lt;") == "<"

    def test_nbsp_decoded(self):
        """Double-encoded nbsp is decoded to non-breaking space character."""
        assert _step1_unescape("&amp;nbsp;") == "\xa0"

    def test_ampersand(self):
        """Ampersand is decoded correctly."""
        assert _step1_unescape("&amp;") == "&"

    def test_quot(self):
        """Quote entities are decoded."""
        assert _step1_unescape("&quot;hello&quot;") == '"hello"'

    def test_already_plain(self):
        """Plain text without entities is unchanged."""
        assert _step1_unescape("Hello world") == "Hello world"

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert _step1_unescape("") == ""

    def test_idempotent(self):
        """Running _step1_unescape twice gives same result as once."""
        raw = "&amp;lt;test&amp;gt;"
        once = _step1_unescape(raw)
        twice = _step1_unescape(once)
        assert once == twice


class TestStep2StripIframesAndImages:
    """Test removal of iframes and images (Step 2)."""

    def test_iframe_removed(self):
        """Iframe tags are completely removed."""
        html = '<p>Before</p><iframe src="test"></iframe><p>After</p>'
        result = _step2_strip_iframes_and_images(html)
        assert "<iframe" not in result

    def test_img_removed(self):
        """Image tags are completely removed."""
        html = '<p>Before</p><img src="test.jpg"/><p>After</p>'
        result = _step2_strip_iframes_and_images(html)
        assert "<img" not in result

    def test_iframe_content_removed(self):
        """Content inside iframe is also removed."""
        html = '<iframe>Some content here</iframe>'
        result = _step2_strip_iframes_and_images(html)
        assert "Some content here" not in result

    def test_multiple_iframes_and_images(self):
        """Multiple iframes and images are all removed."""
        html = '<iframe></iframe><img/><iframe></iframe><img/>'
        result = _step2_strip_iframes_and_images(html)
        assert "<iframe" not in result
        assert "<img" not in result

    def test_other_tags_preserved(self):
        """Non-iframe/image tags are preserved."""
        html = '<p>Hello</p><div>World</div>'
        result = _step2_strip_iframes_and_images(html)
        assert "<p>" in result or "Hello" in result
        assert "<div>" in result or "World" in result

    def test_no_iframes_or_imgs(self):
        """HTML without iframes or images passes through unchanged (structurally)."""
        html = '<p>Simple <b>text</b></p>'
        result = _step2_strip_iframes_and_images(html)
        # BeautifulSoup may reformat HTML, so check that essential content is there
        assert "Simple" in result
        assert "text" in result


class TestStep3ExtractText:
    """Test plain text extraction (Step 3)."""

    def test_strips_tags(self):
        """HTML tags are stripped, text is extracted."""
        html = "<p>Hello <b>world</b></p>"
        result = _step3_extract_text(html)
        assert "<" not in result
        assert "Hello" in result
        assert "world" in result

    def test_separator_space(self):
        """Separator space is applied between block elements."""
        html = "<p>First</p><p>Second</p>"
        result = _step3_extract_text(html)
        assert "First" in result
        assert "Second" in result
        # Space separator should join them
        assert result.count(" ") >= 1

    def test_empty_html(self):
        """Empty HTML returns empty string."""
        assert _step3_extract_text("") == ""

    def test_nested_tags(self):
        """Deeply nested tags yield only inner text."""
        html = "<div><p><span><b>Deep</b></span></p></div>"
        result = _step3_extract_text(html)
        assert result.strip() == "Deep"

    def test_multiple_elements(self):
        """Multiple elements are all extracted."""
        html = "<li>Item 1</li><li>Item 2</li><li>Item 3</li>"
        result = _step3_extract_text(html)
        assert "Item 1" in result
        assert "Item 2" in result
        assert "Item 3" in result


class TestStep4NormalizeWhitespace:
    """Test whitespace normalization (Step 4)."""

    def test_collapses_spaces(self):
        """Multiple spaces collapse to single space."""
        assert _step4_normalize_whitespace("hello   world") == "hello world"

    def test_collapses_newlines(self):
        """Multiple newlines collapse to single space."""
        assert _step4_normalize_whitespace("hello\n\nworld") == "hello world"

    def test_collapses_tabs(self):
        """Tabs collapse to single space."""
        assert _step4_normalize_whitespace("hello\t\tworld") == "hello world"

    def test_strips_nbsp(self):
        """Non-breaking space is collapsed and handled."""
        assert _step4_normalize_whitespace("hello\xa0world") == "hello world"

    def test_strips_leading_whitespace(self):
        """Leading whitespace is stripped."""
        assert _step4_normalize_whitespace("  hello") == "hello"

    def test_strips_trailing_whitespace(self):
        """Trailing whitespace is stripped."""
        assert _step4_normalize_whitespace("hello  ") == "hello"

    def test_mixed_whitespace(self):
        """Mixed whitespace types are all normalized."""
        assert _step4_normalize_whitespace("\t\n  hello \xa0 world \n\t") == "hello world"

    def test_only_whitespace(self):
        """String with only whitespace becomes empty."""
        assert _step4_normalize_whitespace("   \n\t\xa0  ") == ""

    def test_preserves_internal_spaces(self):
        """Single spaces between words are preserved."""
        assert _step4_normalize_whitespace("hello world test") == "hello world test"


class TestStep5NormalizeUnicodePunctuation:
    """Test Unicode punctuation normalization (Step 5)."""

    def test_left_single_quote(self):
        """Left single quotation mark is converted to apostrophe."""
        assert _step5_normalize_unicode_punctuation("\u2018hello\u2019") == "'hello'"

    def test_right_single_quote(self):
        """Right single quotation mark is converted to apostrophe."""
        assert _step5_normalize_unicode_punctuation("don\u2019t") == "don't"

    def test_left_double_quote(self):
        """Left double quotation mark is converted to ASCII quote."""
        assert _step5_normalize_unicode_punctuation("\u201chello\u201d") == '"hello"'

    def test_right_double_quote(self):
        """Right double quotation mark is converted to ASCII quote."""
        result = _step5_normalize_unicode_punctuation("She said \u201chello\u201d")
        assert '\u201d' not in result
        assert '"' in result

    def test_em_dash(self):
        """Em dash is converted to hyphen."""
        assert _step5_normalize_unicode_punctuation("word\u2014word") == "word-word"

    def test_en_dash(self):
        """En dash is converted to hyphen."""
        assert _step5_normalize_unicode_punctuation("word\u2013word") == "word-word"

    def test_ellipsis(self):
        """Ellipsis is converted to three dots."""
        assert _step5_normalize_unicode_punctuation("wait\u2026") == "wait..."

    def test_plain_ascii_unchanged(self):
        """Plain ASCII text is unchanged."""
        text = "Hello, world! This is a test."
        assert _step5_normalize_unicode_punctuation(text) == text

    def test_mixed_replacements(self):
        """A string with multiple Unicode chars is fully normalized."""
        text = "He said \u201cdon\u2019t\u2026\u201d\u2014amazing!"
        result = _step5_normalize_unicode_punctuation(text)
        assert "\u201c" not in result
        assert "\u2019" not in result
        assert "\u2026" not in result
        assert "\u2014" not in result
        assert 'He said "don\'t..."-amazing!' == result


class TestPreprocessDescription:
    """Test the full 5-step preprocessing pipeline."""

    def test_full_pipeline_double_encoded_html(self):
        """Full pipeline handles double-encoded HTML with iframes and curly quotes."""
        raw = (
            '&amp;lt;p&amp;gt;Software Engineer needed&amp;lt;/p&amp;gt;'
            '&lt;iframe src=&quot;ad&quot;&gt;&lt;/iframe&gt;'
            'Salary: $100k&amp;nbsp;\u2013&amp;nbsp;$150k'
        )
        result = preprocess_description(raw)
        # Should have no HTML tags
        assert "<" not in result or not result.startswith("<")
        assert ">" not in result or not result.endswith(">")
        # Should have no iframe tag or attribute
        assert "iframe" not in result
        # Curly dash should be normalized
        assert "\u2013" not in result
        # Should have readable content
        assert "Software" in result or "Engineer" in result

    def test_full_pipeline_curly_quotes(self):
        """Full pipeline converts curly quotes to ASCII."""
        raw = f'&lt;p&gt;It\u2019s called \u201cJobMatch\u201d&lt;/p&gt;'
        result = preprocess_description(raw)
        assert "'" in result
        assert '"' in result
        assert "\u2019" not in result
        assert "\u201c" not in result

    def test_full_pipeline_none_input(self):
        """Full pipeline handles None input gracefully."""
        assert preprocess_description(None) == ""

    def test_full_pipeline_empty_string(self):
        """Full pipeline handles empty string."""
        assert preprocess_description("") == ""

    def test_full_pipeline_only_whitespace(self):
        """Full pipeline collapses whitespace-only input to empty."""
        assert preprocess_description("   \n\t  ") == ""

    def test_full_pipeline_realistic_description(self):
        """Full pipeline handles a realistic job description."""
        raw = (
            "&lt;p&gt;We\u2019re looking for a Senior Python Engineer&lt;/p&gt;"
            "&lt;ul&gt;&lt;li&gt;5+ years experience&lt;/li&gt;"
            "&lt;li&gt;Django/FastAPI&lt;/li&gt;&lt;/ul&gt;"
            "&lt;p&gt;Salary: $120k&amp;nbsp;\u2013&amp;nbsp;$160k&lt;/p&gt;"
            "&lt;iframe&gt;ad&lt;/iframe&gt;"
        )
        result = preprocess_description(raw)
        # Should be plain text with no HTML
        assert "<" not in result
        assert ">" not in result
        assert "iframe" not in result
        # Should have readable content
        assert "Senior" in result
        assert "Python" in result
        assert "5" in result


class TestDbMigration:
    """Test database column migration."""

    def test_adds_cleaned_description_column(self, tmp_db):
        """Run preprocessing adds cleaned_description column."""
        run_preprocessing(tmp_db)
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(jobs)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        assert "cleaned_description" in columns

    def test_adds_preprocessed_column(self, tmp_db):
        """Run preprocessing adds preprocessed column."""
        run_preprocessing(tmp_db)
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(jobs)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        assert "preprocessed" in columns

    def test_migration_idempotent(self, tmp_db):
        """Running preprocessing twice doesn't raise on duplicate columns."""
        run_preprocessing(tmp_db)
        # Should not raise
        run_preprocessing(tmp_db)

    def test_migration_on_existing_columns(self, tmp_db):
        """Preprocessing works if columns already exist."""
        # Manually add columns to simulate pre-migrated DB
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("ALTER TABLE jobs ADD COLUMN cleaned_description TEXT")
        cursor.execute("ALTER TABLE jobs ADD COLUMN preprocessed INTEGER DEFAULT 0")
        conn.commit()
        conn.close()
        # Should not raise
        run_preprocessing(tmp_db)


class TestBatchProcessing:
    """Test batch processing and database updates."""

    def test_processes_all_jobs(self, tmp_db):
        """All inserted jobs are processed."""
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        # Insert 10 test jobs
        for i in range(10):
            cursor.execute(
                "INSERT INTO jobs (external_id, board_token, description) "
                "VALUES (?, ?, ?)",
                (f"ext_{i}", "token", f"<p>Job {i}</p>"),
            )
        conn.commit()
        conn.close()

        run_preprocessing(tmp_db)

        # Verify all processed
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE preprocessed=1")
        count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE preprocessed=0")
        unprocessed = cursor.fetchone()[0]
        conn.close()

        assert count == 10
        assert unprocessed == 0

    def test_batch_chunking(self, tmp_db):
        """Batch processing handles multiple chunks correctly."""
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        # Insert CHUNK_SIZE + 1 jobs (501 by default)
        for i in range(501):
            cursor.execute(
                "INSERT INTO jobs (external_id, board_token, description) "
                "VALUES (?, ?, ?)",
                (f"ext_{i}", "token", f"<p>Job {i}</p>"),
            )
        conn.commit()
        conn.close()

        run_preprocessing(tmp_db)

        # All should be processed
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE preprocessed=1")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 501

    def test_idempotent_second_run(self, tmp_db):
        """Second run of preprocessing does nothing (all already processed)."""
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        for i in range(5):
            cursor.execute(
                "INSERT INTO jobs (external_id, board_token, description) "
                "VALUES (?, ?, ?)",
                (f"ext_{i}", "token", f"<p>Job {i}</p>"),
            )
        conn.commit()
        conn.close()

        # First run
        run_preprocessing(tmp_db)

        # Get cleaned_description values
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT id, cleaned_description FROM jobs")
        first_run = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()

        # Second run (should process 0 jobs)
        run_preprocessing(tmp_db)

        # Verify cleaned_description unchanged
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT id, cleaned_description FROM jobs")
        second_run = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()

        assert first_run == second_run

    def test_null_description_handled(self, tmp_db):
        """Jobs with NULL description are handled gracefully."""
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO jobs (external_id, board_token, description) "
            "VALUES (?, ?, ?)",
            ("ext_null", "token", None),
        )
        cursor.execute(
            "INSERT INTO jobs (external_id, board_token, description) "
            "VALUES (?, ?, ?)",
            ("ext_valid", "token", "<p>Valid job</p>"),
        )
        conn.commit()
        conn.close()

        # Should not raise
        run_preprocessing(tmp_db)

        # Verify both processed
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE preprocessed=1")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 2

    def test_cleaned_descriptions_correct(self, tmp_db):
        """Cleaned descriptions are correctly preprocessed."""
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO jobs (external_id, board_token, description) "
            "VALUES (?, ?, ?)",
            ("ext_1", "token", "&lt;p&gt;Python&amp;nbsp;Engineer&lt;/p&gt;"),
        )
        conn.commit()
        conn.close()

        run_preprocessing(tmp_db)

        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT cleaned_description FROM jobs WHERE external_id=?", ("ext_1",))
        row = cursor.fetchone()
        conn.close()

        cleaned = row["cleaned_description"]
        assert "<" not in cleaned
        assert "Python" in cleaned
        assert "Engineer" in cleaned
