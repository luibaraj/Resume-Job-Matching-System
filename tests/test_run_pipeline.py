"""Tests for the standalone pipeline runner script.

Tests cover CLI argument parsing, step selection, env loading, and error handling.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import time

import pytest

# Add project root to path to import the runner
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import run_pipeline


class TestStepRegistry:
    """Tests for the step registry."""

    def test_step_names_match_expected_order(self):
        """Step names are in the correct order."""
        expected_names = [
            "collect",
            "preprocess",
            "filter",
            "extract",
            "embed",
            "retrieve",
            "rerank",
            "generate",
            "evaluate",
        ]
        assert run_pipeline.STEP_NAMES == expected_names

    def test_nine_steps_total(self):
        """Registry contains exactly 9 steps."""
        assert len(run_pipeline.STEPS) == 9

    def test_evaluate_has_argv_patch_flag(self):
        """Only the evaluate step has needs_argv_patch=True."""
        evaluate_step = next(s for s in run_pipeline.STEPS if s.name == "evaluate")
        assert evaluate_step.needs_argv_patch is True

    def test_non_evaluate_steps_no_argv_patch(self):
        """Non-evaluate steps have needs_argv_patch=False."""
        for step in run_pipeline.STEPS:
            if step.name != "evaluate":
                assert step.needs_argv_patch is False


class TestEnvLoading:
    """Tests for environment variable loading from .env file."""

    def test_load_dotenv_parses_basic_vars(self):
        """_load_dotenv correctly parses KEY=VALUE lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("KEY1=value1\n")
            f.write("KEY2=value2\n")
            f.flush()
            env_path = f.name

        try:
            result = run_pipeline._load_dotenv(env_path)
            assert result == {"KEY1": "value1", "KEY2": "value2"}
        finally:
            os.unlink(env_path)

    def test_load_dotenv_strips_quotes(self):
        """_load_dotenv removes surrounding quotes from values."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write('KEY1="value1"\n')
            f.write("KEY2='value2'\n")
            f.flush()
            env_path = f.name

        try:
            result = run_pipeline._load_dotenv(env_path)
            assert result == {"KEY1": "value1", "KEY2": "value2"}
        finally:
            os.unlink(env_path)

    def test_load_dotenv_skips_blank_lines(self):
        """_load_dotenv ignores blank lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("KEY1=value1\n")
            f.write("\n")
            f.write("   \n")
            f.write("KEY2=value2\n")
            f.flush()
            env_path = f.name

        try:
            result = run_pipeline._load_dotenv(env_path)
            assert result == {"KEY1": "value1", "KEY2": "value2"}
        finally:
            os.unlink(env_path)

    def test_load_dotenv_skips_comments(self):
        """_load_dotenv ignores lines starting with #."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("# This is a comment\n")
            f.write("KEY1=value1\n")
            f.write("#KEY2=value2\n")
            f.write("KEY3=value3  # inline comment not removed\n")
            f.flush()
            env_path = f.name

        try:
            result = run_pipeline._load_dotenv(env_path)
            # Note: inline comments are NOT removed by the parser
            assert result == {"KEY1": "value1", "KEY3": "value3  # inline comment not removed"}
        finally:
            os.unlink(env_path)

    def test_load_dotenv_handles_missing_file(self):
        """_load_dotenv returns empty dict if file doesn't exist."""
        result = run_pipeline._load_dotenv("/nonexistent/path/.env")
        assert result == {}

    def test_apply_dotenv_uses_setdefault(self):
        """_apply_dotenv uses os.environ.setdefault so real env vars win."""
        test_env = {"TEST_KEY": "from_dotenv"}
        os.environ["TEST_KEY"] = "from_environ"

        run_pipeline._apply_dotenv(test_env)

        # Environment variable should NOT be overwritten
        assert os.environ["TEST_KEY"] == "from_environ"

        # Clean up
        del os.environ["TEST_KEY"]


class TestStepSelection:
    """Tests for step selection logic."""

    def test_select_steps_all_when_no_filter(self):
        """With no --steps or --skip-steps, all steps are selected."""
        class Args:
            steps = None
            skip_steps = None

        result = run_pipeline._select_steps(Args())
        assert [s.name for s in result] == run_pipeline.STEP_NAMES

    def test_select_steps_by_name(self):
        """--steps selects only specified steps."""
        class Args:
            steps = "preprocess,extract"
            skip_steps = None

        result = run_pipeline._select_steps(Args())
        assert [s.name for s in result] == ["preprocess", "extract"]

    def test_select_steps_preserves_order(self):
        """--steps preserves order from STEPS tuple regardless of input order."""
        class Args:
            steps = "generate,preprocess,collect"
            skip_steps = None

        result = run_pipeline._select_steps(Args())
        # Order should be collect, preprocess, generate (from STEPS)
        assert [s.name for s in result] == ["collect", "preprocess", "generate"]

    def test_skip_steps_by_name(self):
        """--skip-steps excludes specified steps."""
        class Args:
            steps = None
            skip_steps = "evaluate,collect"

        result = run_pipeline._select_steps(Args())
        expected = [s.name for s in run_pipeline.STEPS if s.name not in {"evaluate", "collect"}]
        assert [s.name for s in result] == expected

    def test_select_steps_invalid_name_raises(self):
        """Invalid step names raise ValueError."""
        class Args:
            steps = "invalid,preprocess"
            skip_steps = None

        with pytest.raises(ValueError, match="Invalid step names"):
            run_pipeline._select_steps(Args())

    def test_skip_steps_invalid_name_raises(self):
        """Invalid step names in --skip-steps raise ValueError."""
        class Args:
            steps = None
            skip_steps = "notavalidstep"

        with pytest.raises(ValueError, match="Invalid step names"):
            run_pipeline._select_steps(Args())


class TestInvoke:
    """Tests for step invocation with error handling."""

    def test_invoke_calls_step_fn(self):
        """_invoke calls the step's fn()."""
        mock_fn = MagicMock()
        step = run_pipeline.Step("test", mock_fn)

        run_pipeline._invoke(step)

        mock_fn.assert_called_once()

    def test_invoke_wraps_system_exit_non_zero(self):
        """_invoke wraps SystemExit(!=0) as RuntimeError."""
        def failing_fn():
            raise SystemExit(1)

        step = run_pipeline.Step("test", failing_fn)

        with pytest.raises(RuntimeError, match="exited with code 1"):
            run_pipeline._invoke(step)

    def test_invoke_treats_exit_zero_as_success(self):
        """_invoke treats SystemExit(0) as success."""
        def exit_zero_fn():
            raise SystemExit(0)

        step = run_pipeline.Step("test", exit_zero_fn)

        # Should not raise
        run_pipeline._invoke(step)

    def test_invoke_patches_argv_for_evaluation(self):
        """_invoke sets sys.argv to ["evaluation"] before calling evaluation step."""
        original_argv = sys.argv[:]
        captured_argv = None

        def capture_argv_fn():
            nonlocal captured_argv
            captured_argv = sys.argv[:]

        step = run_pipeline.Step("test", capture_argv_fn, needs_argv_patch=True)
        run_pipeline._invoke(step)

        # Inside fn, sys.argv should be ["evaluation"]
        assert captured_argv == ["evaluation"]
        # After fn, sys.argv should be restored
        assert sys.argv == original_argv

    def test_invoke_restores_argv_on_error(self):
        """_invoke restores sys.argv even if step fn raises."""
        original_argv = sys.argv[:]

        def failing_fn():
            raise RuntimeError("test error")

        step = run_pipeline.Step("test", failing_fn, needs_argv_patch=True)

        with pytest.raises(RuntimeError):
            run_pipeline._invoke(step)

        # sys.argv should still be restored
        assert sys.argv == original_argv


class TestRunStep:
    """Tests for step execution with retry logic."""

    def test_run_step_succeeds_on_first_try(self):
        """_run_step calls step immediately if successful."""
        mock_logger = MagicMock()
        mock_fn = MagicMock()
        step = run_pipeline.Step("test", mock_fn)

        run_pipeline._run_step(step, mock_logger, retries=1, retry_delay=0.01)

        mock_fn.assert_called_once()
        # No retry warnings should be logged
        for call_obj in mock_logger.warning.call_args_list:
            assert "Retrying" not in str(call_obj)

    def test_run_step_retries_on_failure(self):
        """_run_step retries after failure."""
        mock_logger = MagicMock()
        call_count = 0

        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("first attempt fails")

        step = run_pipeline.Step("test", failing_then_success)

        # Should not raise; second attempt succeeds
        run_pipeline._run_step(step, mock_logger, retries=1, retry_delay=0.01)

        assert call_count == 2
        mock_logger.warning.assert_called_once()

    def test_run_step_raises_after_retries_exhausted(self):
        """_run_step raises after all retries are exhausted."""
        mock_logger = MagicMock()

        def always_fails():
            raise RuntimeError("always fails")

        step = run_pipeline.Step("test", always_fails)

        with pytest.raises(RuntimeError, match="always fails"):
            run_pipeline._run_step(step, mock_logger, retries=1, retry_delay=0.01)

        # Should have called fn twice (1 initial + 1 retry)
        assert mock_logger.warning.call_count == 1
        # Error logged once after exhausting retries
        assert mock_logger.error.call_count == 1

    def test_run_step_respects_retry_delay(self):
        """_run_step sleeps between retries."""
        mock_logger = MagicMock()
        call_times = []

        def failing_then_success():
            call_times.append(time.time())
            if len(call_times) < 2:
                raise RuntimeError("fail")

        step = run_pipeline.Step("test", failing_then_success)
        delay = 0.1

        run_pipeline._run_step(step, mock_logger, retries=1, retry_delay=delay)

        # Time between attempts should be >= delay
        elapsed = call_times[1] - call_times[0]
        assert elapsed >= delay * 0.9  # Allow 10% tolerance for timing variance


class TestCLIParsing:
    """Tests for command-line argument parsing."""

    def test_parse_args_default_values(self):
        """parse_args returns correct defaults."""
        with patch("sys.argv", ["run_pipeline.py"]):
            args = run_pipeline._parse_args()
            assert args.steps is None
            assert args.skip_steps is None
            assert args.retries == 1
            assert args.retry_delay == 300.0

    def test_parse_args_steps_option(self):
        """--steps argument is parsed correctly."""
        with patch("sys.argv", ["run_pipeline.py", "--steps", "collect,preprocess"]):
            args = run_pipeline._parse_args()
            assert args.steps == "collect,preprocess"

    def test_parse_args_skip_steps_option(self):
        """--skip-steps argument is parsed correctly."""
        with patch("sys.argv", ["run_pipeline.py", "--skip-steps", "evaluate"]):
            args = run_pipeline._parse_args()
            assert args.skip_steps == "evaluate"

    def test_parse_args_retries_option(self):
        """--retries argument is parsed correctly."""
        with patch("sys.argv", ["run_pipeline.py", "--retries", "3"]):
            args = run_pipeline._parse_args()
            assert args.retries == 3

    def test_parse_args_retry_delay_option(self):
        """--retry-delay argument is parsed correctly."""
        with patch("sys.argv", ["run_pipeline.py", "--retry-delay", "60"]):
            args = run_pipeline._parse_args()
            assert args.retry_delay == 60.0

    def test_parse_args_env_file_option(self):
        """--env-file argument is parsed correctly."""
        with patch("sys.argv", ["run_pipeline.py", "--env-file", "/custom/path/.env"]):
            args = run_pipeline._parse_args()
            assert args.env_file == "/custom/path/.env"


class TestMainFunction:
    """Tests for the main orchestration function."""

    def test_main_returns_zero_on_success(self):
        """main() returns 0 when all steps succeed."""
        mock_steps = [
            run_pipeline.Step("step1", MagicMock()),
            run_pipeline.Step("step2", MagicMock()),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("")
            env_path = f.name

        try:
            with patch.object(run_pipeline, "STEPS", tuple(mock_steps)):
                with patch("sys.argv", ["run_pipeline.py", "--env-file", env_path]):
                    result = run_pipeline.main()
                    assert result == 0
        finally:
            os.unlink(env_path)

    def test_main_returns_one_on_step_failure(self):
        """main() returns 1 when a step fails."""
        def failing_fn():
            raise RuntimeError("step failed")

        mock_steps = [
            run_pipeline.Step("step1", MagicMock()),
            run_pipeline.Step("step2", failing_fn),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("")
            env_path = f.name

        try:
            with patch.object(run_pipeline, "STEPS", tuple(mock_steps)):
                with patch("sys.argv", ["run_pipeline.py", "--env-file", env_path]):
                    result = run_pipeline.main()
                    assert result == 1
        finally:
            os.unlink(env_path)

    def test_main_stops_on_first_failure(self):
        """main() stops pipeline when a step fails."""
        mock_fn1 = MagicMock()
        mock_fn2 = MagicMock()

        def failing_fn():
            raise RuntimeError("step failed")

        mock_fn3 = MagicMock()

        mock_steps = [
            run_pipeline.Step("step1", mock_fn1),
            run_pipeline.Step("step2", failing_fn),
            run_pipeline.Step("step3", mock_fn3),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("")
            env_path = f.name

        try:
            with patch.object(run_pipeline, "STEPS", tuple(mock_steps)):
                with patch("sys.argv", ["run_pipeline.py", "--env-file", env_path]):
                    result = run_pipeline.main()

            # step1 should be called, step2 fails, step3 should NOT be called
            mock_fn1.assert_called_once()
            mock_fn3.assert_not_called()
            assert result == 1
        finally:
            os.unlink(env_path)
