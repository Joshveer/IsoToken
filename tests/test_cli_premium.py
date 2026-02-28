"""
Tests for CLI: one-shot via `run`, --help, interactive session init.
All tests use mock backends to avoid real API calls.
"""

import os
from unittest.mock import patch

from typer.testing import CliRunner

runner = CliRunner()


def _mock_make_run_node(**kwargs):
    def run_node(node, context, shared_prefill=None):
        return "mock answer"
    return run_node


def test_one_shot_returns_output():
    from cli import app
    with patch("engine.make_run_node", _mock_make_run_node):
        result = runner.invoke(app, ["--backend", "openai", "run", "Hello?"], env={"OPENAI_API_KEY": "sk-test"})
    assert result.exit_code == 0, result.output + (result.stderr or "")
    assert "Answer" in result.output or "mock answer" in result.output


def test_no_backend_shows_error():
    from cli import app
    result = runner.invoke(app, ["run", "Hello?"], env={
        "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "", "OLLAMA_MODEL": "", "ISOTOKEN_LLM_URL": "",
    })
    assert result.exit_code != 0
    out = result.output.lower() + (result.stderr or "").lower()
    assert "backend" in out or "configuration" in out


def test_help_is_clean():
    from cli import app
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "isotoken" in result.output.lower() or "IsoToken" in result.output


def test_run_command_works():
    from cli import app
    with patch("engine.make_run_node", _mock_make_run_node):
        result = runner.invoke(app, ["--backend", "openai", "run", "Hello?"], env={"OPENAI_API_KEY": "sk-test"})
    assert result.exit_code == 0, result.output + (result.stderr or "")
    assert "Answer" in result.output or "mock answer" in result.output


def test_interactive_session_init():
    from interactive import InteractiveSession
    with patch("engine.make_run_node", _mock_make_run_node):
        session = InteractiveSession({"backend": "openai", "api_key": "sk-test"})
    assert session._engine is not None
    assert session._console is not None
