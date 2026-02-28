"""
Tests for backends.py: make_run_node, backend creation, unknown backend.
"""

import pytest


def test_make_run_node_openai_returns_callable():
    from backends import make_run_node
    fn = make_run_node("openai", api_key="sk-test", model="gpt-4o")
    assert callable(fn)


def test_make_run_node_anthropic_returns_callable():
    from backends import make_run_node
    fn = make_run_node("anthropic", api_key="sk-test", model="claude-sonnet-4-20250514")
    assert callable(fn)


def test_make_run_node_ollama_returns_callable():
    from backends import make_run_node
    fn = make_run_node("ollama", model="llama3", host="http://localhost:11434")
    assert callable(fn)


def test_make_run_node_openai_compatible_returns_callable():
    from backends import make_run_node
    fn = make_run_node("openai_compatible", base_url="http://localhost:8000", model="test")
    assert callable(fn)


def test_make_run_node_unknown_backend_raises():
    from backends import make_run_node
    with pytest.raises(ValueError, match="Unknown backend"):
        make_run_node("unknown_backend")


def test_make_run_node_openai_no_key_raises():
    from backends import make_run_node
    with pytest.raises(ValueError, match="api_key"):
        make_run_node("openai")


def test_make_run_node_ollama_no_model_raises():
    from backends import make_run_node
    with pytest.raises(ValueError, match="model"):
        make_run_node("ollama")
