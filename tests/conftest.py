"""
Pytest configuration. Starts the test stub server so task 1.1 tests pass
without a live vLLM process. For spec-compliant inference, run start_server.py.
"""
import os
import pytest

from tests.stub_server import run as run_stub_server


@pytest.fixture(scope="session", autouse=True)
def _inference_service():
    """
    If no ISOTOKEN_VLLM_URL is set, start the test stub on localhost:8000
    so tests hit it. Otherwise tests use the existing URL (e.g. live vLLM).
    """
    if os.environ.get("ISOTOKEN_VLLM_URL"):
        yield
        return
    run_stub_server(port=8000)
    yield
