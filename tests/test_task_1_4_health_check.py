"""
Task 1.4: Add a minimal health check that validates stack (vLLM + PEFT + single base model).
SPEC: Stack — HuggingFace Transformers, vLLM (or TGI), PEFT (LoRA), Ray.
Test: Health endpoint passes when stack is correct and fails when component is missing.
"""

import os
import pytest

# Base URL for the inference service (stub or vLLM). Conftest starts stub when ISOTOKEN_VLLM_URL unset.
HEALTH_BASE_URL = os.environ.get("ISOTOKEN_VLLM_URL", "http://localhost:8000")


def test_health_passes_when_stack_correct():
    """
    When vLLM (or stub), PEFT, and single base model are present, GET /health returns 200.
    SPEC: Health check validates stack (vLLM + PEFT + single base model).
    """
    try:
        import requests
    except ImportError:
        pytest.fail("requests is required; install it")
    # Ensure we don't simulate missing components
    for key in ("ISOTOKEN_HEALTH_SKIP_VLLM", "ISOTOKEN_HEALTH_SKIP_PEFT", "ISOTOKEN_HEALTH_SKIP_SINGLE_BASE"):
        os.environ.pop(key, None)
    url = f"{HEALTH_BASE_URL.rstrip('/')}/health"
    response = requests.get(url, timeout=5)
    assert response.status_code == 200, (
        f"Health must return 200 when stack is correct; got {response.status_code}"
    )
    data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    assert data.get("vllm") is True and data.get("peft") is True and data.get("single_base_model") is True, (
        "Health body must indicate vllm, peft, single_base_model all true when stack is correct"
    )


def test_health_fails_when_component_missing():
    """
    When a stack component is missing, GET /health returns non-200 and indicates the failure.
    SPEC: Health endpoint fails when component is missing.
    """
    try:
        import requests
    except ImportError:
        pytest.fail("requests is required; install it")
    os.environ["ISOTOKEN_HEALTH_SKIP_PEFT"] = "1"
    try:
        url = f"{HEALTH_BASE_URL.rstrip('/')}/health"
        response = requests.get(url, timeout=5)
        assert response.status_code != 200, (
            "Health must not return 200 when a component is missing (simulated: PEFT)"
        )
        data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        assert data.get("peft") is False or "peft" in str(data).lower(), (
            "Health body must indicate which component is missing"
        )
    finally:
        os.environ.pop("ISOTOKEN_HEALTH_SKIP_PEFT", None)
