"""
Task 1.1: Deploy base Llama-3-70B via vLLM (or TGI).
SPEC: Stack is vLLM (or TGI); single base model (e.g. meta-llama/Meta-Llama-3-70B).
Test: Service accepts requests and returns completions; no new transformer/serving engine.
"""

import os
import pytest


# Base URL for the inference service (vLLM or TGI). Must be set when service is deployed.
VLLM_BASE_URL = os.environ.get("ISOTOKEN_VLLM_URL", "http://localhost:8000")


def test_service_accepts_requests_and_returns_completions():
    """
    Service must accept a completion request and return non-empty completions.
    SPEC: Service accepts requests and returns completions.
    Fails until base Llama-3-70B is deployed via vLLM (or TGI).
    """
    try:
        import requests
    except ImportError:
        pytest.fail("requests is required to test the inference service; install it")

    prompt = "Hello, world."
    payload = {
        "prompt": prompt,
        "max_tokens": 16,
    }
    # vLLM and TGI expose OpenAI-compatible completion endpoints
    url = f"{VLLM_BASE_URL.rstrip('/')}/v1/completions"
    response = requests.post(url, json=payload, timeout=30)

    assert response.status_code == 200, (
        f"Service must accept requests and return 200; got {response.status_code}"
    )
    data = response.json()
    assert "choices" in data and len(data["choices"]) > 0, (
        "Response must contain completions (OpenAI-compatible shape)"
    )
    completion_text = data["choices"][0].get("text", "")
    assert completion_text is not None and len(completion_text.strip()) > 0, (
        "Service must return non-empty completion text"
    )


def test_service_uses_vllm_or_tgi_no_custom_serving_engine():
    """
    Stack must be vLLM or TGI only; no new transformer or serving engine.
    SPEC: No new transformer, scheduler kernel, LoRA math, or serving engine.
    Stack: HuggingFace Transformers, vLLM (or TGI), PEFT (LoRA), Ray.
    """
    try:
        import requests
    except ImportError:
        pytest.fail("requests is required; install it")

    # Service must identify as vLLM or TGI (e.g. via /health or /info)
    health_url = f"{VLLM_BASE_URL.rstrip('/')}/health"
    try:
        response = requests.get(health_url, timeout=5)
    except requests.RequestException as e:
        pytest.fail(
            f"Cannot reach inference service at {VLLM_BASE_URL}; "
            f"deploy base runtime (task 1.1). Error: {e}"
        )

    assert response.status_code == 200, (
        f"Health check must return 200; got {response.status_code}"
    )
    body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    # vLLM often exposes "vllm" in health/info; TGI may expose "tgi" or similar
    backend = body.get("backend") or body.get("model_engine") or body.get("framework") or ""
    backend_lower = str(backend).lower()
    assert "vllm" in backend_lower or "tgi" in backend_lower or "text-generation" in backend_lower, (
        f"Service must be vLLM or TGI (no custom serving engine); got backend info: {body}"
    )


def test_stack_uses_vllm_or_tgi_dependencies_no_custom_engine():
    """
    Project must depend on vLLM or TGI; must not ship a custom transformer/serving engine.
    SPEC: Stack only — HuggingFace Transformers, vLLM (or TGI), PEFT (LoRA), Ray.
    No new transformer, scheduler kernel, LoRA math, or serving engine.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    deps_content = ""
    for filename in ("requirements.txt", "pyproject.toml"):
        path = os.path.join(root, filename)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                deps_content += f.read()
            break
    assert deps_content, (
        "Project must declare dependencies in requirements.txt or pyproject.toml"
    )
    deps_lower = deps_content.lower()
    assert "vllm" in deps_lower or "text-generation-inference" in deps_lower or "tgi" in deps_lower, (
        "SPEC: stack must include vLLM or TGI; add vllm or text-generation-inference to dependencies"
    )
