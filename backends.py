"""
LLM backends for IsoToken. API backends return a simple run_node callable.
The local backend returns a LocalBackend object with run_node plus
prefill_shared_kv, decode_with_kv, and forward_batch for optimizations.
"""

from typing import Any, Callable

import requests


def _build_prompt(node: dict, context: dict) -> str:
    """Build full prompt for a node: dependency context (if any) + node prompt."""
    prompt = node.get("prompt", "")
    if not context:
        return prompt
    ctx_parts = [str(v) for v in context.values()]
    return "Context from previous steps:\n\n" + "\n\n---\n\n".join(ctx_parts) + "\n\n---\n\n" + prompt


def _run_node_openai(api_key: str, model: str) -> Callable[..., str]:
    """Return a run_node callable that uses OpenAI Chat Completions."""
    def run_node(node: dict, context: dict, shared_prefill: Any = None) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = _build_prompt(node, context)
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        return (r.choices[0].message.content or "").strip()
    return run_node


def _run_node_anthropic(api_key: str, model: str) -> Callable[..., str]:
    """Return a run_node callable that uses Anthropic Messages API."""
    def run_node(node: dict, context: dict, shared_prefill: Any = None) -> str:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        prompt = _build_prompt(node, context)
        r = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        text = r.content[0].text if r.content else ""
        return text.strip()
    return run_node


def _run_node_ollama(host: str, model: str) -> Callable[..., str]:
    """Return a run_node callable that uses Ollama Python SDK."""
    def run_node(node: dict, context: dict, shared_prefill: Any = None) -> str:
        import ollama
        client = ollama.Client(host=host)
        prompt = _build_prompt(node, context)
        r = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return (r.get("message", {}).get("content", "") or "").strip()
    return run_node


def _run_node_openai_compatible(base_url: str, model: str | None, api_key: str | None) -> Callable[..., str]:
    """Return a run_node callable that uses an OpenAI-compatible API (vLLM, LM Studio, etc.)."""
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    chat_url = url + "/chat/completions"

    def run_node(node: dict, context: dict, shared_prefill: Any = None) -> str:
        prompt = _build_prompt(node, context)
        body = {
            "model": model or "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = requests.post(chat_url, json=body, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        return (message.get("content") or "").strip()
    return run_node


def make_run_node(
    backend: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    model_id: str | None = None,
    base_url: str | None = None,
    host: str | None = None,
    adapters: dict[str, str] | None = None,
):
    """
    Create a run_node callable (or LocalBackend object) for the given LLM backend.
    backend: "openai" | "anthropic" | "ollama" | "openai_compatible" | "local"
    For "local": returns a LocalBackend instance (has .run_node, .prefill_shared_kv, .forward_batch).
    For API backends: returns a plain callable.
    """
    if backend == "openai":
        if not api_key:
            raise ValueError("OpenAI backend requires api_key (set OPENAI_API_KEY)")
        return _run_node_openai(api_key, model or "gpt-4o")
    if backend == "anthropic":
        if not api_key:
            raise ValueError("Anthropic backend requires api_key (set ANTHROPIC_API_KEY)")
        return _run_node_anthropic(api_key, model or "claude-sonnet-4-20250514")
    if backend == "ollama":
        if not model:
            raise ValueError("Ollama backend requires model (set OLLAMA_MODEL)")
        return _run_node_ollama(host or "http://localhost:11434", model)
    if backend == "openai_compatible":
        if not base_url:
            raise ValueError("openai_compatible backend requires base_url (set ISOTOKEN_LLM_URL)")
        return _run_node_openai_compatible(base_url, model, api_key)
    if backend == "local":
        mid = model_id or model
        if not mid:
            raise ValueError("Local backend requires model_id (--model <hf_model_id>)")
        from local_backend import LocalBackend
        return LocalBackend(mid, adapters=adapters)
    raise ValueError(f"Unknown backend: {backend!r}. Use: openai, anthropic, ollama, openai_compatible, local")
