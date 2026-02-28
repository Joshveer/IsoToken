"""
Tests for local_backend.py: model loading, run_node, shared KV, co-batch.
Uses sshleifer/tiny-gpt2 (tiny, no GPU needed).
"""

import pytest

MODEL_ID = "sshleifer/tiny-gpt2"


def test_local_backend_loads_model():
    from local_backend import LocalBackend
    backend = LocalBackend(MODEL_ID)
    assert backend.model is not None
    assert backend.tokenizer is not None


def test_run_node_returns_string():
    from local_backend import LocalBackend
    backend = LocalBackend(MODEL_ID)
    node = {"prompt": "Hello", "adapter": "default"}
    result = backend.run_node(node, {})
    assert isinstance(result, str)
    assert len(result) > 0


def test_prefill_shared_kv_returns_cache():
    from local_backend import LocalBackend
    backend = LocalBackend(MODEL_ID)
    kv = backend.prefill_shared_kv("Some context")
    assert kv is not None
    # past_key_values can be a tuple (older transformers) or DynamicCache (newer)
    assert hasattr(kv, "__len__") or hasattr(kv, "key_cache")


def test_prefill_shared_kv_reusable():
    from local_backend import LocalBackend
    backend = LocalBackend(MODEL_ID)
    kv = backend.prefill_shared_kv("Context")
    r1 = backend.decode_with_kv("Hello", kv)
    r2 = backend.decode_with_kv("World", kv)
    assert isinstance(r1, str)
    assert isinstance(r2, str)


def test_decode_with_kv_returns_string():
    from local_backend import LocalBackend
    backend = LocalBackend(MODEL_ID)
    kv = backend.prefill_shared_kv("Test context")
    result = backend.decode_with_kv("Some prompt", kv)
    assert isinstance(result, str)


def test_forward_batch_returns_list():
    from local_backend import LocalBackend
    backend = LocalBackend(MODEL_ID)
    prompts = ["Hello", "World"]
    adapters = ["default", "default"]
    results = backend.forward_batch(prompts, adapters)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)


def test_forward_batch_mismatched_lengths_raises():
    from local_backend import LocalBackend
    backend = LocalBackend(MODEL_ID)
    with pytest.raises(ValueError):
        backend.forward_batch(["a", "b"], ["default"])


def test_make_run_node_local_returns_backend():
    from backends import make_run_node
    from local_backend import LocalBackend
    result = make_run_node("local", model_id=MODEL_ID)
    assert isinstance(result, LocalBackend)


def test_make_run_node_local_no_model_raises():
    from backends import make_run_node
    with pytest.raises(ValueError, match="model_id"):
        make_run_node("local")
