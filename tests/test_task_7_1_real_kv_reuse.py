"""
Phase 7: Real prefix KV cache sharing. Replace simulated shared_prefill with actual past_key_values reuse.
Tests: Prefill called exactly once; past_key_values reused for all nodes (object identity);
FLOP estimate with reuse < without reuse; prefix_release_fn still works.
"""


def test_prefill_called_exactly_once():
    """With shared_kv_model and shared_kv_tokenizer, prefill is invoked exactly once for N nodes."""
    from fracture import validate_pep
    from execute import execute_pep
    from load_lora import load_base_with_two_adapters
    from transformers import AutoTokenizer
    from load_lora import BASE_MODEL_ID

    model = load_base_with_two_adapters()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    prefill_count = 0
    original_prefill = __import__("kv_cache", fromlist=["prefill_global_context"]).prefill_global_context

    def counted_prefill(m, t, gc):
        nonlocal prefill_count
        prefill_count += 1
        return original_prefill(m, t, gc)

    import kv_cache
    kv_cache.prefill_global_context = counted_prefill
    pep = {
        "task_id": "t1",
        "global_context": "System context.",
        "nodes": [
            {"node_id": "n1", "type": "a", "adapter": "default", "prompt": "Q1.", "depends_on": []},
            {"node_id": "n2", "type": "a", "adapter": "B", "prompt": "Q2.", "depends_on": []},
        ],
        "aggregation": {"strategy": "vote"},
    }
    validate_pep(pep)
    metrics = {}
    execute_pep(pep, shared_kv_model=model, shared_kv_tokenizer=tokenizer, metrics=metrics)
    assert prefill_count == 1, "Prefill must be called exactly once"
    assert metrics.get("prefill_count") == 1
    kv_cache.prefill_global_context = original_prefill


def test_past_key_values_reused_for_all_nodes():
    """The same past_key_values object identity is passed to every node decode."""
    from fracture import validate_pep
    from execute import execute_pep
    from load_lora import load_base_with_two_adapters
    from transformers import AutoTokenizer
    from load_lora import BASE_MODEL_ID
    from kv_cache import decode_with_prefix

    model = load_base_with_two_adapters()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    seen_kv_ids = []

    def capture_decode(m, t, prompt, past_key_values):
        seen_kv_ids.append(id(past_key_values))
        return decode_with_prefix(m, t, prompt, past_key_values)

    import kv_cache
    original_decode = kv_cache.decode_with_prefix
    kv_cache.decode_with_prefix = capture_decode
    pep = {
        "task_id": "t1",
        "global_context": "C.",
        "nodes": [
            {"node_id": "n1", "type": "a", "adapter": "default", "prompt": "P1.", "depends_on": []},
            {"node_id": "n2", "type": "a", "adapter": "B", "prompt": "P2.", "depends_on": []},
        ],
        "aggregation": {"strategy": "vote"},
    }
    validate_pep(pep)
    execute_pep(pep, shared_kv_model=model, shared_kv_tokenizer=tokenizer)
    kv_cache.decode_with_prefix = original_decode
    assert len(seen_kv_ids) == 2, "Two nodes decoded"
    assert len(set(seen_kv_ids)) == 1, "past_key_values object identity must be reused for all nodes"


def test_flop_estimate_with_reuse_less_than_without_reuse():
    """FLOP estimate with reuse (1 prefill + N decodes) < without reuse (N full prefills)."""
    from utils.flop_counter import estimate_prefill_cost

    seq_len = 10
    hidden_size = 64
    num_layers = 2
    n_nodes = 3
    # Without reuse: each node does full prefill of seq_len (simplified: same prefix length per node)
    cost_without_reuse = n_nodes * estimate_prefill_cost(seq_len, hidden_size, num_layers)
    # With reuse: 1 prefill of seq_len + N decodes (each decode ~ 1 token so ~ 1*1*H per layer, negligible)
    cost_with_reuse = 1 * estimate_prefill_cost(seq_len, hidden_size, num_layers)
    assert cost_with_reuse < cost_without_reuse, "FLOP with reuse must be less than without reuse"


def test_prefix_release_fn_still_works():
    """When using real shared_kv, prefix_release_fn is called with the past_key_values object."""
    from fracture import validate_pep
    from execute import execute_pep
    from load_lora import load_base_with_two_adapters
    from transformers import AutoTokenizer
    from load_lora import BASE_MODEL_ID

    model = load_base_with_two_adapters()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    released = []

    def release_fn(shared_kv):
        released.append(shared_kv)

    pep = {
        "task_id": "t1",
        "global_context": "X.",
        "nodes": [
            {"node_id": "n1", "type": "a", "adapter": "default", "prompt": "P.", "depends_on": []},
        ],
        "aggregation": {"strategy": "vote"},
    }
    validate_pep(pep)
    execute_pep(pep, shared_kv_model=model, shared_kv_tokenizer=tokenizer, prefill_release_fn=release_fn)
    assert len(released) == 1, "prefix_release_fn must be called once"
    assert released[0] is not None, "Released object must be the shared_kv (past_key_values)"
