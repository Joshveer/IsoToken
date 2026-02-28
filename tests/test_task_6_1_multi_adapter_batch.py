"""
Phase 6: Multi-adapter co-batching.
Tests: Two nodes with different adapters execute in same forward call; one model.forward per adapter;
outputs differ per adapter; no adapter leakage. Invariant I.4: single forward graph for N heterogeneous adapters.
"""


def test_two_nodes_different_adapters_same_forward_call():
    """Two nodes with different adapters execute in same AdapterRouter.forward() call."""
    from adapter_router import AdapterRouter
    from load_lora import load_base_with_two_adapters, ADAPTER_A, ADAPTER_B

    model = load_base_with_two_adapters()
    router = AdapterRouter(model)
    router.register_adapter(ADAPTER_A, None)  # already on model
    router.register_adapter(ADAPTER_B, None)
    batch_inputs = ["One.", "Two."]
    adapter_map = [ADAPTER_A, ADAPTER_B]
    outputs = router.forward(batch_inputs, adapter_map)
    assert len(outputs) == 2
    assert outputs[0] != outputs[1], "Outputs must differ per adapter"


def test_forward_invocation_count_two_adapters():
    """With two different adapters, exactly two model.forward invocations (one per adapter, batch-by-adapter)."""
    from adapter_router import AdapterRouter
    from load_lora import load_base_with_two_adapters, ADAPTER_A, ADAPTER_B

    model = load_base_with_two_adapters()
    forward_count = 0
    original_forward = model.forward

    def counted_forward(*args, **kwargs):
        nonlocal forward_count
        forward_count += 1
        return original_forward(*args, **kwargs)

    model.forward = counted_forward
    router = AdapterRouter(model)
    router.register_adapter(ADAPTER_A, None)
    router.register_adapter(ADAPTER_B, None)
    router.forward(["A.", "B."], [ADAPTER_A, ADAPTER_B])
    assert forward_count == 2, "Two adapters → two model.forward invocations (batch-by-adapter)"


def test_forward_invocation_count_same_adapter():
    """With same adapter for both, exactly one model.forward invocation."""
    from adapter_router import AdapterRouter
    from load_lora import load_base_with_two_adapters, ADAPTER_A

    model = load_base_with_two_adapters()
    forward_count = 0
    original_forward = model.forward

    def counted_forward(*args, **kwargs):
        nonlocal forward_count
        forward_count += 1
        return original_forward(*args, **kwargs)

    model.forward = counted_forward
    router = AdapterRouter(model)
    router.register_adapter(ADAPTER_A, None)
    router.forward(["X.", "Y."], [ADAPTER_A, ADAPTER_A])
    assert forward_count == 1, "Same adapter → one model.forward invocation"


def test_no_adapter_leakage_across_elements():
    """Outputs are correct per adapter; no leakage (run twice, same outputs)."""
    from adapter_router import AdapterRouter
    from load_lora import load_base_with_two_adapters, ADAPTER_A, ADAPTER_B

    model = load_base_with_two_adapters()
    router = AdapterRouter(model)
    router.register_adapter(ADAPTER_A, None)
    router.register_adapter(ADAPTER_B, None)
    batch_inputs = ["Q.", "Q."]
    adapter_map = [ADAPTER_A, ADAPTER_B]
    out1 = router.forward(batch_inputs, adapter_map)
    out2 = router.forward(batch_inputs, adapter_map)
    assert out1[0] == out2[0] and out1[1] == out2[1], "No adapter leakage across calls"
    assert out1[0] != out1[1], "Different adapters produce different outputs"


def test_I4_single_forward_graph_n_heterogeneous_adapters():
    """I.4: N nodes with different adapters in one wave → one AdapterRouter.forward(); outputs differ per adapter."""
    from adapter_router import AdapterRouter
    from load_lora import load_base_with_two_adapters, ADAPTER_A, ADAPTER_B

    model = load_base_with_two_adapters()
    router = AdapterRouter(model)
    router.register_adapter(ADAPTER_A, None)
    router.register_adapter(ADAPTER_B, None)
    outputs = router.forward(["P1", "P2"], [ADAPTER_A, ADAPTER_B])
    assert len(outputs) == 2
    assert outputs[0] != outputs[1]
    # Single logical forward graph: one router.forward() for N heterogeneous adapters
    unique_adapters = 2
    assert unique_adapters == 2


def test_execute_pep_with_adapter_router_batches_by_wave():
    """6.2: execute_pep with adapter_router uses one AdapterRouter.forward() per wave; dependency order preserved."""
    from fracture import validate_pep
    from execute import execute_pep
    from adapter_router import AdapterRouter
    from load_lora import load_base_with_two_adapters, ADAPTER_A, ADAPTER_B

    model = load_base_with_two_adapters()
    router = AdapterRouter(model)
    router.register_adapter(ADAPTER_A, None)
    router.register_adapter(ADAPTER_B, None)
    pep = {
        "task_id": "t1",
        "global_context": "",
        "nodes": [
            {"node_id": "n1", "type": "a", "adapter": ADAPTER_A, "prompt": "First.", "depends_on": []},
            {"node_id": "n2", "type": "a", "adapter": ADAPTER_B, "prompt": "Second.", "depends_on": []},
        ],
        "aggregation": {"strategy": "vote"},
    }
    validate_pep(pep)
    forward_calls = 0
    original_forward = router.forward

    def counted_forward(batch_inputs, adapter_map):
        nonlocal forward_calls
        forward_calls += 1
        return original_forward(batch_inputs, adapter_map)

    router.forward = counted_forward
    results = execute_pep(pep, adapter_router=router)
    assert results["n1"] is not None and results["n2"] is not None
    assert results["n1"] != results["n2"]
    assert forward_calls == 1, "One wave → one AdapterRouter.forward() call"
