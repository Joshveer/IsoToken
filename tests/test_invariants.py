"""
Invariants & constraints (testable). SPEC: stack, one adapter per thread, parallelize only when tasks >= 2.
"""

import os


def test_I1_stack_only_allowed_deps_no_custom_engine():
    """
    I.1 Stack check: only HuggingFace Transformers, vLLM (or TGI), PEFT, Ray in use;
    no custom transformer/scheduler/LoRA math/serving engine.
    Test: Dependency/import audit or build manifest.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    deps_content = ""
    for filename in ("requirements.txt", "pyproject.toml"):
        path = os.path.join(root, filename)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                deps_content += f.read()
            break
    assert deps_content, "Project must declare dependencies (requirements.txt or pyproject.toml)"
    deps_lower = deps_content.lower()
    assert "transformers" in deps_lower, "SPEC: HuggingFace Transformers required"
    assert "peft" in deps_lower, "SPEC: PEFT required"
    assert (
        "vllm" in deps_lower or "text-generation-inference" in deps_lower or "tgi" in deps_lower
    ), "SPEC: vLLM or TGI required"
    # No custom serving engine or custom transformer package in deps
    assert "custom_serving" not in deps_lower and "custom_transformer" not in deps_lower, (
        "No custom transformer/serving engine"
    )


def test_I2_one_adapter_active_per_thread_no_mixed_output():
    """
    I.2 One adapter active per thread at runtime.
    Test: Multiple nodes with different adapters produce correct adapter-specific outputs; no mixed weights.
    (With shared in-process model, parallel dispatch is serialized per adapter; we verify adapter-specific output.)
    """
    from execute import execute_pep, run_node_with_model
    from load_lora import load_base_with_two_adapters, ADAPTER_A, ADAPTER_B

    model = load_base_with_two_adapters()
    run_node = run_node_with_model(model)
    pep = {
        "task_id": "t1",
        "global_context": "",
        "nodes": [
            {"node_id": "n1", "type": "a", "adapter": ADAPTER_A, "prompt": "Q.", "depends_on": []},
            {"node_id": "n2", "type": "a", "adapter": ADAPTER_B, "prompt": "Q.", "depends_on": []},
        ],
        "aggregation": {"strategy": "vote"},
    }
    # Sequential run: each node gets its adapter set before run; no cross-thread adapter state
    result = execute_pep(pep, run_node=run_node, parallel=False)
    assert result["n1"] != result["n2"], "Different adapters must produce different outputs; no mixed weights"
    # Run again: same outputs (no leakage from previous run)
    result2 = execute_pep(pep, run_node=run_node, parallel=False)
    assert result["n1"] == result2["n1"] and result["n2"] == result2["n2"], "One adapter active per run; no cross-run leakage"


def test_I3_single_node_no_parallel_dispatch():
    """
    I.3 If tasks < 2, run sequential.
    Test: Single-node PEP never triggers parallel dispatch.
    """
    from fracture import fracture, validate_pep
    from execute import execute_pep

    pep = fracture("What is 2+2?")
    validate_pep(pep)
    assert len(pep["nodes"]) == 1, "Single-question yields one node"
    ran = []
    def run_node(node, context):
        ran.append(node["node_id"])
        return "out"
    result = execute_pep(pep, run_node=run_node, parallel=True)
    assert len(result) == 1 and len(ran) == 1, "Single-node PEP runs one node (sequential path)"


def test_I4_single_forward_graph_n_heterogeneous_adapters():
    """I.4: Single forward graph for N heterogeneous adapters (Phase 6 co-batching)."""
    from adapter_router import AdapterRouter
    from load_lora import load_base_with_two_adapters, ADAPTER_A, ADAPTER_B

    model = load_base_with_two_adapters()
    router = AdapterRouter(model)
    router.register_adapter(ADAPTER_A, None)
    router.register_adapter(ADAPTER_B, None)
    outputs = router.forward(["P1", "P2"], [ADAPTER_A, ADAPTER_B])
    assert len(outputs) == 2 and outputs[0] != outputs[1], "N heterogeneous adapters → one router.forward(); outputs differ per adapter"


def test_I3_two_independent_nodes_parallel_dispatch():
    """
    I.3 Parallelize when 2+ independent nodes.
    Test: 2+ independent nodes do trigger parallel dispatch.
    """
    import time
    from fracture import validate_pep
    from execute import execute_pep

    delay = 0.05
    def run_node(node, context):
        time.sleep(delay)
        return node["node_id"]
    pep = {
        "task_id": "t1",
        "global_context": "",
        "nodes": [
            {"node_id": "n1", "type": "a", "adapter": "a", "prompt": "p1", "depends_on": []},
            {"node_id": "n2", "type": "a", "adapter": "a", "prompt": "p2", "depends_on": []},
        ],
        "aggregation": {"strategy": "vote"},
    }
    validate_pep(pep)
    start = time.perf_counter()
    result = execute_pep(pep, run_node=run_node, parallel=True)
    elapsed = time.perf_counter() - start
    assert set(result.keys()) == {"n1", "n2"}, "Both nodes run"
    assert elapsed < 1.5 * delay, "Independent nodes run in parallel (not 2× sequential)"
