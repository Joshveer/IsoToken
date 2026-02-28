"""
Task 1.3: Implement hot-swap of LoRA adapters (swap without full restart).
SPEC: One adapter active per thread; clear weight deltas between runs.
Test: Swap adapter A → B → A; outputs match expected adapter behavior; no cross-adapter leakage.
"""

import pytest

from load_lora import (
    ADAPTER_A,
    ADAPTER_B,
    get_active_adapter,
    load_base_with_two_adapters,
    run_forward_logits,
    set_active_adapter,
)


def test_swap_a_to_b_to_a_outputs_match_expected_adapter_behavior():
    """
    After swapping A → B → A, output with A must match output when A was active before B.
    SPEC: Hot-swap without full restart; no cross-adapter leakage (one adapter active per thread).
    """
    model = load_base_with_two_adapters()
    prompt = "The answer is"
    set_active_adapter(model, ADAPTER_A)
    logits_a1 = run_forward_logits(model, prompt)
    set_active_adapter(model, ADAPTER_B)
    run_forward_logits(model, prompt)
    set_active_adapter(model, ADAPTER_A)
    logits_a2 = run_forward_logits(model, prompt)
    assert logits_a1 == logits_a2, (
        "After swap A→B→A, output with A must match initial A (no cross-adapter leakage)"
    )


def test_adapters_a_and_b_produce_different_outputs():
    """
    Adapters A and B must produce distinct outputs so swap is observable.
    SPEC: Dynamic weight swapping; one adapter active per thread.
    """
    model = load_base_with_two_adapters()
    prompt = "The answer is"
    set_active_adapter(model, ADAPTER_A)
    logits_a = run_forward_logits(model, prompt)
    set_active_adapter(model, ADAPTER_B)
    logits_b = run_forward_logits(model, prompt)
    assert logits_a != logits_b, (
        "Adapters A and B must produce distinct outputs (swap is effective)"
    )


def test_one_adapter_active_after_swap():
    """
    Only one adapter is active at a time after set_active_adapter.
    SPEC: Only one adapter active per thread; clear weight deltas between runs.
    """
    model = load_base_with_two_adapters()
    set_active_adapter(model, ADAPTER_A)
    assert get_active_adapter(model) == ADAPTER_A
    set_active_adapter(model, ADAPTER_B)
    assert get_active_adapter(model) == ADAPTER_B
    set_active_adapter(model, ADAPTER_A)
    assert get_active_adapter(model) == ADAPTER_A
