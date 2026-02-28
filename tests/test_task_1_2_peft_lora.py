"""
Task 1.2: Integrate PEFT and load at least one LoRA adapter onto the base model.
SPEC: LoRA adapters via PEFT; one shared base model; no new LoRA math.
Test: Adapter loads; forward pass produces distinct output vs base-only.
"""

import pytest


def test_adapter_loads():
    """
    PEFT must load at least one LoRA adapter onto the base model without error.
    SPEC: LoRA adapters via PEFT, multi-adapter injection via dynamic weight swapping.
    """
    from load_lora import load_base_with_lora
    model = load_base_with_lora()
    assert model is not None
    # Model must have LoRA layers (PEFT leaves adapter modules attached)
    assert hasattr(model, "peft_config") or any(
        "lora" in name.lower() for name, _ in model.named_modules()
    )


def test_forward_pass_produces_distinct_output_vs_base_only():
    """
    Forward pass with adapter must produce output distinct from base-only.
    SPEC: Foundation — one shared base model, LoRA adapters via PEFT.
    Compare logits so distinctness is not masked by greedy decoding.
    """
    from load_lora import load_base, load_base_with_lora, run_forward, run_forward_logits
    base = load_base()
    with_lora = load_base_with_lora()
    prompt = "The answer is"
    out_base = run_forward(base, prompt, max_new_tokens=5)
    out_lora = run_forward(with_lora, prompt, max_new_tokens=5)
    assert out_base is not None and out_lora is not None
    logits_base = run_forward_logits(base, prompt)
    logits_lora = run_forward_logits(with_lora, prompt)
    assert logits_base != logits_lora, (
        "Adapter forward pass must produce distinct output vs base-only (logits differ)"
    )
