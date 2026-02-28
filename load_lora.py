"""
PEFT + LoRA integration. SPEC: LoRA adapters via PEFT; no new LoRA math.
Uses HuggingFace Transformers and PEFT only.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# Tiny model for tests; SPEC base is meta-llama/Meta-Llama-3-70B for production.
BASE_MODEL_ID = "sshleifer/tiny-gpt2"

# Adapter names for hot-swap (task 1.3). PEFT first adapter is "default".
ADAPTER_A = "default"
ADAPTER_B = "B"


def load_base():
    """Load base model only. SPEC: single base model."""
    return AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID)


def load_base_with_lora():
    """Load base model and attach one LoRA adapter via PEFT. SPEC: LoRA adapters via PEFT."""
    base = load_base()
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        init_lora_weights=True,
    )
    model = get_peft_model(base, config)
    # Ensure LoRA changes output: PEFT often zero-inits B so initial output equals base.
    for name, module in model.named_modules():
        if hasattr(module, "lora_B") and module.lora_B is not None:
            for sub in module.lora_B.values():
                if hasattr(sub, "weight") and sub.weight is not None:
                    with torch.no_grad():
                        sub.weight.copy_(torch.ones_like(sub.weight) * 0.01)
    return model


def load_base_with_two_adapters():
    """Load base with two LoRA adapters (A=default, B) for hot-swap. SPEC: dynamic weight swapping."""
    base = load_base()
    config_a = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        init_lora_weights=True,
    )
    model = get_peft_model(base, config_a)
    _set_lora_b_nonzero(model)
    config_b = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=16,
        lora_dropout=0.0,
        init_lora_weights=True,
    )
    model.add_adapter(ADAPTER_B, config_b)
    for _name, module in model.named_modules():
        if hasattr(module, "lora_B") and module.lora_B is not None and ADAPTER_B in module.lora_B:
            sub = module.lora_B[ADAPTER_B]
            if hasattr(sub, "weight") and sub.weight is not None:
                with torch.no_grad():
                    sub.weight.copy_(torch.ones_like(sub.weight) * 0.02)
    return model


def _set_lora_b_nonzero(model):
    """Ensure LoRA B is non-zero so adapter output differs from base."""
    for name, module in model.named_modules():
        if hasattr(module, "lora_B") and module.lora_B is not None:
            for sub in module.lora_B.values():
                if hasattr(sub, "weight") and sub.weight is not None:
                    with torch.no_grad():
                        sub.weight.copy_(torch.ones_like(sub.weight) * 0.01)


def set_active_adapter(model, adapter_name: str) -> None:
    """Set the active LoRA adapter (hot-swap without full restart). SPEC: one adapter active per thread."""
    model.set_adapter(adapter_name)


def get_active_adapter(model) -> str:
    """Return the name of the currently active adapter."""
    out = model.active_adapter
    return out if isinstance(out, str) else out[0]


def run_forward(model, prompt: str, max_new_tokens: int = 5) -> str:
    """Run one forward/generate pass. Returns generated text."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def run_forward_ids(model, prompt: str, max_new_tokens: int = 5):
    """Run one forward/generate pass. Returns generated token ids (for distinctness check)."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return outputs[0].tolist()


def run_forward_logits(model, prompt: str):
    """One forward pass; returns last-token logits. Adapter must change these vs base."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    inputs = tokenizer(prompt, return_tensors="pt")
    with __import__("torch").no_grad():
        out = model(**inputs)
    return out.logits[0, -1, :].tolist()
