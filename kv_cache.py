"""
Real prefix KV cache sharing. Phase 7: prefill once, capture past_key_values;
decode per node with same shared_kv. No recompute of prefix per node.
"""

import torch


def prefill_global_context(model, tokenizer, global_context: str):
    """
    Run model(global_context_ids, use_cache=True); return past_key_values.
    If global_context is empty, use a single token so cache structure is valid.
    """
    if not global_context or not global_context.strip():
        global_context = tokenizer.eos_token or "<|endoftext|>"
    inputs = tokenizer(global_context, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    if "attention_mask" in inputs:
        attention_mask = inputs["attention_mask"].to(model.device)
    else:
        attention_mask = None
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    return out.past_key_values


def decode_with_prefix(model, tokenizer, prompt: str, past_key_values):
    """
    Run model(node_ids, past_key_values=shared_kv, use_cache=True).
    Returns last-token logits (list) for the prompt. Does not mutate past_key_values.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    if "attention_mask" in inputs:
        attention_mask = inputs["attention_mask"].to(model.device)
    else:
        attention_mask = None
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=True,
        )
    logits = out.logits
    last_idx = logits.shape[1] - 1
    return logits[0, last_idx, :].tolist()
