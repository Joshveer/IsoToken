"""
FLOP estimates for attention. Used to show cost with vs without prefix KV reuse.
"""


def estimate_prefill_cost(seq_len: int, hidden_size: int, num_layers: int) -> float:
    """
    Estimate prefill FLOPs for causal self-attention over a sequence of length seq_len.
    Simplified: O(seq_len^2) per layer for attention (QK^T + softmax + scale), plus linear.
    Returns a scalar cost (e.g. FLOPs).
    """
    if seq_len <= 0:
        return 0.0
    # Per layer: Q,K,V projections ~ 3 * seq_len * hidden_size^2; attention QK^T ~ seq_len^2 * hidden_size; output ~ seq_len * hidden_size^2
    attn_flops_per_layer = 4 * seq_len * hidden_size * hidden_size + 2 * seq_len * seq_len * hidden_size
    return float(num_layers * attn_flops_per_layer)
