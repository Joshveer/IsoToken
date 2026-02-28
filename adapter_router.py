"""
AdapterRouter: multi-adapter co-batching over one shared base model.
SPEC Phase 6: Multiple PEP nodes with different LoRA adapters execute in a single forward batch.
Uses HuggingFace Transformers + PEFT only; no custom CUDA kernels.
No global adapter state mutation during any model.forward (set_adapter only between forwards).
"""

from collections import defaultdict


class AdapterRouter:
    """
    Routes batch inputs to the correct adapter; runs one model.forward per unique adapter
    (batch-by-adapter) so heterogeneous adapters are supported without mutating adapter state
    inside a forward pass.
    """

    def __init__(self, base_model):
        """
        base_model: PEFT model with one or more adapters already loaded (or to be registered).
        """
        self._model = base_model
        self._adapters = {}  # name -> config or path (for already-loaded, store name only)

    def register_adapter(self, name: str, lora_path_or_config) -> None:
        """
        Register an adapter by name. If lora_path_or_config is None, adapter is assumed
        already on the model (e.g. from get_peft_model / add_adapter).
        """
        self._adapters[name] = lora_path_or_config

    def forward(self, batch_inputs: list, adapter_map: list) -> list:
        """
        Run forward for each (input, adapter) pair. Groups by adapter and runs one
        model.forward per unique adapter (no set_active_adapter inside forward).
        Returns outputs in same order as batch_inputs.
        """
        if len(batch_inputs) != len(adapter_map):
            raise ValueError("batch_inputs and adapter_map must have same length")
        if not batch_inputs:
            return []

        # Group indices by adapter
        by_adapter = defaultdict(list)
        for i, adapter in enumerate(adapter_map):
            by_adapter[adapter].append(i)

        # One forward per adapter; collect (index, output)
        results = [None] * len(batch_inputs)
        for adapter, indices in by_adapter.items():
            inputs_for_adapter = [batch_inputs[i] for i in indices]
            # Set adapter before forward (not during); then run forward for this sub-batch
            self._model.set_adapter(adapter)
            outputs = self._forward_batch(inputs_for_adapter)
            for idx, out in zip(indices, outputs):
                results[idx] = out
        return results

    def _forward_batch(self, inputs: list) -> list:
        """Run one model.forward for all inputs (same adapter). Returns list of last-token logits."""
        if not inputs:
            return []
        import torch
        from load_lora import BASE_MODEL_ID
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        encoded = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            out = self._model(**encoded)
        logits = out.logits
        # Last non-padding token per sequence
        batch_size = logits.shape[0]
        results = []
        for i in range(batch_size):
            last_idx = (encoded["attention_mask"][i] != 0).sum().item() - 1
            results.append(logits[i, last_idx, :].tolist())
        return results
