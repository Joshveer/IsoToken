"""
Local model backend for IsoToken. Loads a HuggingFace model + optional LoRA
adapters via PEFT. Provides shared KV prefix, adapter switching, co-batching.
"""

from collections import defaultdict
from typing import Any


def _check_deps():
    try:
        import torch
        import transformers
        import peft
    except ImportError as e:
        raise ImportError(
            "Local backend requires: torch, transformers, peft, accelerate. "
            "Install with: pip install torch transformers peft accelerate"
        ) from e


class LocalBackend:
    """
    Load a HuggingFace model locally with optional LoRA adapters.
    Provides run_node, shared KV prefill, and co-batching.
    """

    def __init__(self, model_id: str, adapters: dict[str, str] | None = None):
        _check_deps()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model_id = model_id
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(model_id)
        self._adapter_names: list[str] = []

        if adapters:
            from peft import PeftModel
            items = list(adapters.items())
            self._model = PeftModel.from_pretrained(self._model, items[0][1])
            self._adapter_names.append(items[0][0])
            for name, path in items[1:]:
                self._model.load_adapter(path, name)
                self._adapter_names.append(name)

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def set_adapter(self, name: str) -> None:
        if self._adapter_names:
            self._model.set_adapter(name)

    def run_node(self, node: dict, context: dict, shared_prefill: Any = None) -> str:
        """Set adapter from node, generate text, return decoded string."""
        adapter = node.get("adapter", "default")
        if adapter != "default" and self._adapter_names:
            self.set_adapter(adapter)

        prompt = node.get("prompt", "")
        if context:
            ctx_parts = [str(v) for v in context.values()]
            prompt = "\n\n".join(ctx_parts) + "\n\n" + prompt

        return self._generate(prompt)

    def _generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self._model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][input_ids.shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def prefill_shared_kv(self, context: str):
        """Run one prefill pass and return past_key_values for reuse."""
        import torch
        if not context or not context.strip():
            context = self._tokenizer.eos_token or "<|endoftext|>"
        inputs = self._tokenizer(context, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._model.device)
        with torch.no_grad():
            out = self._model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        return out.past_key_values

    def decode_with_kv(self, prompt: str, past_key_values) -> str:
        """Decode using shared past_key_values (no prefix recomputation)."""
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._model.device)
        with torch.no_grad():
            out = self._model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
            )
        logits = out.logits
        last_token_id = logits[0, -1, :].argmax().item()
        return self._tokenizer.decode([last_token_id], skip_special_tokens=True)

    def forward_batch(self, prompts: list[str], adapters: list[str]) -> list[str]:
        """
        Co-batch: group prompts by adapter, one forward per unique adapter,
        reassemble results in original order.
        """
        import torch
        if len(prompts) != len(adapters):
            raise ValueError("prompts and adapters must have same length")
        if not prompts:
            return []

        by_adapter: dict[str, list[int]] = defaultdict(list)
        for i, adapter in enumerate(adapters):
            by_adapter[adapter].append(i)

        results = [None] * len(prompts)
        for adapter, indices in by_adapter.items():
            if adapter != "default" and self._adapter_names:
                self.set_adapter(adapter)
            batch_prompts = [prompts[i] for i in indices]
            encoded = self._tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True,
            )
            input_ids = encoded["input_ids"].to(self._model.device)
            attention_mask = encoded["attention_mask"].to(self._model.device)
            with torch.no_grad():
                out = self._model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits
            for j, idx in enumerate(indices):
                last_pos = attention_mask[j].sum().item() - 1
                token_id = logits[j, last_pos, :].argmax().item()
                results[idx] = self._tokenizer.decode([token_id], skip_special_tokens=True)
        return results
