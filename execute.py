"""
Execute PEP: dispatch nodes respecting depends_on, parallel within waves via asyncio.
"""

import asyncio
import time
from collections import deque

from fracture import is_pep_dag
from logger import log_interaction


def _result_to_context(val):
    """Extract string for downstream context; support dict {output, logprobs}."""
    if isinstance(val, dict) and "output" in val:
        return val["output"]
    return val if isinstance(val, str) else str(val)


def _confidence_from_output(output) -> float | None:
    """Mean logprob if output is dict with logprobs; else None."""
    if isinstance(output, dict) and output.get("logprobs"):
        lp = output["logprobs"]
        return sum(lp) / len(lp) if lp else None
    return None


def execute_pep(pep: dict, run_node=None, parallel=True, prefill_fn=None, slice_output=None, metrics=None, prefill_release_fn=None, interaction_log_path=None, local_backend=None):
    """Sync wrapper: run execute_pep_async via asyncio.run."""
    return asyncio.run(execute_pep_async(
        pep, run_node=run_node, parallel=parallel, prefill_fn=prefill_fn,
        slice_output=slice_output, metrics=metrics, prefill_release_fn=prefill_release_fn,
        interaction_log_path=interaction_log_path, local_backend=local_backend,
    ))


async def execute_pep_async(pep: dict, run_node=None, parallel=True, prefill_fn=None, slice_output=None, metrics=None, prefill_release_fn=None, interaction_log_path=None, local_backend=None):
    """
    Execute all nodes of a PEP (async). Respects depends_on; independent nodes
    in the same wave run in parallel via asyncio.gather.

    When local_backend is provided (LocalBackend instance), uses optimized paths:
    - Shared KV prefix: prefill once, decode per node with shared past_key_values
    - Co-batching: forward_batch for waves with multiple nodes
    Otherwise uses the standard parallel asyncio.gather path (API backends).
    """
    if not is_pep_dag(pep):
        raise ValueError("PEP must be acyclic")
    nodes = pep["nodes"]
    node_by_id = {n["node_id"]: n for n in nodes}
    if run_node is None:
        def run_node(n, ctx, shared_prefill=None):
            return None

    # Local backend: shared KV prefix path
    if local_backend is not None and hasattr(local_backend, "prefill_shared_kv"):
        global_context = pep.get("global_context") or ""
        shared_kv = local_backend.prefill_shared_kv(global_context)
        if metrics is not None:
            metrics["prefill_count"] = 1
        waves = _execution_waves(nodes)
        results = {}
        for wave in waves:
            if len(wave) > 1 and hasattr(local_backend, "forward_batch"):
                # Co-batch: all nodes in wave via one forward per adapter
                prompts = []
                adapters = []
                for nid in wave:
                    node = node_by_id[nid]
                    dep_outputs = {d: results[d] for d in node.get("depends_on", []) if d in results}
                    prompt = node["prompt"]
                    if dep_outputs:
                        ctx_str = "\n".join(_result_to_context(dep_outputs[d]) for d in node.get("depends_on", []) if d in dep_outputs)
                        prompt = f"{ctx_str}\n\n{prompt}"
                    prompts.append(prompt)
                    adapters.append(node.get("adapter", "default"))
                batch_out = await asyncio.to_thread(local_backend.forward_batch, prompts, adapters)
                for nid, raw in zip(wave, batch_out):
                    results[nid] = slice_output(raw) if slice_output else raw
                    if interaction_log_path:
                        node = node_by_id[nid]
                        log_interaction(nid, node.get("adapter", "default"), prompts[wave.index(nid)], results[nid], node.get("depends_on", []), None, None, path=interaction_log_path)
            else:
                for nid in wave:
                    node = node_by_id[nid]
                    dep_outputs = {d: results[d] for d in node.get("depends_on", []) if d in results}
                    prompt = node["prompt"]
                    if dep_outputs:
                        ctx_str = "\n".join(_result_to_context(dep_outputs[d]) for d in node.get("depends_on", []) if d in dep_outputs)
                        prompt = f"{ctx_str}\n\n{prompt}"
                    if node.get("adapter", "default") != "default" and hasattr(local_backend, "set_adapter"):
                        local_backend.set_adapter(node["adapter"])
                    t0 = time.perf_counter()
                    raw = local_backend.decode_with_kv(prompt, shared_kv)
                    elapsed = time.perf_counter() - t0
                    results[nid] = slice_output(raw) if slice_output else raw
                    if interaction_log_path:
                        log_interaction(nid, node.get("adapter", "default"), prompt, results[nid], node.get("depends_on", []), elapsed, None, path=interaction_log_path)
        if prefill_release_fn is not None:
            prefill_release_fn(shared_kv)
        return results

    # Standard path: prefill_fn (legacy) or parallel asyncio.gather (API backends)
    shared_prefill = prefill_fn(pep["global_context"]) if prefill_fn else None
    if metrics is not None and prefill_fn is not None:
        metrics["prefill_count"] = 1

    waves = _execution_waves(nodes)
    results = {}
    for wave in waves:
        if parallel and len(wave) > 1:
            outputs = await asyncio.gather(*[
                asyncio.to_thread(_run_one, node_by_id[nid], results, run_node, shared_prefill)
                for nid in wave
            ])
            for nid, raw in zip(wave, outputs):
                results[nid] = slice_output(raw) if slice_output else raw
            if interaction_log_path:
                for nid in wave:
                    node = node_by_id[nid]
                    dep_outputs = {d: results[d] for d in node.get("depends_on", []) if d in results}
                    prompt = node["prompt"]
                    if dep_outputs:
                        ctx_str = "\n".join(_result_to_context(dep_outputs[d]) for d in node.get("depends_on", []) if d in dep_outputs)
                        prompt = f"{ctx_str}\n\n{prompt}"
                    log_interaction(nid, node.get("adapter", "default"), prompt, results[nid], node.get("depends_on", []), None, _confidence_from_output(results[nid]), path=interaction_log_path)
        else:
            for nid in wave:
                node = node_by_id[nid]
                dep_outputs = {d: results[d] for d in node.get("depends_on", []) if d in results}
                prompt = node["prompt"]
                if dep_outputs:
                    ctx_str = "\n".join(_result_to_context(dep_outputs[d]) for d in node.get("depends_on", []) if d in dep_outputs)
                    prompt = f"{ctx_str}\n\n{prompt}"
                t0 = time.perf_counter()
                raw = _call_run_node(run_node, node, dep_outputs, shared_prefill)
                elapsed = time.perf_counter() - t0
                results[nid] = slice_output(raw) if slice_output else raw
                if interaction_log_path:
                    log_interaction(nid, node.get("adapter", "default"), prompt, results[nid], node.get("depends_on", []), elapsed, _confidence_from_output(results[nid]), path=interaction_log_path)
    if prefill_release_fn is not None and shared_prefill is not None:
        prefill_release_fn(shared_prefill)
    return results


def _call_run_node(run_node, node, dep_outputs, shared_prefill):
    """Call run_node with 2 or 3 args for backward compatibility."""
    try:
        return run_node(node, dep_outputs, shared_prefill)
    except TypeError:
        return run_node(node, dep_outputs)


def _run_one(node, results, run_node, shared_prefill=None):
    """Run a single node (for parallel submission)."""
    dep_outputs = {d: results[d] for d in node.get("depends_on", []) if d in results}
    return _call_run_node(run_node, node, dep_outputs, shared_prefill)


def _execution_waves(nodes: list) -> list[list[str]]:
    """Partition nodes into waves: wave i can run when all waves 0..i-1 are done."""
    node_by_id = {n["node_id"]: n for n in nodes}
    in_degree = {n["node_id"]: len([d for d in n.get("depends_on", []) if d in node_by_id]) for n in nodes}
    waves = []
    while True:
        ready = [nid for nid, deg in in_degree.items() if deg == 0]
        if not ready:
            break
        waves.append(ready)
        for nid in ready:
            in_degree[nid] = -1
            for n in nodes:
                if nid in n.get("depends_on", []):
                    in_degree[n["node_id"]] -= 1
    return waves if sum(len(w) for w in waves) == len(nodes) else [[n["node_id"] for n in nodes]]
