"""
Execute PEP: dispatch nodes respecting depends_on (order or parallel where allowed).
SPEC: Per-node execution; shared KV; rank-aware scheduling. Stack: vLLM/TGI, PEFT, Ray or async.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

from fracture import is_pep_dag
from scheduler import Scheduler


def run_node_with_model(model):
    """
    Return a run_node(node, context) callable for use with execute_pep.
    Per-node: set active adapter from node["adapter"], run decode, return output.
    SPEC: one adapter active per thread; no cross-thread adapter state.
    """
    from load_lora import set_active_adapter, run_forward_logits

    def run_node(node, context):
        set_active_adapter(model, node["adapter"])
        return run_forward_logits(model, node["prompt"])
    return run_node


def execute_pep(pep: dict, run_node=None, parallel=True, prefill_fn=None, slice_output=None, metrics=None, prefill_release_fn=None, adapter_router=None, scheduler=None, shared_kv_model=None, shared_kv_tokenizer=None):
    """
    Execute all nodes of a PEP. Respects depends_on: run a node only after its dependencies
    have completed. Independent nodes in the same wave run in parallel when parallel=True.
    When adapter_router is provided, each wave is split into microbatches via scheduler (Phase 8);
    each microbatch is one AdapterRouter.forward() call (Phase 6 co-batching). If scheduler is None, a default Scheduler() is used.
    When shared_kv_model and shared_kv_tokenizer are provided (Phase 7): prefill once with model(global_context_ids, use_cache=True),
    capture past_key_values; for each node decode with model(node_ids, past_key_values=shared_kv, use_cache=True). Do not recompute prefix per node.
    run_node(node, context, shared_prefill) -> output; context is dict of node_id -> output for depends_on.
    If prefill_fn is given, it is called once with pep["global_context"]; result passed as shared_prefill to run_node.
    If slice_output(raw) is given, stored results (and thus context passed to dependents) are slice_output(raw); no full CoT.
    If metrics is a dict, prefill_count is set when prefill_fn or real KV path is used (1 for shared prefix).
    If prefill_release_fn(handle) is given, called after all nodes complete with shared_prefill (KV GC).
    Returns dict node_id -> output (sliced when slice_output is provided).
    """
    if not is_pep_dag(pep):
        raise ValueError("PEP must be acyclic")
    nodes = pep["nodes"]
    node_by_id = {n["node_id"]: n for n in nodes}
    if run_node is None:
        def run_node(n, ctx, shared_prefill=None):
            return None

    # Phase 7: real prefix KV reuse
    if shared_kv_model is not None and shared_kv_tokenizer is not None:
        from kv_cache import prefill_global_context, decode_with_prefix
        from load_lora import set_active_adapter
        global_context = pep.get("global_context") or ""
        shared_kv = prefill_global_context(shared_kv_model, shared_kv_tokenizer, global_context)
        if metrics is not None:
            metrics["prefill_count"] = 1
        waves = _execution_waves(nodes)
        results = {}
        for wave in waves:
            wave = _sort_wave_by_cost(wave, node_by_id)
            for nid in wave:
                node = node_by_id[nid]
                dep_outputs = {d: results[d] for d in node.get("depends_on", []) if d in results}
                prompt = node["prompt"]
                if dep_outputs:
                    ctx_str = "\n".join(str(dep_outputs[d]) for d in node.get("depends_on", []) if d in dep_outputs)
                    prompt = f"{ctx_str}\n\n{prompt}"
                set_active_adapter(shared_kv_model, node["adapter"])
                raw = decode_with_prefix(shared_kv_model, shared_kv_tokenizer, prompt, shared_kv)
                results[nid] = slice_output(raw) if slice_output else raw
        if prefill_release_fn is not None and shared_kv is not None:
            prefill_release_fn(shared_kv)
        return results

    shared_prefill = prefill_fn(pep["global_context"]) if prefill_fn else None
    if metrics is not None and prefill_fn is not None:
        metrics["prefill_count"] = 1

    waves = _execution_waves(nodes)
    results = {}
    sched = scheduler if scheduler is not None else (Scheduler() if adapter_router is not None else None)
    for wave in waves:
        wave = _sort_wave_by_cost(wave, node_by_id)
        if adapter_router is not None:
            wave_nodes = [node_by_id[nid] for nid in wave]
            batches = sched.schedule(wave_nodes) if sched else [wave_nodes]
            for batch in batches:
                batch_inputs = []
                batch_nids = [n["node_id"] for n in batch]
                for node in batch:
                    dep_outputs = {d: results[d] for d in node.get("depends_on", []) if d in results}
                    prompt = node["prompt"]
                    if dep_outputs:
                        ctx_str = "\n".join(str(dep_outputs[d]) for d in node.get("depends_on", []) if d in dep_outputs)
                        prompt = f"{ctx_str}\n\n{prompt}"
                    batch_inputs.append(prompt)
                adapter_map = [n["adapter"] for n in batch]
                raw_outputs = adapter_router.forward(batch_inputs, adapter_map)
                for i, nid in enumerate(batch_nids):
                    raw = raw_outputs[i]
                    results[nid] = slice_output(raw) if slice_output else raw
        elif parallel and len(wave) > 1:
            with ThreadPoolExecutor(max_workers=len(wave)) as ex:
                futures = {
                    ex.submit(_run_one, node_by_id[nid], results, run_node, shared_prefill): nid
                    for nid in wave
                }
                for fut in as_completed(futures):
                    nid = futures[fut]
                    raw = fut.result()
                    results[nid] = slice_output(raw) if slice_output else raw
        else:
            for nid in wave:
                node = node_by_id[nid]
                dep_outputs = {d: results[d] for d in node.get("depends_on", []) if d in results}
                raw = _call_run_node(run_node, node, dep_outputs, shared_prefill)
                results[nid] = slice_output(raw) if slice_output else raw
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


def _node_cost(node: dict) -> float:
    """SPEC: cost = adapter_rank × token_budget; lower cost first."""
    rank = node.get("adapter_rank", 1)
    budget = node.get("token_budget", 1)
    return rank * budget


def _sort_wave_by_cost(wave: list[str], node_by_id: dict) -> list[str]:
    """Sort node ids in wave by ascending cost (adapter_rank × token_budget)."""
    return sorted(wave, key=lambda nid: _node_cost(node_by_id[nid]))


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
            in_degree[nid] = -1  # mark done
            for n in nodes:
                if nid in n.get("depends_on", []):
                    in_degree[n["node_id"]] -= 1
    return waves if sum(len(w) for w in waves) == len(nodes) else [[n["node_id"] for n in nodes]]


def _topological_order(nodes: list) -> list[str]:
    """Return node_ids in topological order (dependencies before dependents)."""
    node_by_id = {n["node_id"]: n for n in nodes}
    in_degree = {n["node_id"]: len([d for d in n.get("depends_on", []) if d in node_by_id]) for n in nodes}
    q = deque(nid for nid, deg in in_degree.items() if deg == 0)
    order = []
    while q:
        nid = q.popleft()
        order.append(nid)
        for n in nodes:
            if nid in n.get("depends_on", []):
                succ = n["node_id"]
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    q.append(succ)
    return order if len(order) == len(nodes) else [n["node_id"] for n in nodes]
