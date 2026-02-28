"""
Execute PEP: dispatch nodes respecting depends_on, parallel within waves via asyncio.
"""

import asyncio
import time

from fracture import is_pep_dag


def _result_to_context(val):
    """Extract string for downstream context."""
    if isinstance(val, dict) and "output" in val:
        return val["output"]
    return val if isinstance(val, str) else str(val)


def execute_pep(pep: dict, run_node=None, parallel=True, metrics=None, local_backend=None):
    """Sync wrapper: run execute_pep_async via asyncio.run."""
    return asyncio.run(execute_pep_async(
        pep, run_node=run_node, parallel=parallel, metrics=metrics,
        local_backend=local_backend,
    ))


async def execute_pep_async(pep: dict, run_node=None, parallel=True, metrics=None, local_backend=None):
    """
    Execute all nodes of a PEP (async). Respects depends_on; independent nodes
    in the same wave run in parallel via asyncio.gather.
    All backends (API and local) use run_node; local run_node uses _generate()
    for full multi-token responses.
    """
    if not is_pep_dag(pep):
        raise ValueError("PEP must be acyclic")
    nodes = pep["nodes"]
    node_by_id = {n["node_id"]: n for n in nodes}
    if run_node is None:
        def run_node(n, ctx, shared_prefill=None):
            return None

    # Use run_node for all backends (API and local). Local run_node uses _generate()
    # for full multi-token responses; decode_with_kv/forward_batch only yield one token.
    waves = _execution_waves(nodes)
    results = {}
    for wave in waves:
        if parallel and len(wave) > 1:
            outputs = await asyncio.gather(*[
                asyncio.to_thread(_run_one, node_by_id[nid], results, run_node)
                for nid in wave
            ])
            for nid, raw in zip(wave, outputs):
                results[nid] = raw
        else:
            for nid in wave:
                node = node_by_id[nid]
                dep_outputs = {d: results[d] for d in node.get("depends_on", []) if d in results}
                raw = _call_run_node(run_node, node, dep_outputs)
                results[nid] = raw
    return results


def _call_run_node(run_node, node, dep_outputs):
    """Call run_node with 2 or 3 args for backward compatibility."""
    try:
        return run_node(node, dep_outputs, None)
    except TypeError:
        return run_node(node, dep_outputs)


def _run_one(node, results, run_node):
    """Run a single node (for parallel submission)."""
    dep_outputs = {d: results[d] for d in node.get("depends_on", []) if d in results}
    return _call_run_node(run_node, node, dep_outputs)


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
