"""
Benchmark prefill: measure prefill count for shared-prefix run. SPEC: O(C²) + Σ O(C_i²) vs baseline O((kC)²).
"""

from fracture import fracture, validate_pep
from execute import execute_pep


def benchmark_prefill(num_nodes: int = 3):
    """
    Run execute_pep with shared prefix; return metrics (prefill_count, recorded).
    Results recorded for comparison vs sequential baseline (N prefills).
    """
    pep = fracture("What is A? What is B? What is C?")
    # Ensure we have at least num_nodes (e.g. 3)
    while len(pep["nodes"]) < num_nodes:
        pep["nodes"].append({
            "node_id": "n%i" % (len(pep["nodes"]) + 1),
            "type": "analysis",
            "adapter": "logic_lora",
            "prompt": "p",
            "depends_on": [],
        })
    validate_pep(pep)
    metrics = {}

    def prefill_fn(global_context):
        return {"block_id": 1}

    def run_node(node, context, shared_prefill=None):
        return node["node_id"]

    execute_pep(pep, run_node=run_node, prefill_fn=prefill_fn, metrics=metrics)
    result = dict(metrics)
    result["recorded"] = True
    return result
