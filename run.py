"""
End-to-end: prompt → fracture → execute_pep → aggregate → final answer.
Output format: {"answer": str}.
"""

from fracture import fracture, validate_pep
from execute import execute_pep
from aggregate import aggregate_by_pep


def run(prompt: str, run_node=None):
    """
    Full pipeline: fracture(prompt) → execute_pep → aggregate_by_pep.
    Returns {"answer": str}. run_node(node, context) optional; default returns placeholder.
    """
    pep = fracture(prompt)
    validate_pep(pep)
    if run_node is None:
        def run_node(n, ctx):
            return "output"
    results = execute_pep(pep, run_node=run_node, parallel=False)
    final = aggregate_by_pep(pep, results)
    return {"answer": final}
