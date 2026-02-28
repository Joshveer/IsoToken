"""
Rank-aware microbatch scheduler. Phase 8: group similar-cost nodes, avoid mixing
high-rank with low-rank in same microbatch to optimize heterogeneous LoRA workloads.
"""

from collections import defaultdict


def _node_cost(node: dict) -> float:
    """Cost = adapter_rank × token_budget. Lower cost first."""
    rank = node.get("adapter_rank", 1)
    budget = node.get("token_budget", 1)
    return rank * budget


class Scheduler:
    """
    Schedules nodes into microbatches. Heuristic: cost = adapter_rank × token_budget;
    group similar-cost nodes; avoid mixing high-rank with low-rank in same microbatch.
    Returns microbatches in deterministic order (ascending rank, then by node_id within batch).
    """

    def schedule(self, nodes: list) -> list:
        """
        Partition nodes into microbatches. Nodes with same adapter_rank are grouped
        together; high-rank nodes are isolated from low-rank. Returns List[List[node]].
        """
        if not nodes:
            return []
        # Group by adapter_rank (so similar cost / same rank stay together; high rank isolated)
        by_rank = defaultdict(list)
        for node in nodes:
            rank = node.get("adapter_rank", 1)
            by_rank[rank].append(node)
        # Order batches by ascending rank (low cost first); within each batch sort by node_id for determinism
        batches = []
        for rank in sorted(by_rank.keys()):
            batch = sorted(by_rank[rank], key=lambda n: n.get("node_id", ""))
            batches.append(batch)
        return batches
