"""
Phase 11: Full interaction graph logging. Tests: log file created; one entry per node; fields correct.
Logging does not alter execution results.
"""

import json
import os
import tempfile


def test_log_file_created():
    """When interaction_log_path is set, log file is created."""
    from execute import execute_pep

    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "logs", "interaction_graph.jsonl")
        pep = {
            "nodes": [
                {"node_id": "n1", "adapter": "a", "prompt": "p1", "depends_on": []},
            ],
            "global_context": "",
        }
        def run_node(node, context, shared_prefill=None):
            return "out1"
        execute_pep(pep, run_node=run_node, parallel=False, interaction_log_path=log_path)
        assert os.path.isfile(log_path), "Log file must be created"


def test_one_entry_per_node():
    """One JSONL entry per node execution."""
    from execute import execute_pep

    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "graph.jsonl")
        pep = {
            "nodes": [
                {"node_id": "n1", "adapter": "a", "prompt": "p1", "depends_on": []},
                {"node_id": "n2", "adapter": "a", "prompt": "p2", "depends_on": []},
            ],
            "global_context": "",
        }
        def run_node(node, context, shared_prefill=None):
            return f"out_{node['node_id']}"
        execute_pep(pep, run_node=run_node, parallel=False, interaction_log_path=log_path)
        with open(log_path, encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 2, "Must have one entry per node (2 nodes)"


def test_fields_correct():
    """Each log entry has node_id, adapter, input, output, deps, latency, confidence."""
    from execute import execute_pep

    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "graph.jsonl")
        pep = {
            "nodes": [
                {"node_id": "n1", "adapter": "adapter_x", "prompt": "hello", "depends_on": []},
            ],
            "global_context": "",
        }
        execute_pep(pep, run_node=lambda n, c, s=None: "world", parallel=False, interaction_log_path=log_path)
        with open(log_path, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert record["node_id"] == "n1"
        assert record["adapter"] == "adapter_x"
        assert record["input"] == "hello"
        assert record["output"] == "world"
        assert record["deps"] == []
        assert "latency" in record
        assert "confidence" in record


def test_logging_does_not_alter_execution_results():
    """Results with and without interaction_log_path are identical."""
    from execute import execute_pep

    pep = {
        "nodes": [
            {"node_id": "n1", "adapter": "a", "prompt": "p1", "depends_on": []},
            {"node_id": "n2", "adapter": "a", "prompt": "p2", "depends_on": ["n1"]},
        ],
        "global_context": "",
    }
    def run_node(node, context, shared_prefill=None):
        return node["node_id"] + "_result"
    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "graph.jsonl")
        results_no_log = execute_pep(pep, run_node=run_node, parallel=False)
        results_with_log = execute_pep(pep, run_node=run_node, parallel=False, interaction_log_path=log_path)
    assert results_no_log == results_with_log, "Logging must not alter execution results"
