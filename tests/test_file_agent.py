"""
Tests for file-aware agent execution: fracture with files, aggregate pass-through, engine file write.
"""

import os
import tempfile


def test_fracture_with_files_creates_one_node_per_file():
    from fracture import fracture
    files = {"src/a.py": "x = 1", "src/b.py": "y = 2", "src/c.py": "z = 3"}
    pep = fracture("Add type hints", files=files)
    assert len(pep["nodes"]) == 3
    for node in pep["nodes"]:
        assert "file_path" in node
        assert node["type"] == "file_edit"
        assert "```" in node["prompt"]


def test_aggregate_file_nodes_returns_outputs_dict():
    from aggregate import aggregate_by_pep
    pep = {
        "task_id": "t1",
        "global_context": "",
        "nodes": [
            {"node_id": "n1", "type": "file_edit", "adapter": "default", "prompt": "...", "depends_on": [], "file_path": "a.py"},
            {"node_id": "n2", "type": "file_edit", "adapter": "default", "prompt": "...", "depends_on": [], "file_path": "b.py"},
        ],
        "aggregation": {"strategy": "vote"},
    }
    outputs = {"n1": "new a content", "n2": "new b content"}
    result = aggregate_by_pep(pep, outputs)
    assert isinstance(result, dict)
    assert result["n1"] == "new a content"
    assert result["n2"] == "new b content"


def test_engine_run_writes_files_with_mock_backend():
    from engine import IsoTokenEngine
    from unittest.mock import patch

    def mock_make_run_node(**kwargs):
        def run_node(node, context, shared_prefill=None):
            return "```python\n# modified\n```"
        return run_node

    with tempfile.TemporaryDirectory() as d:
        fpath = os.path.join(d, "test.py")
        with open(fpath, "w") as f:
            f.write("# original\n")

        with patch("engine.make_run_node", mock_make_run_node):
            engine = IsoTokenEngine(llm_backend={"backend": "openai", "api_key": "fake"})
            result = engine.run("Refactor", files=[fpath])

        assert fpath in result["files_changed"]
        with open(fpath) as f:
            assert "# modified" in f.read()
        assert "metrics" in result


def test_engine_run_text_prompt_returns_answer():
    from engine import IsoTokenEngine
    from unittest.mock import patch

    def mock_make_run_node(**kwargs):
        def run_node(node, context, shared_prefill=None):
            return "The answer is 42"
        return run_node

    with patch("engine.make_run_node", mock_make_run_node):
        engine = IsoTokenEngine(llm_backend={"backend": "openai", "api_key": "fake"})
        result = engine.run("What is the meaning of life?")

    assert result["answer"] == "The answer is 42"
    assert result["files_changed"] == []
    assert "metrics" in result
