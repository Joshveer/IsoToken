"""
Tests for engine.py with local backend: initialization, run, distill.
"""

import json
import os
import tempfile

MODEL_ID = "sshleifer/tiny-gpt2"


def test_engine_local_initializes():
    from engine import IsoTokenEngine
    engine = IsoTokenEngine(llm_backend={"backend": "local", "model_id": MODEL_ID})
    assert engine._is_local
    assert engine._local_backend is not None


def test_engine_local_run_returns_answer():
    from engine import IsoTokenEngine
    engine = IsoTokenEngine(llm_backend={"backend": "local", "model_id": MODEL_ID})
    result = engine.run("Hello world")
    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert "metrics" in result
    assert result["metrics"]["backend"] == "local"
    assert result["metrics"]["prefill_count"] == 1


def test_engine_local_run_with_files():
    from engine import IsoTokenEngine
    engine = IsoTokenEngine(llm_backend={"backend": "local", "model_id": MODEL_ID})
    with tempfile.TemporaryDirectory() as d:
        fpath = os.path.join(d, "test.py")
        with open(fpath, "w") as f:
            f.write("x = 1\n")
        result = engine.run("Modify the file", files=[fpath])
        assert "metrics" in result
        assert result["metrics"]["num_agents"] == 1


def test_engine_local_distill():
    from engine import IsoTokenEngine
    engine = IsoTokenEngine(llm_backend={"backend": "local", "model_id": MODEL_ID})
    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "runs.jsonl")
        with open(log_path, "w") as f:
            f.write('{"input": "q", "target": "a"}\n')
        out_dir = os.path.join(tmp, "student")
        engine.distill(log_path, out_dir, max_steps=1)
        assert os.path.isdir(out_dir)
        assert len(os.listdir(out_dir)) > 0
