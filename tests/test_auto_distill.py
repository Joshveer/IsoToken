"""
Tests for auto-distillation: distillation_log_path on engine, auto_distill_threshold.
"""

import json
import os
import tempfile
from unittest.mock import patch


def _mock_make_run_node(**kwargs):
    def run_node(node, context, shared_prefill=None):
        return "mock answer"
    return run_node


def test_engine_logs_runs_when_distill_log_set():
    from engine import IsoTokenEngine
    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "runs.jsonl")
        with patch("engine.make_run_node", _mock_make_run_node):
            engine = IsoTokenEngine(
                llm_backend={"backend": "openai", "api_key": "fake"},
                distillation_log_path=log_path,
            )
            engine.run("Hello?")
            engine.run("World?")
        assert os.path.isfile(log_path)
        with open(log_path) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 2
        r0 = json.loads(lines[0])
        assert r0["input"] == "Hello?"
        assert "target" in r0


def test_engine_no_log_when_distill_log_not_set():
    from engine import IsoTokenEngine
    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "runs.jsonl")
        with patch("engine.make_run_node", _mock_make_run_node):
            engine = IsoTokenEngine(llm_backend={"backend": "openai", "api_key": "fake"})
            engine.run("Hello?")
        assert not os.path.exists(log_path)


def test_auto_distill_triggers_at_threshold():
    from engine import IsoTokenEngine
    distill_calls = []
    original_distill = IsoTokenEngine.distill

    def mock_distill(self, log_path, output_dir, max_steps=100):
        distill_calls.append((log_path, output_dir))

    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "runs.jsonl")
        with patch("engine.make_run_node", _mock_make_run_node), \
             patch.object(IsoTokenEngine, "distill", mock_distill):
            engine = IsoTokenEngine(
                llm_backend={"backend": "openai", "api_key": "fake"},
                distillation_log_path=log_path,
                auto_distill_threshold=3,
                auto_distill_output=os.path.join(tmp, "student"),
            )
            engine.run("q1?")
            engine.run("q2?")
            assert len(distill_calls) == 0
            engine.run("q3?")
            assert len(distill_calls) == 1
            assert distill_calls[0][0] == log_path


def test_auto_distill_triggers_again_at_multiples():
    from engine import IsoTokenEngine
    distill_calls = []

    def mock_distill(self, log_path, output_dir, max_steps=100):
        distill_calls.append(1)

    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "runs.jsonl")
        with patch("engine.make_run_node", _mock_make_run_node), \
             patch.object(IsoTokenEngine, "distill", mock_distill):
            engine = IsoTokenEngine(
                llm_backend={"backend": "openai", "api_key": "fake"},
                distillation_log_path=log_path,
                auto_distill_threshold=2,
            )
            for i in range(6):
                engine.run(f"q{i}?")
            assert len(distill_calls) == 3
