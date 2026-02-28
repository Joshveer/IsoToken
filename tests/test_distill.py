"""
Tests for distill.py: collect_data and train_student.
"""

import json
import os
import tempfile


def test_collect_data_reads_jsonl():
    from distill import collect_data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"input": "q1", "target": "a1"}\n')
        f.write('{"input": "q2", "target": "a2"}\n')
        f.write('{"input": "q3", "target": "a3"}\n')
        path = f.name
    try:
        ds = collect_data(path)
        assert len(ds) == 3
        assert "input" in ds.column_names
        assert "target" in ds.column_names
        assert ds[0]["input"] == "q1"
        assert ds[2]["target"] == "a3"
    finally:
        os.unlink(path)


def test_collect_data_normalizes_keys():
    from distill import collect_data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"prompt": "hello", "answer": "world"}\n')
        path = f.name
    try:
        ds = collect_data(path)
        assert ds[0]["input"] == "hello"
        assert ds[0]["target"] == "world"
    finally:
        os.unlink(path)


def test_train_student_saves_adapter():
    from distill import train_student
    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "runs.jsonl")
        with open(log_path, "w") as f:
            f.write('{"input": "q", "target": "a"}\n')
        out_dir = os.path.join(tmp, "adapter")
        train_student("sshleifer/tiny-gpt2", log_path, out_dir, max_steps=1)
        assert os.path.isdir(out_dir)
        files = os.listdir(out_dir)
        assert len(files) > 0
