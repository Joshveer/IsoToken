"""
Tests for tools.py: read_files, write_file, parse_code_block, build_file_prompt.
"""

import os
import tempfile

import pytest


def test_read_files_reads_real_files():
    from tools import read_files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("hello = 1\n")
        path = f.name
    try:
        result = read_files([path])
        assert path in result
        assert result[path] == "hello = 1\n"
    finally:
        os.unlink(path)


def test_read_files_glob_expands():
    from tools import read_files
    with tempfile.TemporaryDirectory() as d:
        for name in ("a.py", "b.py"):
            with open(os.path.join(d, name), "w") as f:
                f.write(f"# {name}\n")
        result = read_files([os.path.join(d, "*.py")])
        assert len(result) == 2


def test_read_files_missing_raises():
    from tools import read_files
    with pytest.raises(FileNotFoundError):
        read_files(["nonexistent_file_xyz_123.py"])


def test_write_file_creates_dirs_and_writes():
    from tools import write_file
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sub", "deep", "file.txt")
        write_file(path, "content")
        assert os.path.isfile(path)
        with open(path) as f:
            assert f.read() == "content"


def test_parse_code_block_extracts_fenced():
    from tools import parse_code_block
    response = 'Here is the code:\n```python\ndef foo():\n    pass\n```\nDone.'
    assert parse_code_block(response) == "def foo():\n    pass"


def test_parse_code_block_no_block_returns_stripped():
    from tools import parse_code_block
    assert parse_code_block("  just text  ") == "just text"


def test_parse_code_block_multiple_returns_first():
    from tools import parse_code_block
    response = '```\nfirst\n```\nand\n```\nsecond\n```'
    assert parse_code_block(response) == "first"


def test_build_file_prompt_contains_all_parts():
    from tools import build_file_prompt
    result = build_file_prompt("src/foo.py", "x = 1", "Add type hints")
    assert "Add type hints" in result
    assert "src/foo.py" in result
    assert "x = 1" in result
    assert "```" in result


def test_discover_files_finds_files_under_root():
    from tools import discover_files
    with tempfile.TemporaryDirectory() as d:
        for name in ("a.py", "b.md", "c.txt"):
            with open(os.path.join(d, name), "w") as f:
                f.write("x")
        paths = discover_files(root=d)
        assert len(paths) == 3
        assert any("a.py" in p for p in paths)
        assert any("b.md" in p for p in paths)
        assert any("c.txt" in p for p in paths)


def test_discover_files_skips_excluded_dirs():
    from tools import discover_files
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "top.py"), "w") as f:
            f.write("x")
        os.makedirs(os.path.join(d, ".git"))
        with open(os.path.join(d, ".git", "config.py"), "w") as f:
            f.write("x")
        paths = discover_files(root=d)
        assert len(paths) == 1
        assert "top.py" in paths[0]
