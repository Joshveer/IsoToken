"""
File tools for IsoToken agents. Read files, write files, parse code blocks from LLM responses.
"""

import glob
import os
import re

# Directories to skip when discovering "all files"
DEFAULT_EXCLUDE_DIRS = frozenset({
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".tox",
    "dist", "build", ".eggs", "site-packages", ".mypy_cache", ".ruff_cache",
})
# Extensions to include when discovering (None = all text-like files we can read)
DEFAULT_DISCOVER_EXTENSIONS = (
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".sh", ".bash", ".zsh", ".js", ".ts", ".jsx", ".tsx", ".html", ".css",
    ".c", ".h", ".cpp", ".hpp", ".rs", ".go", ".rb", ".java", ".kt",
)


def discover_files(
    root: str | None = None,
    extensions: tuple[str, ...] | None = DEFAULT_DISCOVER_EXTENSIONS,
    exclude_dirs: frozenset[str] | None = DEFAULT_EXCLUDE_DIRS,
    max_files: int = 200,
) -> list[str]:
    """
    Discover files under root (default: cwd). Returns list of paths.
    Skips exclude_dirs, optionally filters by extension, caps at max_files.
    """
    root = os.path.abspath(root or os.getcwd())
    if not os.path.isdir(root):
        return []
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in (exclude_dirs or set())]
        for name in filenames:
            if len(out) >= max_files:
                return out
            if extensions and not any(name.endswith(ext) for ext in extensions):
                continue
            path = os.path.join(dirpath, name)
            if os.path.isfile(path):
                out.append(path)
    return out


def read_files(paths: list[str]) -> dict[str, str]:
    """
    Read files from a list of paths. Supports glob patterns (e.g. "src/*.py").
    Returns dict of resolved_path -> file content.
    Raises FileNotFoundError for paths that match nothing.
    """
    result: dict[str, str] = {}
    for pattern in paths:
        expanded = glob.glob(pattern, recursive=True)
        if not expanded:
            if os.path.isfile(pattern):
                expanded = [pattern]
            else:
                raise FileNotFoundError(f"No files matched: {pattern}")
        for fpath in expanded:
            if os.path.isfile(fpath):
                with open(fpath, encoding="utf-8") as f:
                    result[fpath] = f.read()
    return result


def write_file(path: str, content: str) -> None:
    """Write content to path, creating parent directories if needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


_CODE_BLOCK_RE = re.compile(r"```(?:\w*)\n(.*?)```", re.DOTALL)


def parse_code_block(response: str) -> str:
    """
    Extract the first fenced code block from an LLM response.
    Returns the code inside the block (without the ``` markers).
    If no code block found, returns the full response stripped.
    """
    match = _CODE_BLOCK_RE.search(response)
    if match:
        return match.group(1).strip()
    return response.strip()


def build_file_prompt(file_path: str, content: str, task: str) -> str:
    """
    Format a prompt for an LLM agent that should modify a file.
    Includes the task, file path, and file content.
    """
    return (
        f"{task}\n\n"
        f"File: {file_path}\n\n"
        f"```\n{content}\n```\n\n"
        f"Return the complete modified file inside a single fenced code block."
    )
