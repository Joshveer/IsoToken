"""IsoToken CLI entry point. Run with: isotoken or python -m isotoken"""

import sys
from pathlib import Path

# Ensure project root is on path so "cli" is importable when running installed script
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
