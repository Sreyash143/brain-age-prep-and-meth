"""pytest setup.  Run from the project root (the folder holding config_v3.py / core_v3.py):

    python -m pytest v5_pkg_tests -q          # or:  python -m pytest tests -q
Set V4_PROJECT_DIR to point at the project root if you run from elsewhere.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOTS = [Path(os.environ["V4_PROJECT_DIR"])] if os.environ.get("V4_PROJECT_DIR") else []
ROOTS += [Path.cwd(), HERE.parent, HERE.parent.parent]
for r in ROOTS:
    if (r / "core_v3.py").exists() and str(r) not in sys.path:
        sys.path.insert(0, str(r))
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
