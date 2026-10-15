"""Paths within the leakcheck repository. Scripts may be launched from any working directory."""
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
RESULTS = ROOT / "results" / "revision"
PACKAGE = ROOT
