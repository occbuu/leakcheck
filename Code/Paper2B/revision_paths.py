"""Paths shared by the author workspace and the standalone leakcheck repository."""
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
RESULTS = ROOT / 'results' / 'paper2b_revision_v1'
PACKAGE = ROOT if (ROOT / 'src' / 'leakcheck').is_dir() else ROOT / 'Paper2B' / 'leakcheck'
MANUSCRIPT = ROOT / 'Paper2B' / 'SX' / 'Paper2B_leakcheck'
SUBMISSION = MANUSCRIPT / 'revision_v1'
