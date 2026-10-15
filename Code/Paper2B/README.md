# Revision analysis

From the repository root, install `Code/Paper2B/requirements_analysis.txt`, then run `python Code/Paper2B/analyse_revision.py` and `python Code/Paper2B/make_flow.py`. Paths are resolved by `revision_paths.py`; outputs are in `results/paper2b_revision_v1`.

The analysis uses the original result CSV and manifest directly. Cached predictions are checked against reconstructed record IDs, order, labels and F1. Dataset-held-out folds and cluster resamples retain both members of each pair. Existing fold timings are retained when cached predictions are reused.

The main workspace keeps manuscript/response builders under `Code/Paper2B`. They require private journal email and LaTeX sources and are intentionally not distributed as public analysis code.
