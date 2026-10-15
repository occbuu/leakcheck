# Training / export

This folder rebuilds the screening model that ships inside the `leakcheck` package. A user who only wants to *screen* a reported result does not need these scripts; `pip install -e .` is enough.

The training stage fits a gradient-boosted classifier on the paired corpus (`results/raw_results.csv`): each comparison contributes one resample-then-split record and one train-only record, matched on everything a paper would report except the resulting score. Validation is leave-one-dataset-out. The fitted trees are then exported as JSON so inference depends only on NumPy.

```bash
pip install -r requirements.txt
python 04_export_model.py
```

That writes `src/leakcheck/model.json` at the repository root. Regression fixtures stored in the JSON are what `leakcheck.selftest()` checks.

Shared experimental machinery lives in `leakage_lib.py` (the three orderings: leaked / correct / none). `analysis.ipynb` reproduces the diagnostic plots. Dataset inventory: `../data/processed/manifest.csv`.

Runtime versions used for the reported run are recorded in `../results/summary.json`.

## October revision

The current revision is reproduced by `Paper2B/analyse_revision.py` and `Paper2B/make_flow.py`; see `Paper2B/README.md`. Outputs are written to `../results/paper2b_revision_v1`. The export script above refits the original model and is not required to reproduce the revision analyses.
