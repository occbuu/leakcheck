# Revision analysis

From the repository root:

```text
python -m pip install -r Code/revision/requirements_analysis.txt
python Code/revision/analyse_revision.py
python Code/revision/make_flow.py
python Code/revision/make_interface_assets.py
```

Paths are resolved by `revision_paths.py`. Inputs are `results/raw_results.csv` and `data/processed/manifest.csv`. Outputs are written to `results/revision/`, including `summary.json`.

Cached predictions are reused only when their record identities, order, labels and F1 values match the reconstructed corpus. To refit every held-out fold, move `results/revision/oof_predictions.csv` aside first.

`make_interface_assets.py` draws the reader workflow and the F1-by-imbalance grid from the shipped JSON model. It does not fit a model. The density figure already stored at `results/revision/f1_density.png` is left unchanged.
