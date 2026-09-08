# v1.0.0 — Initial public release

Replication archive for *Leakage makes every problem look equally easy: a multi-dataset measurement of the resampling-before-splitting artefact*.

This release is the version intended for archival (Zenodo DOI) and for reviewers who need to regenerate every number in the paper from the experimental record.

## What's included

- **Corpus.** 41 harmonised binary classification problems (`data/processed/*.npz`, `manifest.csv`): 35 PMLB-derived benchmarks and 6 employee-turnover datasets. Raw third-party dumps are not redistributed; `Code/01_fetch_datasets.py` rebuilds the PMLB half over HTTPS.
- **Experiment.** Full factorial of three evaluation protocols (`leaked` / `correct` / `none`) × 7 classifiers × 4 over-samplers × 10 seeds (`results/raw_results.csv`, 25,620 fitted models). Controlled simulation and partial-oversampling (mitigation) sweeps are included.
- **Analysis.** `Code/03_analysis.py` rebuilds `results/summary.json`, the manuscript tables, and `figures/fig01`–`fig05`. `Code/05_verify_numbers.py` recomputes headline quantities from `raw_results.csv`.
- **Notebook.** `Code/analysis.ipynb` walks through the invariance result interactively.

## Headline numbers in this archive

| Quantity | Value |
|---|---|
| Datasets | 41 |
| Mean leakage premium (minority-class F1) | 0.224 (95% CI 0.220–0.228) |
| Variance ratio (correct / leaked F1) | 4.86 |
| Share of pairs with a positive premium | 92.2% |
| Mean contamination rate | 0.64 |

## How to verify without re-running the factorial

```bash
cd Code
pip install -r requirements.txt
python 03_analysis.py
python 05_verify_numbers.py
```

Re-running the factorial from scratch takes several hours (`run_experiment.py`, two shards, 10 seeds) and is optional.

## Requirements

Python 3.9+; see `Code/requirements.txt`. Versions used for the reported run are recorded in `results/summary.json` under `versions`.

## License

BSD 3-Clause.
