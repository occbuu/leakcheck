# leakcheck

Screen reported binary classification results for the controlled resampling-before-splitting signature.

The output is a **screening score**, not a calibrated probability that a published study contains leakage. A high score recommends checking the methods; a low score does not establish a correct pipeline. The software contribution is an inspectable summary-screening workflow rather than a new learning algorithm or demonstrated improvement in peer review.

## Current revision

Local code version **1.0.1** uses the unchanged v1.0.0 JSON weights, neutral review bands and strict supported-family handling. No new public release, verified archive or DOI is claimed for this revision. The identifier 10.5281/zenodo.22196497 belongs to the original manuscript's archive metadata; its contents were not verified during this revision.

Project URL: https://github.com/occbuu/leakcheck

## Install and test

From the repository root:

```text
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

For inference, NumPy is the sole required dependency. pandas is optional for table output. Declared minimums are Python 3.9 and NumPy 1.21; the revision was tested on Python 3.12.4 and NumPy 2.4.6. These declarations are not a tested minimum-version matrix.

## Use

```python
import leakcheck
r = leakcheck.check(f1=0.92, imbalance_ratio=5.2,
                    n_minority=237, n_features=44, clf="RandomForest")
print(r.score, r.verdict)
print(r.explain())
assert leakcheck.selftest()
```

```text
leakcheck --f1 0.92 --ir 5.2 --n-minority 237 --n-features 44 --clf RandomForest
leakcheck --file results.csv
```

Input CSV requires `f1`, `imbalance_ratio`, `n_minority`, `n_features` and a supported `clf` on every row. The legacy `Result.probability` property is a compatibility alias for `score`; visible text and table/JSON exports use score terminology.

| Score | Review band |
| --- | --- |
| Below 0.35 | Low screening score |
| 0.35 to below 0.65 | Intermediate screening score |
| At least 0.65 | High screening score - methods check recommended |

These are operational conventions, not validated diagnostic cut-offs or cost optima.

## Input conventions

Supply binary minority-class F1 on [0,1], majority/minority ratio, minority count before resampling, and predictor count after encoding. Macro, weighted, micro and majority-class F1 are not substitutes. Supported families are 5NN, DecTree, GradBoost, LogReg, RandForest, SVM-RBF and XGBoost, with recognised spellings. Missing or unsupported families raise an explicit error; neural transformers, LightGBM and AdaBoost are not silently mapped to random forests or other families.

Nonfinite inputs, invalid ratios and noninteger counts are rejected. Inputs outside the fitted ratio/count/feature ranges produce extrapolation notes; a within-range input is not a validation guarantee.

## Reproduce the revision

```text
python -m pip install -r Code/Paper2B/requirements_analysis.txt
python Code/Paper2B/analyse_revision.py
python Code/Paper2B/make_flow.py
```

The scripts locate files relative to the repository, independent of the current working directory. Existing complete-pair records are read from `results/raw_results.csv` and `data/processed/manifest.csv`. No duplicated corpus inputs or separate software snapshot is required.

Revision outputs are in `results/paper2b_revision_v1/`: out-of-fold predictions, baselines, per-dataset and per-resampler metrics, reliability, grouped bootstrap, thresholds, prevalence scenarios, sensitivity, errors, three vector figures and measured environment/provenance.

The analysis reuses the cached prediction file only if its record identities, order, labels and F1 values match the reconstructed source corpus. To refit all folds, move `oof_predictions.csv` aside first. The analysis retains paired records within each source-held-out fold and within 1,000 dataset-cluster bootstrap samples. Bootstrap intervals condition on fixed held-out predictions and exclude model-refitting and independent-source uncertainty.

## Evidence and limits

The controlled corpus contains 11,200 matched comparisons (22,400 labelled records) from 41 datasets. Full-model source-held-out AUC is 0.7951, with a cluster interval of 0.7427-0.8488; AP is 0.7764 and Brier loss 0.1974. F1-only AUC is 0.7939 and logistic AUC 0.7957; greater complexity has not demonstrated a discrimination advantage. All 41 source results, including low-AUC datasets, are supplied.

Whole-source harmonisation precedes the original splits. The train-only label refers specifically to resampling order; it does not certify fully independent preprocessing. Before-split test data also have different composition and prevalence. The paired contrast learns a bundled protocol signature and does not isolate contamination causally.

External diagnostic validity, literature prevalence, target-population calibration, multiclass/raw-modality use and review-time benefits have not been established. PPV/NPV tables are hypothetical transport scenarios. Hosted CI and a numerical coverage percentage have not been established.

The 18 local regression tests cover fixtures, aliases, unsupported families, validation, score wording, warnings, CLI and batch consistency. Stored rounded fixtures use tolerance 1e-6; separate matrix export parity was 2.22e-16. The original export script is `Code/04_export_model.py`; running it retrains and overwrites JSON weights and is unnecessary for inference or revision analysis.

## Repository layout

```text
src/leakcheck/                  package and unchanged JSON weights
tests/                        package tests
Code/                          original training/export and notebook
Code/Paper2B/                  current revision analysis and path resolver
data/processed/                original processed corpus and manifest
results/raw_results.csv        original experimental record
results/paper2b_revision_v1/    current revision results and figures
LICENSE.txt                    BSD licence with current author notices
LICENSE                        original licence notices retained
```

## Authors

- **Hieu Le Ngoc** (corresponding) — Faculty of Information Technology 2, Posts and Telecommunications Institute of Technology. ORCID: [0000-0002-1133-1433](https://orcid.org/0000-0002-1133-1433). Email: [lnhieu@ptit.edu.vn](mailto:lnhieu@ptit.edu.vn)
- **Trung Huynh Ngoc Thanh** — Faculty of Data Science, University of Finance – Marketing. ORCID: [0009-0001-5785-6630](https://orcid.org/0009-0001-5785-6630). Email: [hnttrung@ufm.edu.vn](mailto:hnttrung@ufm.edu.vn)

Questions about the software: [occbuu@gmail.com](mailto:occbuu@gmail.com)

## Funding

This research is funded by University of Finance – Marketing, HCMC Vietnam. This study is supported by Posts and Telecommunications Institute of Technology (PTIT), Vietnam.

## Licence

BSD 3-Clause. See [`LICENSE.txt`](LICENSE.txt) (also [`LICENSE`](LICENSE)).
