# Leakage premium — replication code

Code, processed corpus and experimental record for:

> Leakage makes every problem look equally easy: a multi-dataset measurement
> of the resampling-before-splitting artefact

## Layout

| Path | Contents |
|---|---|
| `Code/` | Corpus builder, factorial, simulation, mitigation, analysis |
| `data/processed/` | Harmonised 41-dataset corpus (`.npz` + `manifest.csv`) |
| `results/` | `raw_results.csv`, `summary.json`, tables |
| `figures/` | Figures used in the paper |

Raw third-party dumps are not redistributed. `Code/01_fetch_datasets.py`
re-downloads the PMLB sources.

## Reproduce from the archived results

```bash
cd Code
pip install -r requirements.txt
python 03_analysis.py
python 05_verify_numbers.py
```

## Reproduce the experiment from scratch

```bash
cd Code
python 01_fetch_datasets.py
OMP_NUM_THREADS=1 python run_experiment.py --shard 0 --nshards 2 --seeds 10 &
OMP_NUM_THREADS=1 python run_experiment.py --shard 1 --nshards 2 --seeds 10 &
wait
python run_simulation.py --seeds 3
python run_mitigation.py --seeds 3
python 03_analysis.py
```

Notebook: `Code/analysis.ipynb`.

## GitHub + Zenodo DOI

```bash
git init
git add -A
git commit -m "Initial public release"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

Then zenodo.org → GitHub → enable this repository → GitHub Release `v1.0.0`.
Put the DOI in the paper and in `CITATION.cff`.

## License

BSD 3-Clause. See `LICENSE`.
