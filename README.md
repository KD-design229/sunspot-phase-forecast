# Solar-activity forecasting: data, preprocessing, training and paper figures

This repository reproduces the computations underlying a research manuscript
using authentic WDC-SILSO monthly international sunspot numbers, version 2.0.

## Files and tasks

| File | Purpose |
|------|---------|
| `SILSO_monthly_1990_2025_snapshot.csv` | Frozen 432-month observations, 1990-2025 |
| `prepare_silso_data.py` | Parse official raw SILSO monthly TXT or download a *candidate* refreshed snapshot; compare without overwriting study data |
| `run_recent_sunspot_study.py` | Validate input, construct lookback windows, chronological splits, training-only normalization, Ridge/MLP/LSTM/GRU fitting and one-step test metrics |
| `run_supplemental_experiments.py` | Three rolling test periods, 12/24/36/60-month windows, validation-based selection, block bootstrap, yearly metrics and weighted smoothing |
| `verify_results.py` | Check the frozen snapshot SHA256 and recompute 2021-2025 test RMSE values |
| `rebuild_publication_figures.py` | Draw all 10 publication figures with `paper_color_library.zip` palette `issue_01_008` without retraining or changing predictions |
| `figures_publication/` | Publication-ready PNGs |
| `figures/` | Earlier original outputs and learning curves used to colorize plot 5 |

## Python environment

Python 3.11+ recommended. From this directory:

```bash
python -m pip install -r requirements.txt
python verify_results.py
python rebuild_publication_figures.py
```

For fresh model training, back up the saved output files or run on a copy,
then execute in this order:

```bash
python run_recent_sunspot_study.py
python run_supplemental_experiments.py
python rebuild_publication_figures.py
```

Optional official-source check (network required; revisions may change values):

```bash
python run_recent_sunspot_study.py --verify-official
```

To parse an already downloaded official raw file into a *candidate* CSV:

```bash
python prepare_silso_data.py --source SN_m_tot_V2.0.txt
```

Do not switch in the candidate CSV when reproducing the published numerical
results. The saved output and trained seed selection come from the frozen
snapshot; PyTorch and BLAS versions can alter fresh training results slightly.

## Interpretation / provenance

The test scheme is *one-month-ahead* using previous observed months, NOT a
multi-year forecast made without future measurements. Trend smoothing uses
observations on both sides of the center and is purely for retrospective
visualization, not model input. The longer-window and rolling-fold experiments
select hyperparameters using validation data, not the held-out test period.

Source: WDC-SILSO / Royal Observatory of Belgium, Brussels.
Monthly version 2.0: https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.txt
DOI: https://doi.org/10.24414/qnza-ac80
Data reuse license: CC BY-NC 4.0 (see DATA_ATTRIBUTION.md).
