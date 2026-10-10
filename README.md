# Continuous Indian Sign Language Recognition with Adaptive Multimodal Fusion

This project has two parts:
- **A research study:** a pre-registered, statistically tested evaluation of adaptive (reliability-aware) multimodal fusion for continuous ISL recognition.
- **A working demo:** OpenCV apps that recognise ISL sign sequences from test videos and live from a webcam.

**Headline results** (all on held-out data; details in [reports/FINAL_REPORT.md](reports/FINAL_REPORT.md)):
- **Study, ISL-CSLTR, unseen signers:** adaptive fusion 84.9 % WER vs 84.0 % for the best baseline (p = 0.30). No model beats a constant "I YOU" output (82.5 %). **Negative result.**
- **Demo, INCLUDE, 24 signs:** 93.1 % isolated-sign accuracy and 7.1 % WER on chains of held-out clips (fixed fusion; adaptive 88.2 % / 11.4 %).
- **Publication readiness:** 15 / 45, so do not write the adaptive-fusion paper yet. A negative-result / benchmark-audit paper may be viable after the experiments listed in the report.

## Documents
| File | Contents |
|---|---|
| [reports/FINAL_REPORT.md](reports/FINAL_REPORT.md) | Full research report: problem, related work, protocol, results, statistics, robustness, ablation, error and fusion analysis, efficiency, limitations, readiness score, next steps |
| [reports/PREREGISTRATION.md](reports/PREREGISTRATION.md) | Analysis plan written before training, plus dated amendments (all deviations) |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Environment, exact commands, seeds, measured costs, **time and compute estimates for full training** |
| [DATA.md](DATA.md) | Datasets, licences (both CC BY 4.0), preprocessing, label corrections, splits, ethics |
| [MODEL_CARD.md](MODEL_CARD.md) | Intended use, metrics, limitations, failure modes |
| [references.bib](references.bib) | Citations (entries marked VERIFY need checking before submission) |
| `results/tables.md`, `results/summary.json`, `results/error_analysis.json`, `results/fusion_analysis.json`, `results/efficiency.json` | Generated result tables and analyses |
| `results/runs/` | Per-run predictions and metrics for every study run (JSON) |

## Quick start
```bash
conda env create -f environment.yml
```
```bash
conda activate isl-cslr
```
```bash
python demo/app.py
```
```bash
python demo/live.py
```
- `app.py` replays held-out test videos with live recognition. Keys: SPACE pause, N/P next/previous, D switch dataset, F fixed/adaptive, Q quit.
- `live.py` runs on your webcam. Raise your hands and sign one of the 24 listed signs; L shows a reference clip to copy.

The trained models, landmark files and shrunk demo videos are already in this folder, so the demo runs without downloading anything. To rebuild everything from scratch, follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

**Webcam notes:**
- Allow camera access for your terminal app: System Settings → Privacy & Security → Camera.
- Live accuracy on a new person in front of a webcam has **not** been measured and will be lower than the test numbers.

## Repository layout
```
demo/      app.py (test-video demo), live.py (webcam), train_include.py, train_demo.py
src/       data processing, models (fusion variants), training, metrics, statistics, configs, corruptions
scripts/   data download (byte-range), landmark/feature extraction, study runner, analysis
data/      gloss table + metadata, demo videos (640 px), INCLUDE splits, signer correction
features/  extracted MediaPipe landmarks (demo datasets)
models/    MediaPipe task files, trained checkpoints (models/include, models/demo)
results/   metrics, tables, analyses, per-run study results
reports/   final report, pre-registration
```

## Working together
- **Sharing through git:** large regenerable folders (videos, features) are listed in `.gitignore`. Share `models/`, `features/` and `data/*/videos` separately (for example a cloud-drive zip, about 190 MB), or regenerate them with the scripts.
- **Licence:** choose one for the *code* before publishing; MIT and Apache-2.0 are common choices. The data stays CC BY 4.0 and must be cited (`references.bib`).
- **Before submitting anything:**
  - re-check the `VERIFY` entries in `references.bib`;
  - state the limitations in `MODEL_CARD.md`;
  - do not present the demo as an interpreter for Deaf users.
