# Reproducibility

## Environment
- **Tested on:** Apple M2 (8 cores, 8 GB unified memory, no CUDA), macOS 26.1, Python 3.11.17, PyTorch 2.14.1 (CPU), MediaPipe 1.1.0.
- **Setup:**
  ```bash
  conda env create -f environment.yml
  ```
  ```bash
  conda activate isl-cslr
  ```
- **MediaPipe model files** in `models/`: `pose_landmarker_full.task` and `hand_landmarker.task`, from Google's `storage.googleapis.com/mediapipe-models/...` "latest". They are needed for any landmark extraction.
- **DINOv2 ViT-S/14:** only the research study's RGB stream uses it. It downloads automatically through `torch.hub` into `models/torch/`.
- **Determinism:**
  - Every run seeds PyTorch, NumPy and the batch sampler; the seed is in each result file.
  - Splits are deterministic: the split seed is 0, and demo sequences use seeds 100, 200 and 7.
  - Image corruptions are seeded by (sequence id, condition).
  - CPU floating-point results can differ in the last digits across machines.

## A. Live / video demo (what the app uses)
```bash
python scripts/fetch_videos.py data/demo_sentences.txt
```
```bash
python scripts/extract_video_lm.py
```
```bash
python demo/train_demo.py --seeds 0-2
```
```bash
python scripts/fetch_include.py
```
```bash
python scripts/extract_video_lm.py --videos data/include/videos --out features/include_lm.pkl
```
```bash
python demo/train_include.py --fusion trel_gate --seed 0
```
```bash
python demo/train_include.py --fusion concat --seed 0
```
```bash
python demo/app.py
```
```bash
python demo/live.py
```

1. `fetch_videos.py` downloads 279 ISL-CSLTR videos (3.1 GB transferred, 41 MB stored). `extract_video_lm.py` then writes `features/videos_lm.pkl`.
2. `train_demo.py` trains the ISL-CSLTR sentence models into `models/demo/` and `results/demo/metrics.json`.
3. `fetch_include.py` downloads the 440 INCLUDE clips (6.1 GB transferred, 48 MB stored). The second `extract_video_lm.py` call extracts their landmarks.
4. Run `train_include.py` for seeds 0, 1 and 2 with each fusion (`trel_gate`, `concat`); outputs go to `models/include/` and `results/include/`.
5. `app.py` replays held-out test videos; `live.py` runs on the webcam.

## B. Research study (ISL-CSLTR key-frames, pre-registered in `reports/PREREGISTRATION.md`)
The study's raw frames, features and checkpoints were deleted to save space; the per-run result JSONs are kept in `results/runs/`.

**1. Download the key-frames.** The archive's first 1.33 GB hold all of them; fetching that range with curl is much faster than streaming.
```bash
U=$(curl -sI "https://data.mendeley.com/public-files/datasets/kcmpdxky7p/files/c14c8953-be53-4c7a-85d5-5bcd7e79d86e/file_downloaded" | awk '/^location/{print $2}' | tr -d '\r')
```
```bash
curl -r 0-1330115747 -o data/_frames_part.bin "$U"
```
```bash
LOCAL_PART=data/_frames_part.bin python scripts/fetch_frames.py data/frames
```
After it finishes, delete `data/_frames_part.bin`.

**2. Extract features for the clean condition.** Run both landmark shards, then the RGB stage, then merge, then the hand crops:
```bash
python scripts/extract_features.py clean --stage lm --shard 0 --nshards 2
```
```bash
python scripts/extract_features.py clean --stage lm --shard 1 --nshards 2
```
```bash
python scripts/extract_features.py clean --stage rgb
```
```bash
python scripts/extract_features.py clean --stage merge
```
```bash
python scripts/extract_features.py clean --stage handrgb
```
Repeat for each image corruption (`lowlight uneven hand_occ body_occ background blur distance viewpoint combined`).

**3. Check the signer mapping**, which writes `results/signer_check.jpg`:
```bash
python scripts/verify_signers.py
```

**4. Train and evaluate.** The runner skips runs that already exist, so it can resume. Repeat for the other protocols: `cross_domain`, `random`, `unseen_sentence`.
```bash
python scripts/run.py --configs L_trel_gate G_concat A_rgb I_xattn --protocols loso --seeds 0-4 --workers 4
```

**5. Analyse.**
```bash
python scripts/analyze.py
```
```bash
python scripts/error_analysis.py L_trel_gate G_concat A_rgb
```
```bash
python scripts/efficiency.py
```
`analyze.py` writes `results/tables.md` and `results/summary.json`. `error_analysis.py` writes `results/error_analysis.json`.

Configurations are defined in `src/configs.py`. The pre-registration and its amendments record every protocol decision and when it was made.

## C. Measured cost on the test machine
| Step | Measured time |
|---|---|
| MediaPipe pose (full) + hands, VIDEO mode, 640×360 | ~13 frames/s per process; 4 processes ≈ 3× |
| DINOv2 ViT-S/14, 224 px | 21–30 img/s on the M2 GPU (idle); ~8 img/s on CPU under load |
| Study run (40 epochs, ~470 key-frame sequences) | 95–120 s each with 4 parallel workers on an idle machine; 7–17 min under heavy memory contention |
| ISL-CSLTR video demo model (198 training videos) | 31–41 s per run |
| INCLUDE 24-sign model (30 epochs × 800 random chains) | 8–9 min per run with 2 jobs in parallel; ~23–25 min with 6 in parallel |
| Inference | 2–10 ms per sentence (model only); live app ~13.5 frames/s, limited by tracking |
| Downloads | Mendeley 0.2–1.8 MB/s (parallel ranges help); Zenodo ~0.33 MB/s per connection, ~1.5–3 MB/s with 12 parallel |

## D. How long would it take to fully train?
Estimates are built from the measurements above; ranges reflect network speed and how busy the machine is. A GPU only speeds up model training: landmark extraction (MediaPipe) is CPU-bound and downloads are network-bound.

| Goal | Data | This M2 (8 GB, CPU) | GPU workstation (1× 24 GB GPU, 16-core CPU) |
|---|---|---|---|
| **Rebuild the demo** as shipped (2 datasets × 2 fusions × 3 seeds) | 9 GB download | **~3–4 h**: download 1.5–2 h, landmarks ~30 min, training ~1 h | ~2 h, mostly download |
| **Complete the pre-registered study** (~30 configs × 7 LOSO folds × 5 seeds + 10 seeds for the primary pair + 3 other protocols ≈ 1,250 runs; features for 10 image conditions) | 1.3 GB | **~1–1.5 days** if dedicated: features 10–15 h, training 10–12 h. 2–4× longer if other heavy apps are running. | **~6–8 h**: features 3–4 h, training 2–3 h |
| **Study on full-rate ISL-CSLTR video** (all 97 sentences, 687 videos) | 7.6 GB | **~8–12 h**: download 1.5–3 h, landmarks ~30 min, ~1 min per run so the full ablation grid ≈ 5–8 h | ~3–4 h |
| **Full INCLUDE** (263 signs, 4,292 clips) sign-sequence model, 3 seeds × 2 fusions | 56.8 GB | **~1–2 days**: download 6–10 h, landmarks 2–3 h, training ~1.5–3 h per run (extrapolated) | ~10–14 h, mostly download and extraction; ~15 min per run |
| **End-to-end RGB video CSLR** (fine-tune S3D or a video transformer, as SOTA systems on PHOENIX do) | any | **not feasible** (memory, no CUDA) | Needs ≥24 GB VRAM: hours per run, days for a seeded grid |

**Minimum hardware** to reproduce everything except end-to-end video training:
- 8 GB RAM and 4 CPU cores; no GPU needed.
- About 15 GB of free disk while downloading, about 2 GB afterwards.
- Close other memory-heavy apps: running `llama-server` and Docker at the same time slowed this machine 3–10×.
