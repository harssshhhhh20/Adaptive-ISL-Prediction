# Paper kit, for whoever writes the paper

Everything here comes from experiments in this folder. Full details, statistics and caveats are in `reports/FINAL_REPORT.md`, and that report is the source of truth if anything here disagrees.

## 1. Working title
> **Lightweight Landmark-Based Indian Sign Language Recognition: From Isolated Signs to Continuous Sequences, and When Simple Fusion Beats Adaptive Fusion**

## 2. Draft abstract (≈150 words; adjust after the remaining experiments)
> We present a lightweight Indian Sign Language (ISL) recogniser that runs entirely on a laptop CPU. MediaPipe pose and hand landmarks feed a 1.3M-parameter Transformer trained with CTC. On the official INCLUDE-50 benchmark it reaches 96.2 ± 1.6 % test accuracy over 3 seeds, on par with the 94.5 % reported by the dataset authors, after 5 minutes of training. Because it is trained with CTC, it also decodes sequences of signs without boundary supervision (8.1 % WER on sequences built from held-out isolated clips). Natural continuous signing remains much harder. We further test, under a pre-registered protocol, whether reliability-aware adaptive fusion of RGB, pose and hand streams improves recognition. It does not: fixed fusion matched or outperformed adaptive fusion in all four settings, and learned modality weights were unrelated to errors. Finally, on natural continuous ISL sentences (ISL-CSLTR) no model surpasses a constant-output baseline on unseen signers. Continuous sentence recognition is therefore not solved at this data scale, and we recommend reporting such baselines. Code and protocols are released.

## 3. Contributions (claim exactly these)
**Framing:** the headline result is **isolated-sign** recognition. "Continuous" applies to the model (CTC, no boundaries) and to joined-clip sequences only; on natural continuous sentences the result is negative.

1. A CPU-only landmark CTC recogniser **on par with** the published INCLUDE-50 baseline (96.2 ± 1.6 vs 94.5 %), trained in 5 min. It is **not** "state of the art" (see §6).
2. Boundary-free decoding of *sign sequences built from isolated clips* (INCLUDE-50 chains: WER 8.1 ± 0.8). Not natural continuous sentences.
3. A pre-registered negative result: adaptive, reliability-aware fusion ≤ fixed fusion, with a mechanism analysis (gate collapse in 3 of 7 folds; weights uncorrelated with errors, ρ = 0.04).
4. An evaluation audit of ISL-CSLTR: trivial and closed-set baselines that future work should report.

## 4. Paper outline → where the content is
| Section | Source in this folder |
|---|---|
| 1 Introduction | FINAL_REPORT §2, §4 |
| 2 Related work | FINAL_REPORT §3 (comparability table), `references.bib` |
| 3 Data | `DATA.md`, FINAL_REPORT §5; INCLUDE-50 split counts in §A1b |
| 4 Method | FINAL_REPORT §8, `MODEL_CARD.md` (architecture), `src/models.py` |
| 5 Protocol and statistics | FINAL_REPORT §6, `reports/PREREGISTRATION.md` |
| 6.1 INCLUDE-50 results | `paper/tables.tex` (Table `tab:include50`), `paper/figures/fig_include50_accuracy.pdf` |
| 6.2 Fixed vs adaptive | `paper/tables.tex` (Table `tab:fusion`), `paper/figures/fig_fixed_vs_adaptive.pdf`, FINAL_REPORT §10–11 |
| 6.3 Fusion-weight analysis | FINAL_REPORT §11.1, `results/fusion_analysis.json`, `results/figures/weights_by_condition_L_trel_gate.png` |
| 6.4 ISL-CSLTR audit | FINAL_REPORT §9–10, §13 (error analysis), `results/tables.md` |
| 7 Efficiency | FINAL_REPORT §14, `results/efficiency.json`, `REPRODUCIBILITY.md` §C |
| 8 Limitations and ethics | FINAL_REPORT §15, `DATA.md` §4, `MODEL_CARD.md` |

Regenerate figures and tables after any new run:
```bash
python paper/make_figures.py
```

## 5. Key numbers (all on held-out test data)
| What | Value |
|---|---|
| INCLUDE-50 accuracy, fixed / adaptive | **96.2 ± 1.6 %** / 95.0 ± 1.3 % (paper baseline 94.5 %) |
| INCLUDE-50 chains: WER (fixed / adaptive) | 8.1 ± 0.8 / 10.9 ± 2.8 |
| INCLUDE-50 chains: exact sequences (fixed / adaptive) | 81.2 % / 75.0 % |
| Training time per run (Apple M2 CPU) | 5.0 min (fixed), 5.9 min (adaptive) |
| Model size | 1.33 M params; 43–67 MFLOPs per sentence; live ≈ 13 fps (tracking-bound) |
| ISL-CSLTR, unseen signer (LOSO), adaptive vs best baseline | 84.9 vs 84.0 WER; Δ +0.9 [−1.2, +2.9]; p = 0.30 |
| ISL-CSLTR constant "I YOU" baseline | 82.5 WER |
| ISL-CSLTR signer-seen: adaptive vs RGB / vs fixed | −7.9 WER (p_Holm = 0.006) / −2.9 (p = 0.12, not significant) |

## 6. Allowed wording vs do-not-claim
| ✅ Write | ❌ Do not write |
|---|---|
| "on par with the published INCLUDE-50 baseline" | "state of the art", "outperforms prior work" (newer INCLUDE-50 papers have not been surveyed) |
| "continuous recognition of sign sequences built from isolated clips" | "continuous / natural ISL sentence recognition" for the INCLUDE results (the chains have no co-articulation) |
| "adaptive fusion did not improve over fixed fusion" | "adaptive fusion improves robustness" (not shown; image-level corruptions were never run) |
| "live demo; live accuracy not evaluated" | any live-accuracy number |
| "3 seeds" | implying 5–10 seeds |
| "a research prototype for a closed vocabulary" | "sign-language interpreter" or "accessibility tool" |

## 7. Still to do before submission
These are listed in priority order; commands are in `REPRODUCIBILITY.md`, and Claude can run them.
1. **Literature check:** recent INCLUDE-50 / INCLUDE results. This decides the wording of contribution 1.
2. **5 seeds** for the INCLUDE-50 fixed and adaptive runs, plus a paired significance test. Per-clip predictions must be saved first (a small code change). About 30 min.
3. **INCLUDE-50 ablations:** hand-only, pose-only, no augmentation, BiLSTM backbone. About 1 h.
4. **Full INCLUDE (263 signs)** against the published 85.6 %. About a weekend, mostly downloading.
5. **Measured end-to-end live latency.**
6. **Fill in the `VERIFY` entries** in `references.bib` (author lists).
7. **Choose a code licence** (MIT or Apache-2.0) and decide where to host code and models.

## 8. Submission checklist
- [ ] Anonymise for double-blind review: remove names and paths, and use an anonymous code link.
- [ ] Cite both datasets (CC BY 4.0) and MediaPipe.
- [ ] Ethics statement: identifiable signers, no Deaf community involvement, not an interpreter.
- [ ] Limitations section, including the ISL-CSLTR negative result.
- [ ] Reproducibility: seeds, hardware, training times, pre-registration link.
- [ ] Venues: CVPR/ECCV/ICCV sign-language workshops, the ACL-family Workshop on Sign Language Processing, NCVPRIPG / CVIP / INDICON, or IEEE Access. Check current deadlines.
