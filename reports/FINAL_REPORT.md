# Adaptive Multimodal Fusion for Continuous Indian Sign Language Recognition: a pre-registered evaluation

*The pre-registration is in `reports/PREREGISTRATION.md`; deviations from it are listed in §6.4.*

## 1. Abstract
We tested whether **adaptive, reliability-aware multimodal fusion** improves continuous Indian Sign Language (ISL) recognition. The fusion combines RGB, body-pose and hand-landmark streams with per-frame weights conditioned on features, quality signals and temporal context. The baselines were RGB-only, fixed concatenation fusion and cross-modal attention fusion.

**Setup.** We used ISL-CSLTR, the only public continuous-ISL corpus with gloss labels: 663 sentence sequences, 97 sentences, 162 glosses and 7 signers. We pre-registered a signer-independent (leave-one-signer-out) protocol, a single primary metric (WER) and the statistical tests.

**The primary result is negative.**
- The proposed model reached 84.9 % WER against 84.0 % for the strongest baseline, chosen by validation score (RGB-only). The difference is +0.9 points (95 % CI −1.2 to +2.9; paired permutation p = 0.30).
- No model beat a **constant system that always outputs "I YOU" (82.5 % WER)**.
- With signers seen in training, adaptive fusion beat RGB-only (75.8 vs 83.7 % WER, Holm p = 0.006) but not fixed fusion (78.7 %, p = 0.12).
- A closed-set sentence classifier beat every continuous model (61.0 % WER), showing that the corpus rewards sentence identification rather than sign recognition.
- The learned fusion weights moved in the right direction when a modality was removed. However, they collapsed onto a single modality in 3 of 7 folds, and they were unrelated to recognition errors (ρ = 0.04).

**Exploratory follow-up.**
- On INCLUDE isolated signs, the same landmark CTC pipeline reaches 93.1 % sign accuracy on a 24-sign subset.
- On the full official **INCLUDE-50 benchmark** it reaches **96.2 ± 1.6 %** (fixed fusion, 3 seeds; 5 minutes of CPU training per run). That is on par with the 94.5 % published by the dataset authors.
- Fixed fusion again beat adaptive fusion there.

**Scope.** The strong numbers above are for *isolated* signs and for sequences of joined isolated clips. Natural continuous ISL sentence recognition was not achieved.

**Conclusion.** On the data available, adaptive fusion did not produce a statistically significant or practically meaningful improvement over fixed fusion.

## 2. Problem definition
Continuous sign language recognition (CSLR) maps an unsegmented video of a signed sentence to its gloss sequence, with no sign boundaries given at training or test time.

For modalities $m\in\{\text{RGB},\text{pose},\text{hand}\}$ with per-frame features $F_m(t)$, adaptive fusion learns weights $w_m(t)\ge 0$ with $\sum_m w_m(t)=1$ and computes $F(t)=\sum_m w_m(t)F_m(t)$. The question is whether this helps, especially under unseen signers, domain shift and degraded inputs.

## 3. Related work and comparability
| Work | Dataset | Split | Metric | Reported | Comparable? |
|---|---|---|---|---|---|
| SignFlow, IEEE Access 2025 (RGB + pose, CNN + Transformer, CTC) | authors' own continuous-ISL dataset | not stated in abstract | WER | 19 | **No**: different data, unknown split |
| ISL-CSLTR (Elakkiya & Natarajan, 2021) | ISL-CSLTR | no official split | — | no published CSLR WER found | Same data, nothing to compare against |
| iSign (ACL Findings 2024), ISLTranslate (2023), CISLR (EMNLP 2022) | ISL translation / isolated words | official | BLEU / accuracy | — | **No**: different tasks |
| INCLUDE (ACM MM 2020) | INCLUDE, 263 isolated signs | official | accuracy | 85.6 % (full), 94.5 % (INCLUDE-50) | Only our exploratory §A1, and on a 24-sign subset, so **not equal** |
| STMC (AAAI 2020): multi-cue RGB + pose | PHOENIX-2014 / 2014T (German SL) | official | WER | 20.7 / 21.0 test (as cited) | **No** |
| SignVTCL (2024): video + keypoints + flow | PHOENIX-2014T | official | WER | 19.0 test | **No** |
| TwoStream-SLR (NeurIPS 2022): RGB + keypoint streams with lateral connections | PHOENIX, CSL-Daily | official | WER | — | Re-implemented at feature level, **not trained** (compute) |
| Signer-invariant Conformer (ICCV 2025 workshop) | Isharah (Saudi SL), signer-independent track | official | WER | 13.07 | **No**: best signer-independent figure found, but a different language |

No published number is directly comparable to ours. All claims below are against baselines reproduced here under identical protocols.

## 4. Research motivation
Sign recognition depends on hands, body and face, and each visual stream fails in different situations: motion blur, hands leaving the frame, lighting, background. A fusion model that downweights an unreliable stream frame by frame should, in principle, be more robust than fixed fusion. That is most likely to matter for unseen signers and degraded video, which is exactly where deployed ISL systems fail.

## 5. Dataset
**ISL-CSLTR** (see `DATA.md`): sentence key-frames, 663 sequences, 18,863 frames (median 24 per sentence), 97 sentences, 162 glosses (338 tokens, mean 3.5 per sentence), 7 signers.
- Signers 1–2 are native signers recorded on a phone (240×320); signers 3–7 are students recorded on a DSLR (1920×1080).
- Every signer signs each sentence once.
- **115 of 162 glosses occur in only one sentence.**
- Label fixes: one misspelled folder, and one sequence filed under the wrong signer, found by appearance clustering and confirmed visually.

**Modalities**, extracted once per frame:
1. DINOv2 ViT-S/14 frame embedding: CLS + mean patch, temporally centred.
2. MediaPipe Pose: 25 upper-body points, body-normalised.
3. MediaPipe Hands: 2 × 21 points, global position plus hand-local shape.
4. Hand-crop DINOv2 embeddings (`hrgb`), post hoc.
5. Ten observable quality signals: brightness, contrast, sharpness, pose visibility, hand detection and confidence, and landmark jitter.

**Tracking quality.** MediaPipe finds ≥ 1 hand in 58–74 % of DSLR frames and 89–96 % of phone frames.

## 6. Experimental protocol
### 6.1 Splits
All splits are asserted disjoint in code.
| Protocol | Train / val / test | Status |
|---|---|---|
| **LOSO (primary)** | 5 signers / 1 held-out signer / 1 held-out signer; 7 folds, so every sequence is tested once | run |
| Random, signer-dependent | per sentence ≈ 5/1/1 sequences | run |
| Cross-domain DSLR→phone | signers 3–6 / 7 / 1–2 | **not run** |
| Unseen sentences | 3 folds × 6 held-out sentences; every test gloss occurs in training | **not run** |

### 6.2 Training
- Objective: CTC with greedy decoding; AdamW with a OneCycle schedule (lr 3e-3, 40 epochs, batch 16, dropout 0.3).
- Checkpoint selection: by validation WER.
- Augmentation (identical for all models): speed 0.8–1.2×, 10 % frame drop, landmark noise, modality dropout p = 0.15.
- Multimodal models also get auxiliary per-modality CTC heads (weight 0.5).
- Backbone: per-modality encoder → fusion → temporal conv → 2-layer pre-norm Transformer (d = 192, 4 heads) → CTC.

### 6.3 Metrics and statistics
- **Metric:** corpus gloss WER (primary); also CER, exact-sentence accuracy and gloss precision / recall / F1.
- **Primary test:** paired approximate-randomisation test over sequences, using seed-averaged errors (10,000 permutations).
- **Uncertainty and robustness checks:**
  - 95 % sentence-cluster bootstrap CIs;
  - exact sign-flip test over the 7 test signers;
  - seed-level Wilcoxon test with Cohen's d_z and rank-biserial r.
- **Multiple comparisons:** Holm correction within families; Friedman test for the ablation.

### 6.4 Deviations from the pre-registration
See amendments 1–3 in `reports/PREREGISTRATION.md`.
- **Backbone:** BiLSTM → Transformer, because the BiLSTM did not fit the training set.
- **RGB features:** temporal centring of the frame embeddings.
- **Seeds:** 3 instead of 5/10.
- **Robustness:** no image corruptions were run, so it covers only the feature-level conditions.
- **Unrun protocols and configurations:** cross-domain, unseen-sentence, TwoStream, and most ablations.
- **Post hoc:** the hand-crop stream was added after Stage 1.

## 7. Baselines
| ID | Inputs | Fusion | Role |
|---|---|---|---|
| A_rgb | RGB | — | Baseline 1 |
| C_hand / F_pose_hand | landmarks | — / concat | Baseline 2 (skeleton) |
| G_concat | RGB + pose + hand | concatenation + projection | Baseline 3: fixed fusion |
| I_xattn | RGB + pose + hand | MulT-style cross-modal attention | Baseline 5: existing attention fusion |
| twostream | RGB + pose + hand | TwoStream-SLR-style lateral streams and ensemble | Baseline 4: implemented, **not trained** |
| sentcls | RGB + pose + hand | closed-set sentence classifier | Diagnostic |
| constant | — | — | Outputs the single training-optimal string, "I YOU" |

## 8. Proposed method
**L_trel_gate: temporal reliability-aware fusion.**
- **Gate inputs:** for each frame, the gate takes the concatenated modality features plus the 10 quality signals.
- **Weights:** a linear layer and a BiGRU over time produce per-modality logits, then a softmax gives $w_m(t)$; the fused feature is $\sum_m w_m(t)F_m(t)$.
- **Cost:** 1.81 M parameters, 67.2 MFLOPs per sentence; G_concat has 1.82 M parameters and 65.6 MFLOPs. The capacities match, so differences come from the mechanism.
- **Other gate variants implemented:** feature-only gate, quality-only gate, uncertainty-aware (auxiliary-head entropy), and sequence-level attention. They are not trained in the final grid.

## 9. Main results
### 9.1 Signer-independent (LOSO), clean test, 3 seeds
| Config | WER ↓ | 95 % CI | CER ↓ | Sentence acc. | Gloss F1 | Val WER | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| *Constant "I YOU"* | **82.5** | – | 80.7 | 0.0 | 26.4 | – | 0 |
| A_rgb | 84.0 ± 1.1 | [81.2, 86.7] | 80.0 | 3.0 | 27.3 | 76.4 | 1.19 M |
| I_xattn | 83.7 ± 1.2 | [80.7, 86.7] | 79.7 | 0.7 | 24.1 | 81.3 | 2.27 M |
| G_concat | 84.6 ± 1.4 | [81.1, 88.1] | 77.0 | 2.3 | 23.1 | 77.6 | 1.82 M |
| **L_trel_gate (proposed)** | 84.9 ± 2.4 | [79.0, 90.7] | 77.1 | 2.3 | 21.7 | 76.4 | 1.81 M |
| Q hand-crops + pose + hand, concat (post hoc) | 86.6 ± 1.1 | [83.9, 89.3] | 81.4 | 0.6 | 21.3 | 82.2 | 1.82 M |
| Q hand-crops + pose + hand, trel_gate (post hoc, 2 seeds) | 86.4 ± 3.9 | wide | 77.6 | 2.9 | 21.9 | 75.0 | 1.81 M |

### 9.2 Signer-dependent (random split), 3 seeds
| Config | WER ↓ |
|---|---:|
| sentcls (closed-set) | **61.0 ± 3.6** |
| C_hand (hand landmarks only) | 72.6 ± 2.1 |
| L_trel_gate | 75.8 ± 2.1 |
| G_concat | 78.7 ± 3.3 |
| A_rgb | 83.7 ± 3.1 |
| constant "I YOU" | 82.5 |

## 10. Statistical analysis
**Primary comparison: L_trel_gate vs the strong baseline (LOSO, clean).** The strong baseline is A_rgb, chosen by validation WER from A_rgb 76.4, G_concat 77.6 and I_xattn 81.3.
- **Pooled WER:** 84.86 vs 83.98, so Δ = **+0.88 points** (the proposed model is *worse*). The 95 % sentence-cluster bootstrap CI is [−1.18, +2.93].
- **Relative WER reduction:** −1.1 % (CI [−3.6, +1.4]).
- **Paired permutation test** (663 sequences): **p = 0.30**.
- **Per test signer,** Δ = S1 −4.3, S2 +0.2, S3 +5.9, S4 +5.2, S5 −4.5, S6 +2.9, S7 +1.0; exact sign-flip p = 0.55.
- **Seed level:** 84.86 ± 2.37 vs 83.98 ± 1.10; Wilcoxon p = 0.75; d_z = 0.38; rank-biserial r = 0.33.
- **Verdict:** none of the pre-registered thresholds is met (minimum: ≥ 5 % relative reduction and p < 0.05).

**Generalisation family** (Holm-corrected; only the random protocol was run):
- vs A_rgb: Δ = −7.9 [−12.9, −2.8], p_Holm = 0.006 (significant).
- vs G_concat: Δ = −2.9 [−6.3, +0.5], p_Holm = 0.12 (not significant).

**Ablation family:** Friedman over A_rgb, G_concat and L_trel_gate gives χ² = 1.0, p = 0.59. No pairwise difference is significant.

## 11. Robustness analysis (feature-level conditions, LOSO)
| Config | clean | RGB missing | pose missing | hand missing | 2× faster | 1.5× slower |
|---|---:|---:|---:|---:|---:|---:|
| A_rgb | 84.0 | 99.7 | 84.0 | 84.0 | 83.7 | 89.3 |
| G_concat | 84.6 | 84.4 | 85.1 | 86.9 | 84.8 | 86.3 |
| I_xattn | 83.7 | 84.6 | 85.0 | 85.4 | 89.1 | 85.6 |
| L_trel_gate | 84.9 | 83.7 | 85.4 | 86.0 | 85.6 | 86.9 |

**Adaptive vs fixed fusion** (Holm-corrected over 10 tests): no condition differs significantly; all |Δ| ≤ 0.9 points, all p_Holm ≥ 0.56.

**Adaptive vs RGB-only:** adaptive wins only when RGB is removed (trivially) and on slowed signing (Δ −2.3, p_Holm = 0.016).

**Not tested:** image-level lighting, occlusion, background, blur, distance and viewpoint. The code exists (`src/corruptions.py`), but the features were never extracted.

### 11.1 Does the adaptive mechanism behave sensibly? (seed 0, 113,481 frames)
- **Response to a missing modality:** removing a modality lowers its weight (RGB 0.36 → 0.21, hand 0.32 → 0.24, pose 0.31 → 0.27) and shifts weight to the remaining streams. The response is only partial: weights do not go to zero.
- **Tracking quality:** the hand weight rises with the hand-detection rate (sequence-level ρ = 0.38), and the RGB weight with brightness (ρ = 0.29).
- **No link to errors:** gate confidence (entropy) is unrelated to sequence WER (ρ = 0.04).
- **Collapse:** in 3 of 7 folds the gate collapsed onto a single modality (weight ≈ 1.0), and which one varied: hand, RGB and pose in different folds. So "adaptive" weighting is unstable across training runs, not a consistent reliability estimate.

## 12. Ablation study
| Planned ablation | Status | Result |
|---|---|---|
| A RGB · B pose · C hand | A run (LOSO); C run (random only); B not run | A 84.0 (LOSO); C 72.6 (random), the best single modality |
| D RGB+pose · E RGB+hand · F pose+hand | not run in the study | — |
| G RGB+pose+hand fixed | run | 84.6 |
| H fixed scalar weights · I attention | I_xattn run; H and I_seq not run | I_xattn 83.7 |
| J feature gate · K reliability gate | not run | — |
| L temporal reliability gate (proposed) | run | 84.9 |
| Hand-crop RGB stream (post hoc) | run | 86.4–86.6: worse than whole-frame RGB on LOSO |
| Parameter-matched and no-augmentation controls | not run | — |

The improvement cannot be attributed to the adaptive mechanism, because there is no improvement to attribute.

## 13. Error analysis (LOSO, all seeds)
- **Most confused glosses are pronouns:** YOU→I (56), IT→YOU (30), YOU→NOT (30), YOU→ME (27), ME→YOU (21). Deletions dominate substitutions; the most deleted glosses are I (263), YOU (222), NOT (148) and WHAT (98).
- **Rare glosses are almost never recovered.** Recall is 8.5 % for glosses that occur in one sentence (115 glosses), 9.3 % for 2–3 sentences, and 26.8 % for ≥ 4 sentences. The model mostly learns frequent function signs.
- **Errors grow towards the end of a sentence** (sentences of ≥ 3 glosses): 73 % on the first gloss, 86 % in the middle, 94 % on the last. Late signs are usually dropped.
- **Signer- and domain-specific failures:**
  - WER by test signer: S1 87.8, S2 87.4 (phone), S3 82.1, S4 83.6, S5 81.8, S6 82.2, S7 89.2.
  - Phone versus DSLR overall: 87.6 vs 83.7.
  - S7 has the darkest recordings, and the phone signers are both low-resolution and native signers, which is the confound noted in §5.
- **Short sentences:** a single-gloss sentence ("congratulations") gets WER > 100 % from insertions.
- **Tracking failures** cannot be separated from recognition failures in the remaining files. Detection rates are reported in §5.

## 14. Efficiency
| Model | Params | MFLOPs per sentence (T = 28) | CPU latency (ms per sentence) |
|---|---:|---:|---:|
| A_rgb | 1.19 M | 30.7 | 1.4–2.0 |
| C_hand | 1.08 M | 24.3 | 1.3–2.8 |
| G_concat | 1.82 M | 65.6 | 2.9–4.8 |
| L_trel_gate | 1.81 M | 67.2 | 3.9–10.1 |
| I_xattn | 2.27 M | 106.3 | 3.9–10.2 |
| twostream | 4.53 M | 108.9 | 5.8–15.7 |

- **Latency ranges** come from two measurements on a loaded M2.
- **The gate's cost:** it adds about 2 % FLOPs, but its sequential BiGRU roughly doubles latency.
- **Front-end cost dominates:**
  - MediaPipe pose + hands: about 75 ms per frame (13 fps).
  - DINOv2 ViT-S: 33–48 ms per frame on the M2 GPU, about 125 ms on a loaded CPU.
  - The landmark-only pipeline runs live at about 13 fps on the M2.
- **Training cost:** 95–120 s per run (4 parallel workers, idle machine). The full measured cost table is in `REPRODUCIBILITY.md`.

## 15. Limitations
1. **Tiny, closed corpus:** 97 sentences, one take per signer per sentence, and 115 single-sentence glosses. The standard splits partly measure sentence identification; a closed-set classifier beats all CSLR models.
2. **Sparse key-frames** (median 24 per sentence) instead of video, plus frozen front-ends on CPU-only hardware. Absolute WERs are not comparable to end-to-end systems.
3. **Statistical power:** 3 seeds and an incomplete grid; no image corruptions, cross-domain or unseen-sentence runs; TwoStream not trained.
4. **Confound:** the phone recordings are also the native signers.
5. **Post hoc additions:** the hand-crop stream and the demo studies were added after seeing results, so they are exploratory.
6. **Community involvement:** no Deaf signers or ISL linguists were involved.

## 16. Reproducibility
- **Environment and commands:** `environment.yml`, `requirements.txt`, and every command in `REPRODUCIBILITY.md`.
- **Configurations:** `src/configs.py`.
- **Seeds:** stored in every run file.
- **Raw results:** per-run predictions and metrics in `results/runs/` (JSON); tables are regenerated by `scripts/analyze.py`.
- **Data:** public under CC BY 4.0 and re-downloadable by script. Intermediate features and study checkpoints were deleted to save space; regenerating them takes about 1–1.5 days on the test machine, see the estimates.

## 17. Comparison with published work
- **No directly comparable result exists** (§3).
- The only ISL continuous-recognition WER found (SignFlow, 19) uses a private dataset with an unknown split.
- PHOENIX numbers (19–21 % WER) come from a different language and recording setup, roughly 10× more training sentences, and end-to-end fine-tuned video backbones.
- **What our numbers do show:** on ISL-CSLTR, signer-independent CSLR with frozen features is at the level of a trivial constant baseline. Any future claim on this corpus should report that baseline and the closed-set sentence-classifier diagnostic.

## 18. Recommended next experiments
1. **Finish the pre-registered grid with ≥ 5 seeds:** cross-domain, unseen sentences, sentcls and C_hand on LOSO, TwoStream, and the J/K gate ablations. About 1 day on the M2.
2. **Re-run the study on the full-rate ISL-CSLTR videos** (all 97 sentences). The 40-sentence video models reached 64.6 % WER (signer-seen). The key-frame models scored 76–79 % on the 97-sentence signer-dependent split. These are not directly comparable, because the vocabularies differ, but they suggest dense tracking helps.
3. **Use a larger signer pool or pre-training.** INCLUDE-50 is now ready to use (96 % accuracy). One option: a landmark encoder pre-trained on all of INCLUDE (263 signs, 4,292 clips), then fine-tuned for CTC on sentences.
4. **Extract the image-corruption conditions,** the experiment where adaptive fusion should help most if it helps at all.
5. **Involve Deaf ISL signers** in data collection and evaluation (repeated takes, natural sentences).

## Appendix A: exploratory demo studies (not pre-registered)
### A1. INCLUDE sign sequences
- **Setup:** 24 signs; official split restricted to them (313 / 40 / 87 clips). Trained with CTC on random chains of 1–5 training clips; tested on 87 isolated test clips and on 400 chains of test clips.
- **Results** (3 seeds):

| Model | Isolated-sign accuracy | Sequence WER ↓ | Exact sequences |
|---|---:|---:|---:|
| Fixed fusion | **93.1 ± 2.0 %** | **7.1 ± 1.2** | **82.0 %** |
| Adaptive fusion | 88.2 ± 1.3 % | 11.4 ± 2.1 | 71.7 % |

- **Limitations:** chains lack co-articulation, so these are not natural continuous sentences, and the result is not comparable to the INCLUDE paper's full 263-sign task.

### A1b. INCLUDE-50 (official benchmark, all 50 signs), run 2026-10-11
- **Setup:** the official AI4Bharat split (689 / 77 / 192 clips). Same landmark CTC pipeline, trained on random chains of 1–4 training clips (25 % single clips).
- **Checkpoint selection:** by validation WER; the test clips were never used for training or selection.
- **Benchmark metric:** accuracy on the 192 test clips, where the predicted sign is the class with the highest CTC likelihood among the 50.

| Model (3 seeds) | Test accuracy, closed-set | Free-decoding accuracy | Sequence WER ↓ (400 test chains) | Exact sequences | Train time per run (M2) |
|---|---:|---:|---:|---:|---:|
| Fixed fusion | **96.2 ± 1.6 %** | 92.5 ± 1.6 % | **8.1 ± 0.8** | **81.2 %** | 5.0 min |
| Adaptive fusion | 95.0 ± 1.3 % | 89.6 ± 2.1 % | 10.9 ± 2.8 | 75.0 % | 5.9 min |
| INCLUDE paper, best reported | 94.5 % | — | — | — | — |

- **Comparison with the published result:** fixed fusion is on par with the INCLUDE authors' best reported INCLUDE-50 accuracy (94.5 %), on the same split and metric.
- **No improvement claimed:**
  - 3 seeds give a 95 % t-interval of roughly [92.2, 100].
  - With 192 test clips, the binomial standard error is about 1.4 points.
  - We have not surveyed later INCLUDE-50 results, so no state-of-the-art claim is made.
- **The point of interest:** this level is reached with a 1.3 M-parameter landmark model trained in 5 minutes on a laptop CPU.
- **Adaptive fusion** is again below fixed fusion on every metric.

### A2. ISL-CSLTR videos, 40 sentences
- **Setup:** one held-out video per sentence; each test video is a person never seen signing that sentence.
- **Results:**
  - Fixed fusion: WER 64.6 ± 4.8, closed-vocabulary sentence accuracy 30.8 %.
  - Adaptive fusion: WER 71.6 ± 2.9, closed-vocabulary sentence accuracy 30.0 %.

Both appendix studies agree with the main result: simple fixed fusion is at least as good as the proposed adaptive fusion.
