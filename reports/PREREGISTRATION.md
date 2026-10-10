# Pre-registered analysis plan

Written 2026-10-09 03:25 IST, **before any model was trained or evaluated**. If the analysis later departs from this plan, the final report will say so.

## Dataset
ISL-CSLTR (Elakkiya & Natarajan, Mendeley Data, DOI 10.17632/kcmpdxky7p.1, CC BY 4.0).
- We use the sentence-level frame sequences: 101 sentences × 7 signers, about 663 sequences.
- Folders 1–2 are phone recordings and folders 3–7 are DSLR recordings. The folder→signer mapping was checked visually.
- Gloss targets come from `ISL Corpus sign glosses.csv` after the normalisation documented in `src/glosses.py`. As a sensitivity check, results are re-scored with raw glosses.

## Primary endpoint (one metric, one protocol, one comparison)
- **Metric:** gloss Word Error Rate (WER). It is corpus-level (total edits / total reference glosses) with greedy CTC decoding.
- **Protocol:** signer-independent leave-one-signer-out (LOSO) with 7 folds. The validation signer is also excluded from training, and the test condition is clean.
- **Proposed model:** `L_trel_gate`. It uses RGB + pose + hand with temporal, reliability-aware adaptive fusion.
- **Strong baseline:** the configuration in `BASELINE_POOL` (`src/configs.py`) with the lowest mean **validation** WER on LOSO. It is chosen without looking at test results.
- **Seeds:** 10 per model for the primary comparison and 5 for all other configurations.

## Statistical tests for the primary comparison
1. **Main test: paired approximate-randomization (permutation) test on corpus WER.** For each test sequence, each model's errors are averaged over seeds. Sequences are the paired units. We use 10,000 random swaps and a two-sided test. This is the standard significance test for ASR/CSLR system comparison.
2. **Confidence interval: 95% cluster bootstrap over sentences.** We draw 10,000 resamples of the 101 sentences, keeping every sequence of a resampled sentence. This gives a CI on the WER difference and on the relative WER reduction. Sentence clustering accounts for the dependence between repeated signings of the same sentence.
3. **Signer-level robustness check.** We compute the WER difference per test signer (7 values) and apply an exact sign-flip permutation test. It has only 128 permutations, so its smallest attainable two-sided p-value is 1/64. It checks that the gain is not driven by a single signer.
4. **Seed-level check.** We compute per-seed pooled LOSO WER for both models (10 paired values) and apply a Wilcoxon signed-rank test, reporting the mean ± SD and a 95% t-interval.
5. **Effect sizes.** We report the absolute WER difference, the relative WER reduction, Cohen's d_z on the per-seed differences, and the rank-biserial correlation for the Wilcoxon test.

Significance threshold: α = 0.05 (two-sided) for the primary comparison.

## Secondary analyses (Holm–Bonferroni corrected within each family)
- **Family R (robustness):** proposed vs strong baseline, and adaptive (`L_trel_gate`) vs fixed (`G_concat`). Each is tested on the 9 image corruptions plus the combined condition, 3 missing-modality conditions, and 2 speed conditions, all with LOSO test signers. Test 1 is used for each.
- **Family G (generalisation):** the same two comparisons on the cross-domain (DSLR→phone), signer-dependent random, and unseen-sentence protocols.
- **Family A (ablation):** a Friedman test over the A–L configurations, with LOSO (fold, seed) runs as blocks. Post-hoc tests are Holm-corrected pairwise Wilcoxon tests against `L_trel_gate`.

## Success thresholds (revised from the brief, see below)
| Level | Requirement on the primary comparison |
|---|---|
| Minimum | ≥ 5 % relative WER reduction **and** p < 0.05 **and** the bootstrap CI excludes 0 |
| Strong | ≥ 10 % relative reduction, the gain holds on cross-domain, and inference cost is less than 2× the baseline |
| Excellent | ≥ 15 % relative reduction, **or** baseline-level clean WER with significant gains under the robustness family |

**Why the thresholds are relative to a reproduced baseline:**
- No published continuous-recognition WER exists for ISL-CSLTR under any documented split. SignFlow (IEEE Access 2025, WER 19) uses a different, private ISL dataset.
- So the only scientifically valid comparison is against baselines reproduced here under an identical protocol.

## Known threats, declared in advance
1. **Closed sentence set.** 117 of the 167 glosses occur in only one sentence. Under the random, LOSO and cross-domain protocols every test sentence also appears in training, so WER partly measures sentence identification. The unseen-sentence protocol (21 eligible sentences) is the only test of genuine compositional CSLR, and it has low statistical power.
2. **Sparse frames.** Sequences are released key-frames (median about 24 per sentence), not full-rate video.
3. **Signer group and camera are confounded.** The phone signers (1–2) are also the native Deaf signers, so the cross-domain result mixes a signer-population shift with a camera shift.
4. **Frozen features.** The RGB backbone (DINOv2 ViT-S/14) and MediaPipe are frozen because the hardware is an Apple M2 with 8 GB. Absolute WERs are therefore not comparable to end-to-end fine-tuned systems.
5. **Synthetic corruptions.** The image corruptions are synthetic, although they are applied before landmark and feature extraction so their effect on the trackers is real. Corruption types are never seen in training. The only training augmentation is feature-level, identical for every model.

---
## Amendment 1, 2026-10-09 05:50 IST. Made before any test-set evaluation, using only train/validation data.

1. **Data facts found during extraction.**
   - Frames exist for only 97 of the 101 sentences (663 sequences). The 4 sentences with no frames are dropped from the vocabulary, leaving 162 glosses.
   - One folder name is misspelled (`which collegeschool are you from`) and is mapped to the gloss table.
   - An appearance-embedding check plus manual inspection found exactly one mislabelled sequence: `do you need something|4` shows signer 3. The correction lives in `data/signer_map.json` and is used by every split.
   - Phone recordings (signers 1–2) are only 240×320 px; DSLR frames are 1920×1080, stored at 640×360.
2. **Backbone changed from 1D-CNN + BiLSTM to 1D-CNN + 2-layer Transformer encoder for all models.**
   - In pilots on one LOSO fold, the BiLSTM/CTC models did not fit even the *training* set: train WER stayed around 86% after 30–60 epochs, under several learning rates, dropouts and batch sizes.
   - The Transformer reached about 6% train WER.
   - The BiLSTM is kept as a backbone ablation (`*_lstm`).
3. **Per-sequence temporal centring of the RGB embeddings.** Consecutive DINOv2 frame embeddings have a mean cosine similarity of 0.977, so they are dominated by static appearance. Without centring, no model fit the training set.
4. **Added the `sentcls` diagnostic (closed-set sentence classifier)** to measure how much of the WER on the standard protocols can be achieved by sentence identification alone. It is not a candidate for the strong baseline.
5. **Expectation recorded before testing.** A linear probe on pooled features reaches only 10–20% sentence accuracy on an unseen validation signer (chance 1%). Absolute signer-independent WERs are therefore expected to be high (≈ 75–90%). Differences between fusion methods at that level may be small relative to noise.

---
## Amendment 2: written 2026-10-10, AFTER the Stage-1 test results had been seen (post hoc)
- **Hand-appearance stream added.**
  - Stage 1 showed the whole-frame DINOv2 embedding carries little sign information. A hand-appearance stream was therefore added: DINOv2 CLS embeddings of two 112-px hand crops located from the pose wrist and finger points (`hrgb`; configs `Q_*`).
  - The decision was made after seeing test results, so the `Q_*` comparisons are **exploratory**, not confirmatory.

## Amendment 3: deviations forced by compute, recorded 2026-10-10
- **Fewer seeds.** 3 seeds were run instead of the planned 5 (10 for the primary pair). `Q_hrgb_pose_hand_trel` has complete LOSO coverage for only 2 seeds.
- **No image corruptions.** The 9 image-level corruptions were never extracted. Robustness is limited to the feature-level conditions (missing modality, 2× faster, 1.5× slower).
- **Grid incomplete.** Not run: cross-domain (DSLR→phone), unseen-sentence, TwoStream, H/I_seq/J/K ablations, and the BiLSTM backbone controls.
- **Primary test unchanged.** The primary comparison and its tests were run exactly as specified, only with 3 seeds.
- **Exploratory demo studies.** The models trained on ISL-CSLTR *videos* (40 sentences) and on INCLUDE (24 signs) are exploratory and not part of this pre-registered study.
