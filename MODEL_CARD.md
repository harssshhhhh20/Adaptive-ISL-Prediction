# Model card: continuous ISL recognisers (landmark CTC models)

## Model details
- **Pipeline:**
  1. MediaPipe Pose (full) and Hands, at 12.5 frames/s.
  2. Body-normalised features: 25 upper-body points, plus 2 × 21 hand points with global and hand-local coordinates.
  3. A per-modality encoder (LayerNorm → Linear → temporal conv).
  4. Fusion, one of two kinds:
     - **fixed:** concatenation and a linear projection;
     - **adaptive (`trel_gate`):** per-frame softmax weights computed by a BiGRU over the features plus 10 observable quality signals.
  5. Temporal conv, then a 2-layer pre-norm Transformer (d = 192, 4 heads).
  6. CTC head with greedy decoding. Training adds auxiliary per-modality CTC heads.
- **Size:** about 1.3 M parameters (pose + hand) and about 45–67 MFLOPs per sentence. CPU inference takes 2–10 ms per sentence.
- **Checkpoints:**
  - `models/include/{trel_gate,concat}_s{0,1,2}.pt`: 24 INCLUDE signs.
  - `models/demo/{adaptive,fixed}_s{0,1,2}.pt`: 40 ISL-CSLTR sentences, 55 glosses.
- **Training:** AdamW, OneCycle schedule (lr 3e-3), dropout 0.3, and feature-level augmentation (speed 0.8–1.2×, 10 % frame drop, landmark noise, modality dropout p = 0.15). The checkpoint with the best validation WER is kept.

## Intended use
Research and teaching demonstrations of continuous sign recognition over a **closed, small vocabulary**.

**Not intended** for interpreting real conversations, for accessibility services, or for any decision affecting Deaf people.

## Evaluation (held-out test data, mean ± SD over 3 seeds)
| Model | Data / split | Metric | Fixed fusion | Adaptive fusion |
|---|---|---|---:|---:|
| INCLUDE, 24 signs | official test clips (87), isolated | sign accuracy | **93.1 ± 2.0 %** | 88.2 ± 1.3 % |
| INCLUDE, 24 signs | 400 chains of test clips | WER ↓ | **7.1 ± 1.2** | 11.4 ± 2.1 |
| INCLUDE, 24 signs | 400 chains of test clips | exact sequence | **82.0 %** | 71.7 % |
| ISL-CSLTR, 40 sentences | 1 held-out video per sentence (40) | WER ↓ | **64.6 ± 4.8** | 71.6 ± 2.9 |
| ISL-CSLTR, 40 sentences | same | closed-vocabulary sentence accuracy | 30.8 % | 30.0 % |
| ISL-CSLTR, 97 sentences (research study, key-frames, RGB + pose + hand) | leave-one-signer-out | WER ↓ | 84.6 ± 1.4 | 84.9 ± 2.4 (always outputting "I YOU" scores 82.5) |

## Limitations and known failure modes
- **INCLUDE sequences are artificial.** They are joined isolated signs with no co-articulation between them, and live webcam accuracy on new users has **not been measured**.
- **Unseen people.** Recognition of signers absent from training is poor on natural sentences: no better than a constant output on the 97-sentence corpus.
- **Common errors:**
  - pronoun confusions (YOU↔I, ME↔YOU);
  - dropped glosses, especially sentence-final ones;
  - glosses that occur in only one training sentence are recalled at 7–13 %.
- **Hand tracking:** MediaPipe finds at least one hand in only about 70 % of DSLR frames, because hands at rest leave the frame and fast motion blurs them.
- **Adaptive fusion** did not outperform fixed fusion in any experiment. Its weights collapsed onto a single modality in 3 of 7 study folds.
- **Population:** trained on a few Deaf students and volunteers from South India, so it may not transfer to other ISL dialects, signing styles, ages or camera setups.

## Ethical considerations
The video data show identifiable people and is licensed CC BY 4.0 for research. Do not present this system as a working sign-language interpreter. Deaf signers were not involved in design or evaluation; see `DATA.md`.
