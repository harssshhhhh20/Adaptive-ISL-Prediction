# Data statement

Both datasets are public and licensed **CC BY 4.0**. You may redistribute and publish with attribution: cite both papers in `references.bib` in any publication. This repository does **not** redistribute the raw videos; the scripts download them from the original hosts. The small processed files kept here (landmarks, shrunk 640-px videos) are derivatives under the same licence.

## 1. ISL-CSLTR (sentence-level ISL)
- **Source:** Elakkiya R. & Natarajan B., Mendeley Data, DOI [10.17632/kcmpdxky7p.1](https://data.mendeley.com/datasets/kcmpdxky7p/1), 2021. CC BY 4.0. The archive is one 8.9 GB zip.
- **Content:**
  - 101 English sentences with one gloss string each (`data/raw_meta/ISL Corpus sign glosses.csv`).
  - 7 signers, recorded on 28-12-2019: 2 native signers from a school for the Deaf (Andhra Pradesh) and 5 student volunteers (SASTRA University).
  - 687 sentence videos and 18,863 sentence key-frames.
- **What we used:**
  - **Research study:** the sentence *key-frame* subset only, 663 sequences (median 24 frames) for 97 sentences. 4 of the 101 sentences have no frames. It was streamed by HTTP byte range (`scripts/fetch_frames.py`) and deleted after feature extraction; re-download it to reproduce.
  - **Demo:** 279 sentence *videos* for 40 sentences chosen for gloss overlap (`data/demo_sentences.txt`), fetched by byte range (`scripts/fetch_videos.py`) and stored at 640 px in `data/videos/`. One malformed 3-minute recording was excluded (`i do not like it (2)`, against a median of 4 s).
- **Signer identity:**
  - Frame folders 1–7 correspond to signers. Folders 1–2 are portrait phone recordings at 240×320; folders 3–7 are DSLR recordings at 1920×1080.
  - An appearance-embedding check (`scripts/verify_signers.py`) plus visual inspection found **one mislabelled sequence**: `do you need something|4` is signer 3. The correction is in `data/signer_map.json`.
  - Videos carry no signer labels.
- **Gloss normalisation** (`src/glosses.py`):
  - `DO NOT / DONOT / DONT → NOT`, `CONGRATULATIIONS → CONGRATULATIONS`, `(AGE) → AGE`, `XXXXXXXX → NAME`, and punctuation removed.
  - Result: 162 glosses and 338 tokens for the 97 sentences. 115 of the 162 glosses occur in exactly one sentence.
  - One folder name is misspelled (`which collegeschool are you from`) and is mapped to the table.
- **Known properties that limit what can be claimed:**
  - It is a closed set of 97 sentences, with each signer signing each sentence once (some phone recordings have extra takes).
  - Under any split, every test video is a person–sentence pair never seen in training.
  - Camera type is confounded with signer population: phone = native signers, DSLR = students.

## 2. INCLUDE (isolated ISL signs)
- **Source:** Sridhar, Ganesan, Kumar & Khapra, *INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition*, ACM Multimedia 2020, DOI 10.1145/3394171.3413528.
  - Data: Zenodo record [4010759](https://zenodo.org/records/4010759), CC BY 4.0.
  - Official splits: [AI4Bharat/INCLUDE](https://github.com/AI4Bharat/INCLUDE) `train_test_paths/`, copied to `data/include/include_{train,val,test}.txt`.
  - Full dataset: 263 signs, 4,292 videos, 56.8 GB, 1920×1080, Deaf students of St. Louis School for the Deaf, Chennai.
- **What we used:** 24 signs (`demo/train_include.py`: `PRON`, `GREET`, `ADJ`), 440 clips, fetched by byte range from 6 category zips (`scripts/fetch_include.py`) and stored at 640 px in `data/include/videos/`.
  - Splits: the **official** train/val/test split, restricted to these signs: 313 / 40 / 87 clips.
  - 3 lines in the official lists point to `Extra/` sub-folders outside our subset; they are ignored.
- **Continuous sequences:** made by joining isolated clips. Training chains use only training clips; test chains use only test clips. They lack natural co-articulation between signs.

## 3. Derived files kept in this repository
| Path | What | Rebuild with |
|---|---|---|
| `features/videos_lm.pkl` | MediaPipe pose + hand landmarks (12.5 fps) for the 279 ISL-CSLTR videos | `scripts/extract_video_lm.py` |
| `features/include_lm.pkl` | Same for the 440 INCLUDE clips | `scripts/extract_video_lm.py --videos data/include/videos --out features/include_lm.pkl` |
| `results/runs/**/seed*.json` | Per-run predictions and metrics of the research study (no checkpoints) | `scripts/run.py` |

## 4. Ethics and responsible use
- **Who was recorded:** the recordings show identifiable people, mostly Deaf signers and students. Use them only for research and education, and do not re-identify people.
- **Not an interpreter:** the models recognise a small closed vocabulary. They are **not** sign-language interpreters and must not be presented or deployed as an accessibility tool for Deaf users.
- **Deaf community involvement:** no Deaf signers or ISL linguists were involved in this project's design or evaluation. A publication should state this, and ideally fix it.
