"""Verify the folder-number -> signer mapping using appearance embeddings.

Each sequence's mean DINOv2 embedding is compared with per-folder centroids (computed without the
sequence itself). Sequences closer to another folder's centroid are flagged and rendered on a
contact sheet next to reference frames of each signer for manual confirmation. The confirmed
mapping is written to data/signer_map.json (sequence id -> signer id).
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

with open(ROOT / "features" / "clean.pkl", "rb") as f:
    feats = pickle.load(f)
ids = sorted(feats)
X = np.stack([feats[i]["rgb"][:, :384].astype(np.float32).mean(0) for i in ids])
X /= np.linalg.norm(X, axis=1, keepdims=True)
folder = np.array([i.split("|")[1] for i in ids])
labels = sorted(set(folder))

pred = []
for k in range(len(ids)):
    cents = []
    for lab in labels:
        m = (folder == lab)
        m[k] = False
        c = X[m].mean(0)
        cents.append(c / np.linalg.norm(c))
    pred.append(labels[int(np.argmax(np.stack(cents) @ X[k]))])
pred = np.array(pred)
flag = [k for k in range(len(ids)) if pred[k] != folder[k]]
print(f"{len(flag)} / {len(ids)} sequences closer to another folder's centroid")
for k in flag:
    print(f"  {ids[k]}  -> looks like signer {pred[k]}")

# contact sheet: one row per flagged sequence, then a reference frame of each signer
frames = ROOT / "data" / "frames"
src = {i: i for i in ids}
sent_dir = {d.lower().strip(): d for d in os.listdir(frames) if (frames / d).is_dir()}


def frame_of(sid):
    sent, fol = sid.split("|")
    d = frames / sent_dir[sent] / fol
    fs = sorted(os.listdir(d))
    return Image.open(d / fs[len(fs) // 2])


W = 110
ref = {lab: frame_of(ids[int(np.where(folder == lab)[0][0])]) for lab in labels}
rows = [ (ids[k], pred[k]) for k in flag ]
sheet = Image.new("RGB", (W * (len(labels) + 1), W * (len(rows) + 1)), "white")
for c, lab in enumerate(labels):
    im = ref[lab].copy(); im.thumbnail((W, W)); sheet.paste(im, ((c + 1) * W, 0))
    ImageDraw.Draw(sheet).text(((c + 1) * W + 3, 3), f"S{lab}", fill="red")
for r, (sid, p) in enumerate(rows, 1):
    im = frame_of(sid); im.thumbnail((W, W)); sheet.paste(im, (0, r * W))
    ImageDraw.Draw(sheet).text((3, r * W + 3), f"{sid.split('|')[1]}->{p}", fill="red")
out = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results" / "signer_check.jpg")
os.makedirs(os.path.dirname(out), exist_ok=True)
sheet.save(out)
json.dump({"flagged": [[ids[k], str(pred[k])] for k in flag]}, open(ROOT / "results" / "signer_flags.json", "w"), indent=1)
print("sheet:", out)
