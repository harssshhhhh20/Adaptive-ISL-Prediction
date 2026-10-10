"""Extract per-frame RGB embeddings, MediaPipe pose + hand landmarks and observable quality
signals for every ISL-CSLTR sentence sequence under one image condition.

Usage: python scripts/extract_features.py <condition> [--limit N]
Writes features/<condition>.pkl  {seq_id: {...}}
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from corruptions import CONDITIONS, SequenceCorruptor  # noqa: E402
from glosses import sentence_key  # noqa: E402

FRAMES = ROOT / "data" / "frames"
FEATS = ROOT / "features"
UPPER_BODY = list(range(0, 25))  # MediaPipe pose indices above the hips (+hips 23, 24)


def list_sequences():
    seqs = []
    for sent in sorted(os.listdir(FRAMES)):
        d = FRAMES / sent
        if not d.is_dir():
            continue
        for signer in sorted(os.listdir(d)):
            files = sorted(f for f in os.listdir(d / signer) if f.lower().endswith(".jpg"))
            if files:
                seqs.append((f"{sentence_key(sent)}|{signer}", [d / signer / f for f in files]))
    return seqs


def make_landmarkers(seg=False):
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "models" / "pose_landmarker_full.task")),
        running_mode=vision.RunningMode.IMAGE, num_poses=1, output_segmentation_masks=seg))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "models" / "hand_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE, num_hands=2, min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3))
    return pose, hands


def run_landmarks(pose, hands, rgb):
    im = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    pr = pose.detect(im)
    p = np.full((33, 4), np.nan, np.float32)
    if pr.pose_landmarks:
        p[:] = [[l.x, l.y, l.z, l.visibility] for l in pr.pose_landmarks[0]]
    hr = hands.detect(im)
    h = np.full((2, 21, 3), np.nan, np.float32)
    hs = np.zeros(2, np.float32)
    cands = []
    for lms, hd in zip(hr.hand_landmarks, hr.handedness):
        arr = np.array([[l.x, l.y, l.z] for l in lms], np.float32)
        cands.append((hd[0].category_name, hd[0].score, arr))
    # slot 0 = signer's left side of image (MediaPipe "Right" on a non-mirrored image), slot 1 = other
    cands.sort(key=lambda c: c[2][:, 0].mean())
    if len(cands) == 2:
        for slot, c in enumerate(cands):
            h[slot], hs[slot] = c[2], c[1]
    elif len(cands) == 1:
        # assign a single hand by which side of the body midline it is on
        c = cands[0]
        mid = np.nanmean(p[[11, 12], 0]) if not np.isnan(p[11, 0]) else 0.5
        slot = 0 if c[2][:, 0].mean() < mid else 1
        h[slot], hs[slot] = c[2], c[1]
    return p, h, hs, (pr.segmentation_masks[0].numpy_view() if pr.segmentation_masks else None)


class RGBEncoder:
    def __init__(self, device):
        import torch
        self.torch = torch
        os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "torch"))
        self.m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval().to(device)
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @staticmethod
    def square(img, size=224):
        h, w = img.shape[:2]
        s = max(h, w)
        canvas = np.zeros((s, s, 3), np.uint8)
        canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
        return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)

    def __call__(self, imgs):
        with self.torch.no_grad():
            return self._encode(imgs)

    def _encode(self, imgs):
        torch = self.torch
        x = torch.from_numpy(np.stack([self.square(i) for i in imgs])).to(self.device)
        x = (x.permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std
        out = []
        for i in range(0, len(x), 32):
            f = self.m.forward_features(x[i:i + 32])
            out.append(torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], 1))
        return torch.cat(out).cpu().numpy().astype(np.float16)


def quality(img, p, hs):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    nz = g[g > 3] if (g > 3).any() else g          # ignore letterbox bars
    vis = p[UPPER_BODY, 3]
    return np.array([
        nz.mean() / 255.0,                                    # brightness
        nz.std() / 128.0,                                     # contrast
        np.log1p(cv2.Laplacian(g, cv2.CV_32F).var()) / 10.0,  # sharpness
        0.0 if np.isnan(vis).all() else float(np.nanmean(vis)),  # pose visibility
        float(hs[0] > 0), float(hs[1] > 0), hs[0], hs[1],     # hand detection + confidence
    ], np.float32)


def crop_box(hand, w, h, pad=1.5, min_px=40):
    xs, ys = hand[:, 0] * w, hand[:, 1] * h
    cx, cy = xs.mean(), ys.mean()
    side = max(np.ptp(xs), np.ptp(ys), min_px / pad) * pad
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    return x0, y0, x0 + int(round(side)), y0 + int(round(side))


def crop(img, box):
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    out = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


# MediaPipe pose points of each hand: (wrist, pinky, index, thumb) for the signer's left / right hand
POSE_HAND = ([15, 17, 19, 21], [16, 18, 20, 22])


def pose_hand_box(p, idx, w, h):
    """Square box around one hand located from pose landmarks (available even when the hand
    landmarker misses the hand); side = 0.6 x shoulder width."""
    if np.isnan(p[idx, 0]).any() or np.isnan(p[[11, 12], 0]).any():
        return None
    pts = p[idx, :2] * [w, h]
    sw = np.linalg.norm((p[11, :2] - p[12, :2]) * [w, h])
    cx, cy = pts.mean(0)
    side = max(0.6 * sw, 32)
    return int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)


def hand_crops(cond, limit=0):
    """DINOv2 CLS embedding of a 112 px crop around each hand, located from that condition's own
    pose landmarks (wrist/finger points), on the same (possibly corrupted) frames a deployed
    system would see. Zeros if the pose was not found."""
    import torch
    torch.set_num_threads(int(os.environ.get("RGB_THREADS", 3)))
    enc = RGBEncoder("cpu")
    with open(FEATS / f"{cond}.pkl", "rb") as f:
        lmfeat = pickle.load(f)
    clean = None
    if cond in ("hand_occ", "combined"):
        with open(FEATS / "clean.pkl", "rb") as f:
            clean = pickle.load(f)
    seg_pose = make_landmarkers(seg=True)[0] if cond in ("background", "combined") else None
    seqs = list_sequences()[: limit or None]
    out, t0 = {}, time.time()
    for k, (sid, files) in enumerate(seqs):
        imgs = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files]
        h, w = imgs[0].shape[:2]
        lms = [{"hands": clean[sid]["hands"][t]} for t in range(len(imgs))] if clean is not None else None
        cor = SequenceCorruptor(cond, sid, len(imgs), h, w, lms)
        P = lmfeat[sid]["pose"]
        crops, slots = [], []
        for t, img in enumerate(imgs):
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            mask = None
            if seg_pose is not None:
                r = seg_pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
                mask = r.segmentation_masks[0].numpy_view() if r.segmentation_masks else np.ones((h, w), np.float32)
            x = cor(t, img, mask)
            for j, idx in enumerate(POSE_HAND):
                box = pose_hand_box(P[t], idx, w, h)
                if box is not None:
                    crops.append(cv2.resize(crop(x, box), (112, 112)))
                    slots.append((t, j))
        emb = np.zeros((len(imgs), 2, 384), np.float16)
        if crops:
            with torch.no_grad():
                xb = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float() / 255
                xb = (xb - enc.mean.cpu()) / enc.std.cpu()
                cls = torch.cat([enc.m.forward_features(xb[i:i + 64])["x_norm_clstoken"] for i in range(0, len(xb), 64)])
            for (t, j), e in zip(slots, cls.numpy()):
                emb[t, j] = e
        out[sid] = emb.reshape(len(imgs), -1)
        if k % 25 == 0:
            print(f"[{cond}/handrgb] {k + 1}/{len(seqs)}  {(time.time() - t0) / (k + 1):.2f}s/seq", flush=True)
    with open(FEATS / f"{cond}.handrgb.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"[{cond}/handrgb] wrote {len(out)} in {time.time() - t0:.0f}s")


def main():
    """Two stages so the memory-heavy RGB encoder runs in a single process:
    --stage lm   MediaPipe landmarks + quality signals (shardable, no PyTorch)
    --stage rgb  DINOv2 embeddings of the identically corrupted frames
    --stage merge  combine both into features/<cond>.pkl"""
    ap = argparse.ArgumentParser()
    ap.add_argument("condition", choices=CONDITIONS)
    ap.add_argument("--stage", choices=["lm", "rgb", "merge", "handrgb"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    cond = args.condition
    FEATS.mkdir(exist_ok=True)

    if args.stage == "handrgb":
        return hand_crops(cond, args.limit)

    if args.stage == "merge":
        lm = {}
        for f in sorted(FEATS.glob(f"{cond}.lm.part*.pkl")):
            with open(f, "rb") as fh:
                lm.update(pickle.load(fh))
        with open(FEATS / f"{cond}.rgb.pkl", "rb") as fh:
            rgb = pickle.load(fh)
        assert set(lm) == set(rgb), (len(lm), len(rgb))
        for k in lm:
            lm[k]["rgb"] = rgb[k]
        with open(FEATS / f"{cond}.pkl", "wb") as fh:
            pickle.dump(lm, fh)
        for f in list(FEATS.glob(f"{cond}.lm.part*.pkl")) + [FEATS / f"{cond}.rgb.pkl"]:
            f.unlink()
        print(f"[{cond}] merged {len(lm)} sequences")
        return

    clean = None
    if cond in ("hand_occ", "combined"):
        with open(FEATS / "clean.pkl", "rb") as f:
            clean = pickle.load(f)
    seg_pose = make_landmarkers(seg=True)[0] if cond in ("background", "combined") else None
    if args.stage == "lm":
        pose, hands = make_landmarkers()
        out_path = FEATS / f"{cond}.lm.part{args.shard}of{args.nshards}.pkl"
    else:
        import torch
        torch.set_num_threads(int(os.environ.get("RGB_THREADS", 3)))
        enc = RGBEncoder(os.environ.get("RGB_DEVICE", "cpu"))
        out_path = FEATS / f"{cond}.rgb.pkl"

    seqs = list_sequences()
    if args.limit:
        seqs = seqs[:args.limit]
    if args.stage == "lm":
        seqs = seqs[args.shard::args.nshards]
    out, t0 = {}, time.time()
    for k, (sid, files) in enumerate(seqs):
        imgs = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files]
        h, w = imgs[0].shape[:2]
        lms = None
        if clean is not None:
            c = clean[sid]
            lms = [{"hands": c["hands"][t]} for t in range(len(imgs))]
        cor = SequenceCorruptor(cond, sid, len(imgs), h, w, lms)
        P, H, HS, Q, seen = [], [], [], [], []
        for t, img in enumerate(imgs):
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            mask = None
            if seg_pose is not None:
                r = seg_pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
                mask = r.segmentation_masks[0].numpy_view() if r.segmentation_masks else np.ones((h, w), np.float32)
            x = cor(t, img, mask)
            if args.stage == "lm":
                p, hh, hs, _ = run_landmarks(pose, hands, x)
                P.append(p); H.append(hh); HS.append(hs); Q.append(quality(x, p, hs))
            else:
                seen.append(x)
        if args.stage == "lm":
            out[sid] = {"pose": np.stack(P), "hands": np.stack(H), "hand_score": np.stack(HS),
                        "quality": np.stack(Q), "size": (w, h)}
        else:
            out[sid] = enc(seen)
        if k % 25 == 0:
            el = time.time() - t0
            print(f"[{cond}/{args.stage}] {k + 1}/{len(seqs)}  {el / (k + 1):.2f}s/seq", flush=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"[{cond}/{args.stage}] wrote {len(out)} sequences to {out_path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
