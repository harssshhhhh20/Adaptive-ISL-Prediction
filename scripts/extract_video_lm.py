"""Dense MediaPipe pose + hand landmarks from the downloaded sentence videos (tracking/VIDEO mode),
every `--stride`-th frame. Writes features/videos_lm.pkl  {"sentence|video": {...}}

The record format matches scripts/extract_features.py so src/data.preprocess() can be reused.
"""
import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "data" / "videos"


class Tracker:
    """Frame-by-frame MediaPipe pose + hand tracking (VIDEO mode), used for offline extraction
    and by the live demo app so both see identical landmarks."""

    def __init__(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        from extract_features import quality
        self.mp, self.quality = mp, quality
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ROOT / "models" / "pose_landmarker_full.task")),
            running_mode=vision.RunningMode.VIDEO, num_poses=1))
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ROOT / "models" / "hand_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO, num_hands=2, min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3, min_tracking_confidence=0.3))

    def __call__(self, bgr, ts_ms):
        """Returns (pose [33,4], hands [2,21,3], hand_score [2], quality [8]) for one frame."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        pr = self.pose.detect_for_video(im, ts_ms)
        p = np.full((33, 4), np.nan, np.float32)
        if pr.pose_landmarks:
            p[:] = [[l.x, l.y, l.z, l.visibility] for l in pr.pose_landmarks[0]]
        hr = self.hands.detect_for_video(im, ts_ms)
        h = np.full((2, 21, 3), np.nan, np.float32)
        hs = np.zeros(2, np.float32)
        cands = sorted(((hd[0].score, np.array([[l.x, l.y, l.z] for l in lms], np.float32))
                        for lms, hd in zip(hr.hand_landmarks, hr.handedness)), key=lambda c: c[1][:, 0].mean())
        if len(cands) == 2:
            for j, (sc, arr) in enumerate(cands):
                h[j], hs[j] = arr, sc
        elif len(cands) == 1:
            sc, arr = cands[0]
            mid = np.nanmean(p[[11, 12], 0]) if not np.isnan(p[11, 0]) else 0.5
            j = 0 if arr[:, 0].mean() < mid else 1
            h[j], hs[j] = arr, sc
        return p, h, hs, self.quality(rgb, p, hs)


def process(path, stride, tracker=None):
    tracker = tracker or Tracker()
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    P, H, HS, Q, k = [], [], [], [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if k % stride == 0:
            p, h, hs, q = tracker(bgr, int(k * 1000 / fps))
            P.append(p); H.append(h); HS.append(hs); Q.append(q)
            size = (bgr.shape[1], bgr.shape[0])
        k += 1
    cap.release()
    return dict(pose=np.stack(P), hands=np.stack(H), hand_score=np.stack(HS), quality=np.stack(Q), size=size,
                fps=fps / stride, n_video_frames=k)


def _job(args):
    sid, path, stride = args
    return sid, process(path, stride)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--videos", default=str(VIDEOS))
    ap.add_argument("--out", default=str(ROOT / "features" / "videos_lm.pkl"))
    a = ap.parse_args()
    out_path = Path(a.out)
    out = pickle.load(open(out_path, "rb")) if out_path.exists() else {}
    jobs = []
    vroot = Path(a.videos)
    for p in sorted(vroot.rglob("*.mp4")):
        rel = p.relative_to(vroot)
        sid = f"{'/'.join(rel.parts[:-1])}|{p.stem}"           # "<sentence or category/sign>|<video>"
        if sid not in out:
            jobs.append((sid, p, a.stride))
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for k, (sid, rec) in enumerate(ex.map(_job, jobs), 1):
            out[sid] = rec
            if k % 20 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)}  {(time.time() - t0) / k:.1f}s/video", flush=True)
                pickle.dump(out, open(out_path, "wb"))
    pickle.dump(out, open(out_path, "wb"))
    print("done", len(out))


if __name__ == "__main__":
    main()
