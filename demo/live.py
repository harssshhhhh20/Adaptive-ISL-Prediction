"""Live ISL sign recognition from the webcam.

Vocabulary: the 50 INCLUDE-50 signs (default) or the 24-sign demo set (--vocab 24). The models were
trained on ISOLATED sign clips (INCLUDE); with CTC they can also decode several signs signed in a
row, but natural continuous signing (with co-articulation) was not part of training, and live
accuracy on new users has not been measured.

    python demo/live.py                 # default camera, 50 signs
    python demo/live.py --vocab 24      # 24-sign model
    python demo/live.py --camera 1      # another camera
    python demo/live.py --camera clip.mp4   # a video file treated as a live stream (testing)

How to use: raise your hands into view and sign one (or a few) of the listed signs, then lower your
hands. Recognition starts automatically when hands are raised and finishes ~0.8 s after they drop.
Press L to watch a reference clip of a sign ([ / ] to change sign) and copy it.

Keys:  SPACE start/stop manually | L learn panel | [ ] previous/next reference sign |
       F fixed/adaptive fusion | C clear history | Q quit
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "scripts", "demo"):
    sys.path.insert(0, str(ROOT / p))
from app import (C_ACC, C_BG, C_HAND, C_MUT, C_OK, C_PANEL, C_POSE, C_TXT, FONT, Include50Recogniser,  # noqa: E402
                 IncludeRecogniser, decode, draw_landmarks, text)
from extract_video_lm import Tracker  # noqa: E402
from train_include import load_clips  # noqa: E402

PROC_SIZE = (640, 360)       # INCLUDE videos were processed at 640x360 (16:9)
SAMPLE_MS = 80               # 12.5 landmark frames per second, as in training
PRE_ROLL, END_GAP, MIN_LEN, MAX_LEN = 6, 10, 8, 160
W, H = 1240, 760
VW, VH = 860, 484


def hands_raised(p, h):
    """A hand is visible and its wrist is above roughly the bottom of the ribcage."""
    present = [j for j in range(2) if not np.isnan(h[j, 0, 0])]
    if not present:
        return False
    if np.isnan(p[[11, 12], 1]).any():
        return True
    sh_y = np.nanmean(p[[11, 12], 1])
    sw = abs(p[11, 0] - p[12, 0])
    return any(h[j, 0, 1] < sh_y + 1.6 * sw for j in present)


class Reference:
    """Looping reference clips of each sign, from the TRAINING split."""

    def __init__(self, vocab, train_clips):
        self.clips = {}
        for sid, g, _ in train_clips:
            self.clips.setdefault(g, sid)
        self.signs = [g for g in vocab if g in self.clips]
        self.i, self.cap, self.cur = 0, None, None

    def frame(self):
        g = self.signs[self.i]
        if self.cur != g or self.cap is None:
            if self.cap:
                self.cap.release()
            sid = self.clips[g]
            self.cap = cv2.VideoCapture(str(ROOT / "data" / "include" / "videos" / sid.split("|")[0] /
                                            (sid.split("|")[1] + ".mp4")))
            self.cur = g
        ok, f = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
        return g, f

    def step(self, d):
        self.i = (self.i + d) % len(self.signs)


def chips(img, glosses, x, y, scale, color):
    for g in glosses:
        (tw, th), _ = cv2.getTextSize(g, FONT, scale, 2)
        if x + tw > VW:
            text(img, "...", (x, y), scale, color, 2)
            break
        cv2.rectangle(img, (x - 6, y - th - 8), (x + tw + 6, y + 8), C_PANEL, -1)
        text(img, g, (x, y), scale, color, 2)
        x += tw + 18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="0")
    ap.add_argument("--fusion", choices=["trel_gate", "concat"], default=None)
    ap.add_argument("--vocab", choices=["50", "24"], default="50")
    ap.add_argument("--snapshots", help="headless test: save frames here instead of opening a window")
    a = ap.parse_args()
    torch.set_num_threads(4)
    src = int(a.camera) if a.camera.isdigit() else a.camera
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Could not open camera/video {a.camera!r}. On macOS allow camera access for your "
                 f"terminal app in System Settings > Privacy & Security > Camera.")
    is_file = not isinstance(src, int)
    if a.vocab == "50":
        from train_include50 import load as load50
        Rec, train_clips = Include50Recogniser, load50()[0]["train"]
    else:
        Rec, train_clips = IncludeRecogniser, load_clips()["train"]
    rec = Rec(a.fusion)
    ref = Reference(rec.itos[1:], train_clips)
    tracker = Tracker()
    t_start = time.time()

    state = "idle"                       # idle -> signing -> idle
    ring = deque(maxlen=PRE_ROLL)        # recent frames kept so the start of a sign is not lost
    buf, gap, live, weights = [], 0, [], None
    history = deque(maxlen=6)
    manual, show_ref, last_sample, fps, t_prev = False, True, -1e9, 0.0, time.time()
    snap_k, frame_no = 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            if is_file:
                break
            continue
        frame_no += 1
        frame = cv2.resize(frame, PROC_SIZE)
        now_ms = int((frame_no * 1000 / (cap.get(cv2.CAP_PROP_FPS) or 25)) if is_file else (time.time() - t_start) * 1000)
        p, h, hs, q = tracker(frame, now_ms)
        sample = now_ms - last_sample >= SAMPLE_MS
        if sample:
            last_sample = now_ms
            fr = (p, h, hs, q)
            active = hands_raised(p, h) or manual
            if state == "idle":
                ring.append(fr)
                if active and sum(hands_raised(x[0], x[1]) for x in list(ring)[-3:]) >= 2 or manual:
                    state, buf, gap, live = "signing", list(ring), 0, []
            else:
                buf.append(fr)
                gap = 0 if active else gap + 1
                if len(buf) % 3 == 0 and len(buf) >= 4:
                    live, weights, _ = decode(rec, *[[x[i] for x in buf] for i in range(4)], PROC_SIZE)
                if (gap >= END_GAP and not manual) or len(buf) >= MAX_LEN:
                    if len(buf) - gap >= MIN_LEN:
                        final, weights, post = decode(rec, *[[x[i] for x in buf] for i in range(4)], PROC_SIZE)
                        if len(final) <= 1 and getattr(rec, "classify", None):
                            top = rec.classify(post, len(buf))    # single attempt: most likely sign
                            final = [f"{top[0][0]} ({100 * top[0][1]:.0f}%)"] if final or top[0][1] > 0.5 else final
                        history.appendleft((time.strftime("%H:%M:%S"), final))
                    state, ring, live = "idle", deque(maxlen=PRE_ROLL), []

        # ---------------------------------------------------------------- drawing
        dt = time.time() - t_prev
        t_prev = time.time()
        fps = 0.9 * fps + 0.1 / max(dt, 1e-3) if fps else 1 / max(dt, 1e-3)
        img = np.full((H, W, 3), C_BG, np.uint8)
        v = frame.copy()
        draw_landmarks(v, p, h)
        v = cv2.flip(cv2.resize(v, (VW, VH)), 1)            # mirror for a natural selfie view
        img[10:10 + VH, 10:10 + VW] = v
        if state == "signing":
            cv2.circle(img, (34, 36), 10, (60, 60, 230), -1)
            text(img, "SIGNING" + (" (manual)" if manual else ""), (52, 43), 0.7, (230, 230, 255), 2)
        else:
            text(img, "Raise your hands and sign", (24, 43), 0.7, C_TXT, 2)
        y0 = VH + 40
        text(img, "LIVE HYPOTHESIS" if state == "signing" else "LAST RESULT", (14, y0), 0.5, C_MUT)
        shown = live if state == "signing" else (history[0][1] if history else [])
        chips(img, shown or ["..."], 20, y0 + 44, 1.05, C_TXT if state == "signing" else C_OK)
        text(img, "History:", (14, y0 + 90), 0.5, C_MUT)
        for k, (ts, g) in enumerate(list(history)[1:4]):
            text(img, f"{ts}  {' '.join(g) or '(nothing recognised)'}", (90 + 0 * k, y0 + 90 + 24 * k), 0.5, C_TXT)

        x = VW + 30
        text(img, "Live ISL Sign Recognition", (x, 38), 0.72, C_TXT, 2)
        text(img, f"{len(rec.itos) - 1}-sign vocabulary, trained on isolated clips;", (x, 60), 0.42, C_MUT)
        text(img, "sign one sign, or a few in a row", (x, 78), 0.42, C_MUT)
        text(img, rec.stats[1].strip() + " (held-out test clips)", (x, 102), 0.42, C_ACC)
        text(img, "Live accuracy on you: not evaluated", (x, 122), 0.42, C_MUT)
        if weights is not None:
            for j, (m, col) in enumerate(zip(["pose", "hand"], [C_POSE, C_HAND])):
                yy = 146 + 22 * j
                cv2.rectangle(img, (x, yy), (x + int(150 * float(weights[j])), yy + 14), col, -1)
                cv2.rectangle(img, (x, yy), (x + 150, yy + 14), C_MUT, 1)
                text(img, f"{m} {float(weights[j]):.2f}", (x + 158, yy + 12), 0.42, C_TXT)
        if show_ref:
            g, rf = ref.frame()
            if rf is not None:
                rh = int(rf.shape[0] * 300 / rf.shape[1])
                img[196:196 + rh, x:x + 300] = cv2.resize(rf, (300, rh))
                text(img, f"Reference: {g}   ([ / ] to change)", (x, 196 + rh + 20), 0.48, C_ACC, 1)
            ytxt = 196 + 169 + 46
        else:
            ytxt = 210
        text(img, "Signs it knows:", (x, ytxt), 0.47, C_TXT)
        words = rec.itos[1:]
        per = 4 if len(words) > 30 else 3
        for k in range(0, len(words), per):
            text(img, "  ".join(words[k:k + per]), (x, ytxt + 20 + 17 * (k // per)), 0.36, C_MUT)
        text(img, f"{fps:4.1f} fps", (x, H - 52), 0.42, C_MUT)
        text(img, "SPACE start/stop  L learn  [ ] sign  F fusion  C clear  Q quit", (x, H - 28), 0.4, C_MUT)

        if a.snapshots:
            if frame_no % 15 == 0:
                Path(a.snapshots).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(a.snapshots) / f"live_{snap_k:03d}.png"), img)
                snap_k += 1
            continue
        cv2.imshow("Live ISL Recognition", img)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            manual = not manual
            if not manual and state == "signing":
                gap = END_GAP                     # finish on next sample
        elif key == ord("l"):
            show_ref = not show_ref
        elif key == ord("]"):
            ref.step(1)
        elif key == ord("["):
            ref.step(-1)
        elif key == ord("c"):
            history.clear()
        elif key == ord("f"):
            rec = Rec("concat" if rec.fusion == "trel_gate" else "trel_gate")
    if a.snapshots:
        print("history:", list(history))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
