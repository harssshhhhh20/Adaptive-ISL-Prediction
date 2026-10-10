"""Live OpenCV demo: continuous ISL recognition running frame-by-frame on held-out test videos.

    python demo/app.py                    # INCLUDE sign sequences (default)
    python demo/app.py --dataset sentences  # ISL-CSLTR natural sentences
    python demo/app.py --video my.mp4     # any video file

Keys:  SPACE pause | N next | P previous | R replay | D switch dataset | F fixed/adaptive fusion | Q quit

Every video shown is from the held-out TEST split (never used for training or model selection).
MediaPipe tracking and the recogniser run live while the video plays; the gloss line is the model's
current hypothesis for what has been signed so far, and is final when the video ends.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "scripts", "demo"):
    sys.path.insert(0, str(ROOT / p))
from data import Vocab, collate, preprocess  # noqa: E402
from extract_video_lm import Tracker  # noqa: E402
from metrics import align_counts  # noqa: E402
from models import build_model, posteriors  # noqa: E402

STRIDE = 2
# Demo sign sequences, built from held-out TEST clips only (clip choice fixed by seed 7).
SEQUENCES = [["HELLO", "HOW-ARE-YOU"], ["GOOD-MORNING", "I", "HAPPY"], ["SHE", "BEAUTIFUL"], ["THANK-YOU"],
             ["HE", "DEAF"], ["GOOD-NIGHT", "WE", "HAPPY"], ["THEY", "RICH"], ["I", "SAD"],
             ["HELLO", "YOU", "PLEASED"], ["GOOD-EVENING", "THEY", "POOR"], ["YOU", "QUIET"], ["IT", "LOUD"],
             ["GOOD-AFTERNOON", "ALRIGHT"], ["WE", "THANK-YOU"], ["HELLO", "SHE", "HAPPY", "THANK-YOU"]]
W, H = 1180, 680            # window
VW, VH = 800, 450           # video area
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
C_BG, C_PANEL, C_TXT, C_MUT = (21, 17, 15), (34, 27, 24), (237, 234, 232), (166, 160, 154)
C_OK, C_BAD, C_ACC, C_POSE, C_HAND = (138, 195, 76), (107, 107, 239), (255, 196, 124), (90, 200, 255), (80, 220, 120)


def load_state(path):
    sd = torch.load(path)
    return {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}


# ----------------------------------------------------------------------------- recognisers
class IncludeRecogniser:
    """Sign-sequence model trained on INCLUDE (official split)."""
    name = "INCLUDE sign sequences (24 signs)"

    def __init__(self, fusion=None):
        from train_include import VOCAB
        self.itos = VOCAB
        runs = [json.loads(p.read_text()) for p in sorted((ROOT / "results" / "include").glob("*.json"))]
        pool = [r for r in runs if fusion is None or r["fusion"] == fusion]
        best = min(pool, key=lambda r: r["val_wer"])              # chosen on validation only
        self.fusion = best["fusion"]
        self.model = build_model(len(VOCAB), ["pose", "hand"], best["fusion"])
        self.model.load_state_dict(load_state(ROOT / "models" / "include" / f"{best['fusion']}_s{best['seed']}.pt"))
        self.model.eval()
        same = [r for r in runs if r["fusion"] == self.fusion]
        self.stats = [f"Test results ({len(same)} seeds, held-out clips):",
                      f"  single-sign accuracy  {np.mean([100 - r['test_isolated']['wer'] for r in same]):.1f}%",
                      f"  sequence gloss acc.   {np.mean([100 - r['test_sequences']['wer'] for r in same]):.1f}%",
                      f"  exact sequences       {np.mean([r['test_sequences']['sent_acc'] for r in same]):.1f}%"]
        self.label = f"{'Adaptive' if self.fusion == 'trel_gate' else 'Fixed'} fusion (seed {best['seed']}, best on validation)"

    def items(self):
        from train_include import load_clips
        test_by = {}
        for sid, g, _ in load_clips()["test"]:
            test_by.setdefault(g, []).append(sid)
        rng = np.random.default_rng(7)
        out = []
        for seq in SEQUENCES:
            seq = [g for g in seq if g in test_by]
            sids = [test_by[g][rng.integers(len(test_by[g]))] for g in seq]
            paths = [ROOT / "data" / "include" / "videos" / s.split("|")[0] / (s.split("|")[1] + ".mp4") for s in sids]
            out.append(dict(paths=paths, ref=seq, title=" ".join(seq)))
        return out

    def sentence(self, post, n):
        return None


class SentenceRecogniser:
    """Natural-sentence model trained on ISL-CSLTR videos (40 sentences)."""
    name = "ISL-CSLTR natural sentences (40)"

    def __init__(self, fusion=None):
        from train_demo import CONFIGS, constrained, demo_table
        self.table, self.constrained = demo_table(), constrained
        self.vocab = Vocab(self.table)
        self.itos = self.vocab.itos
        self.metrics = json.loads((ROOT / "results" / "demo" / "metrics.json").read_text())
        s = self.metrics["summary"]
        names = [m for m in ("adaptive", "fixed") if fusion is None or CONFIGS[m]["fusion"] == fusion]
        cfg_name = min(names, key=lambda m: self.metrics["runs"][s[m]["best_seed_by_val"]]["val_wer"])
        run = s[cfg_name]["best_seed_by_val"]
        cfg = CONFIGS[cfg_name]
        self.fusion = cfg["fusion"]
        self.model = build_model(len(self.vocab), cfg["mods"], cfg["fusion"])
        self.model.load_state_dict(load_state(ROOT / "models" / "demo" / f"{run}.pt"))
        self.model.eval()
        m = s[cfg_name]
        self.stats = [f"Test results (3 seeds, held-out videos):",
                      f"  gloss accuracy        {100 - m['test_wer'][0]:.1f}%",
                      f"  exact sentences       {m['test_sent_acc'][0]:.1f}%",
                      f"  closed-vocab sentence {m['closed_vocab_sent_acc'][0]:.1f}%",
                      "  (each test video = a person the model",
                      "   never saw signing that sentence)"]
        self.label = f"{cfg_name.capitalize()} fusion ({run.split('_')[1]}, best on validation)"

    def items(self):
        out = []
        for sid in self.metrics["split"]["test"]:
            sent, name = sid.split("|")
            out.append(dict(paths=[ROOT / "data" / "videos" / sent / f"{name}.mp4"], ref=self.table[sent], title=f'"{sent}"'))
        return out

    def sentence(self, post, n):
        return self.constrained(post, n, self.table, self.vocab)


# ----------------------------------------------------------------------------- inference
@torch.no_grad()
def decode(rec, P, Hh, HS, Q, size):
    r = dict(pose=np.stack(P), hands=np.stack(Hh), hand_score=np.stack(HS), quality=np.stack(Q), size=size,
             rgb=np.zeros((len(P), 768), np.float16))
    f = preprocess(r)
    out = rec.model(collate([("x", f, [1])]))
    post = posteriors(out)[0]
    ids, prev, gl = post.argmax(-1).tolist(), 0, []
    for t, i in enumerate(ids):
        if i != prev and i != 0:
            gl.append(rec.itos[i])
        prev = i
    w = out["weights"][0, -1].numpy() if out["weights"] is not None else None
    return gl, w, post


# ----------------------------------------------------------------------------- drawing
def text(img, s, xy, scale=0.55, color=C_TXT, thick=1):
    cv2.putText(img, s, xy, FONT, scale, color, thick, cv2.LINE_AA)


def draw_landmarks(img, p, h):
    hh, ww = img.shape[:2]
    if not np.isnan(p[0, 0]):
        for a, b in POSE_EDGES:
            if not np.isnan(p[[a, b], 0]).any():
                cv2.line(img, (int(p[a, 0] * ww), int(p[a, 1] * hh)), (int(p[b, 0] * ww), int(p[b, 1] * hh)), C_POSE, 2, cv2.LINE_AA)
    for j, col in enumerate([C_HAND, (240, 140, 60)]):
        if np.isnan(h[j, 0, 0]):
            continue
        for a, b in HAND_EDGES:
            cv2.line(img, (int(h[j, a, 0] * ww), int(h[j, a, 1] * hh)), (int(h[j, b, 0] * ww), int(h[j, b, 1] * hh)), col, 1, cv2.LINE_AA)
        for x, y, _ in h[j]:
            cv2.circle(img, (int(x * ww), int(y * hh)), 2, col, -1, cv2.LINE_AA)


def fit(frame, w, h):
    fh, fw = frame.shape[:2]
    s = min(w / fw, h / fh)
    img = cv2.resize(frame, (int(fw * s), int(fh * s)))
    canvas = np.zeros((h, w, 3), np.uint8)
    y0, x0 = (h - img.shape[0]) // 2, (w - img.shape[1]) // 2
    canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return canvas


def gloss_chips(img, glosses, ref, x, y, scale=0.9, final=False):
    """Draw glosses as chips; when final, colour by correctness against the reference."""
    remaining = list(ref)
    for g in glosses:
        col = C_TXT
        if final:
            col = C_OK if g in remaining else C_BAD
            if g in remaining:
                remaining.remove(g)
        (tw, th), _ = cv2.getTextSize(g, FONT, scale, 2)
        if x + tw + 20 > VW + 10:
            break
        cv2.rectangle(img, (x - 6, y - th - 8), (x + tw + 6, y + 8), C_PANEL, -1)
        text(img, g, (x, y), scale, col, 2)
        x += tw + 18


def render(state, rec, frame, lm, glosses, weights, item, idx, n_items, fps, final=None, tally=None):
    img = np.full((H, W, 3), C_BG, np.uint8)
    v = fit(frame, VW, VH)
    if lm is not None:
        sub = v.copy()
        fh, fw = frame.shape[:2]
        s = min(VW / fw, VH / fh)
        ox, oy = (VW - int(fw * s)) // 2, (VH - int(fh * s)) // 2
        roi = sub[oy:oy + int(fh * s), ox:ox + int(fw * s)]
        draw_landmarks(roi, *lm)
        v = sub
    img[10:10 + VH, 10:10 + VW] = v
    # --- bottom panel: recognition
    y0 = VH + 30
    text(img, "LIVE RECOGNITION" if final is None else "FINAL OUTPUT", (14, y0), 0.5, C_MUT)
    gloss_chips(img, glosses or ["..."], item["ref"], 20, y0 + 42, 1.0, final=final is not None)
    text(img, "Ground truth:", (14, y0 + 90), 0.5, C_MUT)
    text(img, " ".join(item["ref"]), (130, y0 + 90), 0.6, C_TXT, 1)
    if final is not None:
        e = sum(align_counts(item["ref"], glosses))
        ok = glosses == item["ref"]
        verdict = "EXACT MATCH" if ok else f"{e} error(s) / {len(item['ref'])} glosses"
        text(img, verdict, (14, y0 + 125), 0.6, C_OK if ok else C_BAD, 2)
        if final.get("sentence"):
            text(img, f'Closest known sentence: "{final["sentence"]}"', (260, y0 + 125), 0.55, C_ACC, 1)
    # --- right panel: info
    x = VW + 30
    text(img, "Continuous ISL Recognition", (x, 40), 0.7, C_TXT, 2)
    text(img, rec.name, (x, 68), 0.5, C_ACC)
    text(img, rec.label, (x, 92), 0.45, C_MUT)
    text(img, "MediaPipe pose+hands -> fusion ->", (x, 120), 0.45, C_MUT)
    text(img, "Transformer -> CTC (no boundaries given)", (x, 140), 0.45, C_MUT)
    for k, line in enumerate(rec.stats):
        text(img, line, (x, 180 + 22 * k), 0.47, C_TXT if k == 0 else C_MUT)
    yb = 200 + 22 * len(rec.stats)
    text(img, "Fusion weights (current frame)", (x, yb), 0.47, C_TXT)
    if weights is not None:
        for j, (m, col) in enumerate(zip(["pose", "hand"], [C_POSE, C_HAND])):
            yy = yb + 16 + 26 * j
            cv2.rectangle(img, (x, yy), (x + int(200 * float(weights[j])), yy + 16), col, -1)
            cv2.rectangle(img, (x, yy), (x + 200, yy + 16), C_MUT, 1)
            text(img, f"{m} {float(weights[j]):.2f}", (x + 210, yy + 13), 0.45, C_TXT)
    else:
        text(img, "fixed fusion: no per-frame weights", (x, yb + 28), 0.45, C_MUT)
    text(img, f"Test video {idx + 1}/{n_items}: {item['title'][:34]}", (x, yb + 90), 0.47, C_TXT)
    if tally and tally[1]:
        text(img, f"This session: {100 * (1 - tally[0] / tally[1]):.1f}% glosses correct, {tally[2]}/{tally[3]} exact",
             (x, yb + 114), 0.45, C_MUT)
    text(img, f"{fps:4.1f} fps (tracking + recognition)" + ("   PAUSED" if state.get("paused") else ""), (x, yb + 138), 0.45, C_MUT)
    for k, line in enumerate(["SPACE pause   N next   P previous   R replay",
                              "D switch dataset   F fixed/adaptive   Q quit"]):
        text(img, line, (x, H - 40 + 20 * k), 0.42, C_MUT)
    return img


# ----------------------------------------------------------------------------- main loop
def frames_of(paths):
    for p in paths:
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield cv2.resize(f, (640, int(f.shape[0] * 640 / f.shape[1]))) if f.shape[1] > 640 else f, fps
        cap.release()


def play(item, rec, idx, n_items, show, tally, state):
    """Track + recognise one item live. Returns a key action."""
    tracker = Tracker()
    P, Hh, HS, Q = [], [], [], []
    glosses, weights, lm, t_last, fps, size = [], None, None, time.time(), 0.0, None
    k, frame, post = 0, None, None
    period = 1 / 25
    for frame, vfps in frames_of(item["paths"]):
        t0 = time.time()
        if k % STRIDE == 0:
            p, h, hs, q = tracker(frame, int(k * 1000 / vfps))
            P.append(p); Hh.append(h); HS.append(hs); Q.append(q)
            size = (frame.shape[1], frame.shape[0])
            lm = (p, h)
            if len(P) >= 4 and len(P) % 2 == 0:
                glosses, weights, post = decode(rec, P, Hh, HS, Q, size)
        k += 1
        dt = time.time() - t_last
        t_last = time.time()
        fps = 0.9 * fps + 0.1 * (1 / max(dt, 1e-3)) if fps else 1 / max(dt, 1e-3)
        key = show(render(state, rec, frame, lm, glosses, weights, item, idx, n_items, fps, tally=tally),
                   max(1, int(1000 * (period - (time.time() - t0)))))
        while state.get("paused") and key not in ("pause", "quit", "next", "prev", "replay", "dataset", "fusion"):
            key = show(None, 50)
        if key == "pause":
            state["paused"] = not state.get("paused")
        elif key:
            return key
    glosses, weights, post = decode(rec, P, Hh, HS, Q, size)
    final = dict(sentence=rec.sentence(post, len(P)))
    e = sum(align_counts(item["ref"], glosses))
    tally[0] += e; tally[1] += len(item["ref"]); tally[2] += glosses == item["ref"]; tally[3] += 1
    img = render(state, rec, frame, None, glosses, weights, item, idx, n_items, fps, final=final, tally=tally)
    deadline = time.time() + state["hold"]
    while True:
        key = show(img, 50)
        if key or (state["auto"] and time.time() > deadline):
            return key or "next"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["signs", "sentences"], default="signs")
    ap.add_argument("--video", help="run on any video file instead of the test set")
    ap.add_argument("--auto", action="store_true", help="advance automatically after each video")
    ap.add_argument("--hold", type=float, default=4.0, help="seconds to hold the final result in --auto mode")
    ap.add_argument("--snapshots", help="headless: save one PNG per video to this folder instead of a window")
    a = ap.parse_args()
    torch.set_num_threads(4)
    state = {"auto": a.auto or bool(a.snapshots), "hold": 0 if a.snapshots else a.hold}
    keymap = {ord(" "): "pause", ord("n"): "next", ord("p"): "prev", ord("r"): "replay", ord("d"): "dataset",
              ord("f"): "fusion", ord("q"): "quit", 27: "quit", 3: "next", 2: "prev"}
    snap = {"n": 0, "last": None}

    def show(img, wait_ms):
        if a.snapshots:
            if img is not None:
                snap["last"] = img
            return None
        if img is not None:
            cv2.imshow("Continuous ISL Recognition", img)
        k = cv2.waitKey(wait_ms) & 0xFF
        return keymap.get(k)

    kinds = {"signs": IncludeRecogniser, "sentences": SentenceRecogniser}
    dataset, fusion = a.dataset, None
    rec = kinds[dataset](fusion)
    items = rec.items() if not a.video else [dict(paths=[Path(a.video)], ref=[], title=Path(a.video).name)]
    idx, tally = 0, [0, 0, 0, 0]
    while True:
        act = play(items[idx], rec, idx, len(items), show, tally, state)
        if a.snapshots:
            Path(a.snapshots).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(a.snapshots) / f"{dataset}_{idx:02d}.png"), snap["last"])
            snap["n"] += 1
            if snap["n"] >= 3:
                break
        if act == "quit":
            break
        if act == "prev":
            idx = (idx - 1) % len(items)
        elif act == "next":
            idx = (idx + 1) % len(items)
        elif act in ("dataset", "fusion"):
            if act == "dataset" and not a.video:
                dataset = "sentences" if dataset == "signs" else "signs"
            if act == "fusion":
                fusion = "concat" if rec.fusion == "trel_gate" else "trel_gate"
            rec = kinds[dataset](fusion)
            if not a.video:
                items, idx, tally = rec.items(), 0, [0, 0, 0, 0]
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
