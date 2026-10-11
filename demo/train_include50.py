"""INCLUDE-50 (official AI4Bharat split): CTC sign-sequence model on MediaPipe landmarks.

Benchmark metric (comparable to the INCLUDE paper, 94.5 % on INCLUDE-50): accuracy on the 191
official test clips, where the predicted sign is the one with the highest CTC likelihood among the
50 classes. Also reported: free (unconstrained) decoding accuracy and WER on 400 chains of test
clips (continuous recognition, no boundaries given).

python demo/train_include50.py --fusion concat --seed 0
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import collate, preprocess  # noqa: E402
from metrics import corpus, per_item  # noqa: E402
from models import build_model, ctc_loss, greedy_decode, posteriors  # noqa: E402
from train import Augment  # noqa: E402

SPLITS = ROOT / "data" / "include"


def gloss_of(folder):
    return folder.split(". ", 1)[1].upper().replace(" ", "-")


def load():
    raw = pickle.load(open(ROOT / "features" / "include_lm.pkl", "rb"))
    by_key = {f"{k.split('|')[0]}/{k.split('|')[1]}".lower(): k for k in raw}
    clips, signs = {"train": [], "val": [], "test": []}, set()
    for name in clips:
        for line in open(SPLITS / f"include50_{name}.txt"):
            parts = line.strip().split("/")                    # cat/sign/clip or cat/sign/Extra/clip
            sign = parts[1]
            sid = by_key.get(os.path.splitext(line.strip())[0].lower())
            if sid and len(raw[sid]["pose"]) >= 4:
                clips[name].append((sid, gloss_of(sign), raw[sid]))
                signs.add(gloss_of(sign))
    return clips, ["<blank>"] + sorted(signs)


def concat(records):
    r = {k: np.concatenate([x[k] for x in records]) for k in ("pose", "hands", "hand_score", "quality")}
    r["size"] = records[0]["size"]
    r["rgb"] = np.zeros((len(r["pose"]), 768), np.float16)
    return r


def chains(clips, n, rng, max_len=4):
    out = []
    for k in range(n):
        L = 1 if k % 4 == 0 else int(rng.integers(2, max_len + 1))
        picks = [clips[rng.integers(len(clips))] for _ in range(L)]
        out.append((preprocess(concat([p[2] for p in picks])), [p[1] for p in picks]))
    return out


def batches(items, bs, stoi):
    for i in range(0, len(items), bs):
        yield collate([(f"s{j}", f, [stoi[g] for g in gl]) for j, (f, gl) in enumerate(items[i:i + bs])])


@torch.no_grad()
def evaluate(model, items, vocab, stoi):
    model.eval()
    res = []
    for b in batches(items, 32, stoi):
        for ref, h in zip(b["labels"], greedy_decode(posteriors(model(b)), b["lengths"])):
            res.append(per_item([vocab[i] for i in ref], [vocab[i] for i in h]))
    return corpus(res)


@torch.no_grad()
def classify(model, items, vocab, stoi):
    """Closed-set accuracy: the class whose single-gloss CTC likelihood is highest."""
    model.eval()
    ok = 0
    C = len(vocab) - 1
    for b in batches(items, 16, stoi):
        lp = torch.log(posteriors(model(b)).clamp_min(1e-8)).transpose(0, 1)       # T, B, V
        B = lp.shape[1]
        nll = torch.stack([F.ctc_loss(lp, torch.full((B, 1), c + 1), b["lengths"], torch.ones(B, dtype=torch.long),
                                      blank=0, reduction="none", zero_infinity=True) for c in range(C)], 1)
        pred = nll.argmin(1) + 1
        ok += sum(int(p) == lab[0] for p, lab in zip(pred.tolist(), b["labels"]))
    return 100 * ok / len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion", default="concat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--per_epoch", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    clips, vocab = load()
    stoi = {g: i for i, g in enumerate(vocab)}
    print(f"{len(vocab) - 1} signs; clips train/val/test = {[len(clips[k]) for k in ('train', 'val', 'test')]}", flush=True)
    val_iso = [(preprocess(concat([r])), [g]) for _, g, r in clips["val"]]
    val_seq = chains(clips["val"], 200, np.random.default_rng(100))
    test_iso = [(preprocess(concat([r])), [g]) for _, g, r in clips["test"]]
    test_seq = chains(clips["test"], 400, np.random.default_rng(200))
    mods = ["pose", "hand"]
    model = build_model(len(vocab), mods, a.fusion)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2, foreach=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=a.epochs * ((a.per_epoch + 15) // 16),
                                                pct_start=0.1)
    rng = np.random.default_rng(a.seed)
    aug = Augment(mods, rng)
    best, state, curve, t0 = 1e9, None, [], time.time()
    for ep in range(a.epochs):
        model.train()
        items = [(aug(f), gl) for f, gl in chains(clips["train"], a.per_epoch, rng)]
        for b in batches(items, 16, stoi):
            out = model(b)
            out["aux_weight"] = model.aux_weight
            loss = ctc_loss(out, b)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
        v = evaluate(model, val_iso + val_seq, vocab, stoi)["wer"]
        curve.append((ep, round(v, 2)))
        if v < best:
            best, state = v, {k: x.clone() for k, x in model.state_dict().items()}
        print(f"epoch {ep} loss {loss.item():.3f} val WER {v:.2f} ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(state)
    res = dict(fusion=a.fusion, seed=a.seed, signs=len(vocab) - 1, n_clips={k: len(v) for k, v in clips.items()},
               val_wer=best, val_curve=curve,
               test_isolated_accuracy_closed=classify(model, test_iso, vocab, stoi),
               test_isolated_free=evaluate(model, test_iso, vocab, stoi),
               test_sequences=evaluate(model, test_seq, vocab, stoi),
               train_time_s=time.time() - t0, params=sum(p.numel() for p in model.parameters()))
    name = f"include50_{a.fusion}_s{a.seed}"
    (ROOT / "models" / "include50").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "include50").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ROOT / "models" / "include50" / f"{name}.pt")
    json.dump(dict(res, vocab=vocab), open(ROOT / "results" / "include50" / f"{name}.json", "w"), indent=1)
    print(f"{name}: TEST isolated accuracy {res['test_isolated_accuracy_closed']:.1f}% (closed-set) / "
          f"{100 - res['test_isolated_free']['wer']:.1f}% (free decoding) | sequences WER {res['test_sequences']['wer']:.2f}, "
          f"exact {res['test_sequences']['sent_acc']:.1f}%")


if __name__ == "__main__":
    main()
