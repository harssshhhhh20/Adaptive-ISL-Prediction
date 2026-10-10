"""Continuous recognition of ISL sign sequences built from INCLUDE clips (official split).

Training sequences are random chains of 1-5 *training* clips (half follow simple templates such
as GREETING PRONOUN ADJECTIVE, half are random), regenerated every epoch. The model only gets
the gloss sequence -- never the boundaries -- and is trained with CTC exactly as in the study.
Test: (a) the official test clips as isolated signs, (b) fixed chains built only from test clips.

python demo/train_include.py --fusion trel_gate --seed 0
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import collate, preprocess  # noqa: E402
from metrics import corpus, per_item  # noqa: E402
from models import build_model, ctc_loss, greedy_decode, posteriors  # noqa: E402
from train import Augment  # noqa: E402

FEAT = ROOT / "features" / "include_lm.pkl"
SPLITS = ROOT / "data" / "include"
PRON = ["I", "YOU", "HE", "SHE", "IT", "WE", "THEY"]
GREET = ["HELLO", "HOW-ARE-YOU", "ALRIGHT", "GOOD-MORNING", "GOOD-AFTERNOON", "GOOD-EVENING", "GOOD-NIGHT",
         "THANK-YOU", "PLEASED"]
ADJ = ["LOUD", "QUIET", "HAPPY", "SAD", "BEAUTIFUL", "DEAF", "RICH", "POOR"]
VOCAB = ["<blank>"] + PRON + GREET + ADJ
STOI = {g: i for i, g in enumerate(VOCAB)}


def gloss_of(folder):
    return folder.split(". ", 1)[1].upper().replace(" ", "-")


def load_clips():
    raw = pickle.load(open(FEAT, "rb"))
    split_of = {}
    for name in ("train", "val", "test"):
        for line in open(SPLITS / f"include_{name}.txt"):
            parts = line.strip().split("/")
            if len(parts) != 3:          # "Extra" sub-folders are outside our sign subset
                continue
            cat, sign, f = parts
            split_of[f"{cat}/{sign}|{Path(f).stem}"] = name
    clips = {"train": [], "val": [], "test": []}
    for sid, r in raw.items():
        g = gloss_of(sid.split("|")[0].split("/")[1])
        if g in STOI and sid in split_of and len(r["pose"]) >= 4:
            clips[split_of[sid]].append((sid, g, r))
    return clips


def template(rng):
    t = rng.integers(0, 6)
    P, G, A = (lambda: str(rng.choice(PRON))), (lambda: str(rng.choice(GREET))), (lambda: str(rng.choice(ADJ)))
    return [[G()], [G(), G()], [P(), A()], [G(), P(), A()], [P(), A(), G()], [G(), G(), P(), A()]][t]


def random_seq(rng):
    return [str(x) for x in rng.choice(VOCAB[1:], size=int(rng.integers(1, 6)))]


def concat(records):
    keys = ("pose", "hands", "hand_score", "quality")
    r = {k: np.concatenate([x[k] for x in records]) for k in keys}
    r["size"] = records[0]["size"]
    r["rgb"] = np.zeros((len(r["pose"]), 768), np.float16)
    return r


def build(glosses, by_gloss, rng):
    recs = [by_gloss[g][rng.integers(len(by_gloss[g]))][1] for g in glosses]
    return preprocess(concat(recs)), glosses


def fixed_set(clips, n, seed):
    by = {}
    for sid, g, r in clips:
        by.setdefault(g, []).append((sid, r))
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        gl = template(rng) if k % 2 == 0 else random_seq(rng)
        gl = [g for g in gl if g in by] or [next(iter(by))]
        out.append(build(gl, by, rng))
    return out


def batches(items, bs):
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        yield collate([(f"s{j}", f, [STOI[g] for g in gl]) for j, (f, gl) in enumerate(chunk)])


@torch.no_grad()
def evaluate(model, items):
    model.eval()
    res = []
    for b in batches(items, 32):
        hyps = greedy_decode(posteriors(model(b)), b["lengths"])
        for ref, h in zip(b["labels"], hyps):
            res.append(per_item([VOCAB[i] for i in ref], [VOCAB[i] for i in h]))
    return corpus(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion", default="trel_gate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--per_epoch", type=int, default=800)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    clips = load_clips()
    by_train = {}
    for sid, g, r in clips["train"]:
        by_train.setdefault(g, []).append((sid, r))
    val_seq = fixed_set(clips["val"], 200, 100)
    test_iso = [(preprocess(concat([r])), [g]) for _, g, r in clips["test"]]
    test_seq = fixed_set(clips["test"], 400, 200)
    mods = ["pose", "hand"]
    model = build_model(len(VOCAB), mods, a.fusion)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2, foreach=True)
    steps = a.epochs * ((a.per_epoch + 15) // 16)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(a.seed)
    aug = Augment(mods, rng)
    best, best_state, curve, t0 = 1e9, None, [], time.time()
    for ep in range(a.epochs):
        model.train()
        items = []
        for k in range(a.per_epoch):
            gl = template(rng) if k % 2 == 0 else random_seq(rng)
            f, gl = build(gl, by_train, rng)
            items.append((aug(f), gl))
        for b in batches(items, 16):
            out = model(b)
            out["aux_weight"] = model.aux_weight
            loss = ctc_loss(out, b)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
        v = evaluate(model, val_seq)["wer"]
        curve.append((ep, round(v, 2)))
        if v < best:
            best, best_state = v, {k: x.clone() for k, x in model.state_dict().items()}
        print(f"epoch {ep} loss {loss.item():.3f} val WER {v:.2f} ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    iso = evaluate(model, test_iso)
    seq = evaluate(model, test_seq)
    name = f"{a.fusion}_s{a.seed}"
    (ROOT / "models" / "include").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "include").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ROOT / "models" / "include" / f"{name}.pt")
    res = dict(fusion=a.fusion, seed=a.seed, val_wer=best, val_curve=curve, n_clips={k: len(v) for k, v in clips.items()},
               test_isolated=iso, test_sequences=seq, train_time_s=time.time() - t0,
               params=sum(p.numel() for p in model.parameters()))
    (ROOT / "results" / "include" / f"{name}.json").write_text(json.dumps(res, indent=1))
    print(f"{name}: val WER {best:.2f} | isolated test accuracy {100 - iso['wer']:.1f}% (exact {iso['sent_acc']:.1f}%) "
          f"| sequence test WER {seq['wer']:.2f}, exact-sequence acc {seq['sent_acc']:.1f}%")


if __name__ == "__main__":
    main()
