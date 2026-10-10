"""Train one (config, split, seed) job on clean training data and evaluate on every test condition."""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from data import (FEATS, MODALITIES, Q_HAND, Q_POSE, Q_RGB, SeqDataset, Vocab, collate, load_condition,
                  seq_sentence)
from glosses import load_gloss_table
from metrics import corpus, per_item
from models import SentenceClassifier, build_model, ctc_loss, greedy_decode, posteriors

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
IMAGE_CONDS = ["clean", "lowlight", "uneven", "hand_occ", "body_occ", "background", "blur",
               "distance", "viewpoint", "combined"]
FEATURE_CONDS = ["missing_rgb", "missing_pose", "missing_hand", "fast", "slow"]
Q_OF = {"rgb": Q_RGB, "pose": Q_POSE + [8], "hand": Q_HAND + [9], "hrgb": []}

_CACHE = {}


def features(cond):
    """Preprocessed features for an image condition (cached on disk and in-process)."""
    if cond not in _CACHE:
        pre = FEATS / f"pre_{cond}.pkl"
        if pre.exists():
            with open(pre, "rb") as f:
                d = pickle.load(f)
        else:
            d = load_condition(cond)
            for v in d.values():
                v["rgb"] = v["rgb"].astype(np.float16)
                v["hrgb"] = v["hrgb"].astype(np.float16)
            with open(pre, "wb") as f:
                pickle.dump(d, f)
        if cond != "clean" and len(_CACHE) > 2:      # keep memory bounded per worker
            for k in [k for k in _CACHE if k != "clean"]:
                del _CACHE[k]
        _CACHE[cond] = d
    return _CACHE[cond]


def _resample(f, idx):
    return {k: v[idx] for k, v in f.items()}


def drop_modality(f, m):
    g = dict(f)
    g[m] = np.zeros_like(f[m])
    q = f["quality"].copy()
    q[:, Q_OF[m]] = 0.0
    g["quality"] = q
    return g


def feature_condition(feats, cond):
    out = {}
    for k, f in feats.items():
        T = len(f["rgb"])
        if cond.startswith("missing_"):
            out[k] = drop_modality(f, cond.split("_")[1])
        elif cond == "fast":                         # signing 2x faster: half the frames
            out[k] = _resample(f, np.arange(0, T, 2))
        elif cond == "slow":                         # signing 1.5x slower
            out[k] = _resample(f, np.clip(np.round(np.arange(0, T, 1 / 1.5)).astype(int), 0, T - 1))
    return out


def sent_loss(model, out, batch):
    index = {c: k for k, c in enumerate(model.sent_classes)}
    y = torch.tensor([index[tuple(model.vocab.decode(lab))] for lab in batch["labels"]])
    return torch.nn.functional.cross_entropy(out["sent_logits"], y)


class Augment:
    """Training-time augmentation, identical for every model (multimodal or not)."""

    def __init__(self, mods, rng, mod_drop=0.15, noise=0.05, frame_drop=0.1, speed=(0.8, 1.2)):
        self.mods, self.rng = mods, rng
        self.mod_drop, self.noise, self.frame_drop, self.speed = mod_drop, noise, frame_drop, speed

    def __call__(self, f):
        T = len(f["rgb"])
        r = self.rng
        n = max(4, int(round(T * r.uniform(*self.speed))))
        idx = np.clip(np.round(np.linspace(0, T - 1, n)).astype(int), 0, T - 1)
        if self.frame_drop and len(idx) > 6:
            keep = r.random(len(idx)) > self.frame_drop
            idx = idx[keep] if keep.sum() >= 4 else idx
        g = {k: v[idx].astype(np.float32) for k, v in f.items()}
        for m in ("pose", "hand"):
            g[m] = g[m] + r.normal(0, self.noise, g[m].shape).astype(np.float32) * (g[m] != 0)
        if len(self.mods) > 1 and self.mod_drop:
            drop = [m for m in self.mods if r.random() < self.mod_drop]
            if len(drop) < len(self.mods):
                for m in drop:
                    g = drop_modality(g, m)
        return g


class BucketBatchSampler(torch.utils.data.Sampler):
    """Random batches of similar-length sequences (shuffle, sort within pools of 4 batches)."""

    def __init__(self, lengths, batch_size, seed):
        self.lengths, self.bs, self.rng, self.epoch = np.asarray(lengths), batch_size, np.random.default_rng(seed), 0

    def __iter__(self):
        idx = self.rng.permutation(len(self.lengths))
        pool = self.bs * 4
        batches = []
        for i in range(0, len(idx), pool):
            chunk = sorted(idx[i:i + pool], key=lambda j: self.lengths[j])
            batches += [chunk[k:k + self.bs] for k in range(0, len(chunk), self.bs)]
        for b in self.rng.permutation(len(batches)):
            yield batches[b]

    def __len__(self):
        return (len(self.lengths) + self.bs - 1) // self.bs


class AugDataset(SeqDataset):
    def __init__(self, *a, aug=None, **k):
        super().__init__(*a, **k)
        self.aug = aug

    def __getitem__(self, i):
        sid, f, y = super().__getitem__(i)
        return sid, (self.aug(f) if self.aug else f), y


@torch.no_grad()
def evaluate(model, feats, ids, table, vocab, keep_weights=False):
    model.eval()
    ds = SeqDataset(ids, feats, table, vocab)
    order = sorted(range(len(ds)), key=lambda i: len(feats[ds.ids[i]]["rgb"]))
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, order), batch_size=32, shuffle=False, collate_fn=collate)
    items, weights = [], {}
    for b in dl:
        out = model(b)
        if "sent_logits" in out:
            hyps = [vocab.encode(list(model.sent_classes[k])) for k in out["sent_logits"].argmax(-1).tolist()]
        else:
            hyps = greedy_decode(posteriors(out), b["lengths"])
        for k, (sid, ref, hyp) in enumerate(zip(b["ids"], b["labels"], hyps)):
            it = per_item(vocab.decode(ref), vocab.decode(hyp))
            it["id"], it["hyp"] = sid, " ".join(vocab.decode(hyp))
            items.append(it)
            if keep_weights and out["weights"] is not None:
                L = int(b["lengths"][k])
                weights[sid] = out["weights"][k, :L].numpy().astype(np.float16)
    num = [{k: v for k, v in it.items() if k not in ("id", "hyp")} for it in items]
    return corpus(num), items, weights


def run_job(cfg, split, seed, out_path, eval_conds=None, epochs=None, keep_weights=False):
    epochs = epochs or cfg.get("epochs", 40)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(cfg.get("threads", 2))
    table = load_gloss_table(raw=cfg.get("raw_glosses", False))
    vocab = Vocab(table)
    clean = features("clean")
    mods = cfg["mods"]
    sent_classes = None
    if cfg["fusion"] == "sentcls":
        sent_classes = sorted({tuple(table[seq_sentence(i)]) for i in split["train"]})
        model = SentenceClassifier(len(sent_classes), mods, d=cfg.get("d", 192), p=cfg.get("dropout", 0.3))
        model.sent_classes, model.vocab = sent_classes, vocab
    else:
        model = build_model(len(vocab), mods, cfg["fusion"], d=cfg.get("d", 192), p=cfg.get("dropout", 0.3),
                            **({"backbone": cfg["backbone"]} if "backbone" in cfg else {}))
    nparams = sum(p.numel() for p in model.parameters())
    rng = np.random.default_rng(seed)
    aug = Augment(mods, rng, mod_drop=cfg.get("mod_drop", 0.15)) if cfg.get("augment", True) else None
    train_ds = AugDataset(split["train"], clean, table, vocab, aug=aug)
    sampler = BucketBatchSampler([len(clean[i]["rgb"]) for i in train_ds.ids], cfg.get("batch_size", 16), seed)
    dl = torch.utils.data.DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-3), weight_decay=1e-2, foreach=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.get("lr", 3e-3), total_steps=epochs * len(dl),
                                                pct_start=0.1)
    best, best_state, best_ep, t0 = 1e9, None, -1, time.time()
    curve = []
    for ep in range(epochs):
        model.train()
        for b in dl:
            out = model(b)
            out["aux_weight"] = model.aux_weight
            loss = sent_loss(model, out, b) if sent_classes else ctc_loss(out, b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
        if ep >= 9 and (ep % 2 == 1 or ep == epochs - 1):
            m, _, _ = evaluate(model, clean, split["val"], table, vocab)
            curve.append((ep, round(m["wer"], 2)))
            if m["wer"] < best:
                best, best_ep = m["wer"], ep
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    train_time = time.time() - t0
    model.load_state_dict(best_state)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in best_state.items()},
               out_path.with_suffix(".pt"))
    res = dict(cfg=cfg, split=split["name"], test_ids=split["test"], seed=seed, params=nparams, val_wer=best,
               best_epoch=best_ep, train_time=train_time, val_curve=curve, conds={})
    evaluate_conditions(model, res, clean, table, vocab, eval_conds, keep_weights, out_path)
    return res


def available_conditions():
    return [c for c in IMAGE_CONDS if (FEATS / f"{c}.pkl").exists()] + FEATURE_CONDS


def evaluate_conditions(model, res, clean, table, vocab, conds, keep_weights, out_path):
    """Evaluate on each requested condition not yet in res; write res (+ fusion weights)."""
    test = res["test_ids"]
    wpath = out_path.with_suffix(".weights.pkl")
    weights_all = {}
    if keep_weights and wpath.exists():
        with open(wpath, "rb") as f:
            weights_all = pickle.load(f)
    for cond in (conds or available_conditions()):
        if cond in res["conds"]:
            continue
        if cond == "clean":
            fe = clean
        elif cond in IMAGE_CONDS:
            if not (FEATS / f"{cond}.pkl").exists():
                continue
            fe = features(cond)
        else:
            fe = feature_condition({k: clean[k] for k in test if k in clean}, cond)
        m, items, w = evaluate(model, fe, test, table, vocab, keep_weights=keep_weights)
        res["conds"][cond] = dict(metrics=m, items=[{k: it[k] for k in ("id", "err", "n", "sub", "dele", "ins", "cerr",
                                                                         "cn", "exact", "tp", "nh", "hyp")}
                                                    for it in items])
        if w:
            weights_all[cond] = w
    with open(out_path, "w") as f:
        json.dump(res, f)
    if weights_all:
        with open(wpath, "wb") as f:
            pickle.dump(weights_all, f)


def reevaluate(out_path, conds=None):
    """Load a finished run's checkpoint and evaluate it on newly available conditions."""
    with open(out_path) as f:
        res = json.load(f)
    todo = [c for c in (conds or available_conditions()) if c not in res["conds"]]
    if not todo:
        return 0
    cfg = res["cfg"]
    torch.set_num_threads(cfg.get("threads", 2))
    table = load_gloss_table(raw=cfg.get("raw_glosses", False))
    vocab = Vocab(table)
    model = build_model(len(vocab), cfg["mods"], cfg["fusion"], d=cfg.get("d", 192), p=cfg.get("dropout", 0.3),
                        **({"backbone": cfg["backbone"]} if "backbone" in cfg else {}))
    sd = torch.load(out_path.with_suffix(".pt"))
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    evaluate_conditions(model, res, features("clean"), table, vocab, todo, res["seed"] == 0, out_path)
    return len(todo)
