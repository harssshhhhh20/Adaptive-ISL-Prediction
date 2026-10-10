"""Sequence tensors, modality preprocessing and evaluation splits for ISL-CSLTR."""
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from glosses import load_gloss_table

ROOT = Path(__file__).resolve().parents[1]
FEATS = ROOT / "features"

SIGNERS = [str(i) for i in range(1, 8)]
PHONE = ["1", "2"]                  # portrait phone recordings (verified visually)
DSLR = ["3", "4", "5", "6", "7"]    # landscape DSLR recordings
UPPER = list(range(0, 25))          # MediaPipe pose landmarks above the knees
MODALITIES = ("rgb", "pose", "hand", "hrgb")    # hrgb = DINOv2 embeddings of the two hand crops
# indices into the per-frame quality vector produced by extract_features.quality()
Q_RGB, Q_POSE, Q_HAND = [0, 1, 2], [3], [4, 5, 6, 7]


# ----------------------------------------------------------------------------- preprocessing
def _body_frame(pose, w, h):
    """Per-sequence body-centred coordinate frame from the frames where shoulders were found."""
    sh = pose[:, [11, 12], :2] * np.array([w, h])
    ok = ~np.isnan(sh).any(axis=(1, 2))
    if ok.sum() == 0:
        return np.array([w / 2, h / 2]), max(w, h) / 4
    centre = np.median(sh[ok].mean(1), axis=0)
    scale = np.median(np.linalg.norm(sh[ok, 0] - sh[ok, 1], axis=-1))
    return centre, max(scale, 1e-3)


def _jitter(x):
    """Frame-to-frame displacement magnitude (temporal consistency signal); 0 where undefined."""
    d = np.zeros(len(x), np.float32)
    if len(x) > 1:
        v = np.linalg.norm(np.diff(x, axis=0).reshape(len(x) - 1, -1), axis=-1)
        d[1:] = np.nan_to_num(v, nan=0.0)
    return np.clip(d / 5.0, 0, 3)


def preprocess(rec):
    """Raw extractor output -> fixed-size per-frame vectors for each modality + quality."""
    w, h = rec["size"]
    centre, scale = _body_frame(rec["pose"], w, h)
    pose = rec["pose"][:, UPPER]
    pxy = (pose[..., :2] * np.array([w, h]) - centre) / scale
    pose_feat = np.concatenate([pxy, pose[..., 3:4]], -1).reshape(len(pose), -1)
    pose_ok = ~np.isnan(pose[:, 0, 0])

    hands = rec["hands"]
    hxy = (hands[..., :2] * np.array([w, h]) - centre) / scale            # global position
    wrist = hxy[:, :, :1]
    size = np.linalg.norm(hxy[:, :, 9:10] - wrist, axis=-1, keepdims=True) + 1e-3
    hloc = (hxy - wrist) / size                                          # hand shape
    hand_feat = np.concatenate([hxy, hloc], -1).reshape(len(hands), -1)

    q = rec["quality"].copy()
    jit = np.stack([_jitter(np.where(pose_ok[:, None], pxy.reshape(len(pxy), -1), np.nan)),
                    _jitter(hxy.reshape(len(hxy), -1))], 1)
    rgb = rec["rgb"].astype(np.float32)
    out = {
        # per-sequence temporal centring: frame embeddings are ~0.98 cosine-similar within a sequence,
        # dominated by static signer/background appearance; centring keeps what changes over time
        "rgb": rgb - rgb.mean(0, keepdims=True),
        "pose": np.nan_to_num(pose_feat, nan=0.0).astype(np.float32),
        "hand": np.nan_to_num(hand_feat, nan=0.0).astype(np.float32),
        "quality": np.concatenate([q, jit], 1).astype(np.float32),
    }
    if "hrgb" in rec:
        hr = rec["hrgb"].astype(np.float32)
        present = (np.abs(hr).sum(-1, keepdims=True) > 0)
        mu = (hr * present).sum(0, keepdims=True) / np.maximum(present.sum(0, keepdims=True), 1)
        out["hrgb"] = (hr - mu) * present            # same temporal centring as the frame embedding
    else:
        out["hrgb"] = np.zeros((len(rgb), 768), np.float32)
    out["pose"] = np.clip(out["pose"], -6, 6)
    out["hand"] = np.clip(out["hand"], -6, 6)
    return out


def load_raw(cond):
    with open(FEATS / f"{cond}.pkl", "rb") as f:
        raw = pickle.load(f)
    hp = FEATS / f"{cond}.handrgb.pkl"
    if hp.exists():
        with open(hp, "rb") as f:
            for k, v in pickle.load(f).items():
                raw[k]["hrgb"] = v
    return raw


def load_condition(cond):
    return {k: preprocess(v) for k, v in load_raw(cond).items()}


# ----------------------------------------------------------------------------- vocabulary
class Vocab:
    def __init__(self, table):
        toks = sorted({g for seq in table.values() for g in seq})
        self.itos = ["<blank>"] + toks
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, seq):
        return [self.stoi[t] for t in seq]

    def decode(self, ids):
        return [self.itos[i] for i in ids]

    def __len__(self):
        return len(self.itos)


def seq_sentence(sid):
    return sid.split("|")[0]


# Folder number = signer, except where the appearance check (scripts/verify_signers.py) and manual
# inspection showed a sequence filed under the wrong folder.
_SIGNER_FIX = json.loads((ROOT / "data" / "signer_map.json").read_text()) if (ROOT / "data" / "signer_map.json").exists() else {}


def seq_signer(sid):
    return _SIGNER_FIX.get(sid, sid.split("|")[1])


# ----------------------------------------------------------------------------- splits
def _unseen_sentence_folds(table, n_folds=3):
    """Sentence-disjoint folds. Only sentences whose every gloss also occurs in some *other*
    sentence are eligible as test items, and a fold is only accepted if every test gloss still
    occurs in that fold's training sentences. Sentences with identical gloss strings share a fold."""
    sent_count = Counter(g for v in table.values() for g in set(v))
    eligible = sorted(s for s, v in table.items() if all(sent_count[g] >= 2 for g in v))
    groups = defaultdict(list)
    for s in eligible:
        groups[tuple(table[s])].append(s)
    groups = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    folds = [[] for _ in range(n_folds)]
    for g in groups:
        for f in sorted(range(n_folds), key=lambda i: len(folds[i])):
            trial = folds[f] + g
            train_glosses = Counter(t for s, v in table.items() if s not in trial for t in v)
            if all(train_glosses[t] > 0 for s in trial for t in table[s]):
                folds[f] = trial
                break
    return folds


def make_splits(seq_ids, split_seed=0):
    """Returns {protocol: [ {name, train, val, test} ... ]} with disjointness asserted."""
    table = load_gloss_table()
    by_signer = defaultdict(list)
    for s in seq_ids:
        by_signer[seq_signer(s)].append(s)
    splits = {}

    # 1. Signer-dependent random split (no official split exists): stratified by sentence.
    rng = np.random.default_rng(split_seed)
    tr, va, te = [], [], []
    by_sent = defaultdict(list)
    for s in seq_ids:
        by_sent[seq_sentence(s)].append(s)
    for sent in sorted(by_sent):
        items = sorted(by_sent[sent])
        rng.shuffle(items)
        n = len(items)
        n_te = 1 if n >= 3 else 0
        n_va = 1 if n >= 4 else 0
        te += items[:n_te]; va += items[n_te:n_te + n_va]; tr += items[n_te + n_va:]
    splits["random"] = [dict(name="random", train=tr, val=va, test=te)]

    # 2. Signer-independent leave-one-signer-out; validation signer is also held out of training.
    val_of = {"1": "3", "2": "4", "3": "4", "4": "5", "5": "6", "6": "7", "7": "3"}
    splits["loso"] = []
    for t in SIGNERS:
        v = val_of[t]
        train = [s for s in seq_ids if seq_signer(s) not in (t, v)]
        splits["loso"].append(dict(name=f"loso_s{t}", train=train, val=by_signer[v], test=by_signer[t]))

    # 3. Cross-domain: train on DSLR signers, test on unseen phone-recorded signers.
    splits["cross_domain"] = [dict(name="dslr2phone",
                                   train=[s for s in seq_ids if seq_signer(s) in ("3", "4", "5", "6")],
                                   val=by_signer["7"],
                                   test=[s for s in seq_ids if seq_signer(s) in PHONE])]

    # 4. Unseen sentences (compositional generalisation), signer-dependent.
    splits["unseen_sentence"] = []
    folds = _unseen_sentence_folds(table)
    for i, fold in enumerate(folds):
        fold = set(fold)
        rest = [s for s in seq_ids if seq_sentence(s) not in fold]
        rest_sents = sorted({seq_sentence(s) for s in rest})
        r = np.random.default_rng(split_seed + i)
        val_sents = set(r.choice(rest_sents, size=max(1, len(rest_sents) // 10), replace=False))
        splits["unseen_sentence"].append(dict(
            name=f"unseen_f{i}",
            train=[s for s in rest if seq_sentence(s) not in val_sents],
            val=[s for s in rest if seq_sentence(s) in val_sents],
            test=[s for s in seq_ids if seq_sentence(s) in fold]))

    for proto, lst in splits.items():
        for sp in lst:
            a, b, c = set(sp["train"]), set(sp["val"]), set(sp["test"])
            assert not (a & b or a & c or b & c), sp["name"]
            if proto in ("loso", "cross_domain"):
                sa, sb, sc = ({seq_signer(x) for x in z} for z in (a, b, c))
                assert not (sa & sb or sa & sc or sb & sc), f"signer leak in {sp['name']}"
            if proto == "unseen_sentence":
                assert not ({seq_sentence(x) for x in a} & {seq_sentence(x) for x in c})
    return splits


# ----------------------------------------------------------------------------- batching
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, ids, feats, table, vocab):
        self.ids = [i for i in ids if i in feats]
        self.feats, self.table, self.vocab = feats, table, vocab

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        f = self.feats[sid]
        return sid, f, self.vocab.encode(self.table[seq_sentence(sid)])


def collate(batch):
    sids, feats, labels = zip(*batch)
    T = max(len(f["rgb"]) for f in feats)
    B = len(batch)
    out = {}
    for k in ("rgb", "pose", "hand", "hrgb", "quality"):
        d = feats[0][k].shape[1]
        x = torch.zeros(B, T, d)
        for b, f in enumerate(feats):
            x[b, :len(f[k])] = torch.from_numpy(f[k])
        out[k] = x
    out["lengths"] = torch.tensor([len(f["rgb"]) for f in feats])
    out["targets"] = torch.tensor([t for l in labels for t in l], dtype=torch.long)
    out["target_lengths"] = torch.tensor([len(l) for l in labels])
    out["ids"] = list(sids)
    out["labels"] = list(labels)
    return out
