"""Train the demo models on dense video landmarks for the 40 selected ISL-CSLTR sentences.

Split (stratified by sentence, fixed seed): per sentence 1 video -> test, 1 -> validation, rest
-> train. Test videos are never used for training or model selection. NOTE: this is a
*signer-seen* split -- test signers also appear in training (signing other sentences), so the
numbers are not signer-independent. Reported as such.

python demo/train_demo.py [--seeds 0-2]
Outputs models/demo/*.pt, results/demo/metrics.json, results/demo/predictions.json
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import train  # noqa: E402
from data import Vocab, preprocess, seq_sentence  # noqa: E402
from glosses import load_gloss_table  # noqa: E402
from metrics import corpus, per_item  # noqa: E402
from models import build_model, posteriors  # noqa: E402

OUT_M = ROOT / "models" / "demo"
OUT_R = ROOT / "results" / "demo"
CONFIGS = {
    "adaptive": dict(mods=["pose", "hand"], fusion="trel_gate"),
    "fixed": dict(mods=["pose", "hand"], fusion="concat"),
}


def demo_table():
    keep = [l.strip() for l in open(ROOT / "data" / "demo_sentences.txt") if l.strip()]
    full = load_gloss_table(with_data_only=False)
    return {s: full[s] for s in keep}


def load_feats():
    raw = pickle.load(open(ROOT / "features" / "videos_lm.pkl", "rb"))
    med = np.median([len(r["pose"]) for r in raw.values()])
    dropped = [k for k, r in raw.items() if len(r["pose"]) > 4 * med]   # malformed recordings
    for k in dropped:
        print(f"excluding {k}: {len(raw[k]['pose'])} frames (median {med:.0f})")
        del raw[k]
    feats = {}
    for sid, r in raw.items():
        r = dict(r, rgb=np.zeros((len(r["pose"]), 768), np.float16))
        f = preprocess(r)
        f["rgb"] = f["rgb"].astype(np.float16)
        f["hrgb"] = f["hrgb"].astype(np.float16)
        feats[sid] = f
    return feats, raw


def make_split(ids, seed=0):
    rng = np.random.default_rng(seed)
    by = {}
    for i in ids:
        by.setdefault(seq_sentence(i), []).append(i)
    tr, va, te = [], [], []
    for s in sorted(by):
        items = sorted(by[s])
        rng.shuffle(items)
        te.append(items[0]); va.append(items[1]); tr += items[2:]
    return dict(name="demo", train=tr, val=va, test=te)


def constrained(post, length, table, vocab):
    """Closed-vocabulary decoding: the known sentence with the highest CTC likelihood."""
    lp = torch.log(post[:length].clamp_min(1e-8))[:, None, :]
    best, best_s = None, -1e18
    for s, g in table.items():
        tgt = torch.tensor([vocab.stoi[x] for x in g])[None]
        nll = torch.nn.functional.ctc_loss(lp, tgt, torch.tensor([length]), torch.tensor([len(g)]), blank=0,
                                           reduction="sum", zero_infinity=True)
        if -nll.item() > best_s:
            best, best_s = s, -nll.item()
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-2")
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    table = demo_table()
    train.load_gloss_table = lambda raw=False, **k: table          # restrict vocabulary to the demo set
    vocab = Vocab(table)
    feats, raw = load_feats()
    feats = {k: v for k, v in feats.items() if seq_sentence(k) in table}
    train._CACHE["clean"] = feats
    split = make_split(sorted(feats))
    OUT_M.mkdir(parents=True, exist_ok=True)
    OUT_R.mkdir(parents=True, exist_ok=True)
    print(f"{len(table)} sentences, {len(vocab) - 1} glosses; train {len(split['train'])} / val {len(split['val'])} / test {len(split['test'])} videos")

    metrics, preds = {"split": {k: split[k] for k in ("train", "val", "test")}, "runs": {}}, {}
    for name, cfg in CONFIGS.items():
        for seed in range(lo, hi + 1):
            out = OUT_M / f"{name}_s{seed}.json"
            r = train.run_job(dict(cfg, threads=4), split, seed, out, eval_conds=["clean", "missing_hand", "missing_pose"])
            # closed-vocabulary decoding on the test set
            model = build_model(len(vocab), cfg["mods"], cfg["fusion"])
            sd = torch.load(out.with_suffix(".pt"))
            model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
            model.eval()
            ok = 0
            from data import SeqDataset, collate
            dl = torch.utils.data.DataLoader(SeqDataset(split["test"], feats, table, vocab), batch_size=16, collate_fn=collate)
            with torch.no_grad():
                for b in dl:
                    o = model(b)
                    post = posteriors(o)
                    for k, sid in enumerate(b["ids"]):
                        s = constrained(post[k], int(b["lengths"][k]), table, vocab)
                        ok += s == seq_sentence(sid)
            m = r["conds"]["clean"]["metrics"]
            metrics["runs"][f"{name}_s{seed}"] = dict(
                val_wer=r["val_wer"], test=m, test_missing_hand=r["conds"]["missing_hand"]["metrics"],
                test_missing_pose=r["conds"]["missing_pose"]["metrics"],
                constrained_sentence_acc=100 * ok / len(split["test"]), best_epoch=r["best_epoch"],
                train_time_s=r["train_time"], params=r["params"])
            print(f"{name} seed {seed}: val WER {r['val_wer']:.1f} | TEST WER {m['wer']:.1f}, sentence acc {m['sent_acc']:.1f}%, "
                  f"closed-vocab sentence acc {100 * ok / len(split['test']):.1f}%", flush=True)
    # summary
    summ = {}
    for name in CONFIGS:
        rs = [v for k, v in metrics["runs"].items() if k.startswith(name)]
        summ[name] = {key: (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
                      for key, vals in {"test_wer": [x["test"]["wer"] for x in rs],
                                        "test_sent_acc": [x["test"]["sent_acc"] for x in rs],
                                        "closed_vocab_sent_acc": [x["constrained_sentence_acc"] for x in rs],
                                        "test_wer_missing_hand": [x["test_missing_hand"]["wer"] for x in rs],
                                        "test_wer_missing_pose": [x["test_missing_pose"]["wer"] for x in rs]}.items()}
        summ[name]["best_seed_by_val"] = min((v["val_wer"], k) for k, v in metrics["runs"].items() if k.startswith(name))[1]
    metrics["summary"] = summ
    (OUT_R / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
