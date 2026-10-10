"""Does the adaptive fusion behave sensibly? (section 16 of the brief)

Uses the per-frame fusion weights saved for seed 0 of each LOSO fold and the per-frame quality
signals of the corresponding (possibly corrupted) input. Outputs results/fusion_analysis.json
and figures in results/figures/.
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from configs import CONFIGS  # noqa: E402
from corruptions import SequenceCorruptor  # noqa: E402
from data import seq_signer  # noqa: E402
from train import FEATURE_CONDS, RUNS, feature_condition, features  # noqa: E402

FIG = ROOT / "results" / "figures"
QNAMES = ["brightness", "contrast", "sharpness", "pose_visibility", "hand0_detected", "hand1_detected",
          "hand0_score", "hand1_score", "pose_jitter", "hand_jitter"]
COLORS = {"rgb": "#3b6fb6", "pose": "#d08a2c", "hand": "#3a9a5b"}


def load_weights(config, protocol="loso"):
    """{cond: {sid: [T, M] weights}} merged over folds (each sequence is tested in exactly one fold)."""
    out = defaultdict(dict)
    for wf in sorted((RUNS / config / protocol).glob("*/seed0.weights.pkl")):
        with open(wf, "rb") as f:
            for cond, d in pickle.load(f).items():
                out[cond].update({k: v.astype(np.float32) for k, v in d.items()})
    return dict(out)


def load_items(config, protocol="loso"):
    out = defaultdict(dict)
    for rf in sorted((RUNS / config / protocol).glob("*/seed0.json")):
        r = json.loads(rf.read_text())
        for cond, v in r["conds"].items():
            for it in v["items"]:
                out[cond][it["id"]] = it
    return dict(out)


def quality_for(cond, sids):
    if cond in FEATURE_CONDS:
        clean = features("clean")
        fe = feature_condition({k: clean[k] for k in sids}, cond)
    else:
        fe = features(cond)
    return {k: fe[k]["quality"] for k in sids if k in fe}


def analyse(config):
    mods = CONFIGS[config]["mods"]
    W = load_weights(config)
    items = load_items(config)
    if "clean" not in W:
        print("no weights for", config)
        return None
    res = {"config": config, "mods": mods, "mean_weights": {}, "corr_frame": {}, "corr_seq_error": {}}

    # 1) mean modality weight per condition
    for cond, d in W.items():
        allw = np.concatenate(list(d.values()))
        res["mean_weights"][cond] = dict(zip(mods, allw.mean(0).round(4).tolist()))

    # 2) frame-level Spearman correlation between each modality weight and each quality signal,
    #    pooled over all conditions (variation in quality comes mostly from the corruptions)
    Wf, Qf = [], []
    for cond, d in W.items():
        q = quality_for(cond, list(d))
        for sid, w in d.items():
            if sid in q and len(q[sid]) == len(w):
                Wf.append(w); Qf.append(q[sid])
    Wf, Qf = np.concatenate(Wf), np.concatenate(Qf)
    for i, m in enumerate(mods):
        res["corr_frame"][m] = {}
        for j, qn in enumerate(QNAMES):
            if np.std(Qf[:, j]) > 0:
                r, p = st.spearmanr(Wf[:, i], Qf[:, j])
                res["corr_frame"][m][qn] = dict(rho=float(r), p=float(p))
    res["n_frames"] = int(len(Wf))

    # 3) sequence-level: does putting weight on a degraded modality go with more errors?
    #    and is the gate's confidence (low weight entropy) associated with lower error?
    ent, err, hand_w, hand_det, rgb_w, bright = [], [], [], [], [], []
    for cond, d in W.items():
        q = quality_for(cond, list(d))
        for sid, w in d.items():
            it = items.get(cond, {}).get(sid)
            if it is None or sid not in q:
                continue
            err.append(it["err"] / max(it["n"], 1))
            ent.append(float(np.mean(-(w * np.log(w + 1e-9)).sum(1))))
            if "hand" in mods:
                hand_w.append(w[:, mods.index("hand")].mean()); hand_det.append(q[sid][:, 4:6].mean())
            if "rgb" in mods:
                rgb_w.append(w[:, mods.index("rgb")].mean()); bright.append(q[sid][:, 0].mean())
    r = st.spearmanr(ent, err)
    res["corr_seq_error"]["gate_entropy_vs_seq_wer"] = dict(rho=float(r[0]), p=float(r[1]), n=len(err))
    if hand_w:
        r = st.spearmanr(hand_w, hand_det)
        res["corr_seq_error"]["hand_weight_vs_hand_detection_rate"] = dict(rho=float(r[0]), p=float(r[1]))
    if rgb_w:
        r = st.spearmanr(rgb_w, bright)
        res["corr_seq_error"]["rgb_weight_vs_brightness"] = dict(rho=float(r[0]), p=float(r[1]))

    # 4) within-sequence response to occlusion: weight inside vs outside the occluded frames
    occ = {}
    for cond in ("hand_occ", "body_occ"):
        if cond not in W:
            continue
        inside, outside = [], []
        for sid, w in W[cond].items():
            T = len(w)
            f = features(cond)[sid]
            cor = SequenceCorruptor(cond, sid, T, 1, 1, [{"hands": np.zeros((2, 21, 3))}] * T)
            mask = np.zeros(T, bool)
            mask[list(cor.frames)] = True
            if mask.all() or not mask.any():
                continue
            inside.append(w[mask].mean(0)); outside.append(w[~mask].mean(0))
        inside, outside = np.array(inside), np.array(outside)
        occ[cond] = {}
        for i, m in enumerate(mods):
            d = inside[:, i] - outside[:, i]
            occ[cond][m] = dict(inside=float(inside[:, i].mean()), outside=float(outside[:, i].mean()),
                                mean_diff=float(d.mean()), wilcoxon_p=float(st.wilcoxon(d).pvalue), n=len(d))
    res["occlusion_response"] = occ

    # 5) per-signer mean weights on clean data (do weights shift between recording domains?)
    by_signer = defaultdict(list)
    for sid, w in W["clean"].items():
        by_signer[seq_signer(sid)].append(w.mean(0))
    res["clean_weights_by_signer"] = {s: dict(zip(mods, np.mean(v, 0).round(4).tolist())) for s, v in sorted(by_signer.items())}
    return res, W


def plot(config, res, W):
    FIG.mkdir(parents=True, exist_ok=True)
    mods = res["mods"]
    # bar chart: mean weights per condition
    conds = [c for c in ["clean", "lowlight", "uneven", "hand_occ", "body_occ", "background", "blur", "distance",
                         "viewpoint", "combined", "missing_rgb", "missing_pose", "missing_hand", "fast", "slow"] if c in res["mean_weights"]]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    bottom = np.zeros(len(conds))
    for m in mods:
        v = np.array([res["mean_weights"][c][m] for c in conds])
        ax.bar(conds, v, bottom=bottom, color=COLORS[m], label=m, width=0.7)
        bottom += v
    ax.set_ylabel("mean fusion weight"); ax.set_ylim(0, 1); ax.legend(ncol=3, frameon=False, loc="upper right")
    ax.set_title(f"{config}: mean modality weight by test condition (LOSO, seed 0)")
    plt.xticks(rotation=35, ha="right"); plt.tight_layout()
    fig.savefig(FIG / f"weights_by_condition_{config}.png", dpi=150); plt.close(fig)

    # time course examples: same sequence clean vs hand occlusion, occluded span shaded
    if "hand_occ" in W:
        sids = sorted(W["hand_occ"])[::max(1, len(W["hand_occ"]) // 4)][:4]
        fig, axes = plt.subplots(len(sids), 2, figsize=(11, 2.2 * len(sids)), sharey=True)
        for r, sid in enumerate(sids):
            for c, cond in enumerate(["clean", "hand_occ"]):
                w = W[cond][sid]
                ax = axes[r, c]
                ax.stackplot(np.arange(len(w)), *[w[:, i] for i in range(len(mods))],
                             colors=[COLORS[m] for m in mods], labels=mods, alpha=0.9)
                if cond == "hand_occ":
                    cor = SequenceCorruptor(cond, sid, len(w), 1, 1, [{"hands": np.zeros((2, 21, 3))}] * len(w))
                    fr = sorted(cor.frames)
                    ax.axvspan(fr[0] - 0.5, fr[-1] + 0.5, color="k", alpha=0.15, label="hands occluded")
                ax.set_xlim(0, len(w) - 1); ax.set_ylim(0, 1)
                ax.set_title(f"{sid} — {cond}", fontsize=8)
                if r == 0 and c == 1:
                    ax.legend(fontsize=7, loc="upper right", ncol=4, frameon=False)
        axes[-1, 0].set_xlabel("frame"); axes[-1, 1].set_xlabel("frame")
        plt.tight_layout(); fig.savefig(FIG / f"weights_over_time_{config}.png", dpi=150); plt.close(fig)


def main():
    configs = sys.argv[1:] or ["L_trel_gate", "K_rel_gate", "J_gate"]
    out = {}
    for c in configs:
        r = analyse(c)
        if r:
            out[c], W = r
            plot(c, out[c], W)
            print(json.dumps({k: out[c][k] for k in ("mean_weights", "corr_seq_error", "occlusion_response")}, indent=1)[:3000])
    (ROOT / "results" / "fusion_analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
