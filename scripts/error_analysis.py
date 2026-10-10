"""Quantitative and qualitative error analysis (section 17 of the brief).

python scripts/error_analysis.py [config ...]   -> results/error_analysis.json, results/error_examples.md
Uses all seeds of the LOSO runs (each test sequence's hypotheses from every seed).
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import stats as S  # noqa: E402
from data import PHONE, seq_sentence, seq_signer  # noqa: E402
from glosses import load_gloss_table  # noqa: E402
from train import features  # noqa: E402


def alignment(ref, hyp):
    """Levenshtein alignment as a list of (op, ref_tok, hyp_tok, ref_pos)."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), int)
    D[:, 0] = range(n + 1); D[0, :] = range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("C" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1], i - 1)); i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append(("D", ref[i - 1], None, i - 1)); i -= 1
        else:
            ops.append(("I", None, hyp[j - 1], i)); j -= 1
    return ops[::-1]


def wer_of(items):
    return 100 * sum(e for e, _ in items) / max(1, sum(n for _, n in items))


def analyse(config, cond="clean"):
    table = load_gloss_table()
    runs = S.load(config, "loso")
    seeds = S.seeds_available(runs)
    gloss_sent_freq = Counter(g for v in table.values() for g in set(v))
    recs = []
    for sp in runs.values():
        for s in seeds:
            for it in sp[s]["conds"].get(cond, {}).get("items", []):
                ref = table[seq_sentence(it["id"])]
                hyp = it["hyp"].split()
                recs.append(dict(id=it["id"], seed=s, ref=ref, hyp=hyp, err=it["err"], n=it["n"], ops=alignment(ref, hyp)))
    if not recs:
        return None
    out = {"config": config, "cond": cond, "n_hyps": len(recs), "wer": wer_of([(r["err"], r["n"]) for r in recs])}

    # most confused glosses, deletions and insertions
    subs, dels, ins, ref_count, correct = Counter(), Counter(), Counter(), Counter(), Counter()
    for r in recs:
        for op, a, b, _ in r["ops"]:
            if a:
                ref_count[a] += 1
            if op == "S":
                subs[(a, b)] += 1
            elif op == "D":
                dels[a] += 1
            elif op == "I":
                ins[b] += 1
            elif op == "C":
                correct[a] += 1
    out["top_substitutions"] = [(f"{a} -> {b}", c) for (a, b), c in subs.most_common(15)]
    out["top_deletions"] = dels.most_common(15)
    out["top_insertions"] = ins.most_common(15)
    acc = {g: correct[g] / ref_count[g] for g in ref_count}
    out["gloss_accuracy_by_sentence_frequency"] = {}
    for k, name in [(1, "gloss in 1 sentence"), (2, "gloss in 2-3 sentences"), (4, "gloss in >=4 sentences")]:
        gs = [g for g in acc if (gloss_sent_freq[g] == 1 if k == 1 else (2 <= gloss_sent_freq[g] <= 3 if k == 2 else gloss_sent_freq[g] >= 4))]
        tot = sum(ref_count[g] for g in gs)
        out["gloss_accuracy_by_sentence_frequency"][name] = dict(
            n_glosses=len(gs), recall=100 * sum(correct[g] for g in gs) / max(tot, 1))

    # signer-specific and domain-specific failures
    by_signer = defaultdict(list)
    for r in recs:
        by_signer[seq_signer(r["id"])].append((r["err"], r["n"]))
    out["wer_by_signer"] = {s: round(wer_of(v), 2) for s, v in sorted(by_signer.items())}
    out["wer_phone_vs_dslr"] = dict(phone=wer_of([x for s in PHONE for x in by_signer[s]]),
                                    dslr=wer_of([x for s, v in by_signer.items() if s not in PHONE for x in v]))

    # sentence length and position of errors (boundary effects)
    by_len = defaultdict(list)
    pos_err = defaultdict(lambda: [0, 0])
    for r in recs:
        by_len[len(r["ref"])].append((r["err"], r["n"]))
        L = len(r["ref"])
        for op, a, b, i in r["ops"]:
            if op in ("C", "S", "D") and L >= 3:
                where = "first" if i == 0 else ("last" if i == L - 1 else "middle")
                pos_err[where][0] += op != "C"; pos_err[where][1] += 1
    out["wer_by_ref_length"] = {k: round(wer_of(v), 2) for k, v in sorted(by_len.items())}
    out["error_rate_by_position(len>=3)"] = {k: round(100 * v[0] / v[1], 2) for k, v in pos_err.items()}

    # landmark-tracking failures: per-sequence hand-detection rate vs per-sequence WER
    # (needs the extracted study features; skipped if they have been removed)
    seq_err = defaultdict(list)
    for r in recs:
        seq_err[r["id"]].append(r["err"] / r["n"])
    try:
        fe = features(cond)
    except FileNotFoundError:
        fe = None
    if fe is None:
        by_sent = defaultdict(list)
        for r in recs:
            by_sent[seq_sentence(r["id"])].append((r["err"], r["n"]))
        sw = sorted(((round(wer_of(v), 1), s) for s, v in by_sent.items()), reverse=True)
        out["hardest_sentences"], out["easiest_sentences"] = sw[:10], sw[-10:]
        out["examples"] = {}
        seed0 = [r for r in recs if r["seed"] == seeds[0]]
        rng = np.random.default_rng(0)
        for name, sel in [("correct", [r for r in seed0 if r["err"] == 0]), ("one_error", [r for r in seed0 if r["err"] == 1]),
                          ("bad", [r for r in seed0 if r["err"] >= max(2, r["n"])])]:
            pick = [sel[i] for i in rng.permutation(len(sel))[:6]] if sel else []
            out["examples"][name] = [dict(id=r["id"], ref=" ".join(r["ref"]), hyp=" ".join(r["hyp"])) for r in pick]
        return out
    det, pv, ew = [], [], []
    seq_err = defaultdict(list)
    for r in recs:
        seq_err[r["id"]].append(r["err"] / r["n"])
    for sid, v in seq_err.items():
        q = fe[sid]["quality"]
        det.append(q[:, 4:6].max(1).mean()); pv.append(q[:, 3].mean()); ew.append(np.mean(v))
    r1, r2 = st.spearmanr(det, ew), st.spearmanr(pv, ew)
    out["tracking"] = dict(hand_detect_rate_vs_wer=dict(rho=float(r1[0]), p=float(r1[1])),
                           pose_visibility_vs_wer=dict(rho=float(r2[0]), p=float(r2[1])), n_sequences=len(ew))
    lo = np.array(det) < np.percentile(det, 20)
    out["tracking"]["wer_worst20pct_hand_detection"] = 100 * float(np.mean(np.array(ew)[lo]))
    out["tracking"]["wer_other80pct"] = 100 * float(np.mean(np.array(ew)[~lo]))

    # temporal-length (signing speed proxy): frames per gloss vs WER
    fpg = [len(fe[sid]["rgb"]) / len(table[seq_sentence(sid)]) for sid in seq_err]
    r3 = st.spearmanr(fpg, ew)
    out["frames_per_gloss_vs_wer"] = dict(rho=float(r3[0]), p=float(r3[1]))

    # per-sentence WER: hardest sentences
    by_sent = defaultdict(list)
    for r in recs:
        by_sent[seq_sentence(r["id"])].append((r["err"], r["n"]))
    sw = sorted(((round(wer_of(v), 1), s) for s, v in by_sent.items()), reverse=True)
    out["hardest_sentences"] = sw[:10]
    out["easiest_sentences"] = sw[-10:]
    out["examples"] = {}
    seed0 = [r for r in recs if r["seed"] == seeds[0]]
    rng = np.random.default_rng(0)
    for name, sel in [("correct", [r for r in seed0 if r["err"] == 0]),
                      ("one_error", [r for r in seed0 if r["err"] == 1]),
                      ("bad", [r for r in seed0 if r["err"] >= max(2, r["n"])])]:
        pick = [sel[i] for i in rng.permutation(len(sel))[:6]] if sel else []
        out["examples"][name] = [dict(id=r["id"], ref=" ".join(r["ref"]), hyp=" ".join(r["hyp"])) for r in pick]
    return out


def main():
    configs = sys.argv[1:] or ["L_trel_gate", "G_concat", "A_rgb"]
    res = {}
    for c in configs:
        for cond in ["clean"]:
            try:
                r = analyse(c, cond)
            except FileNotFoundError:
                r = None
            if r:
                res[f"{c}|{cond}"] = r
    (ROOT / "results" / "error_analysis.json").write_text(json.dumps(res, indent=1, default=str))
    lines = ["# Error analysis examples (seed 0, LOSO)\n"]
    for k, r in res.items():
        if not k.endswith("|clean"):
            continue
        lines.append(f"## {k} — WER {r['wer']:.1f}\n")
        for name, ex in r["examples"].items():
            lines.append(f"**{name}**\n")
            lines += [f"- `{e['id']}`  REF: {e['ref']}  |  HYP: {e['hyp'] or '∅'}" for e in ex]
            lines.append("")
    (ROOT / "results" / "error_examples.md").write_text("\n".join(lines))
    for k, r in res.items():
        print(k, json.dumps({x: r.get(x) for x in ("wer", "wer_by_signer", "wer_phone_vs_dslr", "top_substitutions",
                                                   "top_deletions", "gloss_accuracy_by_sentence_frequency",
                                                   "wer_by_ref_length", "error_rate_by_position(len>=3)",
                                                   "hardest_sentences")}, default=str)[:2500])


if __name__ == "__main__":
    main()
