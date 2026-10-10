"""Statistical analysis of run results (see reports/PREREGISTRATION.md for the plan)."""
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as st
from statsmodels.stats.multitest import multipletests

from data import seq_signer

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"


def load(config, protocol):
    """{split: {seed: result_json}}"""
    out = defaultdict(dict)
    base = RUNS / config / protocol
    if not base.exists():
        return {}
    for f in sorted(base.glob("*/seed*.json")):
        r = json.loads(f.read_text())
        out[f.parent.name][r["seed"]] = r
    return dict(out)


def seeds_available(runs):
    sets = [set(v) for v in runs.values()]
    return sorted(set.intersection(*sets)) if sets else []


def item_matrix(runs, cond="clean", seeds=None, key="err"):
    """Per-test-sequence errors averaged over seeds, pooled over all splits of a protocol.
    Returns (ids, mean_err, n, per_seed_err [S, N])."""
    seeds = seeds if seeds is not None else seeds_available(runs)
    rows = {}
    for split, by_seed in runs.items():
        for s in seeds:
            if cond not in by_seed[s]["conds"]:
                continue
            for it in by_seed[s]["conds"][cond]["items"]:
                rows.setdefault((split, it["id"]), {"n": it["n"], "e": {}})["e"][s] = it[key]
    ids = sorted(rows)
    if not ids:
        raise ValueError(f"no evaluated items for condition {cond}")
    E = np.array([[rows[i]["e"].get(s, np.nan) for i in ids] for s in seeds], dtype=float)
    n = np.array([rows[i]["n"] for i in ids], dtype=float)
    return ids, np.nanmean(E, 0), n, E


def wer(err, n):
    return 100.0 * np.sum(err) / np.sum(n)


def per_seed_wer(E, n):
    return np.array([wer(e, n) for e in E])


def paired_permutation(ea, eb, n, iters=10000, seed=0):
    """Approximate randomisation: swap the two systems' outputs per sequence."""
    rng = np.random.default_rng(seed)
    obs = (ea.sum() - eb.sum()) / n.sum()
    d = ea - eb
    flips = rng.choice([-1.0, 1.0], size=(iters, len(d)))
    null = (flips * d).sum(1) / n.sum()
    p = (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (iters + 1)
    return 100 * obs, p


def cluster_bootstrap(ea, eb, n, clusters, iters=10000, seed=0):
    """95% CI of WER(a)-WER(b) and relative reduction (WER(b)-WER(a))/WER(b), resampling clusters."""
    rng = np.random.default_rng(seed)
    cl = np.asarray(clusters)
    uniq = np.unique(cl)
    idx = {c: np.where(cl == c)[0] for c in uniq}
    A = np.array([ea[idx[c]].sum() for c in uniq])
    B = np.array([eb[idx[c]].sum() for c in uniq])
    N = np.array([n[idx[c]].sum() for c in uniq])
    draws = rng.integers(0, len(uniq), size=(iters, len(uniq)))
    a, b, nn = A[draws].sum(1), B[draws].sum(1), N[draws].sum(1)
    diff = 100 * (a - b) / nn
    rel = 100 * (b - a) / np.maximum(b, 1e-9)
    return dict(diff_ci=np.percentile(diff, [2.5, 97.5]).tolist(), rel_ci=np.percentile(rel, [2.5, 97.5]).tolist())


def signflip_exact(diffs):
    diffs = np.asarray(diffs, float)
    obs = abs(diffs.mean())
    null = [abs((np.array(s) * diffs).mean()) for s in itertools.product([-1, 1], repeat=len(diffs))]
    return float(np.mean(np.array(null) >= obs - 1e-12))


def seed_level(wa, wb):
    d = np.asarray(wa) - np.asarray(wb)
    res = dict(mean_a=float(np.mean(wa)), sd_a=float(np.std(wa, ddof=1)), mean_b=float(np.mean(wb)),
               sd_b=float(np.std(wb, ddof=1)), mean_diff=float(d.mean()))
    if len(d) > 1:
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = st.t.ppf(0.975, len(d) - 1)
        res["diff_ci_t"] = [float(d.mean() - t * se), float(d.mean() + t * se)]
        res["cohen_dz"] = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("inf")
        if np.any(d != 0):
            w = st.wilcoxon(wa, wb, zero_method="wilcox")
            res["wilcoxon_p"] = float(w.pvalue)
            pos = np.sum(st.rankdata(np.abs(d[d != 0]))[d[d != 0] > 0])
            neg = np.sum(st.rankdata(np.abs(d[d != 0]))[d[d != 0] < 0])
            res["rank_biserial"] = float((pos - neg) / (pos + neg))
    return res


def ci_mean(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float("nan")] * 2
    se = x.std(ddof=1) / np.sqrt(len(x))
    t = st.t.ppf(0.975, len(x) - 1)
    return [float(x.mean() - t * se), float(x.mean() + t * se)]


def compare(runs_a, runs_b, cond="clean", seeds=None, iters=10000):
    """Full paired comparison of system a vs system b on one protocol/condition (a = proposed)."""
    seeds = seeds or sorted(set(seeds_available(runs_a)) & set(seeds_available(runs_b)))
    ia, ea, n, Ea = item_matrix(runs_a, cond, seeds)
    ib, eb, nb, Eb = item_matrix(runs_b, cond, seeds)
    assert ia == ib, "systems were not evaluated on identical test items"
    obs, p = paired_permutation(ea, eb, n, iters)
    sentences = [i[1].split("|")[0] for i in ia]
    boot = cluster_bootstrap(ea, eb, n, sentences, iters)
    wa, wb = per_seed_wer(Ea, n), per_seed_wer(Eb, n)
    signer = defaultdict(lambda: [0.0, 0.0, 0.0])
    for k, i in enumerate(ia):
        s = signer[seq_signer(i[1])]
        s[0] += ea[k]; s[1] += eb[k]; s[2] += n[k]
    sdiff = {k: 100 * (v[0] - v[1]) / v[2] for k, v in sorted(signer.items())}
    out = dict(cond=cond, seeds=seeds, n_items=len(ia), wer_a=wer(ea, n), wer_b=wer(eb, n), diff=obs,
               rel_reduction=100 * (eb.sum() - ea.sum()) / max(eb.sum(), 1e-9), perm_p=p, **boot,
               seed_level=seed_level(wa, wb), per_signer_diff=sdiff)
    if len(sdiff) >= 5:
        out["signer_signflip_p"] = signflip_exact(list(sdiff.values()))
    return out


def holm(pvals):
    rej, padj, _, _ = multipletests(pvals, method="holm")
    return padj.tolist(), rej.tolist()


def run_level_matrix(configs, protocol, cond="clean"):
    """WER per (split, seed) block for each config -> [blocks, configs] matrix (common blocks only)."""
    per = {}
    for c in configs:
        r = load(c, protocol)
        per[c] = {(sp, s): v["conds"][cond]["metrics"]["wer"] for sp, d in r.items() for s, v in d.items()
                  if cond in v["conds"]}
    blocks = sorted(set.intersection(*[set(v) for v in per.values()]))
    return blocks, np.array([[per[c][b] for c in configs] for b in blocks])


def pooled_wer_by_seed(runs, cond="clean"):
    seeds = seeds_available(runs)
    _, _, n, E = item_matrix(runs, cond, seeds)
    return per_seed_wer(E, n)
