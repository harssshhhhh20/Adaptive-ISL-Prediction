"""Gloss-level CSLR metrics. WER is the primary metric (lower is better)."""
from collections import Counter

import numpy as np


def align_counts(ref, hyp):
    """Levenshtein alignment returning (substitutions, deletions, insertions)."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, S, Dl, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            S += ref[i - 1] != hyp[j - 1]; i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            Dl += 1; i -= 1
        else:
            I += 1; j -= 1
    return S, Dl, I


def edit_distance(a, b):
    return sum(align_counts(a, b))


def per_item(ref, hyp):
    S, D, I = align_counts(ref, hyp)
    rc, hc = " ".join(ref), " ".join(hyp)
    overlap = sum((Counter(ref) & Counter(hyp)).values())
    return dict(err=S + D + I, sub=S, dele=D, ins=I, n=len(ref), cerr=edit_distance(list(rc), list(hc)),
                cn=len(rc), exact=int(ref == hyp), tp=overlap, nh=len(hyp))


def corpus(items):
    """Corpus-level metrics from a list of per_item dicts."""
    s = {k: sum(it[k] for it in items) for k in items[0]}
    prec = s["tp"] / max(1, s["nh"])
    rec = s["tp"] / max(1, s["n"])
    return dict(
        wer=100 * s["err"] / s["n"], sub=100 * s["sub"] / s["n"], dele=100 * s["dele"] / s["n"],
        ins=100 * s["ins"] / s["n"], cer=100 * s["cerr"] / s["cn"], sent_acc=100 * s["exact"] / len(items),
        precision=100 * prec, recall=100 * rec, f1=100 * 2 * prec * rec / max(1e-9, prec + rec),
    )
