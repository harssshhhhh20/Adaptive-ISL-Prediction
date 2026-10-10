"""Run a grid of (config x protocol x split x seed) jobs in parallel worker processes.

python scripts/run.py --configs A_rgb G_concat --protocols loso --seeds 0-4 --workers 4
Existing result files are skipped, so the script can be re-run to resume.
"""
import argparse
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def _work(job):
    from train import run_job
    name, cfg, split, seed, out, conds, keep_w = job
    t = time.time()
    try:
        r = run_job(cfg, split, seed, out, eval_conds=conds, keep_weights=keep_w)
        return f"{name} {split['name']} s{seed}: val {r['val_wer']:.1f} test {r['conds']['clean']['metrics']['wer']:.1f} ({time.time() - t:.0f}s)"
    except Exception:
        return f"FAILED {name} {split['name']} s{seed}\n{traceback.format_exc()}"


def main():
    from configs import CONFIGS
    from data import make_splits
    from train import RUNS, features

    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--protocols", nargs="+", default=["loso"])
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--clean-only", action="store_true", help="evaluate only the clean condition")
    args = ap.parse_args()

    ids = sorted(features("clean").keys())
    splits = make_splits(ids)
    jobs = []
    for name in args.configs:
        cfg = CONFIGS[name]
        for proto in args.protocols:
            for sp in splits[proto]:
                for seed in parse_seeds(args.seeds):
                    out = RUNS / name / proto / sp["name"] / f"seed{seed}.json"
                    if out.exists():
                        continue
                    conds = ["clean"] if args.clean_only else None
                    jobs.append((name, cfg, sp, seed, out, conds, seed == 0))
    print(f"{len(jobs)} jobs", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=8) as pool:
        for k, msg in enumerate(pool.imap_unordered(_work, jobs), 1):
            print(f"[{k}/{len(jobs)}] {msg}", flush=True)


if __name__ == "__main__":
    main()
