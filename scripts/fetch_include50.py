"""Fetch every INCLUDE-50 clip (official split lists) that is not already downloaded, by HTTP byte
range from the Zenodo zips (Zenodo rate-limits, so requests back off on HTTP 429), and store a
640-px copy under data/include/videos/<category>/<sign>/<clip>.mp4.

python scripts/fetch_include50.py [--workers 6]
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_videos import shrink  # noqa: E402
from remote_zip import fetch, members  # noqa: E402

API = "https://zenodo.org/api/records/4010759"
OUT = ROOT / "data" / "include" / "videos"
TMP = ROOT / "data" / "include" / "_raw"


def stem_key(path):
    """'Animals/1. Dog/MVI_8560.MOV' -> 'animals/1. dog/mvi_8560' (split lists and zips differ in extension case)."""
    return os.path.splitext(path)[0].lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    want = {stem_key(l.strip()): l.strip() for n in ("train", "val", "test")
            for l in open(ROOT / "data" / "include" / f"include50_{n}.txt") if l.strip()}
    zips = [f["key"] for f in requests.get(API).json()["files"] if f["key"].endswith(".zip")]
    jobs = []
    for z in zips:
        f, infos = members(f"{API}/files/{z}/content")
        for i in infos:
            k = stem_key(i.filename)
            if k in want and i.filename.count("/") in (2, 3):          # some official clips sit in <sign>/Extra/
                parts = i.filename.split("/")
                dst = OUT.joinpath(*parts[:-1]) / (Path(parts[-1]).stem + ".mp4")
                if not dst.exists():
                    jobs.append((f, i, dst))
                want.pop(k)
    print(f"{len(jobs)} clips to fetch ({sum(i.compress_size for _, i, _ in jobs) / 1e9:.2f} GB); "
          f"{len(want)} listed clips not found in any zip", flush=True)
    TMP.mkdir(parents=True, exist_ok=True)

    def job(f, info, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = TMP / info.filename.replace("/", "__").replace(" ", "_")
        tmp.write_bytes(fetch(f, info))
        n, fps, _ = shrink(tmp, dst)
        tmp.unlink()
        return f"{info.filename}: {n} frames"

    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(job, *j) for j in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                print(f"[{k}/{len(jobs)}] {fu.result()}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{k}/{len(jobs)}] FAILED {e!r}", flush=True)


if __name__ == "__main__":
    main()
