"""Fetch selected ISL-CSLTR sentence videos straight out of the remote 8.9 GB zip by HTTP byte
range (only the chosen files are transferred), then store a 640-px-wide copy.

python scripts/fetch_videos.py <sentence_list.txt> [--workers 6]
"""
import argparse
import io
import os
import struct
import sys
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from glosses import sentence_key  # noqa: E402

URL = "https://data.mendeley.com/public-files/datasets/kcmpdxky7p/files/c14c8953-be53-4c7a-85d5-5bcd7e79d86e/file_downloaded"
SIZE = 8905822924
OUT = ROOT / "data" / "videos"
RAW = ROOT / "data" / "_raw_videos"


def s3_url():
    return requests.head(URL, allow_redirects=False).headers["location"]


class RangeFile:
    """Minimal seekable remote file, enough for zipfile to read the central directory."""

    def __init__(self, url):
        self.url, self.pos = url, 0

    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, o, w=0):
        self.pos = o if w == 0 else (self.pos + o if w == 1 else SIZE + o)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = SIZE - self.pos
        r = requests.get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"}, timeout=120)
        self.pos += len(r.content)
        return r.content


def fetch_member(url, info):
    """Download one member's bytes (local header + data) and decompress it."""
    lo = info.header_offset
    hi = lo + 30 + len(info.orig_filename.encode()) + 2048 + info.compress_size
    for _ in range(6):
        try:
            r = requests.get(url, headers={"Range": f"bytes={lo}-{min(hi, SIZE - 1)}"}, timeout=600)
            if r.status_code == 206:
                break
        except requests.RequestException:
            pass
    buf = r.content
    sig, *_ = struct.unpack("<I", buf[:4])
    assert sig == 0x04034B50, "bad local header"
    nlen, xlen = struct.unpack("<HH", buf[26:30])
    data = buf[30 + nlen + xlen: 30 + nlen + xlen + info.compress_size]
    assert len(data) == info.compress_size, "truncated"
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    assert zlib.crc32(data) & 0xFFFFFFFF == info.CRC, "CRC mismatch"
    return data


def shrink(src, dst, width=640):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, width / max(w, 1))
    size = (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2)
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"avc1"), fps, size)
    if not vw.isOpened():
        vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        vw.write(cv2.resize(f, size, interpolation=cv2.INTER_AREA) if scale < 1 else f)
        n += 1
    vw.release()
    cap.release()
    return n, fps, (w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sentences")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reverse", action="store_true", help="process files in reverse order (run a 2nd fetcher)")
    a = ap.parse_args()
    wanted = {l.strip() for l in open(a.sentences) if l.strip()}
    url = s3_url()
    z = zipfile.ZipFile(RangeFile(url))
    members = [i for i in z.infolist() if "/Videos_Sentence_Level/" in i.filename and i.filename.lower().endswith(".mp4")
               and sentence_key(i.filename.split("/")[2]) in wanted]
    print(f"{len(members)} videos, {sum(i.compress_size for i in members) / 1e9:.2f} GB", flush=True)
    RAW.mkdir(parents=True, exist_ok=True)
    if a.reverse:
        members = members[::-1]

    def job(info):
        sent = sentence_key(info.filename.split("/")[2])
        name = os.path.splitext(os.path.basename(info.filename))[0].replace(" ", "_")
        dst = OUT / sent / f"{name}.mp4"
        tmp_marker = RAW / f"{sent.replace(' ', '_')}__{name}.mp4"
        if dst.exists() or tmp_marker.exists():
            return f"skip {dst.name}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = RAW / f"{sent.replace(' ', '_')}__{name}.mp4"
        tmp.write_bytes(fetch_member(url, info))
        n, fps, orig = shrink(tmp, dst)
        tmp.unlink()
        return f"{sent}/{name}: {n} frames @ {fps:.0f} fps, orig {orig[0]}x{orig[1]}"

    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(job, m) for m in members]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                print(f"[{k}/{len(members)}] {f.result()}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{k}/{len(members)}] FAILED {e!r}", flush=True)


if __name__ == "__main__":
    main()
