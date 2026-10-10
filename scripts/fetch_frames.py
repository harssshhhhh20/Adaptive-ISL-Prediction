"""Stream the Frames_Sentence_Level subset of ISL-CSLTR (Mendeley DOI 10.17632/kcmpdxky7p.1)
out of the remote 8.9 GB zip using HTTP range requests, downscale each frame, and save it.
Only ~1.4 GB is transferred; nothing else in the archive is downloaded."""
import io, os, sys, zipfile, json, requests
from PIL import Image

URL = "https://data.mendeley.com/public-files/datasets/kcmpdxky7p/files/c14c8953-be53-4c7a-85d5-5bcd7e79d86e/file_downloaded"
SIZE = 8905822924
BLOCK = 32 << 20
MAX_SIDE = int(os.environ.get("MAX_SIDE", 640))
# Optional local copy of the archive's leading bytes (where all sentence frames live).
LOCAL_PATH = os.environ.get("LOCAL_PART")
LOCAL = open(LOCAL_PATH, "rb") if LOCAL_PATH else None
LOCAL_SIZE = os.path.getsize(LOCAL_PATH) if LOCAL_PATH else 0


class CachedHTTPFile:
    def __init__(self):
        self.url = self._resolve(); self.pos = 0; self.cache = {}

    def _resolve(self):
        r = requests.head(URL, allow_redirects=False)
        return r.headers["location"]

    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, o, w=0):
        self.pos = o if w == 0 else (self.pos + o if w == 1 else SIZE + o)
        return self.pos

    def _block(self, b):
        if b not in self.cache:
            lo, hi = b * BLOCK, min(SIZE, (b + 1) * BLOCK) - 1
            for attempt in range(5):
                try:
                    r = requests.get(self.url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
                    if r.status_code == 403:
                        self.url = self._resolve(); continue
                    assert r.status_code == 206 and len(r.content) == hi - lo + 1
                    break
                except Exception as e:
                    print("retry", b, e, file=sys.stderr)
            else:
                raise RuntimeError(f"block {b} failed")
            if len(self.cache) > 4:
                self.cache.pop(min(self.cache))
            self.cache[b] = r.content
        return self.cache[b]

    def read(self, n=-1):
        if n is None or n < 0: n = SIZE - self.pos
        if LOCAL and self.pos + n <= LOCAL_SIZE:      # byte range already downloaded locally
            LOCAL.seek(self.pos); data = LOCAL.read(n); self.pos += len(data); return data
        out = bytearray()
        while n > 0 and self.pos < SIZE:
            b, off = divmod(self.pos, BLOCK)
            chunk = self._block(b)[off:off + n]
            out += chunk; self.pos += len(chunk); n -= len(chunk)
        return bytes(out)


def main(out_dir):
    z = zipfile.ZipFile(CachedHTTPFile())
    members = [i for i in z.infolist()
               if "/Frames_Sentence_Level/" in i.filename and i.filename.lower().endswith(".jpg")]
    members.sort(key=lambda i: i.header_offset)
    meta = {}
    for k, info in enumerate(members):
        _, _, sent, folder, fname = info.filename.split("/")
        dst = os.path.join(out_dir, sent.strip(), folder, fname)
        if os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        im = Image.open(io.BytesIO(z.read(info))).convert("RGB")
        meta[f"{sent.strip()}/{folder}"] = im.size
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        im.save(dst, quality=92)
        if k % 500 == 0:
            print(f"{k}/{len(members)}", flush=True)
    with open(os.path.join(out_dir, "_orig_sizes.json"), "w") as f:
        json.dump(meta, f)
    print("done", len(members))


if __name__ == "__main__":
    main(sys.argv[1])
