"""List or fetch members of a remote zip via HTTP range requests (no full download)."""
import struct, sys, time, zipfile, zlib
import requests


def get_range(url, lo, hi, timeout=300, tries=8):
    """Range GET with back-off on rate limiting (HTTP 429 / 5xx) and network errors."""
    wait = 5
    for _ in range(tries):
        try:
            r = requests.get(url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=timeout)
            if r.status_code == 206:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                wait = int(r.headers.get("Retry-After", wait))
        except requests.RequestException:
            pass
        time.sleep(wait)
        wait = min(wait * 2, 120)
    raise RuntimeError(f"range request failed after {tries} tries: {url} {lo}-{hi}")


class RangeFile:
    def __init__(self, url):
        for k in range(8):
            r = requests.head(url, allow_redirects=True)
            if r.status_code == 200:
                break
            time.sleep(int(r.headers.get("Retry-After", 5 * 2 ** k)))
        self.url, self.size, self.pos = r.url, int(r.headers["Content-Length"]), 0
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, o, w=0):
        self.pos = o if w == 0 else (self.pos + o if w == 1 else self.size + o); return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0: return b""
        r = get_range(self.url, self.pos, self.pos + n - 1)
        self.pos += len(r.content); return r.content


def members(url):
    f = RangeFile(url)
    return f, zipfile.ZipFile(f).infolist()


def fetch(f, info):
    lo = info.header_offset
    hi = min(f.size - 1, lo + 30 + len(info.orig_filename.encode()) + 4096 + info.compress_size)
    buf = get_range(f.url, lo, hi, timeout=900).content
    assert struct.unpack("<I", buf[:4])[0] == 0x04034B50
    nlen, xlen = struct.unpack("<HH", buf[26:30])
    data = buf[30 + nlen + xlen: 30 + nlen + xlen + info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    assert zlib.crc32(data) & 0xFFFFFFFF == info.CRC, "CRC mismatch"
    return data


if __name__ == "__main__":
    for url in sys.argv[1:]:
        _, infos = members(url)
        for i in infos:
            print(url.split("/files/")[1].split("/")[0], i.file_size, i.filename, sep="\t")
