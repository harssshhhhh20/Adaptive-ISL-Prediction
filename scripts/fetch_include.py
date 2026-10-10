"""Fetch selected INCLUDE sign folders from the Zenodo zips by byte range, shrink to 640 px."""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_videos import shrink  # noqa: E402
from remote_zip import fetch, members  # noqa: E402

BASE = "https://zenodo.org/api/records/4010759/files/{}/content"
ZIPS = ["Pronouns_1of2.zip", "Pronouns_2of2.zip", "Greetings_1of2.zip", "Greetings_2of2.zip",
        "Adjectives_1of8.zip", "Adjectives_2of8.zip"]
SIGNS = {"40. I", "41. you", "42. he", "43. she", "44. it", "45. we", "47. they",
         "48. Hello", "49. How are you", "50. Alright", "51. Good Morning", "52. Good afternoon",
         "53. Good evening", "54. Good night", "55. Thank you", "56. Pleased",
         "1. loud", "2. quiet", "3. happy", "4. sad", "5. Beautiful", "7. Deaf", "11. rich", "12. poor"}
OUT = ROOT / "data" / "include" / "videos"
TMP = ROOT / "data" / "include" / "_raw"


def main():
    jobs = []
    for z in ZIPS:
        f, infos = members(BASE.format(z))
        for i in infos:
            parts = i.filename.split("/")
            if len(parts) == 3 and parts[1] in SIGNS and parts[2].lower().endswith((".mov", ".mp4")):
                jobs.append((f, i))
    print(len(jobs), "videos", round(sum(i.compress_size for _, i in jobs) / 1e9, 2), "GB", flush=True)
    TMP.mkdir(parents=True, exist_ok=True)

    def job(f, info):
        cat, sign, name = info.filename.split("/")
        dst = OUT / cat / sign / (Path(name).stem + ".mp4")
        if dst.exists():
            return "skip"
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = TMP / f"{cat}_{sign}_{name}".replace(" ", "_")
        tmp.write_bytes(fetch(f, info))
        n, fps, orig = shrink(tmp, dst)
        tmp.unlink()
        return f"{sign}/{name}: {n} frames @ {fps:.0f}"

    with ThreadPoolExecutor(12) as ex:
        futs = [ex.submit(job, f, i) for f, i in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                print(f"[{k}/{len(jobs)}] {fu.result()}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{k}/{len(jobs)}] FAILED {e!r}", flush=True)


if __name__ == "__main__":
    main()
