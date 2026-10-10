"""Parameters, FLOPs and CPU latency of every model configuration, plus feature-extractor cost.

FLOPs: torch.utils.flop_counter for matmul/linear ops, plus analytic counts for LSTM/GRU
(which the counter does not see). Reported per sequence of T=28 frames (corpus mean).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from configs import CONFIGS  # noqa: E402
from models import IN_DIMS, Q_DIM, SentenceClassifier, build_model  # noqa: E402

T, V = 28, 168


def rnn_flops(model, T):
    total = 0
    for m in model.modules():
        if isinstance(m, (nn.LSTM, nn.GRU)):
            gates = 4 if isinstance(m, nn.LSTM) else 3
            dirs = 2 if m.bidirectional else 1
            h = m.hidden_size
            for layer in range(m.num_layers):
                inp = m.input_size if layer == 0 else h * dirs
                total += dirs * T * 2 * gates * (inp * h + h * h)
    return total


def batch(B=1):
    b = {m: torch.randn(B, T, d) for m, d in IN_DIMS.items()}
    b["quality"] = torch.rand(B, T, Q_DIM)
    b["lengths"] = torch.full((B,), T)
    return b


def main():
    torch.set_num_threads(1)
    rows = {}
    for name, cfg in CONFIGS.items():
        if cfg["fusion"] == "sentcls":
            m = SentenceClassifier(97, cfg["mods"]).eval()
        else:
            m = build_model(V, cfg["mods"], cfg["fusion"], d=cfg.get("d", 192),
                            **({"backbone": cfg["backbone"]} if "backbone" in cfg else {})).eval()
        b = batch()
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            m(b)
        flops = fc.get_total_flops() + rnn_flops(m, T)
        with torch.no_grad():
            for _ in range(5):
                m(b)
            ts = []
            for _ in range(50):
                t = time.perf_counter(); m(b); ts.append(time.perf_counter() - t)
        rows[name] = dict(params=sum(p.numel() for p in m.parameters()), mflops=flops / 1e6,
                          latency_ms=1000 * float(np.median(ts)), latency_ms_per_frame=1000 * float(np.median(ts)) / T)
        print(f"{name:24s} {rows[name]['params']:>10,d} params {rows[name]['mflops']:8.1f} MFLOPs "
              f"{rows[name]['latency_ms']:6.2f} ms/seq", flush=True)
    out = ROOT / "results" / "efficiency.json"
    out.parent.mkdir(exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev["models"] = rows
    out.write_text(json.dumps(prev, indent=1))


def extractor_cost():
    """Per-frame cost of the frozen front-ends on this machine (run when the CPU is idle)."""
    import cv2
    sys.path.insert(0, str(ROOT / "scripts"))
    import extract_features as E
    seqs = E.list_sequences()
    imgs = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for _, fs in seqs[::60] for f in fs[:4]]
    pose, hands = E.make_landmarkers()
    torch.set_num_threads(4)
    enc = E.RGBEncoder("cpu")
    res = {}
    t = time.perf_counter(); [pose.detect(E.mp.Image(image_format=E.mp.ImageFormat.SRGB, data=np.ascontiguousarray(i))) for i in imgs]
    res["pose_ms_per_frame"] = 1000 * (time.perf_counter() - t) / len(imgs)
    t = time.perf_counter(); [hands.detect(E.mp.Image(image_format=E.mp.ImageFormat.SRGB, data=np.ascontiguousarray(i))) for i in imgs]
    res["hand_ms_per_frame"] = 1000 * (time.perf_counter() - t) / len(imgs)
    enc(imgs[:4]); t = time.perf_counter(); enc(imgs)
    res["dinov2_ms_per_frame"] = 1000 * (time.perf_counter() - t) / len(imgs)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        enc.m.forward_features(torch.randn(1, 3, 224, 224))
    res["dinov2_gflops_per_frame"] = fc.get_total_flops() / 1e9
    res["n_frames_timed"] = len(imgs)
    print(res)
    out = ROOT / "results" / "efficiency.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev["extractors"] = res
    out.write_text(json.dumps(prev, indent=1))


if __name__ == "__main__":
    extractor_cost() if "--extractors" in sys.argv else main()
