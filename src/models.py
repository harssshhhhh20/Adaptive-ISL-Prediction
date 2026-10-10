"""CSLR models over pre-extracted per-frame features.

All variants share the same modality encoders, temporal backbone (1D-CNN + BiLSTM, the
standard CSLR design of VAC/SMKD/CorrNet) and CTC head. They differ only in how the
modalities are combined, so differences in WER are attributable to the fusion mechanism.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import Q_HAND, Q_POSE, Q_RGB

IN_DIMS = {"rgb": 768, "pose": 75, "hand": 168, "hrgb": 768}
Q_DIM = 10                                    # 8 extractor quality signals + 2 jitter signals
Q_IDX = {"rgb": Q_RGB, "pose": Q_POSE + [8], "hand": Q_HAND + [9], "hrgb": Q_RGB + Q_POSE}


def lengths_mask(lengths, T):
    return torch.arange(T, device=lengths.device)[None] < lengths[:, None]


class TConv(nn.Module):
    """Temporal convolution on (B, T, C) tensors, computed as unfold + linear. Numerically the
    same as nn.Conv1d, but ~5x faster on this CPU build of PyTorch (no oneDNN on macOS arm64)."""

    def __init__(self, cin, cout, k):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.empty(cout, cin, k))
        self.bias = nn.Parameter(torch.zeros(cout))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        p = self.k // 2
        cols = F.pad(x, (0, 0, p, p)).unfold(1, self.k, 1)
        return F.linear(cols.reshape(x.shape[0], x.shape[1], -1), self.weight.reshape(self.weight.shape[0], -1), self.bias)


class ModalityEncoder(nn.Module):
    def __init__(self, din, d, p):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(din), nn.Linear(din, d), nn.GELU(), nn.Dropout(p))
        self.conv = TConv(d, d, 5)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        h = self.inp(x)
        return self.ln(h + F.gelu(self.conv(h)))


class TemporalBackbone(nn.Module):
    def __init__(self, d, p, kind="transformer"):
        super().__init__()
        self.kind = kind
        self.conv = nn.Sequential(TConv(d, d, 5), nn.LayerNorm(d), nn.GELU(), nn.Dropout(p))
        if kind == "bilstm":
            self.rnn = nn.LSTM(d, d // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=p)
        elif kind == "transformer":
            self.pos = nn.Parameter(torch.zeros(1, 256, d))
            layer = nn.TransformerEncoderLayer(d, 4, 2 * d, p, batch_first=True, norm_first=True)
            self.rnn = nn.TransformerEncoder(layer, 2)
        else:
            raise ValueError(kind)

    def forward(self, x, lengths):
        x = self.conv(x)
        if self.kind == "bilstm":
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            y, _ = self.rnn(packed)
            y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, total_length=x.shape[1])
            return y
        mask = ~lengths_mask(lengths, x.shape[1])
        return self.rnn(x + self.pos[:, :x.shape[1]], src_key_padding_mask=mask)


# ----------------------------------------------------------------------------- fusion modules
class ConcatFusion(nn.Module):
    """Fixed fusion: concatenate and project (input-independent combination)."""
    adaptive = False

    def __init__(self, mods, d, **_):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(len(mods) * d, d), nn.GELU())

    def forward(self, feats, q, mask, aux):
        return self.proj(torch.cat(feats, -1)), None


class FixedWeightFusion(nn.Module):
    """Fixed fusion: one learned global weight per modality, identical for every input/frame."""
    adaptive = False

    def __init__(self, mods, d, **_):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(len(mods)))

    def forward(self, feats, q, mask, aux):
        w = torch.softmax(self.alpha, 0)
        B, T = feats[0].shape[:2]
        return sum(wi * f for wi, f in zip(w, feats)), w.expand(B, T, -1)


class SeqAttentionFusion(nn.Module):
    """Attention fusion: input-dependent weights from pooled features, constant over time."""
    adaptive = True

    def __init__(self, mods, d, **_):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(len(mods) * d, d), nn.GELU(), nn.Linear(d, len(mods)))

    def forward(self, feats, q, mask, aux):
        m = mask[..., None].float()
        pooled = [(f * m).sum(1) / m.sum(1).clamp(min=1) for f in feats]
        w = torch.softmax(self.score(torch.cat(pooled, -1)), -1)[:, None].expand(-1, feats[0].shape[1], -1)
        return sum(w[..., i:i + 1] * f for i, f in enumerate(feats)), w


class GateFusion(nn.Module):
    """Adaptive per-frame fusion. Gate inputs are selected by flags:
    feat  - the modality features themselves
    qual  - observable quality signals (brightness, contrast, sharpness, landmark visibility,
            hand detection/confidence, landmark jitter)
    ent   - per-modality prediction uncertainty (entropy of the auxiliary CTC heads, detached)
    temporal - the gate is a BiGRU over time instead of a per-frame MLP."""
    adaptive = True

    def __init__(self, mods, d, feat=True, qual=False, ent=False, temporal=False, **_):
        super().__init__()
        self.mods, self.use_feat, self.use_q, self.use_ent, self.temporal = mods, feat, qual, ent, temporal
        din = (len(mods) * d if feat else 0) + (Q_DIM if qual else 0) + (len(mods) if ent else 0)
        h = d // 2
        self.inp = nn.Sequential(nn.Linear(din, h), nn.GELU())
        if temporal:
            self.rnn = nn.GRU(h, h // 2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(h, len(mods))

    def forward(self, feats, q, mask, aux):
        parts = []
        if self.use_feat:
            parts.append(torch.cat(feats, -1))
        if self.use_q:
            parts.append(q)
        if self.use_ent:
            ent = [(-(a.softmax(-1) * a.log_softmax(-1)).sum(-1, keepdim=True)).detach() for a in aux]
            parts.append(torch.cat(ent, -1) / 3.0)
        h = self.inp(torch.cat(parts, -1))
        if self.temporal:
            h = h + self.rnn(h)[0]
        w = torch.softmax(self.out(h), -1)
        return sum(w[..., i:i + 1] * f for i, f in enumerate(feats)), w


class CrossAttentionFusion(nn.Module):
    """Cross-modal attention (MulT-style): each modality attends over the others, then concat."""
    adaptive = True

    def __init__(self, mods, d, p=0.1, **_):
        super().__init__()
        self.attn = nn.ModuleList(nn.MultiheadAttention(d, 4, dropout=p, batch_first=True) for _ in mods)
        self.ln = nn.ModuleList(nn.LayerNorm(d) for _ in mods)
        self.proj = nn.Sequential(nn.Linear(len(mods) * d, d), nn.GELU())

    def forward(self, feats, q, mask, aux):
        out = []
        pad = ~mask
        for i, f in enumerate(feats):
            others = torch.cat([g for j, g in enumerate(feats) if j != i], 1)
            kpm = torch.cat([pad for j in range(len(feats)) if j != i], 1)
            a, _ = self.attn[i](f, others, others, key_padding_mask=kpm)
            out.append(self.ln[i](f + a))
        return self.proj(torch.cat(out, -1)), None


FUSIONS = {
    "concat": (ConcatFusion, {}),
    "fixed_weight": (FixedWeightFusion, {}),
    "seq_attention": (SeqAttentionFusion, {}),
    "gate": (GateFusion, dict(feat=True)),
    "rel_gate": (GateFusion, dict(feat=True, qual=True)),
    "qual_gate": (GateFusion, dict(feat=False, qual=True)),
    "unc_gate": (GateFusion, dict(feat=True, ent=True)),
    "rel_unc_gate": (GateFusion, dict(feat=True, qual=True, ent=True)),
    "trel_gate": (GateFusion, dict(feat=True, qual=True, temporal=True)),
    "trel_unc_gate": (GateFusion, dict(feat=True, qual=True, ent=True, temporal=True)),
    "xattn": (CrossAttentionFusion, {}),
}


class CSLRModel(nn.Module):
    def __init__(self, vocab_size, mods=("rgb", "pose", "hand"), fusion="concat", d=192, p=0.3,
                 backbone="transformer", aux_weight=0.5):
        super().__init__()
        self.mods, self.fusion_name, self.aux_weight = list(mods), fusion, aux_weight
        self.enc = nn.ModuleDict({m: ModalityEncoder(IN_DIMS[m], d, p) for m in mods})
        multi = len(mods) > 1
        if multi:
            cls, kw = FUSIONS[fusion]
            self.fusion = cls(self.mods, d, p=p, **kw)
            self.aux = nn.ModuleDict({m: nn.Linear(d, vocab_size) for m in mods})
        else:
            self.fusion = None
        self.backbone = TemporalBackbone(d, p, backbone)
        self.head = nn.Linear(d, vocab_size)

    def forward(self, batch):
        lengths = batch["lengths"]
        T = batch["rgb"].shape[1]
        mask = lengths_mask(lengths, T)
        feats = [self.enc[m](batch[m]) for m in self.mods]
        aux_logits, weights = [], None
        if self.fusion is not None:
            aux_logits = [self.aux[m](f) for m, f in zip(self.mods, feats)]
            fused, weights = self.fusion(feats, batch["quality"], mask, aux_logits)
        else:
            fused = feats[0]
        y = self.backbone(fused, lengths)
        return {"logits": self.head(y), "aux": aux_logits, "weights": weights, "lengths": lengths}


class TwoStreamModel(nn.Module):
    """Feature-level re-implementation of the TwoStream-SLR design (Chen et al., NeurIPS 2022):
    one temporal network per stream, bidirectional lateral connections between streams,
    per-stream CTC heads plus a joint head on the concatenation, and an ensemble of all heads'
    posteriors at inference. The original fine-tunes S3D on RGB and keypoint heatmaps end-to-end,
    which is not feasible on this hardware; here each stream starts from the same frozen
    features as every other model in this study."""

    def __init__(self, vocab_size, mods=("rgb", "pose", "hand"), d=192, p=0.3, aux_weight=0.5, **_):
        super().__init__()
        self.mods, self.aux_weight = list(mods), aux_weight
        self.fusion_name = "twostream"
        self.enc = nn.ModuleDict({m: ModalityEncoder(IN_DIMS[m], d, p) for m in mods})
        self.lateral = nn.ModuleDict({m: nn.Linear((len(mods) - 1) * d, d) for m in mods})
        self.back = nn.ModuleDict({m: TemporalBackbone(d, p) for m in mods})
        self.heads = nn.ModuleDict({m: nn.Linear(d, vocab_size) for m in mods})
        self.joint = TemporalBackbone(d, p)
        self.joint_in = nn.Linear(len(mods) * d, d)
        self.head = nn.Linear(d, vocab_size)

    def forward(self, batch):
        lengths = batch["lengths"]
        f = {m: self.enc[m](batch[m]) for m in self.mods}
        g = {m: f[m] + self.lateral[m](torch.cat([f[o] for o in self.mods if o != m], -1)) for m in self.mods}
        ys = {m: self.back[m](g[m], lengths) for m in self.mods}
        joint = self.joint(self.joint_in(torch.cat([ys[m] for m in self.mods], -1)), lengths)
        logits = self.head(joint)
        aux = [self.heads[m](ys[m]) for m in self.mods]
        return {"logits": logits, "aux": aux, "weights": None, "lengths": lengths, "ensemble": True}


def build_model(vocab_size, mods, fusion, **kw):
    if fusion == "twostream":
        return TwoStreamModel(vocab_size, mods, **kw)
    return CSLRModel(vocab_size, mods, fusion, **kw)


def ctc_loss(out, batch):
    def one(logits):
        lp = logits.log_softmax(-1).transpose(0, 1)
        return F.ctc_loss(lp, batch["targets"], out["lengths"], batch["target_lengths"],
                          blank=0, reduction="mean", zero_infinity=True)
    loss = one(out["logits"])
    if out["aux"]:
        loss = loss + out.get("aux_weight", 0.5) * sum(one(a) for a in out["aux"]) / len(out["aux"])
    return loss


def posteriors(out):
    """Frame posteriors used for decoding; TwoStream averages all heads (its ensemble)."""
    p = out["logits"].softmax(-1)
    if out.get("ensemble"):
        p = (p + sum(a.softmax(-1) for a in out["aux"])) / (1 + len(out["aux"]))
    return p


def greedy_decode(post, lengths):
    ids = post.argmax(-1).cpu()
    res = []
    for b, L in enumerate(lengths.tolist()):
        seq, prev = [], -1
        for t in ids[b, :L].tolist():
            if t != prev and t != 0:
                seq.append(t)
            prev = t
        res.append(seq)
    return res


class SentenceClassifier(nn.Module):
    """Diagnostic closed-set baseline: classify the whole sequence as one of the training
    sentences and output that sentence's gloss string. It cannot produce an unseen sentence, so
    it measures how far WER on this corpus can be driven by sentence identification alone."""

    def __init__(self, n_classes, mods=("rgb", "pose", "hand"), d=192, p=0.3, **_):
        super().__init__()
        self.mods, self.fusion_name, self.aux_weight = list(mods), "sentcls", 0.0
        self.enc = nn.ModuleDict({m: ModalityEncoder(IN_DIMS[m], d, p) for m in mods})
        self.proj = nn.Sequential(nn.Linear(len(mods) * d, d), nn.GELU())
        self.backbone = TemporalBackbone(d, p)
        self.att = nn.Linear(d, 1)
        self.cls = nn.Linear(d, n_classes)

    def forward(self, batch):
        lengths = batch["lengths"]
        mask = lengths_mask(lengths, batch["rgb"].shape[1])
        y = self.backbone(self.proj(torch.cat([self.enc[m](batch[m]) for m in self.mods], -1)), lengths)
        a = self.att(y).squeeze(-1).masked_fill(~mask, -1e9).softmax(-1)
        return {"sent_logits": self.cls((a[..., None] * y).sum(1)), "weights": None, "lengths": lengths, "aux": []}
