"""Image-level degradations applied *before* feature/landmark extraction, so their effect on
MediaPipe and the RGB backbone is real rather than simulated at feature level.

Every corruption is deterministic given (sequence id, condition), so all models are evaluated on
the identical degraded inputs. Corruptions that need to know where the hands/body are use the
landmarks extracted from the *clean* frames (``clean_lm``), i.e. occluders are placed where the
hands really are, as a physical occluder would be.
"""
import hashlib

import cv2
import numpy as np

CONDITIONS = ["clean", "lowlight", "uneven", "hand_occ", "body_occ", "background",
              "blur", "distance", "viewpoint", "combined"]
# "combined" = cluttered background + uneven illumination + hand occlusion on half the frames


def _rng(seq_id, cond):
    h = hashlib.sha256(f"{seq_id}|{cond}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _span(rng, T, frac):
    """Contiguous block of frames covering ``frac`` of the sequence."""
    n = max(1, int(round(T * frac)))
    s = int(rng.integers(0, T - n + 1))
    return set(range(s, s + n))


def lowlight(img, rng):
    x = img.astype(np.float32) / 255.0
    x = np.power(x, 1.6) * rng.uniform(0.18, 0.28)
    x = x + rng.normal(0, 0.025, x.shape)          # sensor noise at low exposure
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def uneven(img, rng):
    h, w = img.shape[:2]
    ang = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = (xx / w - 0.5) * np.cos(ang) + (yy / h - 0.5) * np.sin(ang)
    proj = (proj - proj.min()) / (np.ptp(proj) + 1e-6)
    gain = 0.12 + 1.1 * proj                     # one side nearly dark, other side over-lit
    return np.clip(img.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)


def _hand_boxes(lm, h, w, pad=0.6):
    boxes = []
    for hand in lm["hands"]:
        if np.isnan(hand).all():
            continue
        xs, ys = hand[:, 0] * w, hand[:, 1] * h
        cx, cy = xs.mean(), ys.mean()
        s = max(np.ptp(xs), np.ptp(ys)) * (1 + pad) + 8
        boxes.append((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)))
    return boxes


def hand_occ(img, rng, lm):
    out = img.copy()
    h, w = img.shape[:2]
    colour = rng.integers(60, 200, 3).tolist()
    for x0, y0, x1, y1 in _hand_boxes(lm, h, w):
        cv2.rectangle(out, (x0, y0), (x1, y1), colour, -1)
    return out


def body_occ(img, rng, lm, box):
    out = img.copy()
    x0, y0, x1, y1 = box
    out[y0:y1, x0:x1] = np.array(rng.integers(60, 200, 3), dtype=np.uint8)
    return out


def clutter(h, w, rng):
    bg = np.full((h, w, 3), rng.integers(40, 220, 3), np.uint8)
    for _ in range(60):
        c = rng.integers(0, 255, 3).tolist()
        p = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        if rng.random() < 0.5:
            q = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.rectangle(bg, p, q, c, -1)
        else:
            cv2.circle(bg, p, int(rng.integers(5, max(6, w // 6))), c, -1)
    return cv2.GaussianBlur(bg, (3, 3), 0)


def background(img, mask, bg):
    m = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)[..., None]
    return (img * m + bg * (1 - m)).astype(np.uint8)


def motion_blur(img, rng):
    k = int(rng.integers(9, 15))
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0 / k
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    ker = cv2.warpAffine(ker, rot, (k, k))
    ker /= ker.sum()
    return cv2.filter2D(img, -1, ker)


def distance(img, scale):
    """Signer appears ``scale`` times smaller, as if the camera were further away."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    border = np.median(img.reshape(-1, 3), axis=0).astype(np.uint8)
    out = np.empty_like(img)
    out[:] = border
    y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    out[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    return out


def viewpoint(img, M):
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _yaw_homography(h, w, deg):
    """Approximate a horizontal camera rotation by foreshortening one side of the image."""
    d = np.tan(np.radians(abs(deg))) * 0.5
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    if deg > 0:
        dst = np.float32([[0, 0], [w * (1 - d * 0.3), h * d * 0.5], [w * (1 - d * 0.3), h * (1 - d * 0.5)], [0, h]])
    else:
        dst = np.float32([[w * d * 0.3, h * d * 0.5], [w, 0], [w, h], [w * d * 0.3, h * (1 - d * 0.5)]])
    return cv2.getPerspectiveTransform(src, dst)


class SequenceCorruptor:
    """Applies one condition to a whole sequence with sequence-level consistent parameters."""

    def __init__(self, cond, seq_id, T, h, w, clean_lms=None):
        self.cond, self.rng = cond, _rng(seq_id, cond)
        self.clean_lms = clean_lms
        if cond in ("hand_occ", "body_occ", "combined"):
            self.frames = _span(self.rng, T, 0.5)
        if cond == "combined":
            self.bg = clutter(h, w, self.rng)
        if cond == "body_occ":
            bw, bh = int(w * self.rng.uniform(0.35, 0.5)), int(h * self.rng.uniform(0.35, 0.5))
            x0 = int(self.rng.integers(0, w - bw + 1))
            y0 = int(self.rng.integers(int(h * 0.25), max(int(h * 0.25) + 1, h - bh + 1)))
            self.box = (x0, y0, x0 + bw, y0 + bh)
        if cond == "background":
            self.bg = clutter(h, w, self.rng)
        if cond == "distance":
            self.scale = self.rng.uniform(0.45, 0.6)
        if cond == "viewpoint":
            self.M = _yaw_homography(h, w, self.rng.choice([-1, 1]) * self.rng.uniform(25, 35))

    def __call__(self, t, img, mask=None):
        c = self.cond
        if c == "clean":
            return img
        if c == "lowlight":
            return lowlight(img, self.rng)
        if c == "uneven":
            return uneven(img, self.rng)
        if c == "hand_occ":
            return hand_occ(img, self.rng, self.clean_lms[t]) if t in self.frames else img
        if c == "body_occ":
            return body_occ(img, self.rng, None, self.box) if t in self.frames else img
        if c == "background":
            return background(img, mask, self.bg)
        if c == "blur":
            return motion_blur(img, self.rng)
        if c == "distance":
            return distance(img, self.scale)
        if c == "viewpoint":
            return viewpoint(img, self.M)
        if c == "combined":
            x = uneven(background(img, mask, self.bg), self.rng)
            return hand_occ(x, self.rng, self.clean_lms[t]) if t in self.frames else x
        raise ValueError(c)
