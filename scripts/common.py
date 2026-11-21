"""Shared primitives for the knee lesion-fidelity measurement.

Coordinate convention
---------------------
fastMRI+ bounding boxes in Annotations/knee.csv give ``y`` measured from the
BOTTOM of the 320 x 320 reconstruction. Array indexing is top-origin, so every
box must be converted with :func:`box_row_top` before it is used to index an
image. ``registration_check`` verifies this against image anatomy and raises if
the conversion is wrong.
"""
import csv
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, binary_erosion, sobel

IMG = 320                     # side of the scored image, after cropping
JOINT_TOL_PX = 15             # tolerance for the registration check
JOINT_MIN_FRAC = 0.80         # fail below this fraction on-joint-line


# --------------------------------------------------------------------------
# coordinate convention
# --------------------------------------------------------------------------
def box_row_top(y, h, img=IMG):
    """Top-origin row index of a fastMRI+ box whose ``y`` is bottom-origin."""
    return img - y - h


def box_slices(x, y, w, h, img=IMG, corrected=True):
    """Clipped (row_slice, col_slice) for a fastMRI+ box, or None if degenerate."""
    top = box_row_top(y, h, img) if corrected else y
    y0, y1 = max(0, top), min(img, top + h)
    x0, x1 = max(0, x), min(img, x + w)
    if x1 <= x0 or y1 <= y0:
        return None
    return (slice(y0, y1), slice(x0, x1))


def load_boxes(ann_path):
    """{(file, slice): [(x, y, w, h, label), ...]} with raw CSV coordinates."""
    boxes = {}
    with open(ann_path) as f:
        for r in csv.DictReader(f):
            fn = r["file"].strip()
            fn = fn if fn.endswith(".h5") else fn + ".h5"
            if not r["slice"].strip().isdigit():
                continue
            if r.get("x", "").strip() in ("", "-1"):
                continue
            boxes.setdefault((fn, int(r["slice"])), []).append(
                (int(float(r["x"])), int(float(r["y"])),
                 int(float(r["width"])), int(float(r["height"])), r["label"].strip()))
    return boxes


# --------------------------------------------------------------------------
# registration check
# --------------------------------------------------------------------------
JOINT_LABELS = ("Meniscus Tear", "Cartilage - Partial Thickness loss/defect")


def joint_line_row(ref, col_slice, lo=60, hi=260, sigma=2.0):
    """Row of the joint line within the given columns: darkest smoothed row."""
    prof = gaussian_filter1d(ref[:, col_slice].mean(1) / ref.max(), sigma)
    return lo + int(np.argmin(prof[lo:hi]))


def registration_rows(ref, x, y, w, h):
    """Offset of the box centre from the joint line, both conventions."""
    x0, x1 = max(0, x), min(IMG, x + w)
    jl = joint_line_row(ref, slice(x0, x1))
    as_given = abs((y + h / 2.0) - jl)
    corrected = abs((box_row_top(y, h) + h / 2.0) - jl)
    return jl, as_given, corrected


def registration_report(records, tol=JOINT_TOL_PX, min_frac=JOINT_MIN_FRAC, strict=True):
    """Summarise per-label on-joint-line fractions and fail if too low.

    ``records``: iterable of (label, offset_as_given, offset_corrected).
    """
    by_label = {}
    for lab, d_old, d_new in records:
        if lab not in JOINT_LABELS:
            continue
        by_label.setdefault(lab, []).append((d_old, d_new))
    out, failed = {}, []
    for lab, v in sorted(by_label.items()):
        a = np.array(v)
        frac_old = float((a[:, 0] < tol).mean())
        frac_new = float((a[:, 1] < tol).mean())
        out[lab] = {"n": len(v),
                    "frac_within_tol_as_given": frac_old,
                    "frac_within_tol_corrected": frac_new,
                    "median_offset_as_given": float(np.median(a[:, 0])),
                    "median_offset_corrected": float(np.median(a[:, 1]))}
        print(f"  registration | {lab:44s} n={len(v):3d} "
              f"within {tol}px: as-given {100*frac_old:5.1f}%  corrected {100*frac_new:5.1f}%")
        if frac_new < min_frac:
            failed.append((lab, frac_new))
    if failed and strict:
        raise RuntimeError(
            "REGISTRATION CHECK FAILED - boxes are not on the joint line: "
            + "; ".join(f"{l}: {100*f:.1f}% < {100*min_frac:.0f}%" for l, f in failed))
    return out


# --------------------------------------------------------------------------
# MRI operators
# --------------------------------------------------------------------------
def fft2c(x, ax=(-2, -1)):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)


def ifft2c(x, ax=(-2, -1)):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)


def make_mask(k, acs=24, R=4):
    """Equidistant mask with ACS block; signal-free PE lines counted as known."""
    C, H, W = k.shape
    m = np.zeros((H, W), bool)
    m[np.abs(k[0]) < 5e-9] = True
    s = int((W - acs) / 2) - 1
    m[:, s:s + acs] = True
    m[:, 0::R] = True
    return m


def EH(k, S, m):
    return (ifft2c(k * m[None]) * np.conj(S)).sum(0)


def E(x, S, m):
    return fft2c(x[None] * S) * m[None]


def cg_sense(k, S, m, iters=30):
    """Unregularised CG on (E^H E) x = E^H y, zero initialisation."""
    b = EH(k * m[None], S, m)
    x = np.zeros_like(b); r = b.copy(); p = r.copy()
    rs = np.vdot(r, r)
    for _ in range(iters):
        q = EH(E(p, S, m), S, m)
        d = np.vdot(p, q)
        if abs(d) < 1e-30:
            break
        a = rs / d
        x = x + a * p
        r = r - a * q
        rs2 = np.vdot(r, r)
        p = r + (rs2 / rs) * p
        rs = rs2
    return x


def crop320(a):
    c = (a.shape[-1] - IMG) // 2
    return a[..., :, c:c + IMG]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def ssim_map(ref, test, dr):
    K1, K2, s = 0.01, 0.03, 1.5
    C1, C2 = (K1 * dr) ** 2, (K2 * dr) ** 2
    m1, m2 = gaussian_filter(ref, s), gaussian_filter(test, s)
    s11 = gaussian_filter(ref * ref, s) - m1 * m1
    s22 = gaussian_filter(test * test, s) - m2 * m2
    s12 = gaussian_filter(ref * test, s) - m1 * m2
    return ((2 * m1 * m2 + C1) * (2 * s12 + C2)) / ((m1 ** 2 + m2 ** 2 + C1) * (s11 + s22 + C2))


def psnr(ref, test, dr):
    mse = np.mean((ref - test) ** 2)
    return float(20 * np.log10(dr / np.sqrt(mse))) if mse > 0 else float("nan")


def roi_stats(ref, sl, dr):
    p = ref[sl]
    gx, gy = sobel(p, 0), sobel(p, 1)
    mx, mn = p.max(), p.min()
    return {"mean": float(p.mean() / dr), "std": float(p.std() / dr),
            "contrast": float((mx - mn) / (mx + mn + 1e-12)),
            "grad": float(np.sqrt(gx ** 2 + gy ** 2).mean() / dr)}


def appearance_triple(ref, sl, dr):
    """(mean, Michelson contrast, std) of a patch, normalised by dynamic range."""
    p = ref[sl]
    mx, mn = p.max(), p.min()
    return (p.mean() / dr, (mx - mn) / (mx + mn + 1e-12), p.std() / dr)


# --------------------------------------------------------------------------
# control-region sampling
# --------------------------------------------------------------------------
def tissue_mask(ref, dr, thresh=0.08):
    return binary_erosion(ref > thresh * dr, np.ones((5, 5)))


def draw_size_matched(rng, tissue, occupied, bh, bw, k=5, tries=400):
    out = []
    for _ in range(tries):
        if len(out) >= k:
            break
        cy = rng.integers(0, IMG - bh); cx = rng.integers(0, IMG - bw)
        sl = (slice(cy, cy + bh), slice(cx, cx + bw))
        if occupied[sl].any() or tissue[sl].mean() < 0.9:
            continue
        out.append(sl)
    return out


def draw_appearance_matched(rng, ref, dr, body, occupied, bh, bw, target,
                            k=5, candidates=2000):
    """Closest-k same-size patches to ``target`` in standardised appearance space."""
    pos, feats = [], []
    for _ in range(candidates):
        cy = rng.integers(0, IMG - bh); cx = rng.integers(0, IMG - bw)
        sl = (slice(cy, cy + bh), slice(cx, cx + bw))
        if occupied[sl].any() or body[sl].mean() < 0.9:
            continue
        pos.append(sl)
        feats.append(appearance_triple(ref, sl, dr))
    if len(pos) < k:
        return [], None
    F = np.array(feats)
    mu, sd = F.mean(0), F.std(0) + 1e-9
    d = np.linalg.norm((F - mu) / sd - (np.array(target) - mu) / sd, axis=1)
    order = np.argsort(d)[:k]
    return [pos[i] for i in order], F[order]


# --------------------------------------------------------------------------
# inference helpers
# --------------------------------------------------------------------------
def clustered_bootstrap(x, groups, B=5000, seed=0):
    """Percentile CI and two-sided bootstrap p, resampling clusters."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); groups = np.asarray(groups)
    u = np.unique(groups)
    idx = {s: np.where(groups == s)[0] for s in u}
    v = np.empty(B)
    for b in range(B):
        pick = rng.choice(u, len(u), True)
        v[b] = x[np.concatenate([idx[s] for s in pick])].mean()
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2 * min((v <= 0).mean(), (v >= 0).mean())
    return float(lo), float(hi), float(min(p, 1.0))


def patient_level(x, groups):
    """Wilcoxon on per-subject means plus the positive/negative split."""
    from scipy.stats import wilcoxon
    x = np.asarray(x, float); groups = np.asarray(groups)
    means = np.array([x[groups == s].mean() for s in np.unique(groups)])
    npos, nneg = int((means > 0).sum()), int((means < 0).sum())
    try:
        p = float(wilcoxon(means).pvalue)
    except ValueError:
        p = float("nan")
    return {"n_patients": len(means), "mean": float(means.mean()),
            "median": float(np.median(means)), "wilcoxon_p": p,
            "n_pos": npos, "n_neg": nneg}


def verify_registration(cupid_dir, want, boxes, strict=True):
    """Standalone pre-flight check: are the converted boxes on the joint line?

    Loads only the fully-sampled reference of each annotated slice (no
    reconstruction) and compares the box centre with the joint line. Raises
    RuntimeError if fewer than JOINT_MIN_FRAC of meniscus/cartilage boxes land
    within JOINT_TOL_PX of it.
    """
    from pathlib import Path as _P
    records = []
    for fn in sorted(want):
        ann_z = [z for z in want[fn] if (fn, z) in boxes]
        if not ann_z:
            continue
        d = np.load(_P(cupid_dir) / fn.replace(".h5", ".npy"), allow_pickle=True).item()
        order = sorted(want[fn])
        for z in ann_z:
            i = order.index(z)
            K = np.moveaxis(d["kspace"][i], -1, 0)
            S = np.moveaxis(d["coils"][i], -1, 0)
            K = K / np.abs(K).max()
            ref = crop320(np.abs((np.conj(S) * ifft2c(K)).sum(0)))
            for (x, y, w, h, lab) in boxes[(fn, z)]:
                _, d_old, d_new = registration_rows(ref, x, y, w, h)
                records.append((lab, d_old, d_new))
        del d
    print("registration pre-flight:")
    return registration_report(records, strict=strict)
