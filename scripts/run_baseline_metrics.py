#!/usr/bin/env python3
"""Training-free measurement: do global metrics hide lesion-level degradation?

For every test slice: build CUPID's R=4 / 24-ACS equidistant mask, reconstruct with
zero-filling and with CG-SENSE (30 iterations, mu=0 -- the same operator CUPID uses to
make its "DICOM" surrogate in loader.get_x_PI), and score each against the fully-sampled
coil-combined reference both globally and restricted to fastMRI+ bounding boxes.

No torch, no GPU.
"""
import csv, json
from pathlib import Path
import numpy as np
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import common as _C   # shared box-coordinate convention and registration check

SPL = Path(r"D:\Research Results\splits")
CD  = Path(r"D:\Research Results\cupid_data\test")
ANN = Path(r"D:\Research Results\fastmri-plus\Annotations\knee.csv")
OUT = Path(r"D:\Research Results\results"); OUT.mkdir(parents=True, exist_ok=True)

def fft2c(x, ax=(-2,-1)):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)
def ifft2c(x, ax=(-2,-1)):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)

def make_mask(ksp, acs=24, R=4):
    """Mirrors CUPID src/data/loader.py:generate_mask."""
    C, H, W = ksp.shape
    m = np.zeros((H, W), bool)
    m[np.abs(ksp[0]) < 5e-9] = True          # no-signal PE lines count as known
    s = int((W - acs) / 2) - 1
    m[:, s:s+acs] = True
    m[:, 0::R] = True
    return m

def EH(k, S, m): return (ifft2c(k * m[None]) * np.conj(S)).sum(0)
def E(x, S, m):  return fft2c(x[None] * S) * m[None]

def cg_sense(k, S, m, iters=30):
    """CG on (E^H E) x = E^H y, mu = 0 -- CUPID's x_PI."""
    b = EH(k * m[None], S, m)
    x = np.zeros_like(b); r = b.copy(); p = r.copy()
    rs = np.vdot(r, r)
    for _ in range(iters):
        q = EH(E(p, S, m), S, m)
        denom = np.vdot(p, q)
        if abs(denom) < 1e-30: break
        a = rs / denom
        x = x + a * p; r = r - a * q
        rs_new = np.vdot(r, r)
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x

def crop320(a):
    c = (a.shape[-1] - 320) // 2
    return a[..., :, c:c+320]

def psnr(ref, test, dr):
    mse = np.mean((ref - test) ** 2)
    return 20 * np.log10(dr / np.sqrt(mse)) if mse > 0 else float("inf")

def ssim_map(ref, test, dr):
    """Gaussian-weighted SSIM map (skimage-equivalent defaults, sigma=1.5)."""
    from scipy.ndimage import gaussian_filter
    K1, K2, s = 0.01, 0.03, 1.5
    C1, C2 = (K1 * dr) ** 2, (K2 * dr) ** 2
    mu1, mu2 = gaussian_filter(ref, s), gaussian_filter(test, s)
    s11 = gaussian_filter(ref * ref, s) - mu1 * mu1
    s22 = gaussian_filter(test * test, s) - mu2 * mu2
    s12 = gaussian_filter(ref * test, s) - mu1 * mu2
    return ((2*mu1*mu2 + C1) * (2*s12 + C2)) / ((mu1**2 + mu2**2 + C1) * (s11 + s22 + C2))

# fastMRI+ boxes, keyed (file, slice)
boxes = {}
with open(ANN) as f:
    for r in csv.DictReader(f):
        fn = r["file"].strip(); fn = fn if fn.endswith(".h5") else fn + ".h5"
        if not r["slice"].strip().isdigit(): continue
        if r.get("x", "").strip() in ("", "-1"): continue
        _x, _y, _w, _h = (int(float(r["x"])), int(float(r["y"])),
                          int(float(r["width"])), int(float(r["height"])))
        boxes.setdefault((fn, int(r["slice"])), []).append(
            (_x, _C.box_row_top(_y, _h), _w, _h, r["label"].strip()))

rows = []
want = {}
for r in csv.DictReader(open(SPL / "test_slices.csv")):
    want.setdefault(r["file"], []).append(int(r["slice"]))

# --- coordinate registration check: fails loudly if boxes are not on anatomy ---
_C.verify_registration(CD, want, boxes, strict=True)

for fn in sorted(want):
    d = np.load(CD / fn.replace(".h5", ".npy"), allow_pickle=True).item()
    K_all, S_all = d["kspace"], d["coils"]
    for i, z in enumerate(sorted(want[fn])):
        K = np.moveaxis(K_all[i], -1, 0); S = np.moveaxis(S_all[i], -1, 0)
        scale = np.abs(K).max(); K = K / scale
        m = make_mask(K)
        ref = crop320(np.abs((np.conj(S) * ifft2c(K)).sum(0)))
        zf  = crop320(np.abs(EH(K, S, m)))
        cg  = crop320(np.abs(cg_sense(K, S, m)))
        dr = ref.max()
        rec = {"file": fn, "slice": z, "acc": 4, "acs": 24,
               "mask_frac": float(m.mean()),
               "psnr_zf": psnr(ref, zf, dr), "psnr_cg": psnr(ref, cg, dr)}
        for tag, img in (("zf", zf), ("cg", cg)):
            sm = ssim_map(ref, img, dr)
            rec[f"ssim_{tag}"] = float(sm.mean())
            bs = boxes.get((fn, z), [])
            if bs:
                vals, pv, labels = [], [], []
                for (x, y, w, h, lab) in bs:
                    x0, y0 = max(0, x), max(0, y)
                    x1, y1 = min(320, x + w), min(320, y + h)
                    if x1 <= x0 or y1 <= y0: continue
                    vals.append(float(sm[y0:y1, x0:x1].mean()))
                    pv.append(psnr(ref[y0:y1, x0:x1], img[y0:y1, x0:x1], dr))
                    labels.append(lab)
                if vals:
                    rec[f"ssim_box_{tag}"] = float(np.mean(vals))
                    rec[f"psnr_box_{tag}"] = float(np.mean(pv))
                    rec["n_boxes"] = len(vals); rec["labels"] = "|".join(sorted(set(labels)))
        rows.append(rec)
    print(f"{fn}: {len(want[fn])} slices done", flush=True)

keys = sorted({k for r in rows for k in r})
with open(OUT / "baseline_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
n_box = sum(1 for r in rows if "ssim_box_cg" in r)
print(f"\nslices: {len(rows)} | with boxes: {n_box}")
print("wrote", OUT / "baseline_metrics.csv")
