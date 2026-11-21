#!/usr/bin/env python3
"""Lesion ROIs vs size-matched control ROIs on the same slice.

Removes the confound in the global-vs-box comparison: a lesion box sits in
textured tissue, while the global frame includes background air. Here every
lesion box is compared against K control boxes of IDENTICAL size, drawn from
tissue on the same slice, disjoint from every annotated box on that slice.
"""
import csv, json
from pathlib import Path
import numpy as np
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import common as _C   # shared box-coordinate convention and registration check
from scipy.ndimage import gaussian_filter, binary_erosion

SPL = Path(r"D:\Research Results\splits")
CD  = Path(r"D:\Research Results\cupid_data\test")
ANN = Path(r"D:\Research Results\fastmri-plus\Annotations\knee.csv")
OUT = Path(r"D:\Research Results\results"); OUT.mkdir(parents=True, exist_ok=True)
K_CTRL = 5

def fft2c(x, ax=(-2,-1)):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)
def ifft2c(x, ax=(-2,-1)):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)
def make_mask(k, acs=24, R=4):
    C,H,W = k.shape; m = np.zeros((H,W), bool)
    m[np.abs(k[0]) < 5e-9] = True
    s = int((W-acs)/2)-1; m[:, s:s+acs] = True; m[:, 0::R] = True
    return m
def EH(k,S,m): return (ifft2c(k*m[None])*np.conj(S)).sum(0)
def E(x,S,m):  return fft2c(x[None]*S)*m[None]
def cg_sense(k,S,m,iters=30):
    b = EH(k*m[None],S,m); x = np.zeros_like(b); r = b.copy(); p = r.copy(); rs = np.vdot(r,r)
    for _ in range(iters):
        q = EH(E(p,S,m),S,m); d = np.vdot(p,q)
        if abs(d) < 1e-30: break
        a = rs/d; x = x+a*p; r = r-a*q; rs2 = np.vdot(r,r); p = r+(rs2/rs)*p; rs = rs2
    return x
def crop320(a):
    c = (a.shape[-1]-320)//2; return a[..., :, c:c+320]
def ssim_map(ref, test, dr):
    K1,K2,s = 0.01,0.03,1.5; C1,C2 = (K1*dr)**2, (K2*dr)**2
    m1,m2 = gaussian_filter(ref,s), gaussian_filter(test,s)
    s11 = gaussian_filter(ref*ref,s)-m1*m1; s22 = gaussian_filter(test*test,s)-m2*m2
    s12 = gaussian_filter(ref*test,s)-m1*m2
    return ((2*m1*m2+C1)*(2*s12+C2))/((m1**2+m2**2+C1)*(s11+s22+C2))

boxes = {}
with open(ANN) as f:
    for r in csv.DictReader(f):
        fn = r["file"].strip(); fn = fn if fn.endswith(".h5") else fn+".h5"
        if not r["slice"].strip().isdigit(): continue
        if r.get("x","").strip() in ("","-1"): continue
        _x, _y, _w, _h = (int(float(r["x"])), int(float(r["y"])),
                          int(float(r["width"])), int(float(r["height"])))
        boxes.setdefault((fn, int(r["slice"])), []).append(
            (_x, _C.box_row_top(_y, _h), _w, _h, r["label"].strip()))

want = {}
for r in csv.DictReader(open(SPL/"test_slices.csv")):
    want.setdefault(r["file"], []).append(int(r["slice"]))


# --- coordinate registration check: fails loudly if boxes are not on anatomy ---
_C.verify_registration(CD, want, boxes, strict=True)
rng = np.random.default_rng(0)
rows = []
for fn in sorted(want):
    d = np.load(CD/fn.replace(".h5",".npy"), allow_pickle=True).item()
    Ka, Sa = d["kspace"], d["coils"]
    for i, z in enumerate(sorted(want[fn])):
        bs = boxes.get((fn,z), [])
        if not bs: continue
        K = np.moveaxis(Ka[i],-1,0); S = np.moveaxis(Sa[i],-1,0)
        K = K/np.abs(K).max(); m = make_mask(K)
        ref = crop320(np.abs((np.conj(S)*ifft2c(K)).sum(0)))
        cg  = crop320(np.abs(cg_sense(K,S,m)))
        dr = ref.max()
        sm = ssim_map(ref, cg, dr)
        tissue = binary_erosion(ref > 0.08*dr, np.ones((5,5)))   # interior tissue only
        occupied = np.zeros_like(tissue)
        for (x,y,w,h,_) in bs:
            occupied[max(0,y):min(320,y+h), max(0,x):min(320,x+w)] = True
        for (x,y,w,h,lab) in bs:
            x0,y0,x1,y1 = max(0,x),max(0,y),min(320,x+w),min(320,y+h)
            if x1<=x0 or y1<=y0: continue
            bh, bw = y1-y0, x1-x0
            lesion = float(sm[y0:y1, x0:x1].mean())
            ctrl = []
            for _ in range(400):
                if len(ctrl) >= K_CTRL: break
                cy = rng.integers(0, 320-bh); cx = rng.integers(0, 320-bw)
                sl = (slice(cy,cy+bh), slice(cx,cx+bw))
                if occupied[sl].any(): continue
                if tissue[sl].mean() < 0.9: continue
                ctrl.append(float(sm[sl].mean()))
            if ctrl:
                rows.append({"file":fn, "slice":z, "label":lab, "w":bw, "h":bh,
                             "area":bw*bh, "ssim_lesion":lesion,
                             "ssim_ctrl":float(np.mean(ctrl)), "n_ctrl":len(ctrl),
                             "delta":lesion-float(np.mean(ctrl))})
    print(f"{fn} done ({sum(1 for r in rows if r['file']==fn)} ROIs)", flush=True)

keys = list(rows[0]); 
with open(OUT/"roi_controlled.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(f"\nlesion ROIs: {len(rows)} | slices: {len({(r['file'],r['slice']) for r in rows})}")
print("wrote", OUT/"roi_controlled.csv")
