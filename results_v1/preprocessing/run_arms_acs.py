#!/usr/bin/env python3
"""Second experiment batch: reconstruction arm and calibration-size sensitivity.

For every annotated slice, in the same lesion-vs-matched-control design:
  * zero-filled and CG-SENSE at R=4, ACS=24   -> is the split specific to CG-SENSE?
  * CG-SENSE at R=4 with ACS in {12, 24, 48}  -> does calibration quality drive it?
Each region scored with SSIM and PSNR.
"""
import csv
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion

SPL = Path(r"D:\Research Results\splits"); CD = Path(r"D:\Research Results\cupid_data\test")
ANN = Path(r"D:\Research Results\fastmri-plus\Annotations\knee.csv")
OUT = Path(r"D:\Research Results\results")
K_CTRL = 5; ACS_SET = (12, 24, 48)

def fft2c(x, ax=(-2,-1)): return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x,axes=ax),axes=ax,norm="ortho"),axes=ax)
def ifft2c(x, ax=(-2,-1)): return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x,axes=ax),axes=ax,norm="ortho"),axes=ax)
def make_mask(k, acs, R):
    C,H,W = k.shape; m = np.zeros((H,W), bool)
    m[np.abs(k[0]) < 5e-9] = True
    s = int((W-acs)/2)-1; m[:, s:s+acs] = True; m[:, 0::R] = True
    return m
def EH(k,S,m): return (ifft2c(k*m[None])*np.conj(S)).sum(0)
def E(x,S,m): return fft2c(x[None]*S)*m[None]
def cg_sense(k,S,m,iters=30):
    b = EH(k*m[None],S,m); x=np.zeros_like(b); r=b.copy(); p=r.copy(); rs=np.vdot(r,r)
    for _ in range(iters):
        q = EH(E(p,S,m),S,m); d=np.vdot(p,q)
        if abs(d) < 1e-30: break
        a=rs/d; x=x+a*p; r=r-a*q; rs2=np.vdot(r,r); p=r+(rs2/rs)*p; rs=rs2
    return x
def crop320(a):
    c=(a.shape[-1]-320)//2; return a[..., :, c:c+320]
def ssim_map(ref, test, dr):
    K1,K2,s = 0.01,0.03,1.5; C1,C2=(K1*dr)**2,(K2*dr)**2
    m1,m2 = gaussian_filter(ref,s), gaussian_filter(test,s)
    s11=gaussian_filter(ref*ref,s)-m1*m1; s22=gaussian_filter(test*test,s)-m2*m2
    s12=gaussian_filter(ref*test,s)-m1*m2
    return ((2*m1*m2+C1)*(2*s12+C2))/((m1**2+m2**2+C1)*(s11+s22+C2))
def psnr(ref, test, dr):
    mse = np.mean((ref-test)**2)
    return float(20*np.log10(dr/np.sqrt(mse))) if mse > 0 else float("nan")

boxes = {}
with open(ANN) as f:
    for r in csv.DictReader(f):
        fn = r["file"].strip(); fn = fn if fn.endswith(".h5") else fn+".h5"
        if not r["slice"].strip().isdigit() or r.get("x","").strip() in ("","-1"): continue
        boxes.setdefault((fn,int(r["slice"])),[]).append(
            (int(float(r["x"])),int(float(r["y"])),int(float(r["width"])),
             int(float(r["height"])), r["label"].strip()))
want = {}
for r in csv.DictReader(open(SPL/"test_slices.csv")):
    want.setdefault(r["file"],[]).append(int(r["slice"]))

rng = np.random.default_rng(0); rows = []
for fn in sorted(want):
    d = np.load(CD/fn.replace(".h5",".npy"), allow_pickle=True).item()
    Ka, Sa = d["kspace"], d["coils"]
    sl_all = sorted(want[fn])
    for i, z in enumerate(sl_all):
        bs = boxes.get((fn,z), [])
        if not bs: continue
        K = np.moveaxis(Ka[i],-1,0); S = np.moveaxis(Sa[i],-1,0); K = K/np.abs(K).max()
        ref = crop320(np.abs((np.conj(S)*ifft2c(K)).sum(0))); dr = ref.max()
        tissue = binary_erosion(ref > 0.08*dr, np.ones((5,5)))
        occ = np.zeros_like(tissue)
        for (x,y,w,h,_) in bs:
            occ[max(0,y):min(320,y+h), max(0,x):min(320,x+w)] = True
        recons = {}
        m24 = make_mask(K, 24, 4)
        recons["zf_acs24"]  = crop320(np.abs(EH(K, S, m24)))
        for acs in ACS_SET:
            m = make_mask(K, acs, 4)
            recons[f"cg_acs{acs}"] = crop320(np.abs(cg_sense(K, S, m)))
        maps = {k: (ssim_map(ref, v, dr), v) for k, v in recons.items()}
        for (x,y,w,h,lab) in bs:
            x0,y0,x1,y1 = max(0,x),max(0,y),min(320,x+w),min(320,y+h)
            if x1<=x0 or y1<=y0: continue
            bh,bw = y1-y0, x1-x0
            lsl = (slice(y0,y1), slice(x0,x1))
            ctrl = []
            for _ in range(400):
                if len(ctrl) >= K_CTRL: break
                cy = rng.integers(0,320-bh); cx = rng.integers(0,320-bw)
                sl = (slice(cy,cy+bh), slice(cx,cx+bw))
                if occ[sl].any() or tissue[sl].mean() < 0.9: continue
                ctrl.append(sl)
            if not ctrl: continue
            rec = {"file":fn,"slice":z,"label":lab,"area":bw*bh,"slice_pos":i,
                   "n_slices":len(sl_all)}
            for key,(sm,img) in maps.items():
                rec[f"ssim_les_{key}"] = float(sm[lsl].mean())
                rec[f"ssim_ctl_{key}"] = float(np.mean([sm[s].mean() for s in ctrl]))
                rec[f"psnr_les_{key}"] = psnr(ref[lsl], img[lsl], dr)
                rec[f"psnr_ctl_{key}"] = float(np.mean([psnr(ref[s], img[s], dr) for s in ctrl]))
            rows.append(rec)
    print(f"{fn} done", flush=True)

with open(OUT/"arms_acs.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"\nROIs: {len(rows)} | wrote {OUT/'arms_acs.csv'}")
