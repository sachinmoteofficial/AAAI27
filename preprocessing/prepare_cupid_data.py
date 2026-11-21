#!/usr/bin/env python3
"""Build CUPID-format inputs from fastMRI knee CORPD_FBK volumes.

Writes one .npy per subject, in the exact layout src/data/loader.py expects:
a pickled dict {"kspace": [B,H,W,C] complex64, "coils": [B,H,W,C] complex64}.

CUPID generates the R=4 / 24-ACS mask and the CG-SENSE "DICOM" image itself
(loader.generate_mask, loader.get_x_PI), so neither is produced here.

Readout oversampling (kx=640) is removed to H=320, matching the repo's demo file.
Coil maps come from the 24 central ACS lines: Hann-windowed low-resolution coil
images divided by their root-sum-of-squares, zeroed below a relative threshold.
"""
import argparse, csv, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np, h5py

ap = argparse.ArgumentParser()
ap.add_argument("--splits", default=r"D:\Research Results\splits")
ap.add_argument("--knee",   default=r"D:\Research Results\knee")
ap.add_argument("--out",    default=r"D:\Research Results\cupid_data")
ap.add_argument("--acs", type=int, default=24)
ap.add_argument("--readout", type=int, default=320)
ap.add_argument("--thresh", type=float, default=0.02)
a = ap.parse_args()

def ifft2c(x, axes=(-2, -1)):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=axes), axes=axes, norm="ortho"), axes=axes)

def crop_readout(ksp, out):
    img = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(ksp, axes=-2), axis=-2, norm="ortho"), axes=-2)
    s = (img.shape[-2] - out) // 2
    img = img[..., s:s + out, :]
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(img, axes=-2), axis=-2, norm="ortho"), axes=-2)

def acs_maps(ksp, acs, thresh):
    C, H, W = ksp.shape
    st = (W - acs) // 2
    kacs = np.zeros_like(ksp)
    kacs[..., st:st + acs] = ksp[..., st:st + acs] * np.hanning(acs + 2)[1:-1][None, None, :]
    low = ifft2c(kacs)
    rss = np.sqrt((np.abs(low) ** 2).sum(0))
    maps = low / (rss + 1e-12)
    maps[:, rss < thresh * rss.max()] = 0
    return maps.astype(np.complex64)

splits, knee, out = Path(a.splits), Path(a.knee), Path(a.out)
manifest = []
for name in ("train", "test"):
    (out / name).mkdir(parents=True, exist_ok=True)
    wanted = defaultdict(list)
    for r in csv.DictReader(open(splits / f"{name}_slices.csv")):
        wanted[r["file"]].append(int(r["slice"]))
    for fn, slices in sorted(wanted.items()):
        t0 = time.time()
        slices = sorted(slices)
        with h5py.File(knee / fn, "r") as f:
            W = f["kspace"].shape[-1]
            ks = np.empty((len(slices), a.readout, W, 15), np.complex64)
            cs = np.empty_like(ks)
            for i, z in enumerate(slices):
                kc = crop_readout(f["kspace"][z].astype(np.complex64), a.readout)
                ks[i] = np.transpose(kc, (1, 2, 0))                      # C,H,W -> H,W,C
                cs[i] = np.transpose(acs_maps(kc, a.acs, a.thresh), (1, 2, 0))
        np.save(out / name / fn.replace(".h5", ".npy"), {"kspace": ks, "coils": cs})
        manifest.append({"split": name, "file": fn, "slices": len(slices),
                         "shape": list(ks.shape), "seconds": round(time.time() - t0, 1)})
        print(f"{name}/{fn}: {ks.shape} in {time.time()-t0:.1f}s", flush=True)
json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
print("subjects:", len(manifest), "| slices:", sum(m["slices"] for m in manifest))
