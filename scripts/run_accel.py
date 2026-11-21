#!/usr/bin/env python3
"""CG-SENSE acceleration arm with corrected boxes: one R per invocation.

usage: run_accel.py <R>
Produces the same two contrasts as the main job (lesion vs size-matched
controls, and lesion vs the identical box on unannotated slices of the same
patient) with no sign-based grouping anywhere.
"""
import csv, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

R = int(sys.argv[1]) if len(sys.argv) > 1 else int(__import__("os").environ.get("ACCEL_R", 2))
SPL = Path(r"D:\Research Results\splits")
CD = Path(r"D:\Research Results\cupid_data\test")
ANN = Path(r"D:\Research Results\fastmri-plus\Annotations\knee.csv")
OUT = Path(r"D:\Research Results\v2\results")
ACS = 24

boxes = C.load_boxes(ANN)
want = {}
for r in csv.DictReader(open(SPL / "test_slices.csv")):
    want.setdefault(r["file"], []).append(int(r["slice"]))
want = {k: sorted(v) for k, v in want.items()}

rng = np.random.default_rng(0)
regions, placebo, whole = [], [], []
t0 = time.time()
for fn in sorted(want):
    d = np.load(CD / fn.replace(".h5", ".npy"), allow_pickle=True).item()
    Ka, Sa = d["kspace"], d["coils"]
    sl_all = want[fn]
    cache = {}
    for i, z in enumerate(sl_all):
        K = np.moveaxis(Ka[i], -1, 0); S = np.moveaxis(Sa[i], -1, 0)
        K = K / np.abs(K).max()
        m = C.make_mask(K, ACS, R)
        ref = C.crop320(np.abs((np.conj(S) * C.ifft2c(K)).sum(0)))
        dr = float(ref.max())
        cg = C.crop320(np.abs(C.cg_sense(K, S, m)))
        sm = C.ssim_map(ref, cg, dr)
        cache[z] = {"ref": ref.astype(np.float32), "dr": dr, "cg": cg.astype(np.float32),
                    "sm": sm.astype(np.float32), "tissue": C.tissue_mask(ref, dr, 0.08)}
        whole.append({"file": fn, "slice": z, "R": R, "acs": ACS,
                      "mask_frac": float(m.mean()), "ssim_cg": float(sm.mean()),
                      "psnr_cg": C.psnr(ref, cg, dr)})
    del Ka, Sa, d
    ann = {z: boxes.get((fn, z), []) for z in sl_all}
    for z in sl_all:
        bs = ann[z]
        if not bs:
            continue
        c = cache[z]; ref = c["ref"].astype(np.float64); dr = c["dr"]
        sm = c["sm"].astype(np.float64); img = c["cg"].astype(np.float64)
        occ = np.zeros((C.IMG, C.IMG), bool)
        for (x, y, w, h, _) in bs:
            s = C.box_slices(x, y, w, h)
            if s: occ[s] = True
        for (x, y, w, h, lab) in bs:
            sl = C.box_slices(x, y, w, h)
            if sl is None:
                continue
            bh = sl[0].stop - sl[0].start; bw = sl[1].stop - sl[1].start
            ctrl = C.draw_size_matched(rng, c["tissue"], occ, bh, bw, k=5)
            if not ctrl:
                continue
            regions.append({"file": fn, "slice": z, "label": lab, "R": R,
                            "x": x, "row_top": sl[0].start, "w": bw, "h": bh, "area": bw * bh,
                            "ssim_les": float(sm[sl].mean()),
                            "ssim_ctl": float(np.mean([sm[s].mean() for s in ctrl])),
                            "psnr_les": C.psnr(ref[sl], img[sl], dr),
                            "psnr_ctl": float(np.mean([C.psnr(ref[s], img[s], dr) for s in ctrl]))})
            for z2 in sl_all:
                if z2 == z or ann[z2]:
                    continue
                c2 = cache[z2]
                ref2 = c2["ref"].astype(np.float64); dr2 = c2["dr"]
                sm2 = c2["sm"].astype(np.float64); img2 = c2["cg"].astype(np.float64)
                occ2 = np.zeros((C.IMG, C.IMG), bool); occ2[sl] = True
                ctrl2 = C.draw_size_matched(rng, c2["tissue"], occ2, bh, bw, k=5)
                if not ctrl2:
                    continue
                placebo.append({"file": fn, "src_slice": z, "slice": z2, "label": lab, "R": R,
                                "x": x, "row_top": sl[0].start, "w": bw, "h": bh,
                                "ssim_les": float(sm2[sl].mean()),
                                "ssim_ctl": float(np.mean([sm2[s].mean() for s in ctrl2])),
                                "psnr_les": C.psnr(ref2[sl], img2[sl], dr2),
                                "psnr_ctl": float(np.mean([C.psnr(ref2[s], img2[s], dr2) for s in ctrl2]))})
    del cache
    print(f"R={R} {fn} done ({len(regions)} regions, {(time.time()-t0)/60:.1f} min)", flush=True)

def dump(rows, path):
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

dump(regions, OUT / f"accel_regions_R{R}.csv")
dump(placebo, OUT / f"accel_placebo_R{R}.csv")
dump(whole, OUT / f"whole_image_R{R}.csv")
el = (time.time() - t0) / 60
json.dump({"job": f"run_accel_R{R}", "minutes": round(el, 2), "n_regions": len(regions),
           "n_placebo": len(placebo), "n_slices": len(whole)},
          open(OUT / f"timing_accel_R{R}.json", "w"), indent=2)
print(f"\nR={R}: regions {len(regions)} | placebo {len(placebo)} | elapsed {el:.1f} min")
