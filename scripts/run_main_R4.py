#!/usr/bin/env python3
"""Main corrected-coordinate analysis at R=4, ACS=24.

Produces, for every fastMRI+ region on the test split:
  * lesion vs 5 size-matched tissue controls          (zero-filled and CG-SENSE)
  * lesion vs 5 appearance-matched controls           (zero-filled and CG-SENSE)
  * the identical box on every unannotated slice of the same patient (placebo)
  * whole-image metrics per slice
  * the registration check, both coordinate conventions
"""
import csv, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SPL = Path(r"D:\Research Results\splits")
CD = Path(r"D:\Research Results\cupid_data\test")
ANN = Path(r"D:\Research Results\fastmri-plus\Annotations\knee.csv")
OUT = Path(r"D:\Research Results\v2\results")
FIGD = Path(r"D:\Research Results\v2\figure_data")
ACS, R = 24, 4

boxes = C.load_boxes(ANN)
want = {}
for r in csv.DictReader(open(SPL / "test_slices.csv")):
    want.setdefault(r["file"], []).append(int(r["slice"]))
want = {k: sorted(v) for k, v in want.items()}

rng = np.random.default_rng(0)
regions, placebo, whole, regcheck = [], [], [], []
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
        zf = C.crop320(np.abs(C.EH(K, S, m)))
        cg = C.crop320(np.abs(C.cg_sense(K, S, m)))
        sm_zf, sm_cg = C.ssim_map(ref, zf, dr), C.ssim_map(ref, cg, dr)
        cache[z] = {"ref": ref.astype(np.float32), "dr": dr,
                    "zf": zf.astype(np.float32), "cg": cg.astype(np.float32),
                    "sm_zf": sm_zf.astype(np.float32), "sm_cg": sm_cg.astype(np.float32),
                    "tissue": C.tissue_mask(ref, dr, 0.08),
                    "body": C.tissue_mask(ref, dr, 0.02)}
        whole.append({"file": fn, "slice": z, "R": R, "acs": ACS,
                      "mask_frac": float(m.mean()),
                      "ssim_zf": float(sm_zf.mean()), "ssim_cg": float(sm_cg.mean()),
                      "psnr_zf": C.psnr(ref, zf, dr), "psnr_cg": C.psnr(ref, cg, dr)})
    del Ka, Sa, d

    ann = {z: boxes.get((fn, z), []) for z in sl_all}
    for z in sl_all:
        bs = ann[z]
        if not bs:
            continue
        c = cache[z]; ref, dr = c["ref"].astype(np.float64), c["dr"]
        occ = np.zeros((C.IMG, C.IMG), bool)
        for (x, y, w, h, _) in bs:
            s = C.box_slices(x, y, w, h)
            if s: occ[s] = True
        for (x, y, w, h, lab) in bs:
            sl = C.box_slices(x, y, w, h)
            if sl is None:
                continue
            jl, d_old, d_new = C.registration_rows(ref, x, y, w, h)
            regcheck.append({"file": fn, "slice": z, "label": lab, "x": x, "y": y,
                             "w": w, "h": h, "joint_line_row": jl,
                             "offset_as_given": round(d_old, 2),
                             "offset_corrected": round(d_new, 2)})
            bh = sl[0].stop - sl[0].start; bw = sl[1].stop - sl[1].start
            size_ctrl = C.draw_size_matched(rng, c["tissue"], occ, bh, bw, k=5)
            target = C.appearance_triple(ref, sl, dr)
            app_ctrl, app_feat = C.draw_appearance_matched(
                rng, ref, dr, c["body"], occ, bh, bw, target, k=5, candidates=2000)
            if not size_ctrl:
                continue
            ls = C.roi_stats(ref, sl, dr)
            rec = {"file": fn, "slice": z, "label": lab,
                   "x": x, "y_raw": y, "w": bw, "h": bh, "area": bw * bh,
                   "row_top": sl[0].start,
                   "n_size_ctrl": len(size_ctrl), "n_app_ctrl": len(app_ctrl)}
            for k_ in ("mean", "std", "contrast", "grad"):
                rec[f"les_{k_}"] = ls[k_]
            for nm, ctrls in (("size", size_ctrl), ("app", app_ctrl)):
                if not ctrls:
                    continue
                st = [C.roi_stats(ref, s, dr) for s in ctrls]
                for k_ in ("mean", "std", "contrast", "grad"):
                    rec[f"{nm}ctl_{k_}"] = float(np.mean([s[k_] for s in st]))
            for tag in ("cg", "zf"):
                sm = c["sm_" + tag].astype(np.float64)
                img = c[tag].astype(np.float64)
                rec[f"ssim_les_{tag}"] = float(sm[sl].mean())
                rec[f"psnr_les_{tag}"] = C.psnr(ref[sl], img[sl], dr)
                for nm, ctrls in (("size", size_ctrl), ("app", app_ctrl)):
                    if not ctrls:
                        continue
                    rec[f"ssim_{nm}ctl_{tag}"] = float(np.mean([sm[s].mean() for s in ctrls]))
                    rec[f"psnr_{nm}ctl_{tag}"] = float(np.mean([C.psnr(ref[s], img[s], dr) for s in ctrls]))
            rec["size_ctrl_coords"] = ";".join(f"{s[0].start},{s[1].start}" for s in size_ctrl)
            rec["app_ctrl_coords"] = ";".join(f"{s[0].start},{s[1].start}" for s in app_ctrl)
            regions.append(rec)

            for z2 in sl_all:
                if z2 == z or ann[z2]:
                    continue
                c2 = cache[z2]; ref2 = c2["ref"].astype(np.float64); dr2 = c2["dr"]
                occ2 = np.zeros((C.IMG, C.IMG), bool); occ2[sl] = True
                ctrl2 = C.draw_size_matched(rng, c2["tissue"], occ2, bh, bw, k=5)
                if not ctrl2:
                    continue
                prow = {"file": fn, "src_slice": z, "slice": z2, "label": lab,
                        "x": x, "row_top": sl[0].start, "w": bw, "h": bh, "area": bw * bh}
                for tag in ("cg", "zf"):
                    sm2 = c2["sm_" + tag].astype(np.float64)
                    img2 = c2[tag].astype(np.float64)
                    prow[f"ssim_les_{tag}"] = float(sm2[sl].mean())
                    prow[f"psnr_les_{tag}"] = C.psnr(ref2[sl], img2[sl], dr2)
                    prow[f"ssim_sizectl_{tag}"] = float(np.mean([sm2[s].mean() for s in ctrl2]))
                    prow[f"psnr_sizectl_{tag}"] = float(np.mean([C.psnr(ref2[s], img2[s], dr2) for s in ctrl2]))
                placebo.append(prow)
    del cache
    print(f"{fn} done  ({len(regions)} regions, {len(placebo)} placebo, "
          f"{(time.time()-t0)/60:.1f} min)", flush=True)

print("\n--- registration check (corrected convention must pass) ---")
summary = C.registration_report([(r["label"], r["offset_as_given"], r["offset_corrected"])
                                 for r in regcheck])
json.dump(summary, open(OUT / "registration_check.json", "w"), indent=2)

def dump(rows, path):
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

dump(regions, OUT / "regions_R4.csv")
dump(placebo, OUT / "placebo_R4.csv")
dump(whole, OUT / "whole_image_R4.csv")
dump(regcheck, OUT / "registration_per_box.csv")
elapsed = (time.time() - t0) / 60
json.dump({"job": "run_main_R4", "minutes": round(elapsed, 2),
           "n_regions": len(regions), "n_placebo": len(placebo),
           "n_slices": len(whole)}, open(OUT / "timing_main_R4.json", "w"), indent=2)
print(f"\nregions {len(regions)} | placebo {len(placebo)} | slices {len(whole)} "
      f"| elapsed {elapsed:.1f} min")
