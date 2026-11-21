#!/usr/bin/env python3
"""Acceleration arm statistics: label-only grouping, no sign-based groups."""
import csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

OUT = Path(r"D:\Research Results\v2\results")
FIGD = Path(r"D:\Research Results\v2\figure_data")
ACCELS = [2, 4, 8]
B = 5000
MIN_PATIENTS = 3

def load_R(R):
    """Returns (regions, placebo) as lists of dicts with ssim_les / ssim_ctl."""
    if R == 4:
        reg = []
        for r in csv.DictReader(open(OUT / "regions_R4.csv")):
            reg.append({"file": r["file"], "slice": r["slice"], "label": r["label"],
                        "x": r["x"], "row_top": r["row_top"], "w": r["w"], "h": r["h"],
                        "ssim_les": float(r["ssim_les_cg"]), "ssim_ctl": float(r["ssim_sizectl_cg"])})
        pl = []
        for p in csv.DictReader(open(OUT / "placebo_R4.csv")):
            pl.append({"file": p["file"], "src_slice": p["src_slice"], "label": p["label"],
                       "x": p["x"], "row_top": p["row_top"], "w": p["w"], "h": p["h"],
                       "ssim_les": float(p["ssim_les_cg"]), "ssim_ctl": float(p["ssim_sizectl_cg"])})
        return reg, pl
    reg = [dict(r, ssim_les=float(r["ssim_les"]), ssim_ctl=float(r["ssim_ctl"]))
           for r in csv.DictReader(open(OUT / f"accel_regions_R{R}.csv"))]
    pl = [dict(p, ssim_les=float(p["ssim_les"]), ssim_ctl=float(p["ssim_ctl"]))
          for p in csv.DictReader(open(OUT / f"accel_placebo_R{R}.csv"))]
    return reg, pl

def describe(x, g):
    x = np.asarray(x, float); g = np.asarray(g)
    ok = np.isfinite(x); x, g = x[ok], g[ok]
    d = {"n_regions": int(len(x)), "n_patients": int(len(np.unique(g))),
         "mean": float(x.mean())}
    if len(np.unique(g)) >= MIN_PATIENTS:
        lo, hi, p = C.clustered_bootstrap(x, g, B=B)
        d.update({"ci_lo": lo, "ci_hi": hi, "bootstrap_p": p,
                  "patient": C.patient_level(x, g)})
    else:
        d["note"] = f"fewer than {MIN_PATIENTS} patients: no CI or p-value reported"
    return d

summary, rows_fig = {}, []
for R in ACCELS:
    reg, pl = load_R(R)
    key = lambda r, f: (r["file"], r[f], r["label"], str(r["x"]), str(r["row_top"]), str(r["w"]), str(r["h"]))
    pb = {}
    for p in pl:
        pb.setdefault(key(p, "src_slice"), []).append(p)
    subs, labels, d_size, d_plac = [], [], [], []
    for r in reg:
        subs.append(r["file"]); labels.append(r["label"])
        d_size.append(r["ssim_les"] - r["ssim_ctl"])
        q = pb.get(key(r, "slice"), [])
        d_plac.append(r["ssim_les"] - float(np.mean([p["ssim_les"] for p in q])) if q else np.nan)
    subs = np.array(subs); labels = np.array(labels)
    d_size = np.array(d_size); d_plac = np.array(d_plac)
    summary[f"R{R}"] = {"pooled": {"vs_size": describe(d_size, subs),
                                   "vs_placebo": describe(d_plac, subs)},
                        "per_label": {}}
    print(f"\n=== R={R} ===  regions {len(reg)} | patients {len(np.unique(subs))}")
    for nm, arr in (("vs_size", d_size), ("vs_placebo", d_plac)):
        s = summary[f"R{R}"]["pooled"][nm]
        print(f"  pooled {nm:11s} {s['mean']:+.4f} [{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}] "
              f"patient p={s['patient']['wilcoxon_p']:.4f} {s['patient']['n_pos']}+/{s['patient']['n_neg']}-")
    for lab in sorted(set(labels)):
        m = labels == lab
        e = {"n_regions": int(m.sum()), "n_patients": int(len(np.unique(subs[m])))}
        for nm, arr in (("vs_size", d_size), ("vs_placebo", d_plac)):
            e[nm] = describe(arr[m], subs[m])
        summary[f"R{R}"]["per_label"][lab] = e
        row = {"R": R, "label": lab, "n_regions": e["n_regions"], "n_patients": e["n_patients"]}
        for nm in ("vs_size", "vs_placebo"):
            row[f"{nm}_mean"] = round(e[nm]["mean"], 5)
            row[f"{nm}_ci_lo"] = round(e[nm]["ci_lo"], 5) if "ci_lo" in e[nm] else ""
            row[f"{nm}_ci_hi"] = round(e[nm]["ci_hi"], 5) if "ci_hi" in e[nm] else ""
            row[f"{nm}_p"] = round(e[nm]["bootstrap_p"], 5) if "bootstrap_p" in e[nm] else ""
        rows_fig.append(row)
        if e["n_patients"] >= MIN_PATIENTS:
            print(f"    {lab:44s} size {e['vs_size']['mean']:+.4f}  placebo {e['vs_placebo']['mean']:+.4f}")

json.dump(summary, open(OUT / "acceleration_R2_R4_R8.json", "w"), indent=2)
with open(FIGD / "fig3_acceleration_per_label.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_fig[0])); w.writeheader(); w.writerows(rows_fig)

whole = []
for R in ACCELS:
    p = OUT / f"whole_image_R{R}.csv"
    rr = list(csv.DictReader(open(p)))
    rec = {"R": R, "n_slices": len(rr),
           "sampled_fraction": round(float(np.mean([float(x["mask_frac"]) for x in rr])), 4)}
    for tag in ("cg", "zf"):
        if f"ssim_{tag}" in rr[0]:
            rec[f"ssim_{tag}"] = round(float(np.mean([float(x[f"ssim_{tag}"]) for x in rr])), 4)
            rec[f"psnr_{tag}"] = round(float(np.mean([float(x[f"psnr_{tag}"]) for x in rr])), 2)
    whole.append(rec)
with open(OUT / "whole_image_summary.csv", "w", newline="") as fh:
    keys = sorted({k for r in whole for k in r})
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(whole)
print("\nwhole-image summary:")
for rec in whole:
    print("  ", rec)
