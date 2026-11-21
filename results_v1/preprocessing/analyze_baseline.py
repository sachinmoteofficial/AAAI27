#!/usr/bin/env python3
"""Statistics for the global-vs-lesion fidelity gap (training-free arm)."""
import csv, json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

RES = Path(r"D:\Research Results\results")
rows = list(csv.DictReader(open(RES / "baseline_metrics.csv")))
def f(r, k): 
    v = r.get(k, "")
    return float(v) if v not in ("", None) else np.nan

boxed = [r for r in rows if r.get("ssim_box_cg", "") not in ("", None)]
print(f"slices: {len(rows)} | with boxes: {len(boxed)} | subjects: {len({r['file'] for r in rows})}")

out = {"n_slices": len(rows), "n_boxed": len(boxed),
       "n_subjects": len({r["file"] for r in rows})}

for tag, name in (("cg", "CG-SENSE"), ("zf", "zero-filled")):
    g_all = np.array([f(r, f"ssim_{tag}") for r in rows])
    g = np.array([f(r, f"ssim_{tag}") for r in boxed])
    b = np.array([f(r, f"ssim_box_{tag}") for r in boxed])
    gp = np.array([f(r, f"psnr_{tag}") for r in boxed])
    bp = np.array([f(r, f"psnr_box_{tag}") for r in boxed])
    d_ssim, d_psnr = g - b, gp - bp
    st = wilcoxon(g, b)
    # patient-clustered bootstrap over subjects
    subs = np.array([r["file"] for r in boxed]); uniq = np.unique(subs)
    rng = np.random.default_rng(0); boot = []
    for _ in range(2000):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.where(subs == s)[0] for s in pick])
        boot.append(np.mean(d_ssim[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out[tag] = {"ssim_global_all_slices": float(np.mean(g_all)),
                "ssim_global_boxed": float(np.mean(g)), "ssim_box": float(np.mean(b)),
                "ssim_drop": float(np.mean(d_ssim)), "ssim_drop_ci": [float(lo), float(hi)],
                "psnr_global_boxed": float(np.mean(gp)), "psnr_box": float(np.mean(bp)),
                "psnr_drop_db": float(np.mean(d_psnr)),
                "wilcoxon_p": float(st.pvalue),
                "frac_slices_box_worse": float(np.mean(d_ssim > 0))}
    print(f"\n{name}: global SSIM {np.mean(g):.4f} vs in-box {np.mean(b):.4f} "
          f"(drop {np.mean(d_ssim):.4f}, 95% CI {lo:.4f}-{hi:.4f}, p={st.pvalue:.2e})")
    print(f"  PSNR global {np.mean(gp):.2f} dB vs in-box {np.mean(bp):.2f} dB "
          f"(drop {np.mean(d_psnr):.2f} dB); box worse on {100*np.mean(d_ssim>0):.1f}% of slices")

# per-label breakdown (a slice may carry several labels)
per_label = {}
for r in boxed:
    for lab in (r.get("labels") or "").split("|"):
        if lab: per_label.setdefault(lab, []).append(f(r, "ssim_cg") - f(r, "ssim_box_cg"))
print("\nper-label SSIM drop (CG-SENSE):")
for lab, v in sorted(per_label.items(), key=lambda kv: -np.mean(kv[1])):
    if len(v) >= 3:
        print(f"  {lab:48s} n={len(v):3d}  drop={np.mean(v):.4f}")
    per_label[lab] = {"n": len(v), "mean_drop": float(np.mean(v))}
out["per_label"] = per_label
json.dump(out, open(RES / "baseline_stats.json", "w"), indent=2)
print("\nwrote", RES / "baseline_stats.json")
