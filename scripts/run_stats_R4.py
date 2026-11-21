#!/usr/bin/env python3
"""Contrasts and inference for the corrected R=4 analysis."""
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

OUT = Path(r"D:\Research Results\v2\results")
FIGD = Path(r"D:\Research Results\v2\figure_data")
B = 5000
MIN_PATIENTS_FOR_INFERENCE = 3
HIGHLIGHT = ("Cartilage - Partial Thickness loss/defect", "Meniscus Tear")

regions = list(csv.DictReader(open(OUT / "regions_R4.csv")))
placebo = list(csv.DictReader(open(OUT / "placebo_R4.csv")))

def key(r, slice_field):
    return (r["file"], r[slice_field], r["label"], r["x"], r["row_top"], r["w"], r["h"])

pb = {}
for p in placebo:
    pb.setdefault(key(p, "src_slice"), []).append(p)

def f(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else np.nan

rows = []
for r in regions:
    k = key(r, "slice")
    pl = pb.get(k, [])
    rec = {"file": r["file"], "slice": int(r["slice"]), "label": r["label"],
           "area": int(r["area"]), "n_placebo": len(pl),
           "n_app_ctrl": int(r["n_app_ctrl"])}
    for tag in ("cg", "zf"):
        for met in ("ssim", "psnr"):
            les = f(r, f"{met}_les_{tag}")
            szc = f(r, f"{met}_sizectl_{tag}")
            apc = f(r, f"{met}_appctl_{tag}")
            rec[f"{met}_{tag}_les"] = les
            rec[f"{met}_{tag}_vs_size"] = les - szc            # 2a
            rec[f"{met}_{tag}_vs_app"] = les - apc             # 3
            if pl:
                pl_les = np.mean([f(p, f"{met}_les_{tag}") for p in pl])
                pl_d = np.mean([f(p, f"{met}_les_{tag}") - f(p, f"{met}_sizectl_{tag}") for p in pl])
                rec[f"{met}_{tag}_placebo_les"] = float(pl_les)
                rec[f"{met}_{tag}_vs_placebo"] = float(les - pl_les)                  # 2c
                rec[f"{met}_{tag}_diffindiff"] = float((les - szc) - pl_d)            # 2d
    rows.append(rec)

with open(OUT / "contrasts_per_region_R4.csv", "w", newline="") as fh:
    keys = sorted({k for r in rows for k in r})
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

subs = np.array([r["file"] for r in rows])
labels = np.array([r["label"] for r in rows])

CONTRASTS = [("vs_size", "lesion minus size-matched control"),
             ("vs_app", "lesion minus appearance-matched control"),
             ("vs_placebo", "lesion minus same box on unannotated slices"),
             ("diffindiff", "real delta minus placebo delta")]

def describe(x, g, with_inference=True):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, g = x[ok], np.asarray(g)[ok]
    d = {"n_regions": int(len(x)), "n_patients": int(len(np.unique(g))),
         "mean": float(x.mean()) if len(x) else float("nan")}
    if with_inference and len(x) and len(np.unique(g)) >= MIN_PATIENTS_FOR_INFERENCE:
        lo, hi, p = C.clustered_bootstrap(x, g, B=B)
        d.update({"ci_lo": lo, "ci_hi": hi, "bootstrap_p": p})
        d["patient"] = C.patient_level(x, g)
    else:
        d["note"] = f"fewer than {MIN_PATIENTS_FOR_INFERENCE} patients: no CI or p-value reported"
    return d

summary = {"n_regions": len(rows), "n_patients": int(len(np.unique(subs))),
           "pooled": {}, "per_label": {}}
print(f"regions {len(rows)} | patients {len(np.unique(subs))}\n")
for tag in ("cg", "zf"):
    for met in ("ssim", "psnr"):
        for cname, cdesc in CONTRASTS:
            col = f"{met}_{tag}_{cname}"
            x = np.array([r.get(col, np.nan) for r in rows], float)
            key_ = f"{tag}_{met}_{cname}"
            summary["pooled"][key_] = {"description": cdesc, **describe(x, subs)}
            s = summary["pooled"][key_]
            u = " dB" if met == "psnr" else ""
            line = f"{tag:3s} {met:4s} {cname:12s} mean {s['mean']:+8.4f}{u}"
            if "ci_lo" in s:
                line += (f"  CI [{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}]"
                         f"  patient p={s['patient']['wilcoxon_p']:.4f}"
                         f"  {s['patient']['n_pos']}+/{s['patient']['n_neg']}-")
            print(line)
    print()

for lab in sorted(set(labels)):
    m = labels == lab
    npat = int(len(np.unique(subs[m])))
    entry = {"n_regions": int(m.sum()), "n_patients": npat,
             "highlighted": lab in HIGHLIGHT}
    for tag in ("cg", "zf"):
        for met in ("ssim", "psnr"):
            for cname, _ in CONTRASTS:
                col = f"{met}_{tag}_{cname}"
                x = np.array([r.get(col, np.nan) for r in rows], float)[m]
                entry[f"{tag}_{met}_{cname}"] = describe(x, subs[m])
    summary["per_label"][lab] = entry

json.dump(summary, open(OUT / "contrasts_R4.json", "w"), indent=2)

# ---- figure source data ----
fig1 = []
for lab in sorted(set(labels)):
    e = summary["per_label"][lab]
    row = {"label": lab, "n_regions": e["n_regions"], "n_patients": e["n_patients"]}
    for cname in ("vs_size", "vs_app", "vs_placebo"):
        s = e[f"cg_ssim_{cname}"]
        row[f"{cname}_mean"] = round(s["mean"], 5)
        row[f"{cname}_ci_lo"] = round(s["ci_lo"], 5) if "ci_lo" in s else ""
        row[f"{cname}_ci_hi"] = round(s["ci_hi"], 5) if "ci_hi" in s else ""
        row[f"{cname}_p"] = round(s["bootstrap_p"], 5) if "bootstrap_p" in s else ""
    fig1.append(row)
with open(FIGD / "fig1_per_label_forest.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(fig1[0])); w.writeheader(); w.writerows(fig1)

fig2 = []
for s_ in sorted(set(subs)):
    m = subs == s_
    row = {"file": s_, "n_regions": int(m.sum())}
    for cname in ("vs_size", "vs_app", "vs_placebo", "diffindiff"):
        for tag in ("cg", "zf"):
            x = np.array([r.get(f"ssim_{tag}_{cname}", np.nan) for r in rows], float)[m]
            row[f"{tag}_{cname}"] = round(float(np.nanmean(x)), 5)
    fig2.append(row)
with open(FIGD / "fig2_per_patient.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(fig2[0])); w.writeheader(); w.writerows(fig2)

# ---- appearance-matching quality ----
match = {}
for k_ in ("mean", "contrast", "std", "grad"):
    les = np.array([f(r, f"les_{k_}") for r in regions])
    for nm in ("size", "app"):
        ctl = np.array([f(r, f"{nm}ctl_{k_}") for r in regions])
        ok = np.isfinite(les) & np.isfinite(ctl)
        match[f"{k_}_{nm}"] = {"lesion": float(les[ok].mean()), "control": float(ctl[ok].mean()),
                               "diff": float((les - ctl)[ok].mean()),
                               "wilcoxon_p": float(wilcoxon(les[ok], ctl[ok]).pvalue),
                               "n": int(ok.sum())}
json.dump(match, open(OUT / "appearance_match_quality.json", "w"), indent=2)
print("\nappearance matching quality (lesion vs control):")
for k_ in ("mean", "contrast", "std"):
    a, b = match[f"{k_}_size"], match[f"{k_}_app"]
    print(f"  {k_:9s} lesion {a['lesion']:.4f} | size-matched {a['control']:.4f} (p={a['wilcoxon_p']:.2e})"
          f" | appearance-matched {b['control']:.4f} (p={b['wilcoxon_p']:.2e})")
print("\nwrote", OUT / "contrasts_R4.json")
