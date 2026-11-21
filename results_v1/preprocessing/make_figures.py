#!/usr/bin/env python3
"""Figures for the lesion-fidelity paper."""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

RES = Path(r"D:\Research Results\results")
FIG = Path(r"D:\Research Results\figures"); FIG.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(RES / "sweep_covariates.csv")))
ACC = (2, 4, 8)
def arr(k): return np.array([float(r[k]) for r in rows])
subs = np.array([r["file"] for r in rows]); labels = np.array([r["label"] for r in rows])
uniq_s = np.unique(subs); rng = np.random.default_rng(0)

def boot_ci(x, groups, B=5000):
    u = np.unique(groups)
    vals = [np.mean(x[np.concatenate([np.where(groups == s)[0] for s in rng.choice(u, len(u), True)])])
            for _ in range(B)]
    return np.percentile(vals, [2.5, 97.5])

delta = {R: arr(f"ssim_les_R{R}") - arr(f"ssim_ctl_R{R}") for R in ACC}

# ---------- Figure 1: per-label effect at R=4 ----------
d4 = delta[4]
order, stats = [], {}
for lab in np.unique(labels):
    m = labels == lab
    if m.sum() < 5: continue
    lo, hi = boot_ci(d4[m], subs[m])
    stats[lab] = (d4[m].mean(), lo, hi, int(m.sum()))
    order.append(lab)
order.sort(key=lambda l: stats[l][0])
short = {"Cartilage - Partial Thickness loss/defect": "Cartilage defect\n(partial thickness)",
         "Ligament - ACL Low Grade sprain": "ACL sprain\n(low grade)",
         "Ligament - ACL High Grade Sprain": "ACL sprain\n(high grade)",
         "Ligament - MCL Low-Mod Grade Sprain": "MCL sprain\n(low-mod)",
         "Patellar Retinaculum - High grade sprain": "Patellar retinaculum\n(high grade)",
         "Meniscus Tear": "Meniscus tear", "Joint Effusion": "Joint effusion"}
fig, ax = plt.subplots(figsize=(7.0, 3.6))
y = np.arange(len(order))
for i, lab in enumerate(order):
    mu, lo, hi, n = stats[lab]
    c = "#B2182B" if mu < 0 else "#2166AC"
    ax.plot([lo, hi], [i, i], color=c, lw=2.0, solid_capstyle="round", alpha=0.75, zorder=2)
    ax.plot([mu], [i], "o", color=c, ms=7, zorder=3)
    ax.text(hi + 0.012, i, f"n={n}", va="center", ha="left", fontsize=6, color="#555555")
ax.axvline(0, color="#333333", lw=0.9, ls="--", zorder=1)
ax.set_yticks(y); ax.set_yticklabels([short.get(l, l) for l in order])
ax.set_xlabel("SSIM difference, lesion ROI minus matched tissue control")
ax.set_title("Fluid findings exceed matched tissue; structural findings do not")
ax.text(-0.012, -0.95, "lesion worse than control", fontsize=6, color="#B2182B", ha="right")
ax.text(0.012, -0.95, "lesion better than control", fontsize=6, color="#2166AC", ha="left")
ax.margins(0.06); ax.set_ylim(-1.4, len(order) - 0.4)
fig.tight_layout(); fig.savefig(FIG / "fig1_per_label.png", dpi=300); plt.close(fig)

# ---------- Figure 2: acceleration sweep ----------
neg = np.isin(labels, [l for l in stats if stats[l][0] < 0])
pos = np.isin(labels, [l for l in stats if stats[l][0] >= 0])
fig, ax = plt.subplots(figsize=(4.2, 3.2))
for m, lab, c in ((neg, "Structural findings\n(cartilage, ACL)", "#B2182B"),
                  (pos, "Fluid / bulk findings\n(effusion, meniscus)", "#2166AC"),
                  (np.ones(len(rows), bool), "All ROIs", "#777777")):
    mu = [delta[R][m].mean() for R in ACC]
    ci = [boot_ci(delta[R][m], subs[m]) for R in ACC]
    ax.plot(ACC, mu, "o-", color=c, lw=1.8, ms=5, label=lab, zorder=3)
    ax.fill_between(ACC, [c0 for c0, _ in ci], [c1 for _, c1 in ci], color=c, alpha=0.13, lw=0)
ax.axhline(0, color="#333333", lw=0.9, ls="--")
ax.set_xticks(ACC); ax.set_xlabel("Acceleration factor R"); ax.set_xscale("log", base=2)
ax.set_xticks(ACC); ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_ylabel("SSIM, lesion minus control")
ax.set_title("The structural deficit peaks at R=4, not at R=8")
ax.legend(frameon=False, fontsize=6, loc="upper left")
ax.margins(0.06); fig.tight_layout()
fig.savefig(FIG / "fig2_acceleration.png", dpi=300); plt.close(fig)

summary = {"per_label_R4": {l: {"delta": stats[l][0], "ci": [stats[l][1], stats[l][2]],
                               "n": stats[l][3]} for l in order},
           "by_R": {R: {"all": float(delta[R].mean()),
                        "structural": float(delta[R][neg].mean()),
                        "fluid": float(delta[R][pos].mean())} for R in ACC},
           "n_roi": len(rows), "n_subjects": int(len(uniq_s))}
json.dump(summary, open(RES / "figure_summary.json", "w"), indent=2, default=float)
print(json.dumps(summary["by_R"], indent=1))
print("figures ->", FIG)
