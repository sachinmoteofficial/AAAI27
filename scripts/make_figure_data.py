#!/usr/bin/env python3
"""Figure source data: example arrays, orientation panels, registration table."""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SPL = Path(r"D:\Research Results\splits")
CD = Path(r"D:\Research Results\cupid_data\test")
OUT = Path(r"D:\Research Results\v2\results")
FIGD = Path(r"D:\Research Results\v2\figure_data")
ORI = Path(r"D:\Research Results\v2\orientation")
ACS, R = 24, 4

want = {}
for r in csv.DictReader(open(SPL / "test_slices.csv")):
    want.setdefault(r["file"], []).append(int(r["slice"]))
want = {k: sorted(v) for k, v in want.items()}
regions = list(csv.DictReader(open(OUT / "regions_R4.csv")))
contrasts = {(r["file"], r["slice"], r["label"]): r
             for r in csv.DictReader(open(OUT / "contrasts_per_region_R4.csv"))}

_cache = {}
def recon(fn, z):
    if (fn, z) in _cache:
        return _cache[(fn, z)]
    d = np.load(CD / fn.replace(".h5", ".npy"), allow_pickle=True).item()
    i = want[fn].index(z)
    K = np.moveaxis(d["kspace"][i], -1, 0); S = np.moveaxis(d["coils"][i], -1, 0)
    K = K / np.abs(K).max()
    m = C.make_mask(K, ACS, R)
    ref = C.crop320(np.abs((np.conj(S) * C.ifft2c(K)).sum(0)))
    dr = float(ref.max())
    cg = C.crop320(np.abs(C.cg_sense(K, S, m)))
    out = {"ref": ref, "cg": cg, "dr": dr, "err": np.abs(ref - cg),
           "ssim": C.ssim_map(ref, cg, dr)}
    _cache[(fn, z)] = out
    del d
    return out

# ---------- (iii) one cartilage-defect example ----------
cart = [r for r in regions if r["label"].startswith("Cartilage")]
def contrast_of(r):
    c = contrasts.get((r["file"], r["slice"], r["label"]))
    try:
        return float(c["ssim_cg_vs_placebo"])
    except (TypeError, KeyError, ValueError):
        return np.inf
cart.sort(key=contrast_of)
pick = cart[0]
fn, z = pick["file"], int(pick["slice"])
im = recon(fn, z)
for nm in ("ref", "cg", "err", "ssim"):
    np.save(FIGD / f"example_{nm}.npy", im[nm].astype(np.float32))

def parse_coords(s):
    return [tuple(int(v) for v in p.split(",")) for p in s.split(";") if p]

placebo_rows = [p for p in csv.DictReader(open(OUT / "placebo_R4.csv"))
                if p["file"] == fn and p["src_slice"] == str(z) and p["label"] == pick["label"]]
meta = {"file": fn, "slice": z, "label": pick["label"],
        "R": R, "acs": ACS, "image_shape": [C.IMG, C.IMG],
        "box_raw_csv": {"x": int(pick["x"]), "y_bottom_origin": int(pick["y_raw"]),
                        "w": int(pick["w"]), "h": int(pick["h"])},
        "box_array_indices": {"row_top": int(pick["row_top"]), "col_left": int(pick["x"]),
                              "h": int(pick["h"]), "w": int(pick["w"])},
        "size_matched_controls_row_col": parse_coords(pick["size_ctrl_coords"]),
        "appearance_matched_controls_row_col": parse_coords(pick["app_ctrl_coords"]),
        "ssim_lesion_cg": float(pick["ssim_les_cg"]),
        "ssim_size_control_cg": float(pick["ssim_sizectl_cg"]),
        "contrast_vs_placebo": contrast_of(pick),
        "arrays": {k: f"example_{k}.npy" for k in ("ref", "cg", "err", "ssim")}}
if placebo_rows:
    p0 = placebo_rows[len(placebo_rows) // 2]
    meta["placebo_example"] = {"slice": int(p0["slice"]), "row_top": int(p0["row_top"]),
                               "col_left": int(p0["x"]), "h": int(p0["h"]), "w": int(p0["w"]),
                               "ssim_lesion_cg": float(p0["ssim_les_cg"])}
json.dump(meta, open(FIGD / "example_meta.json", "w"), indent=2)
print(f"example: {fn} slice {z} ({pick['label']})")

# ---------- (iv) registration table ----------
reg = list(csv.DictReader(open(OUT / "registration_per_box.csv")))
with open(FIGD / "fig4_registration_offsets.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(reg[0])); w.writeheader(); w.writerows(reg)

# ---------- orientation panels, corrected boxes only ----------
TARGETS = ["Meniscus Tear", "Cartilage - Partial Thickness loss/defect", "Joint Effusion"]
SHORT = {"Meniscus Tear": "meniscus_tear",
         "Cartilage - Partial Thickness loss/defect": "cartilage",
         "Joint Effusion": "joint_effusion"}
for lab in TARGETS:
    cand = [r for r in regions if r["label"] == lab]
    seen, picks = set(), []
    for r in cand:
        if r["file"] not in seen:
            picks.append(r); seen.add(r["file"])
        if len(picks) == 5:
            break
    for r in cand:
        if len(picks) >= 5:
            break
        if r not in picks:
            picks.append(r)
    if not picks:
        continue
    fig, axes = plt.subplots(1, len(picks), figsize=(3.0 * len(picks), 3.4))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, picks):
        d = recon(r["file"], int(r["slice"]))
        ax.imshow(d["ref"], cmap="gray", vmax=d["dr"] * 0.8)
        ax.add_patch(Rectangle((int(r["x"]), int(r["row_top"])), int(r["w"]), int(r["h"]),
                               fill=False, ec="#2EC4B6", lw=1.8))
        ax.set_title(f"{r['file'][4:11]} sl{r['slice']}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{lab} - corrected box position (row_top = 320 - y - h)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ORI / f"corrected_{SHORT[lab]}.png", dpi=130)
    plt.close(fig)
    print("wrote", ORI / f"corrected_{SHORT[lab]}.png")
print("figure data complete")
