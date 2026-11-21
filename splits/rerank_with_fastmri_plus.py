#!/usr/bin/env python3
"""Regenerate the knee split, preferring test subjects that carry fastMRI+ boxes.

Run once Annotations/knee.csv exists. Split stays subject-wise and the slice
budget is unchanged: 15 train x 20 = 300 slices, 19 test x 20 = 380 slices.

A file listed in knee.csv has >=1 bounding box. A file reviewed with no findings
appears only in knee_file_list.csv, so presence in knee.csv is the right key.
"""
import csv, argparse
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--index",     default=r"D:\Research Results\splits\knee_volume_index.csv")
ap.add_argument("--knee-csv",  default=r"D:\Research Results\fastmri-plus\Annotations\knee.csv")
ap.add_argument("--out",       default=r"D:\Research Results\splits")
ap.add_argument("--slices-per-subject", type=int, default=20)
ap.add_argument("--train-subjects",     type=int, default=15)
ap.add_argument("--test-subjects",      type=int, default=19)
ap.add_argument("--seed",               type=int, default=0)
a = ap.parse_args()

idx = {r["file"]: r for r in csv.DictReader(open(a.index))}

annotated = set()
with open(a.knee_csv) as f:
    for row in csv.DictReader(f):
        stem = (row.get("file") or row.get("Filename") or "").strip()
        if stem:
            annotated.add(stem if stem.endswith(".h5") else stem + ".h5")

subs    = sorted(idx)
has_box = [s for s in subs if s in annotated]
no_box  = [s for s in subs if s not in annotated]
print(f"{len(has_box)}/{len(subs)} local subjects carry fastMRI+ boxes")
if len(has_box) < a.test_subjects:
    print(f"WARNING: only {len(has_box)} annotated; filling the rest from unannotated subjects")

rng  = np.random.default_rng(a.seed)
pool = list(rng.permutation(has_box)) + list(rng.permutation(no_box))   # annotated first
test = sorted(pool[:a.test_subjects])
rest = [s for s in subs if s not in set(test)]
train = sorted(list(rng.permutation(rest))[:a.train_subjects])
assert not set(train) & set(test), "subject leak between train and test"

def central(n, k):
    s = (n - k) // 2
    return range(s, s + k)

out = Path(a.out)
for name, group in (("train", train), ("test", test)):
    rows = [[fn, s, idx[fn]["slices"], idx[fn]["coils"], idx[fn]["kx"], idx[fn]["ky"],
             int(fn in annotated)]
            for fn in group
            for s in central(int(idx[fn]["slices"]), a.slices_per_subject)]
    with open(out / f"{name}_slices.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file","slice","n_slices","coils","kx","ky","fastmri_plus"])
        w.writerows(rows)
    (out / f"{name}_subjects.txt").write_text("\n".join(group) + "\n")
    print(f"{name}: {len(group)} subjects, {len(rows)} slices, "
          f"{sum(1 for s in group if s in annotated)} annotated")
