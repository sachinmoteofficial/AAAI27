#!/usr/bin/env python3
"""Measure CUPID's per-slice VRAM and step time on this GPU.

Measures the unrolled network plus the parallel-imaging perturbation loss --
the dominant cost. The compressibility term is NOT included: it needs
pytorch_wavelets (absent here) and runs once per slice at epoch 0, not every
step. Perturbations use random tensors of the published magnitude instead of
the repo's glyph patterns, which is cost-equivalent (identical shapes and ops)
but not bit-identical to training.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(r"D:\Research Results\CUPID")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))
from data.mri_ops import c2r, EHE                      # noqa: E402
from models.unrolled_net import UnrolledNet            # noqa: E402

def generate_mask(ksp, ACS_length, acc_rate, mask_type):
    """Copied verbatim from CUPID src/data/loader.py (importing that module
    pulls in yaml, which is absent from this environment)."""
    _, _, Nro, Npe = ksp.size()
    mask = torch.zeros(size=(Nro, Npe), dtype=torch.bool, device=ksp.device)
    mask[abs(ksp[0, 0, ...]) < 5e-9] = True
    ACS_start = int((Npe - ACS_length) / 2) - 1
    ACS_end = ACS_start + ACS_length
    mask[:, ACS_start:ACS_end] = True
    if mask_type == 'equidistant':
        mask[:, 0::acc_rate] = True
    return mask

dev = torch.device("cuda:0")
d = np.load(r"D:\Research Results\cupid_data\test\file1000041.npy", allow_pickle=True).item()
ksp = torch.from_numpy(d["kspace"][:1]).to(torch.complex64).permute(0, 3, 1, 2).to(dev)
mps = torch.from_numpy(d["coils"][:1]).to(torch.complex64).permute(0, 3, 1, 2).to(dev)
ksp = ksp / ksp.abs().max()
mask = generate_mask(ksp, ACS_length=24, acc_rate=4, mask_type="equidistant")
print(f"slice {tuple(ksp.shape)} | mask sampled {mask.float().mean():.3f}", flush=True)

def build(unrolls, cg_iter):
    cfg = {"nb_unroll_blocks": unrolls,
           "ResNet": {"nb_res_blocks": 15, "channels": 64, "kernel_size": 3, "stride": 1,
                      "padding": 1, "bias": False, "scalar": 0.1, "use_checkpointing": True,
                      "weights_mean": 0.0, "weights_std": 0.05},
           "DC": {"CG_Iter": cg_iter, "mu": 0.05}}
    return UnrolledNet(dev, **cfg).to(dev)

def step(model, opt, nperturb, per_pert_backward, lam=200.0):
    zf = EHE(torch.zeros_like(ksp[:, 0]), mps, mask)      # placeholder x_PI
    zf = EHE(ksp[:, 0] * 0 + zf, mps, mask)
    opt.zero_grad(set_to_none=True)
    unp, _ = model(c2r(zf, dim=1), mps, mask)
    unp_c = torch.complex(unp[:, 0], unp[:, 1])
    rssq = torch.sqrt(torch.sum(mps ** 2, dim=1))
    keep = (rssq.abs() != 0)
    total = 0.0
    for k in range(nperturb):
        p_k = (5e-2) * torch.randn_like(zf)               # cost-equivalent to glyph perturbation
        pin = zf + EHE(p_k, mps, mask)
        pout, _ = model(c2r(pin, dim=1), mps, mask)
        pout_c = torch.complex(pout[:, 0], pout[:, 1])
        l = torch.norm((pout_c - unp_c) - p_k * keep, p=2) / torch.norm(p_k * keep, p=2)
        if per_pert_backward:
            (l * lam / nperturb).backward(retain_graph=(k < nperturb - 1))
        else:
            total = total + l
    if not per_pert_backward:
        (total * lam / nperturb).backward()
    opt.step()

rows = []
BUDGET_S = 420  # stop starting new configs past this total
t_start = time.time()
for unrolls, cg_iter, nperturb, ppb in [(6,6,2,True), (8,6,4,True), (10,8,6,True),
                                        (10,15,6,True), (10,15,6,False)]:
    if time.time() - t_start > BUDGET_S:
        print(f"skip {unrolls}/{cg_iter}/{nperturb}: probe budget spent", flush=True)
        continue
    tag = f"U{unrolls} G{cg_iter} P{nperturb} {'per-pert bwd' if ppb else 'summed bwd'}"
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        model = build(unrolls, cg_iter)
        opt = torch.optim.Adam(model.parameters(), lr=5e-4)
        step(model, opt, nperturb, ppb)                  # warm-up
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        ts = []
        t0 = time.time(); step(model, opt, nperturb, ppb)
        torch.cuda.synchronize(); ts.append(time.time() - t0)
        peak = torch.cuda.max_memory_allocated() / 2**30
        rows.append({"config": tag, "unrolls": unrolls, "cg_iter": cg_iter,
                     "perturbations": nperturb, "per_pert_backward": ppb,
                     "peak_GiB": round(peak, 2), "s_per_step": round(float(np.median(ts)), 3),
                     "h_per_slice_300ep": round(float(np.median(ts)) * 300 / 3600, 3),
                     "status": "ok"})
        print(f"{tag:34s} peak {peak:5.2f} GiB  {np.median(ts):6.3f} s/step  "
              f"{np.median(ts)*300/3600:5.2f} h/slice-run", flush=True)
    except torch.cuda.OutOfMemoryError as e:
        rows.append({"config": tag, "unrolls": unrolls, "cg_iter": cg_iter,
                     "perturbations": nperturb, "per_pert_backward": ppb,
                     "status": "OOM", "error": str(e)[:120]})
        print(f"{tag:34s} OOM", flush=True)
    finally:
        for v in ("model", "opt"):
            if v in dir(): pass
        torch.cuda.empty_cache()

out = Path(r"D:\Research Results\results\gpu_probe.json")
json.dump({"gpu": torch.cuda.get_device_name(0),
           "vram_GiB": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
           "torch": torch.__version__, "runs": rows}, open(out, "w"), indent=2)
print("\nwrote", out)
