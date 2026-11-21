"""Database-mode loader for CUPID (keeps the dataset off the GPU).

WHY: src/data/loader.py:load_data does
        torch.from_numpy(kspace_train).to(device)
    i.e. it moves the WHOLE dataset to the GPU before the DataLoader runs.
    At 320x368x15 complex64 a slice is 13.5 MiB, so 300 slices of kspace+coils
    is 7.9 GiB -- it OOMs a 4 GB card before the model is even allocated.

WHAT: memory-maps the stacked arrays and yields one slice at a time on the CPU.
    Peak host RAM stays at a few hundred MB and VRAM holds only the live batch.

NOT YET RUN: torch was not installable in the sandbox (no egress), so this
    module is unexecuted. Validate it on the first training run.

USE: in src/train.py replace
        ksp_loader, maps_loader, mask = load_data(device, **data_config, **mask_config)
     with
        from data.loader_db import load_data_db
        ksp_loader, maps_loader, mask = load_data_db(
            device,
            r"D:\Research Results\cupid_data\train_kspace_300.npy",
            r"D:\Research Results\cupid_data\train_coils_300.npy",
            **mask_config)
     and move each batch onto the device at the top of the slice loop:
        ksp_batch, maps_batch = ksp_batch.to(device), maps_batch.to(device)
"""
import numpy as np
import torch

from data.loader import generate_mask


class _SliceSet(torch.utils.data.Dataset):
    """Yields [C, H, W] complex64 from a memory-mapped [B, H, W, C] array."""

    def __init__(self, arr):
        self.arr = arr

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, i):
        x = np.ascontiguousarray(self.arr[i])          # [H, W, C]
        return torch.from_numpy(x).to(torch.complex64).permute(2, 0, 1)


def load_data_db(device, kspace_path, coils_path, **mask_config):
    ksp = np.load(kspace_path, mmap_mode="r")
    coils = np.load(coils_path, mmap_mode="r")
    assert ksp.shape == coils.shape, (ksp.shape, coils.shape)

    # the mask only needs one slice to find the no-signal phase-encode lines
    probe = torch.from_numpy(np.ascontiguousarray(ksp[:1])).to(torch.complex64)
    probe = probe.permute(0, 3, 1, 2).to(device)
    mask = generate_mask(probe, **mask_config)
    del probe

    ksp_loader = torch.utils.data.DataLoader(_SliceSet(ksp), batch_size=1, shuffle=False)
    coils_loader = torch.utils.data.DataLoader(_SliceSet(coils), batch_size=1, shuffle=False)
    return ksp_loader, coils_loader, mask
