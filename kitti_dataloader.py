import os
import glob
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, random_split


class Seq02NpZDataset(Dataset):
    """
    A Dataset for sequence 02 .npz files.

    Each .npz file in the seq_dir is expected to contain at least three arrays:
      - `proj_full`: tensor of shape (C, H, W)
      - `proj_mask`: mask of shape (H, W)
      - `proj_labels`: label map of shape (H, W)
    """
    def __init__(
        self,
        seq_dir,
        data_key: str = 'proj_full',
        label_key: str = 'proj_labels',
        mask_key: str = 'proj_mask',
        transform=None
    ):
        self.npz_files = sorted(glob.glob(os.path.join(seq_dir, '*.npz')))
        self.data_key = data_key
        self.label_key = label_key
        self.mask_key = mask_key
        self.transform = transform

        assert len(self.npz_files) > 0, f"No .npz files found in {seq_dir}"

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        arr = np.load(self.npz_files[idx])
        data = arr[self.data_key]       # shape (C, H, W)
        label = arr[self.label_key]     # shape (H, W)
        mask  = arr[self.mask_key]      # shape (H, W)

        # to torch
        data_tensor  = torch.from_numpy(data).float()
        label_tensor = torch.from_numpy(label).long()
        mask_tensor  = torch.from_numpy(mask).bool()

        if self.transform:
            data_tensor, label_tensor, mask_tensor = self.transform(
                data_tensor, label_tensor, mask_tensor
            )

        return {
            "data": data_tensor,
            "label": label_tensor,
            "mask": mask_tensor
        }


def create_dataloaders(
    seq_dir: str,
    batch_size: int = 8,
    val_split: float = 0.2,
    shuffle: bool = True,
    num_workers: int = 4,
    **dataset_kwargs
):
    """
    Create train/val DataLoaders for seq_02 npz data.

    Args:
      - seq_dir: path to the seq_02 folder containing .npz files
      - batch_size: per-GPU batch size
      - val_split: fraction for validation (default 0.2)
      - shuffle: whether to shuffle the training set
      - num_workers: DataLoader workers
      - dataset_kwargs: passed to Seq02NpZDataset (data_key, label_key, mask_key, transform)
    """
    dataset = Seq02NpZDataset(seq_dir, **dataset_kwargs)
    total = len(dataset)
    val_size = int(val_split * total)
    train_size = total - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader


# Example usage:
# seq02_dir = '/path/to/seq_02'
# train_loader, val_loader = create_dataloaders(
#     seq02_dir,
#     batch_size=16,
#     val_split=0.2,
#     shuffle=True,
#     num_workers=4
# )
#
# for inputs, targets, masks in train_loader:
#     # inputs: (B, C, H, W)
#     # targets/masks: (B, H, W)
#     pass
