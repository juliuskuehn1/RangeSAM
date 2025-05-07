import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.utils import *
from preprocess.parser import Parser, SemanticKitti

ARCH = load_yaml("config/arch/LENet.yaml")
DATA = load_yaml("config/labels/semantic-kitti.yaml")
train_data =SemanticKitti(root="dataset",
                            sequences=DATA["split"]["train"],
                            labels=DATA["labels"],
                            color_map=DATA["color_map"],
                            learning_map=DATA["learning_map"],
                            learning_map_inv=DATA["learning_map_inv"],
                            sensor=ARCH["dataset"]["sensor"],
                            max_points=ARCH["dataset"]["max_points"],
                            transform=True,
                            gt=True,
                            drop_few_static_frames=False
                            )
trainloader = torch.utils.data.DataLoader(train_data,
                                                batch_size=1
                                                shuffle=True,
                                                # shuffle=False,
                                                num_workers=8,
                                                pin_memory=True,
                                                drop_last=True)

# Make sure the output directory exists
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Assuming you have the DataLoader already defined
num_images_to_save = 5

# Loop through the DataLoader
for i, sample in enumerate(trainloader):
    if i >= num_images_to_save:
        break

    proj_labels = sample[0].cpu().numpy()     # [H, W]
    proj_mask = sample[2].cpu().numpy()            # [H, W]
    proj_range = sample[8].cpu().numpy()          # [H, W]

    # Apply color map to semantic labels
    color_map = DATA['color_map']
    colored_label = np.zeros((*proj_labels.shape, 3), dtype=np.uint8)
    for label_id, color in color_map.items():
        colored_label[proj_labels == label_id] = color
    colored_label[proj_mask == 0] = 0  # Mask invalid pixels

    # Create a figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(colored_label)
    axs[0].set_title("Semantic Labels (rasterized)")
    axs[0].axis("off")

    im = axs[1].imshow(proj_range, cmap='viridis')
    axs[1].set_title("Range Image")
    axs[1].axis("off")
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

    # Save the figure
    save_path = os.path.join(output_dir, f"sample_{i:02d}.png")
    plt.savefig(save_path)
    plt.close(fig)  # Avoid memory buildup
    print(f"Saved: {save_path}")
