import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np

import satellite_cloud_generator as scg
import random

import torch
from diffusers import UNet2DModel
from src.scheduler import load_scheduler
from src.DDIM_grad_sample import DDIM_sampling  
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def load_mean_results(result_dir):
    result_dir = Path(result_dir)
    metrics_dir = result_dir / "recon"
    metric_files = sorted(metrics_dir.glob("*_metrics.npz"))

    psnr_values = []
    ssim_values = []
    lpips_values = []

    for file in metric_files:
        metrics = np.load(file)
        psnr_values.append(float(metrics["final_psnr"]))
        ssim_values.append(float(metrics["final_ssim"]))
        lpips_values.append(float(metrics["final_lpips"]))

    print(f"Number of images: {len(metric_files)}")
    print(f"PSNR:  {np.mean(psnr_values):.4f} ± {np.std(psnr_values):.4f}")
    print(f"LPIPS: {np.mean(lpips_values):.4f} ± {np.std(lpips_values):.4f}")
    print(f"SSIM:  {np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f}")


def visualize_recon(stem, data_dir, results_dir, fontsize=12):
    data_dir = Path(data_dir)#
    result_dir = Path(results_dir)

    original = np.array(
        Image.open(data_dir / f"{stem}.png").convert("RGB").resize((256, 256))
    ) / 255.0

    measurement = np.load(result_dir / "measure" / f"{stem}_measure.npy").squeeze()
    recon_file = f"{stem}_es_recon.npy" if "weak" in results_dir else f"{stem}_recon.npy"
    reconstruction = np.load(result_dir / "recon" / recon_file).squeeze()

    reconstruction = reconstruction.transpose(1, 2, 0)
    measurement = measurement.transpose(1, 2, 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(original)
    axes[0].set_title("Reference", fontsize=fontsize)

    measurement_vis = (measurement + 1.0) / 2.0
    measurement_vis = np.clip(measurement_vis, 0, 1)

    axes[1].imshow(measurement_vis)
    axes[1].set_title("Measurement", fontsize=fontsize)

    axes[2].imshow(np.clip(reconstruction, 0, 1))
    axes[2].set_title("Reconstruction", fontsize=fontsize)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()



def visualize_process(directory):
    directory = Path(directory)
    files = sorted(directory.glob("*.npy"))

    n_images = len(files)
    fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))

    # Handle the single-image case
    axes = np.atleast_1d(axes)

    for ax, file in zip(axes, files):
        image = np.load(file).squeeze()

        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)

        ax.imshow(image[..., :3])
        ax.set_title(file.name)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_cloud_levels(stem, data_dir):
    data_dir = Path(data_dir)

    image = np.array(
        Image.open(data_dir / f"{stem}.png").convert("RGB")
    ) / 255.0

    # HWC -> BCHW
    image_chw = np.transpose(image, (2, 0, 1))[None, ...]

    # Cloud levels
    levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    fig, axes = plt.subplots(
        1, len(levels) + 1,
        figsize=(21, 4)
    )

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Generate cloud variations
    for i, level in enumerate(levels):
        np.random.seed(42)
        random.seed(42)

        cloudy, mask = scg.add_cloud(
            image_chw,
            min_lvl=0,
            max_lvl=level,
            return_cloud=True
        )

        cloudy = np.transpose(cloudy[0], (1, 2, 0))
        cloudy = np.clip(cloudy, 0, 1)

        axes[i + 1].imshow(cloudy)
        axes[i + 1].set_title(f"Cloud level: {level:.1f}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()


def generate_samples(
    model_dir,
    num_samples=16,
    total_steps=1000,
    inference_steps=1000,
    device="cuda"
):
    model = UNet2DModel.from_pretrained(model_dir)
    model = model.to(device)

    # Initial Gaussian noise
    x_t = torch.randn(
        num_samples,
        model.config.in_channels,
        model.config.sample_size,
        model.config.sample_size,
        device=device
    )

    images, _, _ = DDIM_sampling(
        x_t=x_t,
        model=model,
        total_steps=total_steps,
        inference_steps=inference_steps,
        scheduler_method="linear",
        ddim_method="linear",
        device=torch.device(device),
        clip=True,
        verbose=True,
        enable_grad=False
    )
    images = (images.clamp(-1, 1) + 1) / 2

    grid = make_grid(images.cpu(), nrow=8, normalize=False)
    grid = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    plt.axis("off")
    plt.show()

    return images