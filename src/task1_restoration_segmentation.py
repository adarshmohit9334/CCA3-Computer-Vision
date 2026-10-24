#!/usr/bin/env python3
"""
CCA 3 — Task 1: Image Restoration and Segmentation

Implements:
- Gaussian and salt-and-pepper noise
- Median and Gaussian filtering
- PSNR and SSIM comparison against the clean original
- Otsu thresholding
- Erosion, dilation, opening, closing
- Canny edge detection
- Marker-based watershed segmentation

Run:
    python src/task1_restoration_segmentation.py --image data/original_photo.jpg
Optional:
    python src/task1_restoration_segmentation.py --image data/original_photo.jpg --output outputs/task1 --seed 42
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def read_image(image_path: Path) -> np.ndarray:
    """Read an image as RGB, with a helpful error if the file cannot be opened."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}\n"
            "Check the file path and make sure the image is in data/."
        )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def add_gaussian_noise(image: np.ndarray, rng: np.random.Generator,
                       sigma: float = 20.0) -> np.ndarray:
    """Add Gaussian noise with a configurable standard deviation."""
    noisy = image.astype(np.float32) + rng.normal(0.0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image: np.ndarray, rng: np.random.Generator,
                          amount: float = 0.04) -> np.ndarray:
    """Randomly replace pixels with black or white; amount is the affected fraction."""
    noisy = image.copy()
    height, width = image.shape[:2]
    total = height * width
    count = int(total * amount)
    ys = rng.integers(0, height, count)
    xs = rng.integers(0, width, count)
    half = count // 2
    noisy[ys[:half], xs[:half]] = 0
    noisy[ys[half:], xs[half:]] = 255
    return noisy


def save_rgb(path: Path, image: np.ndarray) -> None:
    """Save an RGB image using OpenCV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def save_figure(path: Path, images: list[tuple[str, np.ndarray]],
                cmap: str | None = None, columns: int = 2) -> None:
    """Save a labeled comparison grid."""
    rows = int(np.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.5 * rows))
    axes = np.array(axes, dtype=object).reshape(-1)
    for ax, (title, image) in zip(axes, images):
        if image.ndim == 2:
            ax.imshow(image, cmap=cmap or "gray", vmin=0, vmax=255)
        else:
            ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def metrics(clean: np.ndarray, restored: np.ndarray) -> tuple[float, float]:
    """Return PSNR and SSIM for RGB images."""
    psnr = float(peak_signal_noise_ratio(clean, restored, data_range=255))
    ssim = float(structural_similarity(clean, restored, channel_axis=2, data_range=255))
    return psnr, ssim


def make_watershed(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a practical marker-based watershed pipeline.
    The Otsu mask is normalized so its smaller polarity is treated as foreground.
    Inspect this output: generic thresholding may not perfectly separate every scene.
    """
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Select the less-populated polarity as a basic foreground heuristic.
    if np.mean(binary > 0) > 0.5:
        binary = cv2.bitwise_not(binary)

    kernel = np.ones((3, 3), np.uint8)
    sure_background = cv2.dilate(binary, kernel, iterations=3)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    if distance.max() <= 0:
        # Graceful fallback if thresholding produced no foreground.
        markers = np.ones(gray.shape, dtype=np.int32)
        cv2.watershed(bgr.copy(), markers)
        boundary = np.zeros(gray.shape, dtype=np.uint8)
        boundary[markers == -1] = 255
        overlay = image_rgb.copy()
        overlay[markers == -1] = [255, 0, 0]
        return overlay, boundary

    _, sure_foreground = cv2.threshold(
        distance, 0.35 * distance.max(), 255, cv2.THRESH_BINARY
    )
    sure_foreground = sure_foreground.astype(np.uint8)
    unknown = cv2.subtract(sure_background, sure_foreground)

    component_count, markers = cv2.connectedComponents(sure_foreground)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed marks region borders with -1.
    markers = cv2.watershed(bgr.copy(), markers.astype(np.int32))
    boundary = np.zeros(gray.shape, dtype=np.uint8)
    boundary[markers == -1] = 255

    overlay = image_rgb.copy()
    overlay[markers == -1] = [255, 0, 0]
    return overlay, boundary


def run(image_path: Path, output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    original = read_image(image_path)
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)

    # A. Restoration
    gaussian_noisy = add_gaussian_noise(original, rng)
    sp_noisy = add_salt_pepper_noise(original, rng)

    gaussian_filtered = cv2.GaussianBlur(gaussian_noisy, (5, 5), 1.2)
    gaussian_median = cv2.medianBlur(gaussian_noisy, 5)
    sp_median = cv2.medianBlur(sp_noisy, 5)
    sp_gaussian = cv2.GaussianBlur(sp_noisy, (5, 5), 1.2)

    save_rgb(output_dir / "01_original.png", original)
    save_rgb(output_dir / "02_gaussian_noise.png", gaussian_noisy)
    save_rgb(output_dir / "03_salt_pepper_noise.png", sp_noisy)

    save_figure(
        output_dir / "04_gaussian_noise_restoration_comparison.png",
        [("Original clean image", original),
         ("Gaussian noise", gaussian_noisy),
         ("Gaussian filter", gaussian_filtered),
         ("Median filter", gaussian_median)],
        columns=2,
    )
    save_figure(
        output_dir / "05_salt_pepper_restoration_comparison.png",
        [("Original clean image", original),
         ("Salt-and-pepper noise", sp_noisy),
         ("Median filter", sp_median),
         ("Gaussian filter", sp_gaussian)],
        columns=2,
    )

    metric_rows = []
    for label, noisy, restored in [
        ("Gaussian noise + Gaussian filter", gaussian_noisy, gaussian_filtered),
        ("Gaussian noise + median filter", gaussian_noisy, gaussian_median),
        ("Salt-and-pepper + median filter", sp_noisy, sp_median),
        ("Salt-and-pepper + Gaussian filter", sp_noisy, sp_gaussian),
    ]:
        noisy_psnr, noisy_ssim = metrics(original, noisy)
        restored_psnr, restored_ssim = metrics(original, restored)
        metric_rows.append({
            "experiment": label,
            "noisy_psnr_db": round(noisy_psnr, 4),
            "noisy_ssim": round(noisy_ssim, 6),
            "restored_psnr_db": round(restored_psnr, 4),
            "restored_ssim": round(restored_ssim, 6),
        })

    with (output_dir / "restoration_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    # B. Thresholding and morphology
    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(blurred_gray, 100, 200)

    save_gray(output_dir / "06_binary_threshold.png", binary)
    save_gray(output_dir / "07_erosion.png", erosion)
    save_gray(output_dir / "08_dilation.png", dilation)
    save_gray(output_dir / "09_opening.png", opening)
    save_gray(output_dir / "10_closing.png", closing)
    save_gray(output_dir / "11_canny_edges.png", edges)

    save_figure(
        output_dir / "morphology_comparison.png",
        [("Otsu binary threshold", binary), ("Erosion", erosion),
         ("Dilation", dilation), ("Opening", opening), ("Closing", closing),
         ("Canny edges", edges)],
        cmap="gray", columns=3,
    )

    # Watershed segmentation
    watershed_overlay, watershed_boundary = make_watershed(original)
    save_rgb(output_dir / "12_watershed.png", watershed_overlay)
    save_gray(output_dir / "13_watershed_boundaries.png", watershed_boundary)
    save_figure(
        output_dir / "watershed_comparison.png",
        [("Original image", original), ("Otsu binary mask", binary),
         ("Watershed boundaries (red)", watershed_overlay),
         ("Watershed boundary mask", watershed_boundary)],
        columns=2,
    )

    print("\nTask 1 completed.")
    print(f"Input image: {image_path}")
    print(f"Image dimensions: {original.shape[1]} x {original.shape[0]}")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Metrics CSV: {(output_dir / 'restoration_metrics.csv').resolve()}")
    print("\nRestoration metrics (actual values from this input):")
    for row in metric_rows:
        print(
            f"- {row['experiment']}: "
            f"PSNR {row['restored_psnr_db']} dB, "
            f"SSIM {row['restored_ssim']}"
        )
    print("\nReview all saved figures before writing conclusions to your report.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CCA 3 Task 1: Image Restoration and Segmentation"
    )
    parser.add_argument("--image", required=True, help="Path to your original photograph")
    parser.add_argument("--output", default="outputs/task1", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable noise")
    args = parser.parse_args()

    run(Path(args.image), Path(args.output), args.seed)


if __name__ == "__main__":
    main()
