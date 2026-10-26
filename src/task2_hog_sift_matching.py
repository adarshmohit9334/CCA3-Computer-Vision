#!/usr/bin/env python3
"""CCA 3 Task 2: HOG, SIFT keypoints, feature matching and rotation test."""
import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog


def load_image(path, max_side=1200):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(
            f"Cannot read {path}. Check the exact filename and extension (.jpg/.jpeg)."
        )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return image


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def save_image(path, image, title):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 7))
    plt.imshow(image, cmap="gray" if image.ndim == 2 else None)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def hog_features(image):
    return hog(gray(image), orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys",
               visualize=False, feature_vector=True)


def save_hog(image, path, title):
    _, visualization = hog(
        gray(image), orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), block_norm="L2-Hys",
        visualize=True, feature_vector=True
    )
    visualization = visualization.astype(np.float32)
    visualization -= visualization.min()
    if visualization.max() > 0:
        visualization /= visualization.max()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input photo")
    axes[1].imshow(visualization, cmap="gray")
    axes[1].set_title("HOG gradient visualization")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sift_features(image):
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError(
            "SIFT is unavailable in this OpenCV build. Run: "
            "python -m pip install --upgrade opencv-python"
        )
    return cv2.SIFT_create(nfeatures=1500).detectAndCompute(gray(image), None)


def save_keypoints(image, keypoints, path, title):
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    marked = cv2.drawKeypoints(
        bgr, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    save_image(path, cv2.cvtColor(marked, cv2.COLOR_BGR2RGB),
               f"{title} ({len(keypoints)} keypoints)")


def match_and_save(image1, image2, path, title):
    kp1, desc1 = sift_features(image1)
    kp2, desc2 = sift_features(image2)
    good = []
    if desc1 is not None and desc2 is not None and len(desc1) and len(desc2):
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        for pair in matcher.knnMatch(desc1, desc2, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good.append(pair[0])
    drawn = cv2.drawMatches(
        cv2.cvtColor(image1, cv2.COLOR_RGB2BGR), kp1,
        cv2.cvtColor(image2, cv2.COLOR_RGB2BGR), kp2,
        good[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    save_image(path, cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB), title)
    return len(kp1), len(kp2), len(good)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image1", default="data/original_photo.jpeg")
    parser.add_argument("--image2", default="data/second_photo.jpeg")
    parser.add_argument("--output", default="outputs/task2")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    image1, image2 = load_image(Path(args.image1)), load_image(Path(args.image2))

    save_image(out / "01_photo1.png", image1, "Photo 1: original input")
    save_image(out / "02_photo2.png", image2, "Photo 2: second view")
    save_hog(image1, out / "03_hog_photo1.png", "HOG features — Photo 1")
    save_hog(image2, out / "04_hog_photo2.png", "HOG features — Photo 2")

    kp1, _ = sift_features(image1)
    kp2, _ = sift_features(image2)
    save_keypoints(image1, kp1, out / "05_sift_keypoints_photo1.png", "SIFT keypoints — Photo 1")
    save_keypoints(image2, kp2, out / "06_sift_keypoints_photo2.png", "SIFT keypoints — Photo 2")

    n1, n2, matches_original = match_and_save(
        image1, image2, out / "07_sift_matches_original_views.png",
        "SIFT matches: original photo views"
    )

    # Rotation robustness test: rotate Photo 2 by 15 degrees and match again.
    h, w = image2.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)
    rotated_bgr = cv2.warpAffine(
        cv2.cvtColor(image2, cv2.COLOR_RGB2BGR), matrix, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    rotated = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2RGB)
    save_image(out / "08_photo2_rotated_15deg.png", rotated,
               "Photo 2 rotated by 15 degrees")
    _, rotated_kp, matches_rotated = match_and_save(
        image1, rotated, out / "09_sift_matches_after_rotation.png",
        "SIFT matches after rotating Photo 2 by 15 degrees"
    )

    metrics = out / "task2_metrics.csv"
    with metrics.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["comparison", "photo1_sift_keypoints", "photo2_sift_keypoints",
                         "photo1_hog_feature_count", "photo2_hog_feature_count",
                         "good_sift_matches"])
        writer.writerow(["Original views", n1, n2, len(hog_features(image1)),
                         len(hog_features(image2)), matches_original])
        writer.writerow(["Photo 1 vs rotated Photo 2 (+15 degrees)", n1, rotated_kp,
                         len(hog_features(image1)), len(hog_features(rotated)),
                         matches_rotated])

    print(f"Task 2 complete. Outputs: {out}")
    print(f"SIFT keypoints: photo 1={n1}, photo 2={n2}")
    print(f"Good matches: original={matches_original}; rotated={matches_rotated}")
    print(f"Metrics CSV: {metrics}")


if __name__ == "__main__":
    main()
