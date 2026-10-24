# CCA 3 — Computer Vision Assignment (Units 3, 4 and 5)

This project is organized around the four tasks in the assignment brief.

## Current status
- [x] Project structure and Task 1 implementation
- [ ] Add your original photograph and second photograph
- [ ] Add your original 15–30 second video
- [ ] Run and verify Task 1 outputs
- [ ] Implement Task 2: HOG and SIFT matching
- [ ] Implement Task 3: Haar cascade and YOLO
- [ ] Implement Task 4: frame differencing, MOG2 and Lucas–Kanade optical flow
- [ ] Complete technical report, GitHub repository and explanation video

## Input files
Place your own, unedited files in `data/`:
- `data/original_photo.jpg`
- `data/second_photo.jpg`
- `data/original_video.mp4`

Use a clear photograph with visible texture and several objects. For watershed, an image with multiple or touching objects is especially helpful. The second photograph should show the same object or scene from a slightly different angle or distance. The video should be approximately 15–30 seconds long and contain a moving person or object.

## macOS setup (VS Code terminal)

1. Install Python 3 if needed. One option is the official installer from https://www.python.org/downloads/macos/ or Homebrew (`brew install python` if Homebrew is installed).
2. Open this project folder in VS Code.
3. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Put the original files in `data/`.
5. Run Task 1:

```bash
python src/task1_restoration_segmentation.py --image data/original_photo.jpg
```

If your photo has a different extension, pass its actual path to `--image`.

Outputs are written to `outputs/task1/`. The script saves figures and a CSV containing the actual PSNR and SSIM measurements.

## Expected Task 1 outputs
- `01_original.png`
- `02_gaussian_noise.png`
- `03_salt_pepper_noise.png`
- `04_gaussian_noise_restoration_comparison.png`
- `05_salt_pepper_restoration_comparison.png`
- `06_binary_threshold.png`
- `07_erosion.png`
- `08_dilation.png`
- `09_opening.png`
- `10_closing.png`
- `11_canny_edges.png`
- `12_watershed.png`
- `13_watershed_boundaries.png`
- `restoration_metrics.csv`

## Important notes
- PSNR and SSIM compare a restored image with the original clean image. Higher values generally indicate closer similarity, but visual quality and the kind of noise also matter.
- Median filtering often handles salt-and-pepper noise well. Gaussian filtering often reduces Gaussian noise, but may blur fine details. Your results may vary by image and filter settings.
- Watershed segmentation depends on the scene and thresholding. Inspect the output and discuss limitations honestly.
- Do not invent measurements or claim that an algorithm succeeded if the output does not support that claim.
- For the final submission, add the remaining tasks, selected outputs, `requirements.txt`, your technical report, the Git repository URL, and the explanation-video link as required by your teacher.
