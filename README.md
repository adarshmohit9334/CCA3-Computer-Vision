# CCA 3 — Computer Vision Assignment (Units 3, 4 and 5)

This repository contains Python implementations for image restoration/segmentation, HOG/SIFT feature extraction, Haar/YOLO object detection, and video motion estimation.

## Inputs (use your own originals)
- `data/original_photo.jpeg` — your original unedited photo.
- `data/second_photo.jpeg` — another photo of the same object/scene from a different view.
- `data/original_video.mp4` — your own unedited 15–30 second video with a moving person/object and a reasonably steady camera.

Do not submit example/synthetic media as if it were your original recording. If your filename differs, pass the actual path with the command-line option.

## macOS / VS Code setup
```bash
cd /Users/adarshkumar/Desktop/CCA3-Computer-Vision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the tasks
```bash
# Task 1 — restoration and segmentation
python src/task1_restoration_segmentation.py --image data/original_photo.jpeg

# Task 2 — HOG, SIFT, matching and rotation robustness
python src/task2_hog_sift_matching.py --image1 data/original_photo.jpeg --image2 data/second_photo.jpeg

# Task 3 — Haar face cascade and pretrained YOLO
python src/task3_object_detection.py --image data/original_photo.jpeg

# Task 4 — frame differencing, MOG2/GMM background subtraction and Lucas–Kanade flow
python src/task4_video_motion.py --video data/original_video.mp4 --max-frames 180 --step 2
```

First YOLO run can download `yolo11n.pt` and may take time. A working internet connection may be required for installation and model download. Haar cascade only detects faces, while YOLO detects classes supported by its pretrained model; they are not identical detectors.

## Output folders
- `outputs/task1/`: noisy/restored images, threshold/morphology/edge/watershed outputs, PSNR/SSIM CSV.
- `outputs/task2/`: HOG visualizations, SIFT keypoints/matches, rotated-image experiment, metrics CSV.
- `outputs/task3/`: annotated Haar and YOLO images, detection CSV files.
- `outputs/task4/`: selected original frames, frame-difference masks, MOG2 masks, Lucas–Kanade vector visualizations, metrics CSV.

## Reporting honestly
Use the generated outputs and CSVs for your comparison. Detection counts, match counts, masks and tracked points depend on the actual input. Do not invent results. If a detector misses an object, describe the actual observed miss; otherwise report the limitation you observed without claiming a miss occurred. Add captions for output figures and explain what they show.

## Submission checklist
- [ ] Original photo and second view saved in `data/`.
- [ ] Original self-recorded video saved in `data/` (keep raw file; avoid committing a large video to Git).
- [ ] All four tasks run; inspect outputs and CSVs.
- [ ] Report contains input details, code/output explanations, comparisons, measurements, conclusion, GitHub URL and video link.
- [ ] Git repository contains source, README, requirements and selected outputs.
- [ ] 5–8 minute spoken explanation video uploaded to the location specified by your instructor.
- [ ] Give the instructor access to the repository/Drive as requested.
