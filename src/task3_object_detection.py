#!/usr/bin/env python3
"""CCA 3 Task 3: Haar cascade (Viola-Jones) and pretrained Ultralytics YOLO."""
import argparse, csv
from pathlib import Path
import cv2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', default='data/original_photo.jpeg')
    p.add_argument('--output', default='outputs/task3')
    p.add_argument('--model', default='yolo11n.pt', help='Ultralytics model name/path; first run may download it')
    p.add_argument('--confidence', type=float, default=0.25)
    args = p.parse_args()
    image_path, out = Path(args.image), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f'Cannot read {image_path}; check filename/extension.')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError('Could not load OpenCV Haar face cascade.')
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    haar = image.copy()
    for i, (x, y, w, h) in enumerate(faces, 1):
        cv2.rectangle(haar, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(haar, f'Haar face {i}', (x, max(20,y-8)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,255,0), 2)
    cv2.imwrite(str(out/'01_haar_face_detection.png'), haar)
    with (out/'haar_detections.csv').open('w', newline='', encoding='utf-8') as f:
        wr=csv.writer(f); wr.writerow(['detector','label','x','y','width','height','confidence'])
        for x,y,w,h in faces: wr.writerow(['Haar cascade','face',int(x),int(y),int(w),int(h),'not provided by Haar cascade'])
    print(f'Haar cascade: {len(faces)} face(s). Saved {out/"01_haar_face_detection.png"}')

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError('Ultralytics is not installed. Run: python -m pip install ultralytics') from e
    model = YOLO(args.model)
    results = model.predict(source=str(image_path), conf=args.confidence, verbose=False)
    result = results[0]
    annotated = result.plot()
    cv2.imwrite(str(out/'02_yolo_detection.png'), annotated)
    with (out/'yolo_detections.csv').open('w', newline='', encoding='utf-8') as f:
        wr=csv.writer(f); wr.writerow(['class_id','class_name','confidence','x1','y1','x2','y2'])
        if result.boxes is not None:
            for box in result.boxes:
                cls=int(box.cls[0].item()); conf=float(box.conf[0].item())
                x1,y1,x2,y2=[float(v) for v in box.xyxy[0].tolist()]
                wr.writerow([cls, model.names.get(cls, str(cls)), round(conf,5), round(x1,2), round(y1,2), round(x2,2), round(y2,2)])
    n = 0 if result.boxes is None else len(result.boxes)
    print(f'YOLO: {n} object(s). Saved {out/"02_yolo_detection.png"}')
    print('Compare Haar face-only detections with YOLO general object detections. A zero-detection result is valid; discuss limitations rather than inventing detections.')
    print(f'CSV files saved in {out}')

if __name__ == '__main__': main()
