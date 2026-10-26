#!/usr/bin/env python3
"""CCA 3 Task 4: frame differencing, MOG2/GMM background subtraction, Lucas-Kanade flow."""
import argparse, csv
from pathlib import Path
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', default='data/original_video.mp4')
    p.add_argument('--output', default='outputs/task4')
    p.add_argument('--max-frames', type=int, default=180, help='Process at most this many frames for speed')
    p.add_argument('--step', type=int, default=2, help='Compare every Nth frame; default 2 for speed')
    args=p.parse_args()
    video=Path(args.video); out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise FileNotFoundError(f'Cannot open {video}. Check the path and video codec.')
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scale=min(1.0, 900/max(width,height)) if max(width,height)>0 else 1.0
    mog=cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
    lk_params=dict(winSize=(21,21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    prev_gray=None; prev_points=None; prev_frame=None; rows=[]; processed=0; sample_index=0
    writer=None
    try:
        while processed < args.max_frames:
            ok, frame=cap.read()
            if not ok: break
            frame=cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))
            if processed % max(1,args.step) != 0:
                processed += 1; continue
            gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # GMM/MOG2 mask, suppress shadow values (127) to retain definite foreground (255).
            fg=mog.apply(frame)
            _, fg=cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
            if prev_gray is None:
                diff=np.zeros_like(gray)
                flow_vis=frame.copy()
                points=cv2.goodFeaturesToTrack(gray, maxCorners=250, qualityLevel=0.01, minDistance=7, blockSize=7)
            else:
                diff=cv2.absdiff(gray, prev_gray)
                _, diff=cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                diff=cv2.dilate(diff, None, iterations=2)
                if prev_points is None or len(prev_points)==0:
                    prev_points=cv2.goodFeaturesToTrack(prev_gray, maxCorners=250, qualityLevel=0.01, minDistance=7, blockSize=7)
                flow_vis=frame.copy()
                if prev_points is not None and len(prev_points)>0:
                    next_points, status, errors=cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)
                    if next_points is not None and status is not None:
                        good_new=next_points[status.flatten()==1]
                        good_old=prev_points[status.flatten()==1]
                        for new, old in zip(good_new, good_old):
                            x,y=new.ravel(); ox,oy=old.ravel()
                            cv2.arrowedLine(flow_vis, (int(ox),int(oy)), (int(x),int(y)), (0,255,0), 1, tipLength=.3)
                            cv2.circle(flow_vis, (int(x),int(y)), 2, (0,0,255), -1)
                        points=good_new.reshape(-1,1,2) if len(good_new) else None
                    else: points=None
                else: points=None
            # Save panels for first frame and then periodic samples.
            if sample_index in (0, 1, 2, 4, 7, 10, 15, 20) or sample_index % 30 == 0:
                cv2.imwrite(str(out/f'frame_{sample_index:04d}_original.png'), frame)
                cv2.imwrite(str(out/f'frame_{sample_index:04d}_frame_difference.png'), diff)
                cv2.imwrite(str(out/f'frame_{sample_index:04d}_mog2_mask.png'), fg)
                cv2.imwrite(str(out/f'frame_{sample_index:04d}_lucas_kanade.png'), flow_vis)
            rows.append([sample_index, round(sample_index*max(1,args.step)/fps,3), int(np.count_nonzero(diff)), int(np.count_nonzero(fg)), 0 if points is None else len(points)])
            prev_gray=gray.copy(); prev_points=points; sample_index+=1; processed+=1
    finally:
        cap.release()
    if not rows: raise RuntimeError('No frames were processed. Check that the video is a valid supported video file.')
    with (out/'task4_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.writer(f); wr.writerow(['sample_index','approx_time_seconds','frame_difference_foreground_pixels','mog2_foreground_pixels','tracked_lk_points']); wr.writerows(rows)
    # A single easy-to-review comparison panel using the last saved samples.
    print(f'Video metadata: {width}x{height}, {fps:.2f} FPS, reported frames={total}')
    print(f'Processed {len(rows)} sampled frames. Saved visual outputs and metrics to {out}')
    print('Interpretation: frame differencing highlights changes between two sampled frames; MOG2 estimates foreground against a learned background; Lucas–Kanade tracks sparse feature-point motion.')

if __name__=='__main__': main()
