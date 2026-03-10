# file: detect_vehicles.py
"""
Simple YOLOv8 vehicle detector (bounding boxes + labels)
Requirements:
    pip install ultralytics opencv-python numpy

Usage:
    python detect_vehicles.py --video data/videos/traffic.mp4 --output output/annotated.mp4
    python detect_vehicles.py --video 0                # use webcam (0)
"""
import argparse
import time
from ultralytics import YOLO
import cv2
import numpy as np
import os

# COCO class ids for common vehicle classes
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 4: airplane, 5: bus, 6: train, 7: truck, ...
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def draw_box(frame, xyxy, label, conf, color=(0,255,0), thick=2):
    """Draw a single bbox with label and confidence."""
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, thick)
    txt = f"{label} {conf:.2f}"
    ((tw, th), _) = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - 18), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, txt, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

def main(args):
    # Initialize model
    model = YOLO(args.model)

    # Video input (path or camera index)
    try:
        src = int(args.video)
    except Exception:
        src = args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {args.video}")

    # Setup output writer if requested
    writer = None
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # use input frame size after reading first frame
        ret, tmp = cap.read()
        if not ret:
            raise RuntimeError("Couldn't read first frame for output sizing.")
        h, w = tmp.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind
        writer = cv2.VideoWriter(args.output, fourcc, args.fps or 20.0, (w, h))

    last_time = time.time()
    frame_count = 0
    fps = 0.0

    print("Starting detection. Press ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # optionally resize for speed (keep aspect)
        if args.width:
            h0, w0 = frame.shape[:2]
            new_w = args.width
            new_h = int(h0 * new_w / w0)
            frame = cv2.resize(frame, (new_w, new_h))

        # Run detection (single-frame inference)
        # results[0].boxes -> contains xyxy, conf, cls
        results = model(frame)  # ultralytics convenience call
        # robust extraction:
        try:
            boxes = results[0].boxes
            xyxys = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
            clss = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else np.array(boxes.cls).astype(int)
        except Exception:
            xyxys, confs, clss = [], [], []

        # Draw detections for vehicle classes only
        for xyxy, conf, clsid in zip(xyxys, confs, clss):
            clsid = int(clsid)
            if clsid in VEHICLE_CLASS_IDS:
                label = VEHICLE_CLASS_IDS[clsid]
                draw_box(frame, xyxy, label, float(conf), color=(0,200,0), thick=2)

        # FPS
        frame_count += 1
        if frame_count >= 5:
            now = time.time()
            fps = frame_count / (now - last_time)
            last_time = now
            frame_count = 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("YOLO Vehicle Detection", frame)
        if writer:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 vehicle detection (bounding boxes)")
    parser.add_argument("--video", type=str, required=True, help="Path to video file or camera index (0)")
    parser.add_argument("--output", type=str, default="", help="Optional output annotated video path (mp4)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics model weight (default: yolov8n.pt)")
    parser.add_argument("--width", type=int, default=960, help="Resize input frame width for speed (keep aspect). Set 0 to skip.")
    parser.add_argument("--fps", type=float, default=20.0, help="Output video FPS (if saving)")
    args = parser.parse_args()

    main(args)