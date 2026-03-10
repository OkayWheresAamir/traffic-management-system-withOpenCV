# src/state_estimator.py
# Robust state estimator: rolling-window arrival/departure rates + unique queue length
# Requirements: ultralytics, opencv-python, numpy
import time, json, os
from ultralytics import YOLO
import cv2

# -------- CONFIG --------
VIDEO = "data/videos/traffic.mp4"
OUTPUT_LOG = "output/state_log.jsonl"
WIDTH, HEIGHT = 800, 450         # resize for inference (tweak if needed)
LINE_A = 150                     # arrival detector (y < LINE_A => FAR region)
LINE_B = 300                     # departure detector (y >= LINE_B => PASSED)
FAR_CONF_THRESH = 0.05          # allow lower confidence only for far detections
NEAR_CONF_THRESH = 0.40          # normal confidence for nearer detections
DETECT_CLASSES = [2, 3, 5, 7]    # car, motorcycle, bus, truck (COCO ids)
MODEL_NAME = "yolov8n.pt"        # small model for CPU runs
WINDOW = 10.0                    # seconds for rolling window (arrival/departure rates)
PRUNE_AGE = WINDOW * 3           # remove track ids not seen for this many seconds

# -------- SETUP --------
os.makedirs("output", exist_ok=True)
model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(VIDEO)

# state trackers
vehicle_zone = {}      # track_id -> zone (0=FAR,1=QUEUE,2=PASSED)
last_seen = {}         # track_id -> last timestamp seen (for pruning)
arrival_times = []     # timestamps when vehicle entered QUEUE (FAR->QUEUE)
departure_times = []   # timestamps when vehicle entered PASSED (QUEUE->PASSED)

last_tick = time.time()

# safety: fallback fps
try:
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
except Exception:
    fps = 10

print("Starting state estimator. Press ESC to quit. Output:", OUTPUT_LOG)

# -------- Main loop --------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # Run tracker/detector with a relatively low global conf; we'll apply per-detection filters
    results = None
    try:
        results = model.track(frame, persist=True, classes=DETECT_CLASSES, conf=0.12)
    except Exception as e:
        # If tracking fails for a frame, skip processing but keep loop running
        print("Tracker error (skipping frame):", e)
        results = None

    current_queue_ids = set()

    boxes = None
    if results is not None:
        try:
            boxes = results[0].boxes
        except Exception:
            boxes = None

    # Robust extraction of ids/coords/conf (works if tensors or lists)
    ids = None; coords = None; confs = None
    if boxes is not None:
        # ids
        try:
            ids = boxes.id.tolist()
        except Exception:
            try:
                ids = [int(x) for x in boxes.id]
            except Exception:
                ids = None
        # coords
        try:
            coords = boxes.xyxy.cpu().numpy().tolist()
        except Exception:
            try:
                coords = boxes.xyxy.tolist()
            except Exception:
                coords = None
        # confs
        try:
            confs = boxes.conf.cpu().numpy().tolist()
        except Exception:
            try:
                confs = boxes.conf.tolist()
            except Exception:
                confs = None

    # Process detections
    if coords is not None:
        for i, bb in enumerate(coords):
            # safe read
            try:
                x1, y1, x2, y2 = [int(v) for v in bb]
            except Exception:
                continue

            conf = float(confs[i]) if confs is not None and i < len(confs) else 1.0

            # ensure id exists
            if ids is None or i >= len(ids):
                continue
            try:
                track_id = int(ids[i])
            except Exception:
                continue

            # bottom-center
            x = int((x1 + x2) / 2)
            y = int(y2)

            # per-zone confidence: allow lower for FAR only
            if y < LINE_A:
                if conf < FAR_CONF_THRESH:
                    continue
            else:
                if conf < NEAR_CONF_THRESH:
                    continue

            # determine current zone (0=FAR,1=QUEUE,2=PASSED)
            if y < LINE_A:
                zone = 0
            elif y < LINE_B:
                zone = 1
            else:
                zone = 2

            prev_zone = vehicle_zone.get(track_id)

            # count events only on transitions FAR->QUEUE and QUEUE->PASSED
            if prev_zone is not None:
                # FAR -> QUEUE = arrival
                if prev_zone == 0 and zone == 1:
                    arrival_times.append(now)
                # QUEUE -> PASSED = departure
                if prev_zone == 1 and zone == 2:
                    departure_times.append(now)
            # if no prev_zone (fresh detection), do NOT count as arrival/departure;
            # this avoids double-counting at video start or on trackers that reassign ids

            # update state
            vehicle_zone[track_id] = zone
            last_seen[track_id] = now

            # if currently in queue, add to current set
            if zone == 1:
                current_queue_ids.add(track_id)

            # draw detection point for demo
            cv2.circle(frame, (x, y), 4, (0, 200, 200), -1)

    # compute unique queue length
    in_queue = current_queue_ids
    queue_length = len(in_queue)

    # prune old timestamps outside rolling window
    cutoff = now - WINDOW
    if arrival_times:
        arrival_times = [t for t in arrival_times if t >= cutoff]
    if departure_times:
        departure_times = [t for t in departure_times if t >= cutoff]

    # compute rates as veh/s over WINDOW
    arrival_rate = len(arrival_times) / WINDOW
    departure_rate = len(departure_times) / WINDOW

    # logging once per second
    if now - last_tick >= 1.0:
        state = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now)),
            "queue_length": queue_length,
            "arrival_rate": round(arrival_rate, 3),
            "departure_rate": round(departure_rate, 3),
            "window_s": WINDOW
        }
        with open(OUTPUT_LOG, "a") as f:
            f.write(json.dumps(state) + "\n")
        print(state)
        last_tick = now

    # draw overlay
    cv2.line(frame, (0, LINE_A), (WIDTH, LINE_A), (255, 0, 0), 2)   # arrival line (blue)
    cv2.line(frame, (0, LINE_B), (WIDTH, LINE_B), (0, 0, 255), 2)   # departure line (red)
    cv2.putText(frame, f"Queue: {queue_length}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Arr/s (w={int(WINDOW)}s): {arrival_rate:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
    cv2.putText(frame, f"Dep/s (w={int(WINDOW)}s): {departure_rate:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 200), 2)

    cv2.imshow("Traffic State", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

    # prune stale tracks from vehicle_zone/last_seen to avoid unbounded growth
    to_remove = [tid for tid, t in last_seen.items() if now - t > PRUNE_AGE]
    for tid in to_remove:
        vehicle_zone.pop(tid, None)
        last_seen.pop(tid, None)

cap.release()
cv2.destroyAllWindows()
print("Finished. State log:", OUTPUT_LOG)