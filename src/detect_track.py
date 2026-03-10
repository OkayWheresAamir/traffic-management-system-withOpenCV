# homography_viz.py
# Enhanced homography visualization:
# - left: camera view with detections & trapezoid
# - right: bird's-eye warped image with transformed bbox polygons, grid & queue shading
#
# Requirements: ultralytics, opencv-python, numpy

from ultralytics import YOLO
import cv2
import numpy as np
import math

# --- CONFIG (tweak these) ---
VIDEO = "data/videos/traffic.mp4"
MODEL_NAME = "yolov8n.pt"
DETECT_CLASSES = [2,3,5,7]   # car, motorcycle, bus, truck

# Source points (image coordinates) - pick 4 points that form a trapezoid over the road plane
# Order: top-left, top-right, bottom-left, bottom-right (in camera image coords)
SRC_POINTS = np.array([
    [140, 150],   # top-left on road (near horizon)
    [500, 150],   # top-right on road
    [40, 350],    # bottom-left (nearer to camera)
    [600, 350]    # bottom-right
], dtype=np.float32)

# Destination bird's-eye image size (pixels) and destination points
DST_WIDTH = 800
DST_HEIGHT = 500
# Here we map to a nice rectangular top-down canvas (choose dst pts spanning the width)
DST_POINTS = np.array([
    [100, 20],                 # top-left
    [DST_WIDTH-100, 20],       # top-right
    [100, DST_HEIGHT-20],      # bottom-left
    [DST_WIDTH-100, DST_HEIGHT-20]
], dtype=np.float32)

# Real world width in meters corresponding to the horizontal span between dst left/right
# Use an estimate (e.g., width of road between points). This enables drawing meter ticks.
REAL_WORLD_WIDTH_M = 12.0    # adjust to approx real road width (meters)
# meters per pixel in bird view (used for grid): meters_per_pixel = REAL_WORLD_WIDTH_M / (dst_x_span)
meters_per_pixel = REAL_WORLD_WIDTH_M / (DST_POINTS[1][0] - DST_POINTS[0][0])

# Visual tuning
BIRD_BG = (30,30,30)
BBOX_COLOR = (0,255,0)
BBOX_THICK = 2
BIRD_BBOX_COLOR = (50,200,50)
QUEUE_ZONE_DEPTH_M = 25.0   # how deep (meters) the queue zone is from stop line in bird world

# --- Setup ---
model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(VIDEO)
H, status = cv2.findHomography(SRC_POINTS, DST_POINTS)

if H is None:
    raise RuntimeError("Homography computation failed. Check SRC_POINTS and DST_POINTS.")

print("Homography ready. Press ESC to quit. Tweak SRC_POINTS / DST_POINTS in the script as needed.")

def warp_to_bird(img):
    bird = cv2.warpPerspective(img, H, (DST_WIDTH, DST_HEIGHT), flags=cv2.INTER_LINEAR)
    return bird

def transform_points(pts):
    """
    pts: Nx2 float32 points in image coords
    returns Nx2 float32 points in bird view coords
    """
    pts = np.array(pts, dtype=np.float32).reshape(-1,1,2)
    mapped = cv2.perspectiveTransform(pts, H)
    mapped = mapped.reshape(-1,2)
    return mapped

def draw_grid(bird, meters_per_pixel, spacing_m=5.0, color=(80,80,80)):
    """Draw horizontal lines every spacing_m meters (in bird-pixels)."""
    spacing_px = int(round(spacing_m / meters_per_pixel))
    if spacing_px <= 0:
        return bird
    h, w = bird.shape[:2]
    # draw horizontal lines (top to bottom)
    for y in range(0, h, spacing_px):
        cv2.line(bird, (0,y), (w,y), color, 1, lineType=cv2.LINE_AA)
    # draw vertical scale ticks near left edge
    num_ticks = int(w / (50))
    # also draw a scale bar in bottom-left
    bar_len_m = 10  # meters
    bar_len_px = int(round(bar_len_m / meters_per_pixel))
    cv2.rectangle(bird, (20, h-40), (20+bar_len_px, h-30), (255,255,255), -1)
    cv2.putText(bird, f"{bar_len_m} m", (25+bar_len_px, h-35), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1)
    return bird

def draw_transformed_bbox_on_bird(bird, bbox_xyxy, color=(0,200,0)):
    # bbox_xyxy: (x1,y1,x2,y2) in image coords
    x1,y1,x2,y2 = bbox_xyxy
    corners = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
    mapped = transform_points(corners)  # 4x2
    # convert to ints
    pts = mapped.astype(int)
    cv2.polylines(bird, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    # fill a translucent polygon for better visibility
    overlay = bird.copy()
    cv2.fillPoly(overlay, [pts], (color[0], color[1], color[2]))
    cv2.addWeighted(overlay, 0.12, bird, 0.88, 0, bird)

def is_point_in_polygon(pt, poly):
    # pt: (x,y), poly: Nx2 int list
    return cv2.pointPolygonTest(np.array(poly, dtype=np.int32), (int(pt[0]),int(pt[1])), False) >= 0

# compute a bird-view queue zone polygon: we take the bottom horizontal line in DST as stop line,
# then create a rectangle of depth QUEUE_ZONE_DEPTH_M meters upward.
stop_line_y = DST_POINTS[2][1]  # use bottom-left y
queue_depth_px = int(round(QUEUE_ZONE_DEPTH_M / meters_per_pixel))
queue_zone_poly = np.array([
    [DST_POINTS[0][0], stop_line_y - queue_depth_px],
    [DST_POINTS[1][0], stop_line_y - queue_depth_px],
    [DST_POINTS[1][0], stop_line_y],
    [DST_POINTS[0][0], stop_line_y]
], dtype=np.int32)

# main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]
    frame = cv2.resize(frame, (640, 360))  # standardize
    display_cam = frame.copy()

    # draw trapezoid (SRC points)
    pts_int = SRC_POINTS.astype(int)
    cv2.polylines(display_cam, [pts_int], True, (0,215,255), 2, lineType=cv2.LINE_AA)
    # Transform trapezoid to bird view
    SRC_POLY = SRC_POINTS.reshape(-1,1,2)
    DST_POLY = cv2.perspectiveTransform(SRC_POLY, H).astype(int)
    # run detections
    results = model.track(frame, persist=True, classes=DETECT_CLASSES, conf=0.2)

    # prepare bird's-eye base (warp of the camera helps make it realistic)
    bird = warp_to_bird(frame)
    # draw transformed trapezoid on bird view
    cv2.polylines(bird, [DST_POLY], True, (0,215,255), 2, lineType=cv2.LINE_AA)
    # darken bird bg to allow overlays
    bird = cv2.cvtColor(bird, cv2.COLOR_BGR2RGB)
    bird = cv2.cvtColor(bird, cv2.COLOR_RGB2BGR)
    # draw nice grid & scale
    bird = draw_grid(bird, meters_per_pixel, spacing_m=5.0, color=(60,60,60))



    # detection overlays
    detected_centers = []
    if results is not None and results[0].boxes is not None:
        for i, bb in enumerate(results[0].boxes.xyxy):
            x1,y1,x2,y2 = map(int, bb.tolist())
            # draw camera bbox
            cv2.rectangle(display_cam, (x1,y1), (x2,y2), BBOX_COLOR, BBOX_THICK)
            # bottom center
            cx = int((x1+x2)/2); cy = int(y2)
            cv2.circle(display_cam, (cx,cy), 4, (0,0,255), -1)

            # map bbox into bird and draw polygon
            try:
                draw_transformed_bbox_on_bird(bird, (x1,y1,x2,y2), color=BIRD_BBOX_COLOR)
                # also map bottom-center for counting
                mapped_pt = transform_points(np.array([[cx,cy]], dtype=np.float32))[0]
                mx,my = int(mapped_pt[0]), int(mapped_pt[1])
                detected_centers.append((mx,my))
                # draw small dot on bird
                cv2.circle(bird, (mx,my), 4, (0,0,255), -1)
            except Exception as e:
                # skip any transform errors gracefully
                pass

    # compute queue count as number of detected centers inside queue_zone_poly
    queue_count = 0
    for (mx,my) in detected_centers:
        if cv2.pointPolygonTest(queue_zone_poly, (mx,my), False) >= 0:
            queue_count += 1

    # annotate bird view with stats

    # combine views side-by-side
    # resize bird to same height as camera
    bird_resized = cv2.resize(bird, (640, 360))
    combined = np.hstack((display_cam, bird_resized))

    # overlay a separator and labels
    cv2.line(combined, (640,0), (640,360), (255,255,255), 2)
    cv2.putText(combined, "Camera View", (20,20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 2)
    cv2.putText(combined, "Bird's-eye View", (660,25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Homography Visualization (Left: Camera | Right: Bird)", combined)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()