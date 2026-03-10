from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

model = YOLO("yolov8n.pt")

video_path = "data/videos/traffic.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))

    results = model.track(frame, persist=True, classes=[2,3,5,7])

    boxes = []

    # collect boxes
    if results[0].boxes is not None:
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            boxes.append([x1, y1, x2, y2])

            # draw individual box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # ---- GROUP DETECTION ----
    if len(boxes) > 0:

        centers = []
        for x1,y1,x2,y2 in boxes:
            cx = (x1+x2)//2
            cy = (y1+y2)//2
            centers.append([cx,cy])

        centers = np.array(centers)

        clustering = DBSCAN(eps=80, min_samples=2).fit(centers)

        labels = clustering.labels_

        unique_groups = set(labels)

        for group in unique_groups:
            if group == -1:
                continue

            group_boxes = [boxes[i] for i in range(len(boxes)) if labels[i]==group]

            gx1 = min(b[0] for b in group_boxes)
            gy1 = min(b[1] for b in group_boxes)
            gx2 = max(b[2] for b in group_boxes)
            gy2 = max(b[3] for b in group_boxes)

            # draw group bounding box
            cv2.rectangle(frame,(gx1,gy1),(gx2,gy2),(255,0,0),3)

    cv2.imshow("Detection + Tracking", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()