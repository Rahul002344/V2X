import cv2
import numpy as np
points = []

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point added: {x}, {y}")

video = "west.mp4"   # change for each video
cap = cv2.VideoCapture(video)

ret, frame = cap.read()

cv2.imshow("Select ROI - Click 4 points", frame)
cv2.setMouseCallback("Select ROI - Click 4 points", click_event)

while True:
    temp = frame.copy()
    for p in points:
        cv2.circle(temp, p, 5, (0,255,0), -1)

    if len(points) == 4:
        cv2.polylines(temp, [np.array(points)], True, (255,0,0), 2)

    cv2.imshow("Select ROI - Click 4 points", temp)

    if cv2.waitKey(1) & 0xFF == 13: # ENTER
        break

cap.release()
cv2.destroyAllWindows()

print("ROI Points:", points)