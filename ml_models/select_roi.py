import cv2

VIDEO_PATH = "D:/data/dataset/traffic.mp4"

points = []

def click_event(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")

        # Draw point
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select ROI", frame)

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()

frame = cv2.resize(frame, (600, 700))
cv2.imshow("Select ROI", frame)
cv2.setMouseCallback("Select ROI", click_event)

print("👉 Click 4 points: Top-Left → Top-Right → Bottom-Right → Bottom-Left")

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nFinal Points:")
print(points)