import cv2

input_video = r"D:\Dataset\20250418_135807.mp4"
output_video = "output_2fps.mp4"

cap = cv2.VideoCapture(input_video)

# Get original FPS
original_fps = cap.get(cv2.CAP_PROP_FPS)
print("Original FPS:", original_fps)

# Set new FPS
new_fps = 2
frame_interval = int(original_fps / new_fps)

# Get frame size
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, new_fps, (width, height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Save only selected frames
    if frame_count % frame_interval == 0:
        out.write(frame)

    frame_count += 1

cap.release()
out.release()
print("Done: Saved as output_2fps.mp4")