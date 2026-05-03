import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# ---------------- SETTINGS ----------------
DISPLAY_WIDTH = 1100
DISPLAY_HEIGHT = 700
PANEL_WIDTH = 400

# ---------------- LOAD ----------------
model = YOLO("C:/Users/RAHUL KUMAR DUTTA/runs/detect/train3/weights/best.pt")
tracker = Sort(iou_threshold=0.3)

cap = cv2.VideoCapture("D:/data/dataset/traffic.mp4")

pts = np.array([
    [230, 692],
    [609, 694],
    [1000, 1180],
    [3, 1170]
], np.int32)

num_zones = 3
counted_ids = set()

# ---------------- VIDEO SAVE ----------------
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("FINAL_OUTPUT.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (w + PANEL_WIDTH, h))

# ---------------- STORAGE ----------------
vehicle_history = {}
vehicle_path = {}
vehicle_heat = {}
zone_history = []

# ---------------- FUNCTIONS ----------------
def get_color(count):
    if count < 3: return (0,255,0)
    elif count < 6: return (0,255,255)
    else: return (0,0,255)

def get_class(x1,y1,x2,y2,detections,class_ids):
    for i,det in enumerate(detections):
        dx1,dy1,dx2,dy2,_ = det
        if abs(x1-dx1)<30 and abs(y1-dy1)<30:
            return class_ids[i]
    return -1

def get_grid_lines(pts,n):
    lines=[]
    for i in range(1,n):
        a=i/n
        xl=int((1-a)*pts[0][0]+a*pts[3][0])
        yl=int((1-a)*pts[0][1]+a*pts[3][1])
        xr=int((1-a)*pts[1][0]+a*pts[2][0])
        yr=int((1-a)*pts[1][1]+a*pts[2][1])
        lines.append(((xl,yl),(xr,yr)))
    return lines

def get_zone(cy,lines):
    for i,((_,y1),_) in enumerate(lines):
        if cy<y1:
            return i
    return len(lines)

# 🔥 DRAW ZONE HEAT
def draw_zone_heat(frame, pts, n, counts):
    overlay = frame.copy()
    left,right = [],[]

    for i in range(n+1):
        a=i/n
        xl=int((1-a)*pts[0][0]+a*pts[3][0])
        yl=int((1-a)*pts[0][1]+a*pts[3][1])
        xr=int((1-a)*pts[1][0]+a*pts[2][0])
        yr=int((1-a)*pts[1][1]+a*pts[2][1])
        left.append((xl,yl))
        right.append((xr,yr))

    for i in range(n):
        poly=np.array([left[i],right[i],right[i+1],left[i+1]])
        cv2.fillPoly(overlay,[poly],get_color(counts[i]))

    return cv2.addWeighted(overlay,0.3,frame,0.7,0)

# 🔥 PANEL
def draw_panel(frame, text_lines):
    h = frame.shape[0]
    panel = np.zeros((h, PANEL_WIDTH, 3), dtype=np.uint8)

    y = 30
    for line, color in text_lines:
        cv2.putText(panel, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 20

    return np.hstack((panel, frame))


# ================= LOOP =================
while True:
    ret,frame=cap.read()
    if not ret:
        break

    results=model(frame)[0]

    detections=[]
    class_ids=[]

    for b in results.boxes:
        x1,y1,x2,y2=b.xyxy[0].cpu().numpy()
        conf=b.conf[0].cpu().numpy()
        cls=int(b.cls[0].cpu().numpy())

        if conf>0.4:
            detections.append([x1,y1,x2,y2,conf])
            class_ids.append(cls)

    detections=np.array(detections) if len(detections) else np.empty((0,5))
    tracks=tracker.update(detections)

    lines=get_grid_lines(pts,num_zones)
    zone_count=[0]*num_zones

    for tr in tracks:
        x1,y1,x2,y2,tid=map(int,tr)
        cx=(x1+x2)//2
        cy=(y1+y2)//2

        if cv2.pointPolygonTest(pts,(cx,cy),False)>=0:
            zone=get_zone(cy,lines)
            zone_count[zone]+=1

            # TRACK
            if tid not in vehicle_history:
                vehicle_history[tid]=[0]*num_zones
                vehicle_path[tid]=[]

            vehicle_history[tid][zone]+=1

            if len(vehicle_path[tid])==0 or vehicle_path[tid][-1]!=zone:
                vehicle_path[tid].append(zone)

            if zone==num_zones-1:
                counted_ids.add(tid)

        cls_id=get_class(x1,y1,x2,y2,detections,class_ids)
        label=model.names[cls_id] if cls_id!=-1 else "unknown"

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,f"{label} ID:{tid}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    # 🔥 HEAT CALC
    for vid,z in vehicle_history.items():
        vehicle_heat[vid]=z[0]*3 + z[1]*2 + z[2]*1

    top=sorted(vehicle_heat.items(),key=lambda x:x[1],reverse=True)

    zone_history.append(zone_count.copy())
    zone_totals=np.sum(zone_history,axis=0)
    most_congested=np.argmax(zone_totals)

    # 🔥 DRAW GRID + HEAT
    frame=draw_zone_heat(frame,pts,num_zones,zone_count)

    for i,(p1,p2) in enumerate(lines):
        cv2.line(frame,p1,p2,get_color(zone_count[i]),4)

    cv2.polylines(frame,[pts],True,(255,255,0),2)

    # 🔥 TEXT PANEL
    text_lines=[]
    text_lines.append(("Vehicle -> Region (Heat)", (255,255,255)))

    for vid in list(vehicle_path.keys())[:10]:
        path=" -> ".join([f"R{z+1}" for z in vehicle_path[vid]])
        heat=vehicle_heat.get(vid,0)
        z=vehicle_history[vid]

        text_lines.append((f"Car({vid}) -> {path} (Heat:{heat}) "
                           f"(R1:{z[0]}, R2:{z[1]}, R3:{z[2]})",(0,255,255)))

    text_lines.append(("",(255,255,255)))
    text_lines.append((f"Most Congested: R{most_congested+1}",(0,0,255)))

    text_lines.append(("",(255,255,255)))
    text_lines.append(("Traffic Contributors:",(255,255,255)))

    for vid,heat in top[:5]:
        text_lines.append((f"Car({vid}): Heat {heat}, Frames in R1:{vehicle_history[vid][0]}",
                           (255,255,0)))

    output=draw_panel(frame,text_lines)

    out.write(output)

    display=cv2.resize(output,(DISPLAY_WIDTH,DISPLAY_HEIGHT))
    cv2.imshow("Traffic Grid",display)

    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()