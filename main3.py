import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sort import Sort

# ---------------- LOAD MODEL ----------------
model = YOLO("C:/Users/RAHUL KUMAR DUTTA/runs/detect/train3/weights/best.pt")

# ---------------- VIDEO INPUT ----------------
cap_N = cv2.VideoCapture("north.mp4")
cap_S = cv2.VideoCapture("south.mp4")
cap_E = cv2.VideoCapture("east.mp4")
cap_W = cv2.VideoCapture("west.mp4")

fps = cap_N.get(cv2.CAP_PROP_FPS)

# ---------------- ROI (FIXED ORDER) ----------------
ROI_N = np.array([(125,815),(731,813),(483,584),(259,589)], np.int32)
ROI_S = np.array([(148,802),(770,810),(773,488),(537,498)], np.int32)
ROI_E = np.array([(6,748),(847,740),(501,416),(198,423)], np.int32)
ROI_W = np.array([(442,566),(793,566),(577,341),(475,347)], np.int32)

# ---------------- TRACKERS ----------------
tracker_N, tracker_S, tracker_E, tracker_W = Sort(), Sort(), Sort(), Sort()

# ---------------- DATA STORAGE ----------------
log_data = []

# ---------------- GRID ----------------
def get_grid_lines(pts):
    lines=[]
    for i in range(1,3):
        a=i/3
        xl=int((1-a)*pts[0][0]+a*pts[3][0])
        yl=int((1-a)*pts[0][1]+a*pts[3][1])
        xr=int((1-a)*pts[1][0]+a*pts[2][0])
        yr=int((1-a)*pts[1][1]+a*pts[2][1])
        lines.append(((xl,yl),(xr,yr)))
    return lines

# ---------------- ZONE DETECTION ----------------
def get_zone(cx, cy, lines):
    for i, ((x1,y1),(x2,y2)) in enumerate(lines):
        if (x2-x1) != 0:
            line_y = y1 + (cx-x1)*(y2-y1)/(x2-x1)
        else:
            line_y = y1
        if cy < line_y:
            return i
    return len(lines)

# ---------------- HEATMAP ----------------
def get_color(c):
    if c < 3: return (0,255,0)
    elif c < 6: return (0,255,255)
    else: return (0,0,255)

def draw_heat(frame, ROI, counts):
    overlay = frame.copy()
    lines = get_grid_lines(ROI)
    pts = ROI.tolist()

    zones = [
        [pts[0], pts[1], lines[0][1], lines[0][0]],
        [lines[0][0], lines[0][1], lines[1][1], lines[1][0]],
        [lines[1][0], lines[1][1], pts[2], pts[3]]
    ]

    for i, poly in enumerate(zones):
        cv2.fillPoly(overlay, [np.array(poly)], get_color(counts[i]))

    return cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

# ---------------- PROCESS FRAME ----------------
def process_frame(frame, tracker, ROI):
    results = model(frame)[0]

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

    lines=get_grid_lines(ROI)
    zone_count=[0,0,0]

    for tr in tracks:
        x1,y1,x2,y2,tid=map(int,tr)
        cx,cy=(x1+x2)//2,(y1+y2)//2

        if cv2.pointPolygonTest(ROI,(cx,cy),False)>=0:
            z=get_zone(cx,cy,lines)
            zone_count[z]+=1

        label="vehicle"
        for i,det in enumerate(detections):
            dx1,dy1,dx2,dy2,_=det
            if abs(x1-dx1)<30:
                label=model.names[class_ids[i]]
                break

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,f"{label} ID:{tid}",
                    (x1,y1-10),0,0.5,(0,255,0),2)

    cv2.polylines(frame,[ROI],True,(255,255,0),2)

    for (p1,p2) in lines:
        cv2.line(frame,p1,p2,(255,255,255),2)

    frame = draw_heat(frame, ROI, zone_count)

    return frame, zone_count

# ---------------- DENSITY ----------------
def compute_density(z):
    return z[2]*3 + z[1]*2 + z[0]*1

# ---------------- SIGNAL ----------------
def compute_signal(d):
    NS=d["N"]+d["S"]
    EW=d["E"]+d["W"]

    total=NS+EW+1e-5
    base,max_t=20,60

    gNS=int(base+(NS/total)*(max_t-base))
    gEW=int(base+(EW/total)*(max_t-base))

    return gNS,gEW

def get_phase(gNS,gEW,frame,fps):
    cycle=int((gNS+gEW)*fps)
    t=frame%cycle

    if t < gNS*fps:
        remaining=int(gNS-(t/fps))
        return "NS",remaining
    else:
        remaining=int(gEW-((t-gNS*fps)/fps))
        return "EW",remaining

# ---------------- MAIN ----------------
frame_id=0
out=None

while True:
    retN,fN=cap_N.read()
    retS,fS=cap_S.read()
    retE,fE=cap_E.read()
    retW,fW=cap_W.read()

    if not (retN and retS and retE and retW):
        break

    fN,zN=process_frame(fN,tracker_N,ROI_N)
    fS,zS=process_frame(fS,tracker_S,ROI_S)
    fE,zE=process_frame(fE,tracker_E,ROI_E)
    fW,zW=process_frame(fW,tracker_W,ROI_W)

    # direction labels
    cv2.putText(fN,"NORTH",(20,40),0,1,(255,255,0),2)
    cv2.putText(fS,"SOUTH",(20,40),0,1,(255,255,0),2)
    cv2.putText(fE,"EAST",(20,40),0,1,(255,255,0),2)
    cv2.putText(fW,"WEST",(20,40),0,1,(255,255,0),2)

    density={
        "N":compute_density(zN),
        "S":compute_density(zS),
        "E":compute_density(zE),
        "W":compute_density(zW)
    }

    gNS,gEW=compute_signal(density)
    phase,remaining=get_phase(gNS,gEW,frame_id,fps)

    # log data
    log_data.append({
        "frame":frame_id,
        "north":density["N"],
        "south":density["S"],
        "east":density["E"],
        "west":density["W"],
        "phase":phase,
        "time":remaining
    })

    grid=np.vstack((np.hstack((fN,fS)),np.hstack((fE,fW))))

    if out is None:
        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        out=cv2.VideoWriter("final_output7.mp4",fourcc,fps,(grid.shape[1],grid.shape[0]))

    out.write(grid)

    display=cv2.resize(grid,(1200,800))

    cv2.putText(display,f"PHASE: {phase}",(50,50),0,1,(0,0,255),3)
    cv2.putText(display,f"TIME: {remaining}s",(50,100),0,1,(0,255,0),3)

    cv2.imshow("SMART TRAFFIC SYSTEM",display)

    if cv2.waitKey(1)&0xFF==27:
        break

    frame_id+=1

# ---------------- SAVE CSV ----------------
pd.DataFrame(log_data).to_csv("traffic_analysis.csv",index=False)

# ---------------- RELEASE ----------------
cap_N.release()
cap_S.release()
cap_E.release()
cap_W.release()
out.release()
cv2.destroyAllWindows()