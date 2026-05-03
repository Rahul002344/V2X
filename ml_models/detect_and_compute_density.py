# detect_and_compute_density.py
import os
import cv2
import math
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from pathlib import Path
from datetime import timedelta

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="video file or frames folder")
    p.add_argument("--outdir", default="./results", help="output folder")
    p.add_argument("--model_name", default="iisc-aim/UVH-26", help="HuggingFace model name or local path")
    p.add_argument("--window_seconds", type=int, default=15, help="aggregation window length in seconds")
    p.add_argument("--fps_override", type=float, default=0, help="if video fps detection fails, set it")
    p.add_argument("--capacity", type=int, default=50, help="camera 'capacity' used to normalize density (tuneable)")
    return p.parse_args()

# map model class names (or COCO ids) to vehicle buckets we care about
VEHICLE_CLASSES = {
    # adapt if model outputs different names; common names: 'car','bus','truck','motorbike','bicycle'
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "motorbike": "2w",
    "motorcycle": "2w",
    "scooter": "2w",
    "bicycle": "2w",
    "autorickshaw": "3w",
    "three_wheeler": "3w",
    "rickshaw": "3w",
}

def ensure_dir(p): 
    Path(p).mkdir(parents=True, exist_ok=True)

def open_video_or_frames(source):
    source = str(source)
    if os.path.isdir(source):
        # list image files sorted
        imgs = sorted([os.path.join(source, f) for f in os.listdir(source)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        return "frames", imgs, None
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {source}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or math.isnan(fps):
            fps = None
        return "video", cap, fps

def frame_gen(source, mode):
    if mode == "frames":
        for path in source:
            img = cv2.imread(path)
            yield img, path, None
    else:
        cap, fps_hint = source
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame, f"frame_{idx:08d}.jpg", None
            idx += 1
        cap.release()

def compute_density(count, capacity):
    # simple normalized density; clamp to 1.0
    return min(1.0, float(count) / max(1.0, capacity))

def save_json_window(outdir, window_index, row):
    ensure_dir(os.path.join(outdir, "density_jsons"))
    path = os.path.join(outdir, "density_jsons", f"window_{window_index:05d}.json")
    with open(path, "w") as f:
        json.dump(row, f, indent=2)

def main():
    args = parse_args()
    outdir = args.outdir
    ensure_dir(outdir)
    ensure_dir(os.path.join(outdir, "samples"))

    # load model (will fetch from HF if given a HF repo id)
    print("Loading model:", args.model_name)
    model = YOLO(args.model_name)  # ultralytics will handle HF repo ids

    mode, source_obj, fps = open_video_or_frames(args.source)
    if args.fps_override and not fps:
        fps = args.fps_override
    if not fps and mode == "video":
        # try to re-open to read fps
        try:
            cap = source_obj
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        except:
            fps = 30.0

    window_frames = max(1, int(round((args.window_seconds or 15) * (fps or 30.0))))
    print(f"Mode={mode} fps={fps} window_frames={window_frames}")

    per_frame_records = []
    window_records = []
    running_counts = []
    cur_window_idx = 0
    cur_window_start_frame = 0

    # simple per-class mapping to counts
    def counts_from_results(results):
        # results is ultralytics results for a frame
        counts = {"car":0,"bus":0,"truck":0,"2w":0,"3w":0,"other":0}
        if results is None:
            return counts
        boxes = results.boxes if hasattr(results, "boxes") else []
        for b in boxes:
            try:
                cls_name = results.names[int(b.cls)]
            except Exception:
                # fallback to label string if provided
                cls_name = str(int(b.cls))
            k = VEHICLE_CLASSES.get(cls_name.lower(), None)
            if k is None:
                counts["other"] += 1
            else:
                counts[k] += 1
        counts["total"] = sum([v for k,v in counts.items() if k!="total" and k!="other"])
        return counts

    # iterate frames
    if mode == "frames":
        frame_list = source_obj
        iterator = enumerate(frame_gen(frame_list, "frames"))
    else:
        cap = source_obj
        iterator = enumerate(frame_gen((cap, fps), "video"))

    for i, (frame, frame_name, _) in tqdm(iterator, desc="Frames processed"):
        # run detection for this frame
        # ultralytics model(frame) returns a Results object (can be batched)
        res = model(frame, imgsz=1280, conf=0.25, verbose=False)  # adjust imgsz/conf as needed
        r0 = res[0]

        counts = counts_from_results(r0)
        density = compute_density(counts["total"], args.capacity)

        ts_seconds = None
        if fps:
            ts_seconds = (i / fps)
            ts_hms = str(timedelta(seconds=int(ts_seconds)))
        else:
            ts_hms = None

        per_frame_records.append({
            "frame_index": i,
            "frame_name": frame_name,
            "timestamp_s": ts_seconds,
            "timestamp_hms": ts_hms,
            "count_car": counts.get("car",0),
            "count_bus": counts.get("bus",0),
            "count_truck": counts.get("truck",0),
            "count_2w": counts.get("2w",0),
            "count_3w": counts.get("3w",0),
            "count_other": counts.get("other",0),
            "total_count": counts.get("total",0),
            "density": density
        })

        running_counts.append(counts.get("total",0))

        # window boundary
        if (i - cur_window_start_frame + 1) >= window_frames:
            # aggregate window
            win_slice = per_frame_records[-window_frames:]
            avg_counts = np.mean([r["total_count"] for r in win_slice])
            avg_density = np.mean([r["density"] for r in win_slice])
            window_start_idx = cur_window_start_frame
            window_end_idx = i
            window_ts_start = win_slice[0]["timestamp_hms"] if win_slice[0]["timestamp_hms"] else ""
            window_ts_end = win_slice[-1]["timestamp_hms"] if win_slice[-1]["timestamp_hms"] else ""
            row = {
                "window_index": cur_window_idx,
                "frame_start_index": window_start_idx,
                "frame_end_index": window_end_idx,
                "timestamp_start": window_ts_start,
                "timestamp_end": window_ts_end,
                "avg_count": float(avg_counts),
                "avg_density": float(avg_density),
                "samples": [win_slice[0]["frame_name"], win_slice[len(win_slice)//2]["frame_name"], win_slice[-1]["frame_name"]]
            }
            window_records.append(row)
            save_json_window(outdir, cur_window_idx, row)
            cur_window_idx += 1
            cur_window_start_frame = i + 1
            running_counts = []

    # save per_frame_counts.csv
    df = pd.DataFrame(per_frame_records)
    csv_pf = os.path.join(outdir, "per_frame_counts.csv")
    df.to_csv(csv_pf, index=False)
    print("Saved", csv_pf)

    # save window_summaries.csv
    dfw = pd.DataFrame(window_records)
    csv_ws = os.path.join(outdir, "window_summaries.csv")
    dfw.to_csv(csv_ws, index=False)
    print("Saved", csv_ws)

    # Save 3 sample frames: lowest density, median, highest density
    if len(df) > 0:
        df_sorted = df.sort_values("density")
        sample_rows = []
        sample_rows.append(df_sorted.iloc[0])
        sample_rows.append(df_sorted.iloc[len(df_sorted)//2])
        sample_rows.append(df_sorted.iloc[-1])
        samples_dir = os.path.join(outdir, "samples")
        ensure_dir(samples_dir)
        for idx, row in enumerate(sample_rows):
            fname = row["frame_name"]
            # if frames folder, fname is a path; if video, stored as synthetic name -> we saved the frames? we didn't.
            # So read frame again by index if video.
            frame_idx = int(row["frame_index"])
            # reopen video to grab frame
            if mode == "video":
                cap = cv2.VideoCapture(args.source)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, f = cap.read()
                cap.release()
                if not ret:
                    continue
                save_p = os.path.join(samples_dir, f"sample_{idx}_{frame_idx:05d}.jpg")
                cv2.putText(f, f"density={row['density']:.3f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
                cv2.imwrite(save_p, f)
            else:
                # frames mode: fname is path
                try:
                    f = cv2.imread(fname)
                    save_p = os.path.join(samples_dir, f"sample_{idx}_{Path(fname).stem}.jpg")
                    cv2.putText(f, f"density={row['density']:.3f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
                    cv2.imwrite(save_p, f)
                except:
                    pass

        print("Saved sample frames to", samples_dir)

    print("Done.")

if __name__ == "__main__":
    main()