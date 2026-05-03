import numpy as np

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) +
              (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return o


class Track:
    def __init__(self, bbox, track_id):
        self.bbox = bbox
        self.id = track_id
        self.age = 0

class Sort:
    def __init__(self, iou_threshold=0.3):
        self.tracks = []
        self.track_id = 0
        self.iou_threshold = iou_threshold

    def update(self, detections):
        updated_tracks = []

        for det in detections:
            x1, y1, x2, y2, conf = det
            matched = False

            for track in self.tracks:
                if iou(track.bbox, det[:4]) > self.iou_threshold:
                    track.bbox = det[:4]
                    track.age = 0
                    updated_tracks.append(track)
                    matched = True
                    break

            if not matched:
                self.track_id += 1
                new_track = Track(det[:4], self.track_id)
                updated_tracks.append(new_track)

        # Increase age of tracks
        for track in self.tracks:
            track.age += 1
            if track.age < 5:
                updated_tracks.append(track)

        self.tracks = updated_tracks

        results = []
        for track in self.tracks:
            x1, y1, x2, y2 = track.bbox
            results.append([x1, y1, x2, y2, track.id])

        return np.array(results)