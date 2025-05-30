import cv2
import numpy as np
import supervision as sv
from collections import defaultdict
from ultralytics import YOLO
import argparse
from norfair.tracker import Detection
from yolox.tracker.byte_tracker import BYTETracker
# from tracker.tracker import BYTETracker, BYTETrackerArgs  # Adjust import if path differs
unique_ids=set()
def get_tracker_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_thresh", type=float, default=0.5)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.8)
    parser.add_argument("--gate_thresh", type=float, default=0.15)
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6)
    parser.add_argument("--min_box_area", type=float, default=10)
    parser.add_argument("--mot20", action='store_true')
    parser.add_argument("--track_type", type=str, default='bytetrack')
    return parser.parse_args([])
def process_video(video_path):
    print("Inside Process Video Function")

    cap = cv2.VideoCapture(video_path)
    model = YOLO("yolov8n.pt")  # Or your specific model path

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Height:{height} and width;{width}")

    # ByteTrack config
    args=get_tracker_args()
    tracker = BYTETracker(args, frame_rate=fps)

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]  # Only use the first result

        # Check if any boxes are detected
        boxes = results.boxes
        if boxes is not None and boxes.xyxy.numel() > 0:
            xyxy = boxes.xyxy.cpu().numpy().reshape(-1, 4)
            confidence = boxes.conf.cpu().numpy()
            class_id = boxes.cls.cpu().numpy()
            vehicle_class_ids = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
            keep = np.isin(class_id, vehicle_class_ids)

            xyxy = xyxy[keep]
            confidence = confidence[keep]
            class_id = class_id[keep]
            detection_objs = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id.astype(int)
            )
        else:
            detection_objs = sv.Detections(
                xyxy=np.empty((0, 4)),
                confidence=np.array([]),
                class_id=np.array([], dtype=int)
            )
        
        # Feed detections into ByteTrack
        if len(detection_objs.xyxy) > 0:
            output_results = np.concatenate((xyxy, confidence[:, None]), axis=1)
        else:
            output_results = np.empty((0, 5))  # shape must match [N, 5]

        online_targets = tracker.update(
        output_results=output_results,
        img_info=(height, width),
        img_size=(height, width)
)
        print(f"Frame {frame_idx}, Online targets: {online_targets}")
        for target in online_targets:
            unique_ids.add(target.track_id)

        print(f"Total unique vehicles so far: {len(unique_ids)}")
        
        # For testing: limit number of frames
        # if frame_idx > 10:
        #     break

        frame_idx += 1
        yield frame  # Or yield detection_objs if that's what you need
    
    cap.release()
