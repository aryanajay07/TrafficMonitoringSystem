import sys
import os
from types import SimpleNamespace

import numpy as np
import supervision as sv
from collections import defaultdict, deque
from user_app.models import Record, Station
from ultralytics import YOLO
import cv2
from django.http import StreamingHttpResponse
from datetime import datetime
import time
import math
from django.conf import settings
from speed_estimation.Number_plate_detection import detect_license_plate_and_number

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ByteTrack')))
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from yolox.utils.visualize import plot_tracking

import argparse

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

# mac_address = (':'.join(re.findall('..', '%012x' % uuid.getnode())))
mac_address = '8c:aa:ce:51:67:e9'
# SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
SOURCE = np.array([[1272, 1178],[1839, 1119],[1839, 1834],[7, 1710]])
TARGET_WIDTH = 6
TARGET_HEIGHT = 18

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ],
    dtype=np.float32
)

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def calculate_speed(coordinates: deque, fps: int) -> float:
    if len(coordinates) < fps / 2:
        return 0.0
    coordinate_start = coordinates[0]
    coordinate_end = coordinates[-1]
    distance = abs(coordinate_end - coordinate_start)
    time = len(coordinates) / fps
    speed = (distance / time) * 3.6  # m/s to km/h
    return speed


class VideoCamera:
    def __init__(self):
        try:
            # Video path check
            video_path = "speed_estimation/Test_video/v3.mp4"
            print(f"Checking video path: {os.path.abspath(video_path)}")
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
                
            print("Opening video file...")
            # self.video = cv2.VideoCapture(video_path)
            self.video = cv2.VideoCapture(0)

            if not self.video.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Load YOLO model from project models directory
            model_path = "models/yolov8m.pt" 
            print(f"Checking YOLO model path: {os.path.abspath(model_path)}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"YOLO model not found at {model_path}")
            
            print("Loading YOLO model...")
            self.model = YOLO(model_path)
            print("YOLO model loaded successfully")
            
            

            # Initialize other components
            print("Initializing tracker and annotator...")
            self.box_annotator = sv.BoxAnnotator(
                thickness=2,
                text_thickness=1,
                text_scale=0.5
            )
            self.view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
            self.tracker_id_to_coordinates = defaultdict(lambda: deque(maxlen=30))
            
            # Get video properties
            self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
            self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_idx = 0
            self.max_test_frames = 100  # Stop after 10 frames for testing

            args=get_tracker_args()
            # Initialize the ByteTrack tracker
            self.byte_tracker = BYTETracker(args,frame_rate=self.fps)

            print(f"Camera initialized successfully:")
            print(f"- Video path: {os.path.abspath(video_path)}")
            print(f"- Video properties: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
            print(f"- Total frames: {self.frame_count}")
            print(f"- YOLO model: {os.path.abspath(model_path)}")
            
        except Exception as e:
            print(f"Error initializing VideoCamera: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def __del__(self):
        if hasattr(self, 'video'):
            self.video.release()

    def get_frame(self):
        try:
            success, frame = self.video.read()
            print(f"Frame {self.frame_idx}: Read success = {success}")
            
            if not success:
                print(f"Failed to read frame {self.frame_idx}, resetting video")
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_idx = 0
                success, frame = self.video.read()
                if not success:
                    print("Failed to read frame after reset")
                    return None

            # Ensure frame is not empty
            if frame is None or frame.size == 0:
                print(f"Empty frame received at index {self.frame_idx}")
                return None

            # Process frame with YOLO
            print(f"Processing frame {self.frame_idx} with YOLO")
            results = self.model(frame, imgsz=1280)[0]
            
            # Convert YOLO results to supervision Detections format
            boxes = results.boxes
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            
            # Create detections object
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidences,
                class_id=class_ids
            )
            
            # Filter for cars (class_id 2)
            detections = detections[detections.class_id == 2]
            print(f"Detected {len(detections)} vehicles in frame {self.frame_idx}")

            if len(detections) > 0:
                # Process each detected vehicle
                dets = np.hstack((detections.xyxy, detections.confidence.reshape(-1, 1)))
                online_targets = self.byte_tracker.update(dets, [self.frame_height, self.frame_width], [self.frame_height, self.frame_width])

                for target in online_targets:
                    x1, y1, x2, y2 = target.tlbr
                    track_id = target.track_id
                    tracker_id=int(track_id)
                    
                    point = np.array([[(x1 + x2) / 2, y2]])

                    # Transform point
                    transformed_point = self.view_transformer.transform_points(point)[0]
                    self.tracker_id_to_coordinates[tracker_id].append(transformed_point[1])

                    # Calculate speed
                    speed = calculate_speed(self.tracker_id_to_coordinates[tracker_id], self.fps)
                    print(f"Vehicle {tracker_id}: Speed = {speed:.1f} km/h")
                    if speed > 5:
                        print("Speed Limit Breached")

                        # Crop the vehicle region from the frame
                        x1_int, y1_int = int(x1), int(y1)
                        x2_int, y2_int = int(x2), int(y2)
                        vehicle_crop = frame[y1_int:y2_int, x1_int:x2_int]
                        print("vehiclecropshape:",vehicle_crop.shape)
                        # cv2.imshow("vehicle_crop",vehicle_crop)
                        # Sanity check
                        if vehicle_crop is not None and vehicle_crop.size > 0:
                            try:
                                _, license_plate = detect_license_plate_and_number(vehicle_crop)
                                print(f"Tracker ID {tracker_id} Plate: {license_plate}")
                            except Exception as e:
                                print(f"OCR error for tracker {tracker_id}: {e}")
                        else:
                            print(f"Vehicle crop is invalid for tracker {tracker_id}")

                    #Draw bounding box
                    cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                    cv2.putText(frame, f"{speed:.1f} km/h", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # Draw detections
                frame = self.box_annotator.annotate(
                    scene=frame, 
                    detections=detections,
                    labels=[f"{speed} km/h" for speed in detections.class_id]
                )

            # Increment frame counter
            self.frame_idx += 1
            # if self.frame_idx >= self.max_test_frames:
            #     print("Reached test frame limit")
            #     return None

            # Resize frame if too large
            max_dimension = 1280
            height, width = frame.shape[:2]
            if width > max_dimension or height > max_dimension:
                scale = max_dimension / max(width, height)
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
                print(f"Resized frame to {int(width*scale)}x{int(height*scale)}")

            # Convert frame to JPEG with quality parameter
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            if not ret:
                print("Failed to encode frame to JPEG")
                return None
            
            print(f"Successfully processed frame {self.frame_idx-1}")    
            return buffer.tobytes()
            
        except Exception as e:
            print(f"Error in get_frame: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
def save_vehicle_photo(frame, bbox, tracker_id, photo_folder, frame_idx):
    x1, y1, x2, y2 = bbox
    x1_int = np.around(x1).astype(int)
    y1_int = np.around(y1).astype(int)
    x2_int = np.around(x2).astype(int)
    y2_int = np.around(y2).astype(int)

    vehicle_crop = frame[y1_int:y2_int, x1_int:x2_int]
    photo_path = os.path.join(photo_folder, f"tracker_{tracker_id}_frame_{frame_idx}.jpg")
    cv2.imwrite(photo_path, vehicle_crop)
    return photo_path

license_detected = defaultdict(lambda: False)  # Track success per tracker ID

def process_video():
    print("Inside Process Video")
    source_video_path = "speed_estimation/Test_video/test1.mp4"
    target_folder = "./Result_video"
    photo_folder = "./Captured_Photos"
    confidence_threshold = 0.3
    iou_threshold = 0.7

    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)
    os.makedirs(photo_folder, exist_ok=True)

    # Construct the full path for the output video file
    target_video_path = os.path.join(target_folder, "output.mp4")

    # Initialize video info
    cap = cv2.VideoCapture(source_video_path)
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    args=get_tracker_args()
    # Initialize ByteTrack tracker
    tracker = BYTETracker(args,frame_rate=fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Height:{height} and width:{width}")
    # Initialize YOLO model
    model = YOLO("models/yolov8m.pt")

    # Initialize annotators
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    # Initialize zone and transformer
    polygon_zone = sv.PolygonZone(polygon=SOURCE, frame_resolution_wh=(width, height))
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=fps))
    photo_captured = set()
    max_speeds = defaultdict(lambda: 0)  # Dictionary to store the maximum speed of each vehicle
    # print(f"Max_speeds:{max_speeds}")#to be deleted

    #  track last frame a retry was attempted
    retry_delay = 10  # Retry after 10 frames
    last_attempt_frame = defaultdict(lambda: -retry_delay)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        results = model(frame, verbose=False)[0]
        # detections = sv.Detections.from_ultralytics(results)
        results = model(frame)[0]  # assuming results is an Ultralytics result object
        VEHICLE_CLASSES = [2, 3, 5, 7]
        xyxy = results.boxes.xyxy.cpu().numpy()
        confidence = results.boxes.conf.cpu().numpy()
        class_id = results.boxes.cls.cpu().numpy()

        # Filter for vehicles only
        vehicle_mask = np.isin(class_id, VEHICLE_CLASSES)
        xyxy = xyxy[vehicle_mask]
        confidence = confidence[vehicle_mask]
        class_id = class_id[vehicle_mask]
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id.astype(int)
        )
        # Process detections
        if len(detections) ==0:
            frame_idx+=1
            continue

        #prepare detections for byteTrack
        dets=np.hstack((detections.xyxy,detections.confidence.reshape(-1,1)))

        #bytetrack expects detections in format :[x1,y1,x2,y2,score]
        online_targets=tracker.update(dets,[height,width],[height,width])
        print("Online_targets:") #to be deleted
        print(online_targets) #to be deleted
        for target in online_targets:
            x1, y1, x2, y2 = target.tlbr
            track_id = target.track_id
            tracker_id=int(track_id)

            #calculate center point 
            center_x=(x1+x2)/2
            center_y=y2
            center_point = np.array([[center_x, center_y]])

            # Transform point and calculate speed
            transformed_point = view_transformer.transform_points(center_point)[0]
            coordinates[tracker_id].append(transformed_point[1])
            speed = calculate_speed(coordinates[tracker_id], fps)

            if speed>max_speeds[tracker_id]:
                max_speeds[tracker_id]=speed

            print(f"Tracker {tracker_id} speed: {speed:.2f} km/h")
            if speed > 0.1 and not license_detected[tracker_id]:
                if frame_idx - last_attempt_frame[tracker_id] < retry_delay:
                    continue
                last_attempt_frame[tracker_id] = frame_idx
                photo_captured.add(tracker_id)
                photo_path=save_vehicle_photo(frame,(x1,y1,x2,y2),tracker_id,photo_folder,frame_idx)

                #crop photo 
                x1=max(0,min(width-1,int(x1)))
                x2 = max(0, min(width, int(x2)))
                y1 = max(0, min(height - 1, int(y1)))
                y2 = max(0, min(height, int(y2)))
                vehicle_crop = frame[y1:y2, x1:x2]

                print("Vehicle shape:", vehicle_crop.shape)
                os.makedirs("debug_output", exist_ok=True)
                cv2.imwrite(f"debug_output/cropped{tracker_id}.jpg", vehicle_crop)
                if vehicle_crop is None or vehicle_crop.size == 0:
                    print(f"Warning: Invalid vehicle_crop for tracker_id {tracker_id}")
                    continue
                _,license_plate=detect_license_plate_and_number(vehicle_crop)

                print(f"Detected license plate for tracker ID {tracker_id}: {license_plate}")
                if license_plate and isinstance(license_plate, str) and len(license_plate.strip()) >= 3:  # Adjust condition as needed
                    license_detected[tracker_id] = True
                    photo_captured.add(tracker_id)  # optional: mark that we got a valid image
                else:
                    print(f"Retrying for tracker ID {tracker_id} in next frames.")
                    continue  # Don't save to DB if license not detected
                

                try:
                    station = Station.objects.get(mac_address=mac_address)
                    Record.objects.create(
                        stationID=station,
                        speed=int(speed),
                        date=datetime.now().date(),
                        count=1,
                        licenseplate_no=license_plate if license_plate else "Unknown",
                        vehicle_image=photo_path
                     )
                except Exception as e:
                    print(f"Error saving in database :{e}")

        # Draw detections and annotations
        labels = [
            f"#{target.track_id} {max_speeds[target.track_id]:.1f} km/h"
            for target in online_targets
        ]
        
        
        # Draw boxes using supervision
        det_xyxy = np.array([target.tlbr for target in online_targets])
        if det_xyxy.size == 0:
            det_xyxy = det_xyxy.reshape(0, 4)  # Make sure empty array has shape (0, 4)
        det_conf = np.ones(len(online_targets)) if len(online_targets) > 0 else np.array([]) # confidence dummy for annotation
        det_class = np.zeros(len(online_targets),dtype=int)  if len(online_targets) > 0 else np.array([])# class dummy

        # print("det_xyxy shape:", det_xyxy.shape)
        # print("det_conf shape:", det_conf.shape)
        # print("det_class shape:", det_class.shape)

        detection_objs = sv.Detections(
            xyxy=det_xyxy,
            confidence=det_conf,
            class_id=det_class
        )

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detection_objs,
            labels=labels
        )

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Yield frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        frame_idx += 1

    cap.release()


if __name__ == "__main__":
    process_video()
