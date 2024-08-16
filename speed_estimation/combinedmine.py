import os
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
    coordinate_start = coordinates[-1]
    coordinate_end = coordinates[0]
    distance = abs(coordinate_start - coordinate_end)
    time = len(coordinates) / fps
    speed = distance / time * 3.6  # Convert m/s to km/h
    return speed

def process_video():
    source_video_path = "speed_estimation/Test_video/Vehicle3.mp4"
    target_folder = "./Result_video"
    photo_folder = "./Captured_Photos"
    confidence_threshold = 0.3
    iou_threshold = 0.7

    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)
    os.makedirs(photo_folder, exist_ok=True)

    # Construct the full path for the output video file
    target_video_path = os.path.join(target_folder, "output.mp4")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    model = YOLO("models/Vehicle_Detection.pt")

    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_activation_threshold=0.7
    )

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    photo_captured = set()
    max_speeds = defaultdict(lambda: 0)  # Dictionary to store the maximum speed of each vehicle

    for frame_idx, frame in enumerate(frame_generator):
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.confidence > confidence_threshold]
        detections = detections[polygon_zone.trigger(detections)]
        detections = detections.with_nms(threshold=iou_threshold)
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        )
        points = view_transformer.transform_points(points=points).astype(int)

        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)

        labels = []
        for tracker_id in detections.tracker_id:
            speed = calculate_speed(coordinates[tracker_id], video_info.fps)
            max_speeds[tracker_id] = max(max_speeds[tracker_id], speed)  # Update maximum speed
            if speed == 0.0:
                labels.append(f"#{tracker_id}")
            else:
                labels.append(f"#{tracker_id} {int(speed)} km/h")
                if speed > 10 and tracker_id not in photo_captured:
                    vehicle_speed = max_speeds[tracker_id]
                    vehicle_tracker_id = tracker_id
                    photo_captured.add(tracker_id)
                    vehicle_cropped = save_vehicle_photo(frame, detections, tracker_id, photo_folder, frame_idx)

                    try:
                        # Constructing the path to save a file in the media directory
                        photo_filename = f"tracker_{tracker_id}_frame_{frame_idx}.jpg"
                        photo_path = os.path.join(settings.MEDIA_ROOT, 'Vehicle_images', photo_filename)

                        # Saving the file
                        cv2.imwrite(photo_path, vehicle_cropped)
                        print(photo_path)
                        # Detect license plate and its number
                        plates, plate_numbers = detect_license_plate_and_number(photo_path)

                        if plates is None or plate_numbers is None:
                            print("No license plates detected.")
                        else:
                            # Save the license plate image and number to the database
                            for i, plate in enumerate(plates):
                                license_plate_filename = f"tracker_{tracker_id}_frame_{frame_idx}_plate_{i}.jpg"
                                license_plate_path = os.path.join(settings.MEDIA_ROOT, 'License_plate_images', license_plate_filename)
                                cv2.imwrite(license_plate_path, plate)
                                
                                # Storing the record in the database
                                new_data = Record(
                                    stationID=Station.objects.get(mac_address=mac_address),
                                    speed=vehicle_speed,
                                    date=datetime.now().date(),
                                    count=5,
                                    vehicle_image=os.path.join('Vehicle_images', photo_filename),
                                    license_plate_image=os.path.join('License_plate_images', license_plate_filename),
                                    liscenseplate_no=plate_numbers[i] if plate_numbers else None
                                )
                                new_data.save() # Save
                    except Station.DoesNotExist:
                        print(f"Station with MAC address {mac_address} does not exist.")
                    except Exception as e:
                        print(f"Error saving Record: {e}")

        annotated_frame = frame.copy()

        # Ensure the number of labels matches the number of detections
        if len(labels) == len(detections):
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )
        else:
            print(f"Number of labels ({len(labels)}) does not match number of detections ({len(detections)})")

        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )

        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        # Convert encoded JPEG frame to bytes
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
    cv2.destroyAllWindows()

def save_vehicle_photo(frame, detections, tracker_id, photo_folder, frame_idx):
    best_y = float('inf')
    best_detection = None
    for detection in detections:
        if detection[4] == tracker_id:
            x1, y1, x2, y2 = detection[0][0], detection[0][1], detection[0][2], detection[0][3]
            if y2 < best_y:
                best_y = y2
                best_detection = detection

    if best_detection is not None:
        x1, y1, x2, y2 = best_detection[0][0], best_detection[0][1], best_detection[0][2], best_detection[0][3]
        x1_int = np.around(x1).astype(int)
        y1_int = np.around(y1).astype(int)
        x2_int = np.around(x2).astype(int)
        y2_int = np.around(y2).astype(int)

        vehicle_crop = frame[y1_int:y2_int, x1_int:x2_int]
        return vehicle_crop

if __name__ == "__main__":
    process_video()

