# from ultralytics import YOLO
# from speed_estimation.Number_plate_detection import detect_license_plate_and_number

# # model = YOLO("models/Number_plate_recognize_last.pt")
# # print(model.names)  # prints class names with their IDs

# img="check.jpg"
# print(img)
# plates,plate_number=detect_license_plate_and_number("check.jpg")
# print(f"number:{plate_number}")

import os
import numpy as np
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Final_project.settings')
django.setup()

from speed_estimation.process_video import process_video

print("Testing purposes")
video_path = "speed_estimation/Test_video/Vehicle1.mp4"
results=process_video(video_path)

for i, frame in enumerate(results):
    print(f"Frame {i} processed.")
    if i >= 4:
        break
# print(results)