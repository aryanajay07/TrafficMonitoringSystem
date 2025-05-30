import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
import matplotlib.pyplot as plt

# ======= Configuration =======
plate_image_path = "plate3.png"
char_detector_model_path = "models/devanagari_ocr_model.keras"
char_classifier_model_path = "models/characterReaderModel.h5"
class_names = ['क', 'को', 'ख', 'ग', 'च', 'ज', 'झ', 'ञ', 'डि', 'त', 'ना', 'प', 'प्र', 'ब', 'बा', 'भे', 'म', 'मे', 'य', 'लु', 'सी', 'सु', 'से', 'ह', '०', '१', '२', '३', '४', '५', '६', '७', '८', '९']  # list of your 34 Devanagari characters
# Replace with actual characters used in your dataset

# ======= Load models =======
yolo_model = YOLO(char_detector_model_path)
classifier = load_model(char_classifier_model_path)

# ======= Load and run detection =======
image = cv2.imread(plate_image_path)
results = yolo_model(image)[0]

char_crops = []

# ======= Crop character regions =======
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cropped_char = image[y1:y2, x1:x2]
    char_crops.append((x1, cropped_char))

# ======= Sort characters left to right =======
char_crops.sort(key=lambda x: x[0])
sorted_crops = [crop[1] for crop in char_crops]

plate_text = ""

# ======= Predict characters =======
for char_img in sorted_crops:
    char_gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    char_resized = cv2.resize(char_gray, (32, 32))  # match your training input size
    char_arr = img_to_array(char_resized) / 255.0
    char_arr = np.expand_dims(char_arr, axis=0)

    prediction = classifier.predict(char_arr)
    predicted_class = class_names[np.argmax(prediction)]
    plate_text += predicted_class

print("🔤 Predicted Plate Text:", plate_text)

# ======= Visualize results =======
annotated_img = image.copy()
for (box, char) in zip(results.boxes, plate_text):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(annotated_img, char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Show image
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
plt.title(f"Predicted Plate: {plate_text}")
plt.axis('off')
plt.show()
