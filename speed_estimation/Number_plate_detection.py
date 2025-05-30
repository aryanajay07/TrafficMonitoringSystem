import cv2
import os
import easyocr
import logging
from ultralytics import YOLO
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths to the files and folders
model_path = 'models/license_detection_model.pt'
output_folder = './Plate_photo'

# Load the YOLOv8 model with error handling
try:
    model = YOLO(model_path)
    logging.info("YOLO model loaded successfully.")
    print("Class names:", model.names)
except Exception as e:
    logging.error(f"Error loading YOLO model: {e}")
    raise

# Initialize EasyOCR reader for both Nepali (Devnagari script) and English
try:
    reader = easyocr.Reader(['en','ne','hi','en'], gpu=True)
except:
    logging.warning("GPU not available for EasyOCR. Using CPU.")
    reader = easyocr.Reader(['en','ne','hi',], gpu=False)


# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)  # ✅ Create output folder if not exists

def detect_license_plate_and_number(image_input):
    print("Inside plate detection")

    # Handle both file paths and image arrays
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
    else:
        image = image_input

    plates = []
    plate_numbers = []

    if image is None:
        logging.error("Error: Invalid image input")
        return None, None

    results = model(image,show=True)
    if len(results) == 0 or not hasattr(results[0], 'boxes'):
        logging.warning("YOLO returned no detections.")
        return None, None
    boxes = results[0].boxes
    # print(boxes)
    for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            h, w = image.shape[:2]
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))
            confidence = box.conf.item()
            class_id = int(box.cls.item())
      
            if class_id == 0 and confidence > 0.3:
                plate = image[y1:y2, x1:x2]
                plates.append(plate)
                print("Before preprocessing:Image ")
                # ✅ Save cropped plate image
                cv2.imwrite(os.path.join(output_folder, f"plate_{len(plates)}.jpg"), plate)
                try:
                    # Resize early for better preprocessing resolution
                    plate = cv2.resize(plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                    # === Preprocessing ===
                    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
                    thresh = cv2.adaptiveThreshold(blurred, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                    processed = cv2.resize(morphed, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

                    # cv2.imshow("Preprocessed Plate", processed)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # Convert back to 3-channel for OCR
                    plate_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

                    # OCR
                    ocr_result = reader.readtext(plate_rgb)

                    print(f"OCR raw result for plate {i}: {ocr_result}")

                    if ocr_result:
                        text = max(ocr_result, key=lambda x: x[2])[1]
                        cleaned = text.replace(' ', '').replace('\n', '').strip()
                        plate_numbers.append(cleaned)
                        # plate_numbers.append(text)

                        # ✅ Draw bounding box and label for visualization
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (255, 255, 255), 2)
                    else:
                        plate_numbers.append(None)
                except Exception as e:
                    logging.warning(f"OCR Error: {e}")
                    plate_numbers.append(None)

    if not plates:
        logging.info("No license plates detected.")
        return None, None

    # ✅ Return all plates and numbers instead of just the first
    return plates, plate_numbers


# Example usage:
# plates, numbers = detect_license_plate_and_number("license-plates-5/test/images/b1a50a3824887ee2_jpg.rf.68a4fd34fce20184287592f2680f895b.jpg")
# if  plates: 
#     for i, num in enumerate(numbers):
#         print(f"Plate {i+1}: {num}")
#         cv2.imshow(f"Plate {i+1}", plates[i])
# cv2.waitKey(2000)
# cv2.destroyAllWindows()
