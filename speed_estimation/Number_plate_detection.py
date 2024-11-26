import cv2
import os
import easyocr
from ultralytics import YOLO
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Paths to the files and folders
model_path = 'models/Number_plate_recognize_last .pt'
output_folder = './Plate_photo'

# Load the YOLOv8 model (make sure to use the correct model file or model name)
model = YOLO(model_path)  # Replace 'yolov8n.pt' with your custom model if needed

# Initialize EasyOCR reader for both Nepali (Devnagari script) and English
reader = easyocr.Reader(['ne', 'en'])

def detect_license_plate_and_number(image_input):
    # Handle both file paths and image arrays
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
    else:
        image = image_input

    plates = []
    plate_numbers = []

    # Debug: Check if the image was loaded successfully
    if image is None:
        print("Error: Invalid image input")
        return None, None

    # Perform detection
    results = model(image)

    # Process the results
    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf.item()
            class_id = int(box.cls.item())

            # Assuming 'license plate' is the class with index 0
            if class_id == 0 and confidence > 0.1:  # Adjust confidence threshold if needed
                # Crop the detected license plate
                plate = image[y1:y2, x1:x2]
                plates.append(plate)

                # Perform OCR on the cropped plate
                try:
                    ocr_result = reader.readtext(plate)
                    if ocr_result:
                        # Take the text with highest confidence
                        text = max(ocr_result, key=lambda x: x[2])[1]
                        
                        # Try to transliterate if the text is in Devanagari
                        try:
                            text = transliterate(text, sanscript.DEVANAGARI, sanscript.ROMAN)
                        except:
                            pass  # Keep original text if transliteration fails
                        
                        plate_numbers.append(text)
                    else:
                        plate_numbers.append(None)
                except Exception as e:
                    print(f"OCR Error: {e}")
                    plate_numbers.append(None)

    if not plates:
        return None, None
    
    return plates[0], plate_numbers[0] if plate_numbers else None

# Optional: Display the result
# cv2.imshow('Result', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
