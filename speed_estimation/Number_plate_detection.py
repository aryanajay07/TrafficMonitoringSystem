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

def detect_license_plate_and_number(image_path):
    image = cv2.imread(image_path)
    plates = []
    plate_numbers = []

    # Debug: Check if the image was loaded successfully
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return None, None

    # Perform detection
    results = model(image)

    # Debug: Check the results of the detection
    print(f"Detection results: {results}")

    # Process the results
    for result in results:
        boxes = result.boxes

        # Debug: Check the detected boxes
        print(f"Detected boxes: {boxes}")

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf.item()
            class_id = int(box.cls.item())

            # Debug: Print box details
            print(f"Box details - x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, confidence: {confidence}, class_id: {class_id}")

            # Assuming 'license plate' is the class with index 0
            if class_id == 0 and confidence > 0.1:  # Adjust confidence threshold if needed
                # Crop the detected license plate
                cropped_plate = image[y1:y2, x1:x2]
                plates.append(cropped_plate)

                # Save the cropped license plate image
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                plate_filename = os.path.join(output_folder, f'plate_{len(plates)}.png')
                cv2.imwrite(plate_filename, cropped_plate)

                # Debug: Check the saved plate image path
                print(f"Saved plate image at: {plate_filename}")

                # Use EasyOCR to detect text in both Devnagari and English
                ocr_results = reader.readtext(cropped_plate, detail=1)

                # Debug: Check the OCR results
                print(f"OCR results: {ocr_results}")

                text_line = ""
                for (bbox, text, prob) in ocr_results:
                    # Detect if the text is Devnagari
                    if any('\u0900' <= char <= '\u097F' for char in text):
                        number_plate = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
                        text_line += f"{number_plate} (Dev)"  # Add transliterated text with "(Dev)" indicator
                    else:
                        text_line += text  # Add original text

                    # Add a separator after each recognized text element (except the last one)
                    if len(ocr_results) > 1 and (bbox, text, prob) != ocr_results[-1]:
                        text_line += " | "

                plate_numbers.append(text_line)

                # Optional: Drawing bounding box and label on the image with combined text
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'License Plate: {text_line}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if not plates:
        # Debug: No license plates detected
        print("No license plates detected.")
        return None, None

    return plates, plate_numbers

# Optional: Display the result
# cv2.imshow('Result', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
