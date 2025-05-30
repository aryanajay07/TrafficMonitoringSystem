# from speed_estimation.Number_plate_detection import detect_license_plate_and_number
# import cv2

# if __name__ == "__main__":
#     test_image_path = 'check1.jpg'

#     image = cv2.imread(test_image_path)
#     if image is None:
#         print("❌ Error: Image not found or invalid path:", test_image_path)
#         exit()
#     else:
#         print("✅ Image successfully loaded:", image.shape)

#     plates, plate_numbers = detect_license_plate_and_number(test_image_path)

#     if plates and any(p is not None for p in plates):
#         print("✅ License Plate(s) Detected!")

#         for i, (plate, text) in enumerate(zip(plates, plate_numbers)):
#             if plate is not None:
#                 print(f"Plate {i+1} Number: {text}")
#                 cv2.imshow(f"Detected Plate {i+1}", plate)
#                 cv2.imwrite(f"detected_plate_{i+1}.jpg", plate)

#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     else:
#         print("⚠️ No license plate detected.")
# import re
# text = ')्याशवच३६२'
# text = re.sub(r'[^\u0900-\u097F\u0966-\u096F]', '', text)  # Keep only Devanagari and digits
# print(text)  # Expected cleanup: 'बाच३६२' (ideal)

import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if cap.isOpened():
    print("Camera opened successfully with DirectShow backend.")
else:
    print("Failed to open camera.")
cap.release()
