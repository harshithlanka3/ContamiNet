from ultralytics import YOLO
import cv2
import os

# 1. Initialize the YOLO-World model
# 'yolov8s-world.pt' is the small, fast version. It will download automatically.
model = YOLO('yolov8s-world.pt') 

# 2. Define the custom open-vocabulary classes
classes = [
    "plastic", 
    "clear plastic container", 
    "transparent plastic box", 
    "empty plastic tray", 
    "plastic packaging",
    "plastic container", 
    # "transparent cup", 
    # "food residue", 
    "liquid in plastic container", 
    # "clamshell box", 
    "plastic water bottle", 
    # "trash on sand", 
    # "greasy", 
    "plastic film",
    "plastic bag", 
    ""
]
model.set_classes(classes)

# 3. Load the image and run detection
image_path = "test1.jpg"  # Replace with your actual image path
original_img = cv2.imread(image_path)

if original_img is None:
    raise FileNotFoundError(f"Could not load image at {image_path}")

# Run inference (conf=0.15 is slightly lowered to catch more edge cases)
results = model.predict(image_path, conf=0.05)

# 4. Filter for the HIGHEST confidence detection
best_box = None
highest_conf = 0.0

for result in results:
    for box in result.boxes:
        conf = float(box.conf[0])
        if conf > highest_conf:
            highest_conf = conf
            best_box = box

# 5. Process ONLY the best detection
if best_box is not None:
    output_dir = "vlm_crops"
    os.makedirs(output_dir, exist_ok=True)
    
    padding_percent = 0.15
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    class_id = int(best_box.cls[0])
    label = classes[class_id] if class_id < len(classes) else "unknown"
    
    # Calculate padding
    width, height = (x2 - x1), (y2 - y1)
    pad_x, pad_y = int(width * padding_percent), int(height * padding_percent)
    
    # Boundary check
    img_h, img_w = original_img.shape[:2]
    px1, py1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    px2, py2 = min(img_w, x2 + pad_x), min(img_h, y2 + pad_y)
    
    # Crop and Save
    crop_img = original_img[py1:py2, px1:px2]
    safe_label = label.replace(" ", "_")
    filename = f"{output_dir}/best_crop_{safe_label}_{highest_conf:.2f}.jpg"
    cv2.imwrite(filename, crop_img)
    
    print(f"🏆 Best Detection: {label} ({highest_conf*100:.1f}%) -> Saved to {filename}")
    
    # ---------------------------------------------------------
    # -> YOUR VLM INTEGRATION GOES HERE <-
    # e.g., vlm_response = analyze_with_vlm(crop_img)
    # ---------------------------------------------------------
    
else:
    print("No objects detected above the confidence threshold.")
        

print("Processing complete. Check the 'vlm_crops' folder.")