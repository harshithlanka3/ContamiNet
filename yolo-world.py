from ultralytics import YOLO
import cv2
import os

# 1. Initialize the YOLO-World model
# 'yolov8s-world.pt' is the small, fast version. It will download automatically.
model = YOLO('yolov8s-world.pt') 

# 2. Define the custom open-vocabulary classes
classes = [
    # Clear/Transparent focus
    "clear plastic container", "transparent plastic bottle", "empty clear plastic box", 
    "clear plastic clamshell", "transparent plastic cup", "clear plastic lid",
    
    # Opaque/Colored focus
    "opaque plastic bottle", "white plastic jug", "black plastic tray", 
    "colored plastic tub", "opaque detergent bottle",
    
    # Specific shapes/states
    "plastic water bottle", "plastic takeout container", "plastic berry container", 
    "crushed plastic bottle", "plastic yogurt tub", "open yogurt tub",
    "dirty plastic container on its side",
    "plastic container on its side",
    "plastic cup looking inside",
    
    # Bags/Films
    "plastic grocery bag", "clear plastic bag", "crumpled plastic wrap"
]

model.set_classes(classes)

# 3. Load the image and run detection
image_path = "test_images/empty-yoghurt-pot-B8K673.jpg"  # Replace with your actual image path
original_img = cv2.imread(image_path)

if original_img is None:
    raise FileNotFoundError(f"Could not load image at {image_path}")

# Run inference (conf=0.15 is slightly lowered to catch more edge cases)
results = model.predict(image_path, conf=0.15)

# 4. Process bounding boxes, apply padding, and crop
padding_percent = 0.15  # 15% padding to capture edges/lids
output_dir = "vlm_crops"
os.makedirs(output_dir, exist_ok=True)

# Loop through the detection results
for result in results:
    for i, box in enumerate(result.boxes):
        # Extract coordinates, class, and confidence
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        label = classes[class_id]
        confidence = float(box.conf[0])
        
        # Calculate padding dimensions
        width = x2 - x1
        height = y2 - y1
        pad_x = int(width * padding_percent)
        pad_y = int(height * padding_percent)
        
        # Apply padding while ensuring we don't go out of image bounds
        img_h, img_w = original_img.shape[:2]
        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(img_w, x2 + pad_x)
        py2 = min(img_h, y2 + pad_y)
        
        # Crop the padded region
        crop_img = original_img[py1:py2, px1:px2]
        
        # Save the crop (or pass it directly to your VLM function in memory)
        safe_label = label.replace(" ", "_")
        filename = f"{output_dir}/crop_{i}_{safe_label}_{confidence:.2f}.jpg"
        cv2.imwrite(filename, crop_img)
        
        print(f"Detected: {label} ({confidence*100:.1f}%) -> Saved to {filename}")
        
        # ---------------------------------------------------------
        # -> YOUR VLM INTEGRATION GOES HERE <-
        # e.g., vlm_response = analyze_with_vlm(crop_img)
        # ---------------------------------------------------------

print("Processing complete. Check the 'vlm_crops' folder.")