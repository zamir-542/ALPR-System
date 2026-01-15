import os
import cv2
import numpy as np
import glob
import logging
import warnings
from pathlib import Path
from ultralytics import YOLO
from paddleocr import PaddleOCR

# --- CONFIGURATION ---
VERSION = 8
MAX_IMAGES = 100
IOU_THRESHOLD = 0.3
CONF_BASE = 0.20
CONF_RETRIES = [0.15, 0.10]

# Paths
MODEL_PATH = f"best ({VERSION}).pt" 
IMAGES_FOLDER = "random_images"
OUTPUT_FOLDER = f"ocr_result{VERSION}"

# Suppress PaddleOCR verbose logging
os.environ['FLAGS_logtostderr'] = '0'
os.environ['FLAGS_stderrthreshold'] = '3'
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddleocr").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return interArea / float(box1Area + box2Area - interArea)

def filter_overlapping_boxes(boxes_data, iou_threshold=0.3):
    """Remove overlapping detections using NMS-like logic."""
    if not boxes_data:
        return []
    
    # Sort by confidence
    boxes_data = sorted(boxes_data, key=lambda x: x['conf'], reverse=True)
    
    keep = []
    while boxes_data:
        current = boxes_data.pop(0)
        keep.append(current)
        
        remaining = []
        for other in boxes_data:
            if calculate_iou(current['box'], other['box']) < iou_threshold:
                remaining.append(other)
        boxes_data = remaining
    
    return keep

def enhance_for_detection(image):
    """Apply CLAHE enhancement to improve detection in poor lighting."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def adjust_brightness(image, gamma=1.0):
    """Apply Gamma Correction."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def sharpen_image(image):
    """Apply a sharpening kernel."""
    kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)

def smart_detect(model, img_path):
    """Try multiple detection strategies in a logical sequence."""
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    
    # Pass 1: Standard confidence (0.20)
    results = model.predict(img, conf=CONF_BASE, verbose=False)
    if len(results[0].boxes) > 0:
        return results, img
    
    # Pass 2: Lower confidence (0.15)
    results = model.predict(img, conf=0.15, verbose=False)
    if len(results[0].boxes) > 0:
        print(f"    ! Found with confidence 0.15")
        return results, img
        
    # Pass 3: Image Enhancements (at 0.15)
    enhancements = [
        ("CLAHE", enhance_for_detection(img)),
        ("Bright γ=1.5", adjust_brightness(img, gamma=1.5)),
        ("Bright γ=1.8", adjust_brightness(img, gamma=1.8)),
        ("Sharpen", sharpen_image(img)),
        ("Dark γ=0.7", adjust_brightness(img, gamma=0.7)),
    ]
    
    for name, enhanced_img in enhancements:
        results = model.predict(enhanced_img, conf=0.15, verbose=False)
        if len(results[0].boxes) > 0:
            print(f"    ! Detected with: {name}")
            return results, enhanced_img
    
    # Pass 4: Resolution Scaling (at 0.15)
    for sz in [480, 1280]:
        results = model.predict(img, conf=0.15, imgsz=sz, verbose=False)
        if len(results[0].boxes) > 0:
            print(f"    ! Detected with: Resolution {sz}")
            return results, img

    # Pass 5: Last resort - Very Low Confidence (0.10)
    results = model.predict(img, conf=0.10, verbose=False)
    if len(results[0].boxes) > 0:
        print("    ! Detected with: Very Low Confidence (0.10)")
    
    return results, img

def clean_plate_text(text):
    """Clean and normalize OCR text."""
    text = text.replace(' ', '').upper()
    cleaned = ''.join([char for char in text if char.isalnum()])
    
    # Common Malaysian plate fix (O -> Q)
    cleaned = cleaned.replace('O', 'Q')
    
    return cleaned

def recognize_plate_paddleocr(reader, plate_crop, temp_path="temp_plate.jpg"):
    """Run OCR on the plate crop."""
    try:
        cv2.imwrite(temp_path, plate_crop)
        result = reader.predict(temp_path)
        
        if result and isinstance(result, list) and len(result) > 0:
            first_result = result[0]
            if 'rec_texts' in first_result and 'rec_scores' in first_result:
                texts = first_result['rec_texts']
                scores = first_result['rec_scores']
                combined_text = ''.join(texts)
                avg_conf = sum(scores) / len(scores) if scores else 0.0
                cleaned_text = clean_plate_text(combined_text)
                return cleaned_text, avg_conf
        return "", 0.0
    except Exception as e:
        print(f"    PaddleOCR Error: {e}")
        return "", 0.0

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print(f"Loading YOLO model: {Path(MODEL_PATH).name}...")
    model = YOLO(MODEL_PATH)
    
    print("Loading PaddleOCR...")
    reader = PaddleOCR(
        use_doc_orientation_classify=False,
        use_textline_orientation=False,
        use_doc_unwarping=False,
        lang='en',
        text_det_unclip_ratio=2.5,
        text_det_box_thresh=0.4
    )
    
    # Get images
    patterns = ['*.jpg', '*.jpeg', '*.png']
    all_images = []
    for p in patterns:
        all_images.extend(glob.glob(os.path.join(IMAGES_FOLDER, p)))
    
    print(f"Found {len(all_images)} images. Processing top {MAX_IMAGES}...")
    print("=" * 60)
    
    detection_count = 0
    for idx, img_path in enumerate(all_images[:MAX_IMAGES], 1):
        filename = Path(img_path).name
        print(f"[{idx}/{MAX_IMAGES}] Processing: {filename}")
        
        results, processing_img = smart_detect(model, img_path)
        if results is None: 
            continue
            
        annotated_img = results[0].plot(labels=False)
        
        # Prepare boxes for IOU filtering
        raw_boxes = []
        for box in results[0].boxes:
            coords = list(map(int, box.xyxy[0].tolist()))
            conf = box.conf[0].item()
            raw_boxes.append({'box': coords, 'conf': conf})
        
        if not raw_boxes:
            print("  ✗ No plates found")
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"res_{idx}_{filename}"), annotated_img)
            continue
            
        filtered_boxes = filter_overlapping_boxes(raw_boxes, iou_threshold=IOU_THRESHOLD)
        detection_count += 1
        print(f"  ✓ {len(filtered_boxes)} unique plate(s) found")
        
        for i, item in enumerate(filtered_boxes):
            x1, y1, x2, y2 = item['box']
            plate_crop = processing_img[y1:y2, x1:x2]
            
            if plate_crop.size == 0: continue
            
            text, ocr_conf = recognize_plate_paddleocr(reader, plate_crop)
            
            if len(text) >= 2:
                print(f"    Plate {i+1}: '{text}' (Conf: {ocr_conf:.2f})")
                
                # Standardized Font Drawing
                font_scale = 1.2
                thickness = 2
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Draw background box
                cv2.rectangle(annotated_img, (x1, y1 - th - 10), (x1 + tw, y1), (0,0,0), -1)
                # Draw text
                cv2.putText(annotated_img, text, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                
                # Save crop
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"plate_{idx}_{i}_{text}.jpg"), plate_crop)
            else:
                print(f"    Plate {i+1}: Unclear/filtered")
        
        # Save full annotated image
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"res_{idx}_{filename}"), annotated_img)

    print(f"\nDone! Processed {detection_count} detections. Results: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
