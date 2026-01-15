import os
import cv2
import numpy as np
import glob
import logging
import warnings
from pathlib import Path
from ultralytics import YOLO
from paddleocr import PaddleOCR

# --- CONFIGURATION (BASELINE) ---
VERSION = 8
MAX_IMAGES = 100
CONF_BASE = 0.20

# Paths
MODEL_PATH = f"best ({VERSION}).pt"
IMAGES_FOLDER = "random_images"
OUTPUT_FOLDER = f"ocr_result{VERSION}_baseline"

# Suppress PaddleOCR verbose logging
os.environ['FLAGS_logtostderr'] = '0'
os.environ['FLAGS_stderrthreshold'] = '3'
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddleocr").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def clean_plate_text(text):
    """Clean and normalize OCR text - Baseline Version."""
    text = text.replace(' ', '').upper()
    cleaned = ''.join([c for c in text if c.isalnum()])
    cleaned = cleaned.replace('O', 'Q')
    return cleaned

def recognize_plate_paddleocr(reader, plate_crop, temp_path="temp_plate_baseline.jpg"):
    """Run OCR on the plate crop - Baseline (No retry/enhancement)."""
    try:
        cv2.imwrite(temp_path, plate_crop)
        result = reader.predict(temp_path)
        
        if result and isinstance(result, list) and len(result) > 0:
            first_result = result[0]
            if 'rec_texts' in first_result and 'rec_scores' in first_result:
                texts = first_result['rec_texts']
                scores = first_result['rec_scores']
                
                combined_text = ''.join(texts)
                cleaned = clean_plate_text(combined_text)
                avg_conf = sum(scores) / len(scores) if scores else 0.0
                
                return cleaned, avg_conf
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
    
    print(f"Found {len(all_images)} images. Processing top {MAX_IMAGES} (BASELINE)...")
    print("=" * 60)
    
    detection_count = 0
    for idx, img_path in enumerate(all_images[:MAX_IMAGES], 1):
        filename = Path(img_path).name
        print(f"[{idx}/{MAX_IMAGES}] Processing: {filename}")
        
        original_img = cv2.imread(img_path)
        if original_img is None: 
            continue
            
        # Baseline: Single pass detection
        results = model.predict(original_img, conf=CONF_BASE, verbose=False)
        annotated_img = results[0].plot(labels=False)
        boxes = results[0].boxes
        
        if not boxes:
            print("  ✗ No plates found")
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"res_{idx}_{filename}"), annotated_img)
            continue
            
        detection_count += 1
        print(f"  ✓ {len(boxes)} plate(s) found")
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            plate_crop = original_img[y1:y2, x1:x2]
            
            if plate_crop.size == 0: continue
            
            text, ocr_conf = recognize_plate_paddleocr(reader, plate_crop)
            
            if len(text) >= 2:
                print(f"    Plate {i+1}: '{text}' (Conf: {ocr_conf:.2f})")
                
                # Standardized Font Drawing
                font_scale = 1.2
                thickness = 2
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Draw on result
                cv2.rectangle(annotated_img, (x1, y1 - th - 10), (x1 + tw, y1), (0,0,0), -1)
                cv2.putText(annotated_img, text, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                
                # Save crop
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"plate_{idx}_{i}_{text}.jpg"), plate_crop)
            else:
                print(f"    Plate {i+1}: Unclear/filtered")
        
        # Save result
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"res_{idx}_{filename}"), annotated_img)

    print(f"\nDone! Baseline results: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
