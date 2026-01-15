"""
ALPR Results Comparison Tool
Creates side-by-side comparisons of baseline and optimized result folders.
"""

import os
import cv2
import numpy as np
import glob
import re
from pathlib import Path

"""
ALPR Results Comparison Tool - Instant Version
Shows side-by-side comparisons immediately.
"""

import os
import cv2
import numpy as np
import glob
import re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# --- CONFIGURATION ---
VERSION = 8
BASELINE_FOLDER = f"ocr_result{VERSION}_baseline"
OPTIMIZED_FOLDER = f"ocr_result{VERSION}"

def extract_image_number(filename):
    match = re.search(r'res_(\d+)_', filename)
    return int(match.group(1)) if match else None

def get_plate_texts(folder, img_num):
    pattern = os.path.join(folder, f"plate_{img_num}_*")
    matches = glob.glob(pattern)
    plates = []
    for m in matches:
        stem = Path(m).stem
        parts = stem.split('_')
        if len(parts) >= 3:
            text = '_'.join(parts[2:])
            if text and text != 'ERROR':
                plates.append(text)
    return plates if plates else ["None"]

def create_comparison_frame(b_path, o_path, img_num):
    """Load and stitch images in real-time with standardized scaling."""
    b_img = cv2.imread(b_path)
    o_img = cv2.imread(o_path)
    if b_img is None or o_img is None: 
        return None

    # Standardize UI scale: Fix total width to 1600px
    UI_WIDTH = 1600
    IMG_WIDTH = (UI_WIDTH - 60) // 2  # Space for margins/gap
    
    h1, w1 = b_img.shape[:2]
    h2, w2 = o_img.shape[:2]
    
    # Resize both to same width while keeping aspect ratio
    b_h = int(h1 * (IMG_WIDTH / w1))
    b_img = cv2.resize(b_img, (IMG_WIDTH, b_h))
    
    o_h = int(h2 * (IMG_WIDTH / w2))
    o_img = cv2.resize(o_img, (IMG_WIDTH, o_h))
    
    # Common canvas height based on tallest image
    target_h = max(b_h, o_h)
    
    # Create canvas
    canvas = np.ones((target_h + 200, UI_WIDTH, 3), dtype=np.uint8) * 40
    
    # Center images vertically if heights differ
    b_y = 100 + (target_h - b_h) // 2
    o_y = 100 + (target_h - o_h) // 2
    
    canvas[b_y:b_y+b_h, 20:20+IMG_WIDTH] = b_img
    canvas[o_y:o_y+o_h, IMG_WIDTH+40:IMG_WIDTH+40+IMG_WIDTH] = o_img
    
    # Metadata
    b_plates = get_plate_texts(BASELINE_FOLDER, img_num)
    o_plates = get_plate_texts(OPTIMIZED_FOLDER, img_num)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Header - Fixed Sizes
    cv2.putText(canvas, f"IMG #{img_num}", (20, 45), font, 1.0, (255, 255, 255), 2)
    cv2.putText(canvas, "BASELINE", (20, 85), font, 0.7, (100, 100, 255), 2)
    cv2.putText(canvas, "OPTIMIZED", (IMG_WIDTH+40, 85), font, 0.7, (100, 255, 100), 2)
    
    # Footer - Fixed Sizes
    y_footer = target_h + 140
    cv2.putText(canvas, f"Plates: {', '.join(b_plates)}", (20, y_footer), font, 0.6, (100, 100, 255), 2)
    cv2.putText(canvas, f"Plates: {', '.join(o_plates)}", (IMG_WIDTH+40, y_footer), font, 0.6, (100, 255, 100), 2)
    
    status = "MATCH" if b_plates == o_plates else "DIFFERENCE"
    color = (100, 255, 100) if status == "MATCH" else (0, 255, 255)
    cv2.putText(canvas, f"Status: {status}", (20, y_footer + 40), font, 0.6, color, 2)
    
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

def main():
    b_files = glob.glob(os.path.join(BASELINE_FOLDER, "res_*.jpg"))
    o_files = glob.glob(os.path.join(OPTIMIZED_FOLDER, "res_*.jpg"))
    
    if not b_files or not o_files:
        print("Error: Could not find results in one or both folders.")
        return

    b_dict = {extract_image_number(f): f for f in b_files}
    o_dict = {extract_image_number(f): f for f in o_files}
    common_ids = sorted(set(b_dict.keys()) & set(o_dict.keys()))
    
    if not common_ids:
        print("No matching image numbers found.")
        return

    state = {'idx': 0}
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.15)
    
    def update():
        ax.clear()
        cid = common_ids[state['idx']]
        frame = create_comparison_frame(b_dict[cid], o_dict[cid], cid)
        if frame is not None:
            ax.imshow(frame)
            ax.set_title(f"Comparison {state['idx']+1}/{len(common_ids)}")
        ax.axis('off')
        fig.canvas.draw()

    def next_img(e=None):
        state['idx'] = (state['idx'] + 1) % len(common_ids)
        update()
        
    def prev_img(e=None):
        state['idx'] = (state['idx'] - 1) % len(common_ids)
        update()

    ax_prev = plt.axes([0.4, 0.05, 0.08, 0.05])
    ax_next = plt.axes([0.5, 0.05, 0.08, 0.05])
    btn_p, btn_n = Button(ax_prev, '< Prev'), Button(ax_next, 'Next >')
    btn_p.on_clicked(prev_img), btn_n.on_clicked(next_img)
    
    update()
    plt.show()

if __name__ == "__main__":
    main()
