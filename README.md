# ALPR-System
Automatic License Plate Recognition for Malaysian vehicles using YOLOv11 and PaddleOCR.
Check our [One page poster](https://www.canva.com/design/DAG-ZI-kBQ4/FgagOW2d4HOLOx88l2l0Ww/edit?utm_content=DAG-ZI-kBQ4&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

# ALPR System: Automatic License Plate Recognition

This project implements a robust Automatic License Plate Recognition (ALPR) system using YOLOv11 for detection and PaddleOCR for character recognition. It features an optimized pipeline designed to handle challenging lighting conditions and overlapping detections.

---

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd ALPR

# 2. Install dependencies
pip install opencv-python numpy ultralytics paddlepaddle paddleocr

# 3. Place your model file (e.g., best (8).pt) in the root directory

# 4. Add test images to random_images/ folder

# 5. Run the optimized system
python Optimized.py
```

---

## 🛠 Setup Instructions

### 1. Prerequisites
- **Python 3.8 - 3.12** is recommended.
- **Conda** (Optional but recommended for environment management).

### 2. Install Dependencies
Open your terminal and run the following commands:

#### Core Graphics & Math
```bash
pip install opencv-python numpy
```

#### Object Detection (YOLOv11)
```bash
pip install ultralytics
```

#### Optical Character Recognition (PaddleOCR)
For **CPU** users:
```bash
pip install paddlepaddle paddleocr
```
For **GPU (CUDA)** users (Ensure you have CUDA installed):
```bash
pip install paddlepaddle-gpu paddleocr
```

### 3. Model Files
Place your trained YOLO weights in the project root directory:
- Default expected: `best (8).pt`
- To use a different model version, update the `VERSION` variable at the top of `Optimized.py` or `Baseline.py`

**Model Training Note**: If you need to train your own model, the YOLO weights were trained on a Malaysian license plate dataset using YOLOv11. You can use Roboflow or similar platforms to prepare your dataset.

---

## 🚀 How to Run the ALPR System

The system consists of two main scripts: `Optimized.py` (the production-ready version) and `Baseline.py` (for performance comparison).

### Running the Optimized System
1. **Automatic Paths**: The scripts use relative paths. As long as your model file (`best (8).pt`) and `random_images` folder are in the same directory as the script, it will work out-of-the-box.
2. **Configuration**: Modify settings at the top of `Optimized.py`:
   - `VERSION`: The number corresponding to your model file (e.g., `8` for `best (8).pt`).
   - `MAX_IMAGES`: Limit the number of images to process (default: 100).
3. Run the script:
   ```bash
   python Optimized.py
   ```

### Running the Baseline System
To establish a performance baseline without optimizations:
```bash
python Baseline.py
```

### Viewing Comparison Results
After running both systems, launch the interactive comparison viewer:
```bash
python compare_results_folders.py
```

This will:
- Load matching result images from both `ocr_result8` and `ocr_result8_baseline` folders
- Display them side-by-side with detected plate text
- Allow navigation with Next/Previous buttons
- Highlight differences between baseline and optimized results

---

## 🧠 Methodology & Innovation

Our system employs a multi-stage pipeline to transform raw images into structured plate numbers. The core innovation is the **Recursive Selection & Enhancement Logic** which allows the system to recover detections in challenging conditions.

### 1. Advanced Detection (The "Smart Detect" Logic)
Unlike standard detection which uses a fixed confidence threshold, our system uses a **Recursive Detection Strategy** that prioritizes quality over raw noise:
- **Phase 1: Standard (0.20)**: Initial attempt on the original image.
- **Phase 2: Low-Conf (0.15)**: A quick second pass on the original image.
- **Phase 3: Image Enhancement**: Applies digital filters to fix lighting, then retries.
- **Phase 4: Resolution Scaling**: Re-scans at different resolutions (480p, 1280p).
- **Phase 5: Last Resort (0.10)**: Drops to a very low threshold if all else fails.

### 2. Image Enhancement Deep Dive 
To handle environmental challenges, we use specific mathematical filters:
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Equalizes contrast in small tiles to prevent over-amplifying noise. It helps "see" plates in heavy shadows or bright glare.
- **Gamma Correction**: Adjusts luminance non-linearly. We use it to brighten dark night-time shots (`γ > 1.0`) or darken overexposed daytime shots (`γ < 1.0`).
- **Sharpening**: Highlights the edges between characters and the plate background, making it easier for the OCR to distinguish similar letters.

### 3. Algorithmic Logic: NMS & IOU Filtering
To handle "double detections" or overlapping boxes, we implemented **Non-Maximum Suppression (NMS)** using **IOU (Intersection over Union)**:
- **IOU** calculates how much two boxes overlap.
- If two boxes overlap significantly (>30%), the system keeps only the one with the highest confidence.
- This ensures each vehicle is processed only once, preventing redundant results.

### 4. Component Breakdown

| Tier | Component | Function |
| :--- | :--- | :--- |
| **Detection** | **YOLOv11** | The "eyes" that find the license plate in the image. |
| **Logic** | **IOU Filter** | The "brain" that cleans up redundant detections. |
| **Enhancement** | **Smart Retries** | The "troubleshooter" that applies CLAHE/Gamma if a plate is missed. |
| **Recognition** | **PaddleOCR** | The "reader" that translates the picture into digital text. |
| **Cleaning** | **Post-Processor** | The "editor" that fixes common errors (e.g., 'O' to 'Q'). |

### 5. Character Recognition & Post-Processing
We utilize **PaddleOCR** for high-accuracy character reading. We found that providing the **raw crop** (BGR) yielded better results than aggressive binary thresholding. Finally, we apply character mapping to correct common OCR errors and filter alphanumeric strings.

---

## 📁 Project Structure
```
ALPR/
├── Optimized.py              # Main ALPR system with all optimizations
├── Baseline.py               # Simple detection + OCR for comparison
├── compare_results_folders.py # Interactive result comparison viewer
├── best (8).pt              # YOLO model weights (place here)
```

---

## 📊 Output

After running the scripts, you'll find:
- **Annotated Images**: Full-size images with detected plates and recognized text overlaid
- **Cropped Plates**: Individual plate crops saved with recognized text in filename
- **Console Logs**: Real-time detection and recognition status for each image

Example output filename: `plate_1_0_ABC1234.jpg` (Image #1, Plate #0, Text: ABC1234)

---

## 🎯 Use Cases

- **Traffic Monitoring**: Automated vehicle identification at checkpoints
- **Parking Management**: Entry/exit tracking in parking facilities
- **Access Control**: Gate automation based on license plate recognition
- **Law Enforcement**: Speed camera and violation tracking systems

---

## 🔬 Performance Notes

The **Optimized** system typically achieves:
- Higher detection rates on challenging images (low light, motion blur, extreme angles)
- Better handling of partially occluded or dirty plates
- More robust performance across varying resolutions

The **Baseline** provides a reference point to measure improvement and is useful for understanding the impact of each optimization phase.

---

## 🙏 Acknowledgments

- **YOLOv11**: Ultralytics for the state-of-the-art object detection framework
- **PaddleOCR**: PaddlePaddle team for the robust OCR engine
- **Dataset**: Malaysian license plate dataset (specify source if applicable)
