# ALPR-System
Automatic License Plate Recognition for Malaysian vehicles using YOLOv11 and PaddleOCR.
Check our [One page poster](https://www.canva.com/design/DAG-ZI-kBQ4/FgagOW2d4HOLOx88l2l0Ww/edit?utm_content=DAG-ZI-kBQ4&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

# ALPR System: Automatic License Plate Recognition

This project implements a robust Automatic License Plate Recognition (ALPR) system using YOLOv11 for detection and PaddleOCR for character recognition. It features an optimized pipeline designed to handle challenging lighting conditions and overlapping detections.

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
Ensure your trained YOLO weights (e.g., `best (8).pt`) are located in the project root directory as specified in the scripts.

---

## 🚀 How to Run the ALPR System

The system consists of two main scripts: `Optimized.py` (the production-ready version) and `Baseline.py` (for performance comparison).

### Running the Optimized System
1. Open `Optimized.py`.
2. Configure the paths at the top of the script:
   - `images_folder`: Directory containing your input images.
   - `output_folder`: Where results and cropped plates will be saved.
   - `version`: The number corresponding to your model file (e.g., `8` for `best (8).pt`).
3. Run the script:
   ```bash
   python Optimized.py
   ```

### Comparing with Baseline
To see the improvement provided by the optimization logic:
```bash
python Baseline.py
```
After running both, you can use `compare_results_folders.py` to analyze the difference in accuracy.

---

## 🧠 Methodology Explanation

Our system employs a multi-stage pipeline to transform raw images into structured plate numbers.

### 1. Advanced Detection (The "Smart Detect" Logic)
Unlike standard detection which uses a fixed confidence threshold, our system uses a **Recursive Detection Strategy**:
- **Initial Pass**: Detects plates at a standard confidence (0.20).
- **Confidence Retry**: If no plate is found, the system progressively lowers the confidence threshold (0.15, 0.10) to catch faint or distant detections.
- **Enhancement Fallback**: If still no plate is found, it automatically applies image enhancements (**CLAHE**, **Gamma Correction**, and **Sharpening**) and retries.
- **Resolution Scaling**: Finally, it attempts detection at multiple resolutions (480p and 1280p) to capture plates that are either too small or too grainy at native resolution.

### 2. Non-Maximum Suppression (NMS) Filtering
To handle "double detections" or slightly overlapping bounding boxes from the ensemble models, we implemented a custom **IOU (Intersection over Union)** filter. This ensures that even if multiple models detect the same plate, only the most confident detection is passed to the OCR stage.

### 3. Character Recognition (PaddleOCR)
We utilize **PaddleOCR** for its high accuracy in reading alphanumeric characters. The system crops the detected plate and passes it to the OCR engine. 
- **Note**: Extensive testing showed that for PaddleOCR, providing the **raw crop** (BGR) often yielded better results than applying aggressive binary thresholding, as the engine's internal preprocessing is highly optimized.

### 4. Post-Processing & Text Cleaning
To minimize "hallucinated" characters and improve data quality, the system applies:
- **Character Mapping**: Automatically corrects common OCR errors (e.g., mapping 'O' to 'Q' for certain plate formats).
- **Alphanumeric Filtering**: Removes special characters and spaces.
- **Contextual Filtering**: Discards results that do not contain at least one digit, significantly reducing false positives from background noise.

---

## 📁 Project Structure
- `Optimized.py`: Main execution script with retry logic and enhancements.
- `Baseline.py`: Standard detection + OCR pipeline without enhancements.
- `compare_results_folders.py`: Utility to compare accuracy between two result sets.
- `random_images/`: Directory for input testing images.
- `ocr_resultX/`: Generated output containing annotated images and cropped plates.
