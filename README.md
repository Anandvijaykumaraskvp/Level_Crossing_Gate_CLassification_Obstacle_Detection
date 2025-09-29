# 🚦 Level Crossing Gate Classification & Obstacle Detection

This project focuses on **automatic detection and classification of railway level crossing gates and obstacles** using deep learning models.  
It combines **image classification** (to determine gate open/closed states) and **object detection** (to detect obstacles such as vehicles, pedestrians, and signals).  
A **GUI app** is also provided for demonstration purposes.

---

## 📂 Project Structure

- **Classification_Railway_Gate_Crossings.ipynb**  
  Jupyter Notebook for training & evaluating YOLOv8/MobieNetv2 classification model (Gate Opened / Gate Closed).

- **Object-Detection_Railway_Crossing (1).ipynb**  
  Jupyter Notebook for YOLOv8/Faster R-CNN based object detection (cars, trucks, pedestrians, signals, gates, etc.).

- **GUI_App.py**  
  GUI demo to upload images, detect gates/obstacles, classify gate status, and validate results.

- **Demo_Video_Level_Crossing_Classification and detection.7z**  
  A short demo video showing classification and detection results.

- **classification_dataset.zip**  
  Dataset for classification (organized into `Gates Closed` and `Gates Opened`).

- **detection_dataset_split.zip**  
  Dataset for object detection (organized in YOLO format: `images/train`, `images/val`, `labels/train`, `labels/val`).
  
- **Please update dataset paths inside the notebooks (.ipynb) or scripts before execution.**
 
- **README.md**  
  Project documentation (you are reading this).

---

## 🛠️ Installation

Clone this repository:
```bash
git clone https://github.com/Anandvijaykumarskvp/Level_Crossing_Gate_Classification_Obstacle_Detection.git
cd Level_Crossing_Gate_Classification_Obstacle_Detection
