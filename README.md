

# 🚗 Lane and Car Detection  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)  
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)  
![License](https://img.shields.io/badge/License-MIT-yellow.svg)  

---

## 📌 Overview  
This project implements a computer vision system capable of **detecting and tracking both lane lines and vehicles on the road in real time** using Python and OpenCV.  

It applies a combination of **image processing, edge detection, Hough Transform, and machine learning models (HOG + SVM / YOLO)** to process dash-cam video footage.  

---

## 🎯 Goals  
- ✅ Detect lane lines using image processing techniques.  
- ✅ Detect and track vehicles using object detection algorithms.  
- ✅ Ensure detection under varying lighting and road conditions.  
- ✅ Integrate lane and vehicle detection into a real-time pipeline.  

---

## 🛠️ Technologies Used  
- **Python** – Core programming language  
- **OpenCV** – Image processing and video analysis  
- **NumPy & Matplotlib** – Numerical operations & visualization  
- **scikit-image / scikit-learn** – Feature extraction & classification (HOG, SVM)  
- **MoviePy** – Video processing  
- **TensorFlow / YOLO (optional)** – Deep learning-based vehicle detection  

---

## 📂 Project Structure  
```bash
CarAndLaneDetection/
│── data/                 # Input videos, test images, datasets
│── outputs/              # Results (processed videos/images)
│── src/                  # Source code
│   ├── lane_detection.py
│   ├── car_detection.py
│   ├── pipeline.py       # Integrated detection pipeline
│── models/               # Trained/Pre-trained models
│── requirements.txt      # Dependencies
│── README.md             # Documentation
⚙️ Installation
Clone the repository:

bash

git clone https://github.com/SiddharthKanojiya/CarAndLaneDetection.git
cd CarAndLaneDetection

▶️ Usage
Run Lane Detection
bash
Copy code
python src/lane_detection.py --input data/solidWhiteRight.mp4 --output outputs/lane_detected.mp4
Run Car Detection
bash
Copy code
python src/car_detection.py --input data/project_video.mp4 --output outputs/cars_detected.mp4
Run Combined Lane + Car Detection
bash
Copy code
python src/pipeline.py --input data/challenge.mp4 --output outputs/final_output.mp4
📊 Results
Lane Detection Example

Car Detection Example

Combined Detection (Cars + Lanes)

🚀 Challenges & Solutions
Lighting Variations: Used adaptive thresholding and dynamic parameter tuning.

Curved / Occluded Roads: Improved Hough Transform + refined lane averaging.

Video Jitter / Fast Vehicles: Applied smoothing and temporal tracking.

✅ Results & Conclusion
Robust lane and vehicle detection across multiple test videos.

Achieved ~99% accuracy on trained SVM vehicle classifier.

Demonstrates potential use in ADAS (Advanced Driver Assistance Systems) and autonomous driving.

📚 References
OpenCV Documentation

Udacity Self-Driving Car Nanodegree Resources

YOLO Object Detection

👨‍💻 Authors
Siddharth Kanojiya

Link to Notebook[Car and Lane Detection](https://drive.google.com/file/d/1BwXfpbCn5McQSLz85EAcJlt4th9kTXMu/view?usp=sharing)


Link to Sample Notebook[Car and Lane Detection](https://drive.google.com/file/d/13Hw9XVC7IV1SNxn8aIYSqXGnjeteELVi/view?usp=sharing)


