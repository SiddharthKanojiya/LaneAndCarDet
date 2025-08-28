

🚗 Lane and Car Detection
📌 Overview

This project implements a computer vision system capable of detecting and tracking both lane lines and vehicles on the road in real time using Python and OpenCV.
It applies a combination of image processing, edge detection, Hough Transform, and machine learning models (HOG + SVM / YOLO) to process dash-cam video footage.

🎯 Goals

Detect lane lines using image processing techniques.

Detect and track vehicles using object detection algorithms.

Ensure detection under varying lighting and road conditions.

Integrate lane and vehicle detection into a real-time pipeline.

🛠️ Technologies Used

Python – Core programming language

OpenCV – Image processing and video analysis

NumPy & Matplotlib – Numerical operations & visualization

scikit-image / scikit-learn – Feature extraction & classification (HOG, SVM)

MoviePy – Video processing

TensorFlow / YOLO (optional) – Deep learning-based vehicle detection

📂 Project Structure
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

git clone https://github.com/username/CarAndLaneDetection.git
cd CarAndLaneDetection


Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt

▶️ Usage
Lane Detection
python src/lane_detection.py --input data/solidWhiteRight.mp4 --output outputs/lane_detected.mp4

Car Detection
python src/car_detection.py --input data/project_video.mp4 --output outputs/cars_detected.mp4

Combined Lane + Car Detection
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

OpenCV Docs: https://docs.opencv.org/

Udacity Self-Driving Car Nanodegree Resources

YOLO Object Detection: https://pjreddie.com/darknet/yolo/

👨‍💻 Authors

Siddharth Kanojiya

Jai Mehta

Link to Bootcamp notebook: https://colab.research.google.com/drive/1mASkzVB-dkqhLGeNVEBkYULWtgQRrBbu?usp=sharing

Link to Sample Notebook:https://drive.google.com/file/d/13Hw9XVC7IV1SNxn8aIYSqXGnjeteELVi/view?usp=sharing



