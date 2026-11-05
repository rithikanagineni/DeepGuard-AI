# 🛡️ DeepGuard AI
### AI-Based Deepfake Detection System

**Detecting Digital Deception. Building Digital Trust.**

DeepGuard AI is an AI-assisted media analysis platform designed to identify potentially manipulated images and videos using computer vision, deep learning, and machine learning techniques.

The system uses **MobileNetV2 for feature extraction** and an **SVM classifier** to analyze facial content and classify it as Real or Fake. It also provides a user-friendly interface for media analysis, visualization of results, analytics, and downloadable reports.

---

## 📌 Project Overview

With the rapid development of Artificial Intelligence, manipulated images and videos, commonly known as deepfakes, are becoming increasingly realistic.

These manipulated media files can be misused to spread misinformation, damage reputations, violate privacy, and create digital security concerns.

DeepGuard AI aims to provide an accessible platform that analyzes uploaded media and helps users identify potentially manipulated content.

### 🎯 Project Objectives

- Detect potentially manipulated images and videos.
- Extract and preprocess representative video frames.
- Identify facial features using computer vision techniques.
- Extract meaningful visual features using MobileNetV2.
- Classify extracted features using an SVM model.
- Display understandable detection results and analysis statistics.
- Maintain analysis history and provide downloadable reports.

---

## ✨ Features

### 1. Interactive Dashboard
- Provides an overview of the DeepGuard AI application.
- Acts as the main entry point for users.
- Provides access to different application modules.

### 2. Media Analyzer
- Supports uploading image and video files.
- Allows users to preview uploaded media.
- Initiates the AI-based detection process.
- Processes media through the detection pipeline.

### 3. AI-Based Deepfake Detection
- Uses OpenCV Haar Cascade for face detection.
- Extracts facial regions from sampled video frames.
- Uses MobileNetV2 for deep feature extraction.
- Applies an RBF-kernel SVM classifier.
- Generates Real/Fake assessments.

### 4. Detection Results
- Displays the classification assessment.
- Presents the model score.
- Provides video-related analysis statistics.
- Combines sampled frame-level outputs into a video-level assessment.

### 5. Analytics Dashboard
- Displays analysis totals.
- Shows Real/Fake classification distribution.
- Provides access to recent analysis history.
- Helps users understand the analysis activity.

### 6. Methodology Section
- Explains the dataset used.
- Describes preprocessing techniques.
- Presents the model architecture and detection workflow.

### 7. Downloadable Reports
- Generates a downloadable summary of an analysis.
- Helps users retain and review detection information.

---

## 🏗️ System Architecture

DeepGuard AI follows a sequential media-processing and classification pipeline.

```text
       User Uploads Media
               |
               v
       Media Preprocessing
               |
               v
        Video Frame Extraction
               |
               v
         Face Detection
        (Haar Cascade)
               |
               v
       Image Preprocessing
      (RGB, Resize, Normalize)
               |
               v
         MobileNetV2
       Feature Extraction
               |
               v
         SVM Classifier
          (RBF Kernel)
               |
               v
       Frame-Level Predictions
               |
               v
       Video-Level Aggregation
               |
               v
        Detection Result
          REAL / FAKE
               |
               v
       Analytics and Reports
```

---

## 🧠 Model and Algorithm

DeepGuard AI combines a pretrained convolutional neural network with a traditional machine learning classifier.

### MobileNetV2

- Used as a pretrained CNN feature extractor.
- Extracts meaningful visual representations from facial images.
- Global average pooling produces a 1,280-dimensional feature vector.
- Helps avoid training a large CNN from scratch.

### Support Vector Machine (SVM)

- Uses an RBF (Radial Basis Function) kernel.
- Receives the features extracted by MobileNetV2.
- Performs the final Real/Fake classification.

### Video-Level Aggregation

The sampled frame-level outputs are combined to generate a video-level assessment.

### Algorithm Workflow

1. Accept an image or video as input.
2. Extract representative frames from the video.
3. Detect the largest face in each sampled frame.
4. Convert detected face crops into RGB format.
5. Resize images to 224 × 224 pixels.
6. Normalize the image data.
7. Extract features using MobileNetV2.
8. Pass the extracted features to the trained SVM classifier.
9. Combine frame-level predictions for video assessment.
10. Display the result through the application.

---

## 📂 Dataset

**Dataset:** FaceForensics++

**Selected subset:** c40 compression subset, Deepfakes category.

### Dataset Preparation

| Description | Quantity |
|---|---:|
| Original videos | 50 |
| Manipulated videos | 50 |
| Frames sampled per video | 10 |
| Processed Real frames | 497 |
| Processed Fake frames | 497 |
| Total processed frames | 994 |
| Image dimensions | 224 × 224 |

### Preprocessing Techniques

- Evenly spaced frame sampling.
- Face detection using OpenCV Haar Cascade.
- Largest-face selection.
- RGB conversion.
- Image resizing.
- Normalization for model input.

Dataset access and usage should follow the applicable FaceForensics++ terms and conditions.

---

## 🛠️ Tools and Technologies

| Category | Technology |
|---|---|
| Programming Language | Python |
| Frontend | HTML, CSS, JavaScript |
| Backend Framework | FastAPI |
| Computer Vision | OpenCV |
| Deep Learning | MobileNetV2 |
| Machine Learning | Support Vector Machine |
| Dataset | FaceForensics++ |
| Classification Kernel | RBF Kernel |

**Additional libraries:** Include the exact dependencies used in your implementation in `requirements.txt`.

---

## 📊 Experimental Results

The current model was evaluated using a held-out video-level test set.

### Test Dataset

- Total test videos: 20
- Real videos: 10
- Fake videos: 10

### Performance Metrics

| Metric | Result |
|---|---:|
| Accuracy | 70% |
| Precision | 70% |
| Recall | 70% |
| F1-Score | 70% |

### Confusion Matrix

| Actual / Predicted | Real | Fake |
|---|---:|---:|
| Real | 7 | 3 |
| Fake | 3 | 7 |

These results represent the current held-out test set only. They do not guarantee equivalent performance on unseen datasets or real-world media.

---

## 🖥️ Application Modules

| Module | Description |
|---|---|
| Dashboard | Application overview and navigation |
| Media Analyzer | Media upload, preview and analysis |
| Detection Result | Classification assessment, score and video statistics |
| Analytics | Analysis totals, classification distribution and recent history |
| Methodology | Dataset, preprocessing and model explanation |
| Report | Downloadable analysis summary |

---

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.x
- pip package manager
- Git
- A code editor such as Visual Studio Code

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the Project

```bash
cd DeepGuard-AI
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Configure the Model and Dataset

Ensure the trained MobileNetV2/SVM artifacts and any required configuration or dataset paths are available in the locations expected by the application.

### Step 6: Run the Application

If your FastAPI application entry point is `main.py`:

```bash
uvicorn main:app --reload
```

Open the local application using the frontend address configured in your project.

**Note:** Update the entry-point command and folder paths according to your actual repository structure.

---

## 📁 Project Structure

```text
DeepGuard-AI/
│
├── frontend/
│   ├── HTML files
│   ├── CSS files
│   └── JavaScript files
│
├── backend/
│   ├── API files
│   └── Application logic
│
├── models/
│   └── Trained model files
│
├── preprocessing/
│   └── Frame and face processing
│
├── dataset/
│   └── Dataset-related files
│
├── requirements.txt
├── README.md
└── .gitignore
```

*Illustrative structure: replace these entries with the actual directories and filenames in your GitHub repository.*

---

## ⚠️ Limitations and Responsible Use

- The current dataset contains a limited selection of original and manipulated videos.
- Performance may vary with compression, lighting, resolution, pose, and unseen manipulation methods.
- Face detection may fail when faces are small, obscured, or outside the expected view.
- A model score should not automatically be interpreted as a calibrated probability.
- The system is intended for AI-assisted screening and should not be treated as conclusive forensic evidence.

Users should treat uncertain or low-quality results as requiring human review.

---

## 🚀 Future Enhancements

- **Explainable Detection:** Show evidence behind an assessment using measured signals.
- **Frame-Level Investigation:** Add an interactive video timeline with timestamps and frame-level scores.
- **Needs Review Category:** Introduce an inconclusive category for borderline or insufficient evidence.
- **Forensic Reports:** Include media details, analysis summaries, limitations, and model information.
- **Model Transparency:** Present evaluation metrics, confusion matrices, and dataset scope.
- **Improved Generalization:** Train and validate with larger and more diverse datasets.

---

## 👥 Team Members

**Department:** Computer Science and Engineering – Artificial Intelligence and Machine Learning

**Project Guide:** Mr. M. Gopal

**Team Members:**
- Nagineni Rithika
- Penuboku Mrudhula
- N. Devi Bhaskar
- P. Praneeth

---

## 📚 References

1. Rössler et al., *FaceForensics++: Learning to Detect Manipulated Facial Images*, ICCV 2019.
2. Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018.
3. OpenCV Official Documentation.
4. Scikit-learn Official Documentation – Support Vector Machines.
5. FastAPI Official Documentation.
6. FaceForensics++ Dataset.

---

## 📌 Project Status

**Current Stage:** Working AI-assisted media analysis pipeline with an implemented application interface and preliminary evaluation results.

**Project:** DeepGuard AI – AI-Based Deepfake Detection System

**Goal:** Detecting Digital Deception. Building Digital Trust.

---

⭐ If you find this project interesting, consider starring the repository!