# 🛡️ DeepGuard AI
### AI-Based Deepfake Detection System

**Detecting Digital Deception. Building Digital Trust.**

DeepGuard AI is an AI-assisted media analysis platform designed to screen images and videos for potentially manipulated facial content using computer vision, deep learning, and machine learning techniques.

The system uses **MobileNetV2 for feature extraction** and an **RBF-kernel SVM classifier**. Results are reported as **Real**, **Fake**, or **Needs Review**; they are an automated screening assessment, not proof of authenticity or manipulation.

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
- Explain which implemented signals were available for each result without presenting unavailable measurements as evidence.
- Maintain browser-local analysis history and provide downloadable forensic reports.

---

## ✨ Features

### 1. Interactive Dashboard
- Provides an overview of the DeepGuard AI application.
- Acts as the main entry point for users.
- Provides access to different application modules.

### 2. Media Analyzer
- Supports uploading image and video files and previews them before analysis.
- Displays available file details, image/video dimensions, and video duration.
- Calculates a SHA-256 fingerprint in the browser for files up to 100 MB; this is a file identifier, not a manipulation detector.
- Initiates the AI-based detection process.
- Processes media through the detection pipeline.

### 3. AI-Based Deepfake Detection
- Uses OpenCV Haar Cascade for face detection.
- Extracts facial regions from sampled video frames.
- Uses MobileNetV2 for deep feature extraction.
- Applies an RBF-kernel SVM classifier.
- Generates Real, Fake, or Needs Review assessments.
- Uses the image result directly and combines usable sampled-frame scores for video results.
- Reviews at most 20 evenly spaced video frames.

### 4. Detection Results
- Displays the classification assessment.
- Presents the bounded Fake-class model score; it is derived from the SVM decision function and is not a calibrated probability.
- Provides available face-detection, frame, and file statistics.
- Shows a “Why this result?” evidence summary. Signals that are not calculated are marked “Not analyzed.”
- Uses Needs Review for image scores from 0.45 through 0.55 inclusive, images without a detectable face, and videos with fewer than three usable sampled face scores.

### 5. Video Frame Investigation
- Displays a clickable timeline for sampled frames, with timestamp, face-detection status, frame assessment, score, and the implementation's suspicion label.
- Shows a selected sampled frame and its face crop when available.
- The suspicion label is a simple mapping from the frame assessment, not a separate forensic model.

### 6. Media Forensics and Reports
- Shows available file and analysis details, including dimensions, duration/FPS for videos, frame counts, and model scores.
- Generates a downloadable HTML forensic report with the result, analysis summary, sampled-frame table, model information, and limitations.
- Embedded EXIF metadata, provenance, lighting/shadows, blending boundaries, and pixel/texture artifacts are not analyzed.
- Metadata note: metadata can provide context but does not by itself prove whether media is authentic.

### 7. Analytics Dashboard
- Summarizes analyses, assessment categories, and available model-score and video-frame statistics with charts and recent-history views.
- Keeps the latest 100 analysis records in the current browser's local storage. History is not shared or synchronized across users or devices.

### 8. Methodology Section
- Explains the dataset used.
- Describes preprocessing techniques.
- Presents the model architecture, held-out video evaluation metrics, and confusion matrix.

### Signals not implemented
The application does not calculate facial-landmark consistency, texture/pixel irregularities, face-boundary blending, lighting/shadow consistency, embedded metadata/provenance, or learned temporal motion consistency. These are not represented as analyzed evidence.

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
        Assessment
 REAL / FAKE / NEEDS REVIEW
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
9. Convert the SVM decision function to a bounded display score using a sigmoid. This score is not calibrated as a probability.
10. For video, aggregate usable sampled face scores using their median; require at least three usable sampled face scores, otherwise return Needs Review.
11. Return Needs Review for image scores in the inclusive 0.45–0.55 band or when no face is detected.
12. Display the result and the evidence actually available for that analysis.

Video analysis samples at most 20 frames. The frame-timeline suspicion label is based on the frame assessment; it is not an independent model output. Evidence strength is a rule-based coverage summary, not a calibrated confidence measure.

---

## 📂 Dataset

**Dataset:** FaceForensics++

**Selected subset:** c40 compression subset, Deepfakes category.

### Dataset Preparation

| Description | Quantity |
|---|---:|
| Original videos | 50 |
| Manipulated videos | 50 |
| Frames sampled per video during preprocessing | Up to 10 |
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
| Deep Learning | TensorFlow / Keras MobileNetV2 |
| Machine Learning | scikit-learn Support Vector Machine |
| Data processing | NumPy, pandas |
| Dataset | FaceForensics++ |
| Classification Kernel | RBF Kernel |
| API server | Uvicorn |
| Training/evaluation plots | Matplotlib |

The runtime model artifact is `models/deepfake_cnn_svm_rigorous.pkl`. MobileNetV2 uses ImageNet weights, which Keras may need to download the first time the application starts.

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

Evaluation was performed on the held-out video-level test set (20 videos: 10 Real and 10 Fake). These project-specific results do not guarantee equivalent performance on unseen datasets or real-world media.

---

## 🖥️ Application Modules

| Module | Description |
|---|---|
| Dashboard | Application overview and navigation |
| Media Analyzer | Media upload, preview and analysis |
| Detection Result | Real/Fake/Needs Review assessment, bounded model score, available evidence, and video statistics |
| Frame Investigation | Interactive timeline of up to 20 sampled video frames |
| Media Forensics | Available file, media, and analysis details; metadata/provenance is not inspected |
| Analytics | Browser-local analysis totals, assessment/score/frame charts, and recent history |
| Methodology | Dataset, preprocessing, model explanation, held-out metrics, and confusion matrix |
| Report | Downloadable HTML forensic analysis report |

---

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 (the project environment used for validation)
- pip package manager
- Git
- A code editor such as Visual Studio Code

### Step 1: Clone the Repository

```bash
git clone https://github.com/rithikanagineni/DeepGuard-AI.git
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
python -m pip install -r requirements.txt
```

### Step 5: Model and Dataset

The checked-in model artifact is loaded from `models/deepfake_cnn_svm_rigorous.pkl`. The raw training dataset is not required to run inference.

### Step 6: Run the Application

```bash
python -m uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). FastAPI serves both the interface and its static assets.

---


## ⚠️ Limitations and Responsible Use

- The current dataset contains a limited selection of original and manipulated videos.
- Performance may vary with compression, lighting, resolution, pose, and unseen manipulation methods.
- Face detection may fail when faces are small, obscured, or outside the expected view.
- A model score should not automatically be interpreted as a calibrated probability.
- The system is intended for AI-assisted screening and should not be treated as conclusive forensic evidence.

Users should treat uncertain or low-quality results as requiring human review.

---

## 🔭 Not Yet Implemented

- Embedded metadata (such as EXIF) extraction and external provenance verification.
- Dedicated facial-landmark, texture/pixel, blending-boundary, or lighting/shadow analysis.
- A learned temporal model for motion and cross-frame consistency.
- Evaluation on larger, more diverse datasets and manipulation methods.

---

## 👥 Team Members

**Department:** Computer Science and Engineering – Artificial Intelligence and Machine Learning

**Project Guide:** Mr. M. Gopal

**Team Members:**
- Nagineni Rithika
- Penuboku Mrudhula
- N. Devi Bhaskar

---

## 📚 References

### Research Papers

1. Rössler, A., et al. (2019). [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/abs/1901.08971). IEEE/CVF International Conference on Computer Vision (ICCV).
2. Sandler, M., et al. (2018). [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html). IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
3. Cortes, C., & Vapnik, V. (1995). [Support-Vector Networks](https://link.springer.com/article/10.1007/BF00994018). Machine Learning, 20, 273–297.
4. Howard, A. G., et al. (2017). [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861).
5. Viola, P., & Jones, M. (2001). [Rapid Object Detection using a Boosted Cascade of Simple Features](https://doi.org/10.1109/CVPR.2001.990517). IEEE Conference on Computer Vision and Pattern Recognition.
6. Goodfellow, I., et al. (2014). [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661). Advances in Neural Information Processing Systems (NeurIPS).

### Technical Documentation

7. [OpenCV Official Documentation](https://docs.opencv.org/).
8. [Scikit-learn Official Documentation – Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html).
9. [FastAPI Official Documentation](https://fastapi.tiangolo.com/).

### Dataset

10. [FaceForensics++ Dataset and Project Repository](https://github.com/ondyari/FaceForensics).

---

## 📌 Project Status

**Current Stage:** Working AI-assisted media analysis pipeline with an implemented application interface, browser-local analytics/history, interactive sampled-frame investigation, HTML reports, and project-specific held-out evaluation results.

**Project:** DeepGuard AI – AI-Based Deepfake Detection System

**Goal:** Detecting Digital Deception. Building Digital Trust.

---

⭐ If you find this project interesting, consider starring the repository!
