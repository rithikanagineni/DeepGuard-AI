import base64
import os
import cv2
import numpy as np
import joblib

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "deepfake_cnn_svm_rigorous.pkl"
)


# ---------------------------------------------------------
# LOAD CNN FEATURE EXTRACTOR
# ---------------------------------------------------------

feature_extractor = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)


# ---------------------------------------------------------
# LOAD SVM MODEL
# ---------------------------------------------------------

svm_model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------
# FACE DETECTOR
# ---------------------------------------------------------

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

if face_detector.empty():
    raise RuntimeError(
        "OpenCV Haar Cascade could not be loaded."
    )


# ---------------------------------------------------------
# FACE DETECTION
# ---------------------------------------------------------

def detect_largest_face(image):

    # Make sure OpenCV receives a NumPy image array
    if image is None:
        return None

    if not isinstance(image, np.ndarray):
        return None

    if image.size == 0:
        return None

    # Convert BGR image to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return None

    # Select largest detected face
    largest_face = max(
        faces,
        key=lambda rect: rect[2] * rect[3]
    )

    x, y, w, h = largest_face

    face = image[
        y:y + h,
        x:x + w
    ]

    if face.size == 0:
        return None

    return face


# ---------------------------------------------------------
# PREPROCESS FACE
# ---------------------------------------------------------

def preprocess_face(face):

    if face is None:
        return None

    if not isinstance(face, np.ndarray):
        return None

    if face.size == 0:
        return None

    # OpenCV uses BGR.
    # MobileNetV2 expects RGB.
    rgb_face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    resized = cv2.resize(
        rgb_face,
        (224, 224)
    )

    image_array = resized.astype(
        np.float32
    )

    # MobileNetV2 preprocessing:
    # [0,255] -> [-1,1]
    processed = preprocess_input(
        image_array
    )

    processed = np.expand_dims(
        processed,
        axis=0
    )

    return processed


# ---------------------------------------------------------
# CNN FEATURE EXTRACTION
# ---------------------------------------------------------

def extract_features(face):

    processed_face = preprocess_face(face)

    if processed_face is None:
        return None

    features = feature_extractor.predict(
        processed_face,
        verbose=0
    )

    return features


# ---------------------------------------------------------
# SVM SCORE
# ---------------------------------------------------------

def svm_score(features):

    if features is None:
        return None

    decision_score = svm_model.decision_function(
        features
    )

    decision_score = float(
        np.asarray(decision_score).reshape(-1)[0]
    )

    # Convert SVM decision score to a bounded
    # score for visualization.
    #
    # IMPORTANT:
    # This is NOT a calibrated probability.
    score = 1.0 / (
        1.0 + np.exp(-decision_score)
    )

    return float(score)


def assessment_from_score(score):

    model_label = "Fake" if score >= 0.5 else "Real"

    if 0.45 <= score <= 0.55:
        return "Needs Review", model_label

    return model_label, model_label


def encode_preview(image, max_dimension=360):

    if image is None or image.size == 0:
        return None

    height, width = image.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))

    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale)))
        )

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 60]
    )

    if not success:
        return None

    return "data:image/jpeg;base64," + base64.b64encode(
        encoded
    ).decode("ascii")


# ---------------------------------------------------------
# IMAGE PREDICTION
# ---------------------------------------------------------

def predict_image(image_path):

    image = cv2.imread(
        image_path
    )

    if image is None:
        return {
            "success": False,
            "status": "invalid_image",
            "message": "The uploaded image could not be read."
        }

    face = detect_largest_face(
        image
    )

    if face is None:
        return {
            "success": True,
            "status": "needs_review",
            "label": "Needs Review",
            "model_label": None,
            "model_score": None,
            "score": None,
            "confidence": None,
            "message": (
                "No clearly detectable face was found "
                "in the uploaded image. Manual review is recommended."
            ),
            "face_detected": False,
            "decision_reason": "No face was available for CNN/SVM classification.",
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "analyzed_frames": 1,
            "face_detected_frames": 0,
            "face_detection_percentage": 0
        }

    features = extract_features(
        face
    )

    if features is None:

        return {
            "success": False,
            "status": "feature_extraction_failed",
            "message": (
                "CNN feature extraction failed."
            ),
            "face_detected": True
        }

    score = svm_score(
        features
    )

    if score is None:

        return {
            "success": False,
            "status": "classification_failed",
            "message": (
                "SVM classification failed."
            ),
            "face_detected": True
        }

    label, model_label = assessment_from_score(score)

    confidence = (
        score * 100
        if model_label == "Fake"
        else (1.0 - score) * 100
    )
    if label == "Needs Review":
        confidence = None

    return {

        "success": True,

        "status": "prediction_complete",

        "label": label,
        "model_label": model_label,

        "score": round(
            score,
            4
        ),

        "model_score": round(
            score * 100,
            2
        ),

        "confidence": (
            round(confidence, 2)
            if confidence is not None
            else None
        ),

        "face_detected": True,
        "decision_reason": (
            "The model score is close to the 0.50 decision boundary."
            if label == "Needs Review"
            else None
        ),
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "analyzed_frames": 1,
        "face_detected_frames": 1,
        "face_detection_percentage": 100

    }


# ---------------------------------------------------------
# VIDEO FRAME EXTRACTION
# ---------------------------------------------------------

def extract_video_frames(
    video_path,
    max_frames=20
):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        return [], {}

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total_frames <= 0:
        cap.release()
        return [], {}

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = None

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        min(
            max_frames,
            total_frames
        ),
        dtype=int
    )

    samples = []

    for index in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        success, frame = cap.read()

        if success and frame is not None:
            samples.append({
                "frame_number": int(index) + 1,
                "timestamp": (
                    round(float(index) / fps, 3)
                    if fps is not None
                    else None
                ),
                "frame": frame
            })

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return samples, {
        "video_fps": round(fps, 3) if fps is not None else None,
        "video_duration": (
            round(total_frames / fps, 3)
            if fps is not None
            else None
        ),
        "video_width": width or None,
        "video_height": height or None,
        "total_video_frames": total_frames
    }


# ---------------------------------------------------------
# VIDEO PREDICTION
# ---------------------------------------------------------

def predict_video(
    video_path,
    max_frames=20,
    min_face_frames=3
):

    samples, media_info = extract_video_frames(
        video_path,
        max_frames=max_frames
    )

    if len(samples) == 0:

        return {
            "success": False,
            "status": "invalid_video",
            "message": (
                "The uploaded video could not be read."
            ),
            "analyzed_frames": 0,
            "face_detected_frames": 0,
            "face_detection_percentage": 0,
            **media_info
        }

    scores = []
    frame_results = []

    analyzed_frames = len(samples)

    face_detected_frames = 0

    for sample in samples:
        frame = sample["frame"]
        frame_result = {
            "frame_number": sample["frame_number"],
            "timestamp": sample["timestamp"],
            "face_detected": False,
            "score": None,
            "model_score": None,
            "model_label": None,
            "assessment": "Not analyzed",
            "suspicion_level": "Unavailable",
            "frame_image": encode_preview(frame)
        }

        face = detect_largest_face(
            frame
        )

        if face is None:
            frame_results.append(frame_result)
            continue

        face_detected_frames += 1
        frame_result["face_detected"] = True
        frame_result["face_crop"] = encode_preview(face, max_dimension=200)

        features = extract_features(
            face
        )

        if features is None:
            frame_results.append(frame_result)
            continue

        score = svm_score(
            features
        )

        if score is not None:
            scores.append(score)
            label, model_label = assessment_from_score(score)
            frame_result.update({
                "score": round(score, 4),
                "model_score": round(score * 100, 2),
                "model_label": model_label,
                "assessment": label,
                "suspicion_level": (
                    "Moderate"
                    if label == "Needs Review"
                    else "High"
                    if model_label == "Fake"
                    else "Low"
                )
            })

        frame_results.append(frame_result)

    face_detection_percentage = (
        face_detected_frames /
        analyzed_frames *
        100
    )

    # -----------------------------------------------------
    # INSUFFICIENT FACE DATA
    # -----------------------------------------------------

    if face_detected_frames < min_face_frames or len(scores) < min_face_frames:

        median_score = (
            float(np.median(scores))
            if scores
            else None
        )
        model_label = (
            "Fake" if median_score >= 0.5 else "Real"
        ) if median_score is not None else None
        real_like_frames = sum(
            frame["assessment"] == "Real"
            for frame in frame_results
        )
        fake_like_frames = sum(
            frame["assessment"] == "Fake"
            for frame in frame_results
        )
        suspicious_frames = sum(
            frame["assessment"] == "Needs Review"
            for frame in frame_results
        )
        real_frames = real_like_frames
        fake_frames = fake_like_frames
        model_agreement_percentage = (
            max(real_frames, fake_frames, suspicious_frames) / len(scores) * 100
            if scores
            else None
        )

        return {
            "success": True,

            "status": "needs_review",

            "message": (
                "The video does not contain enough "
                "clearly detectable facial frames "
                "for a reliable prediction. Manual review is recommended."
            ),

            "label": "Needs Review",
            "model_label": model_label,
            "model_score": (
                round(median_score * 100, 2)
                if median_score is not None
                else None
            ),
            "score": (
                round(median_score, 4)
                if median_score is not None
                else None
            ),
            "confidence": None,
            "median_score": (
                round(median_score, 4)
                if median_score is not None
                else None
            ),
            "real_frames": real_frames,
            "fake_frames": fake_frames,
            "real_like_frames": real_like_frames,
            "fake_like_frames": fake_like_frames,
            "suspicious_frames": suspicious_frames,
            "fake_frame_percentage": (
                round(fake_frames / len(scores) * 100, 2)
                if scores
                else None
            ),
            "real_frame_percentage": (
                round(real_frames / len(scores) * 100, 2)
                if scores
                else None
            ),
            "suspicious_frame_percentage": (
                round(suspicious_frames / len(scores) * 100, 2)
                if scores
                else None
            ),
            "model_agreement_percentage": (
                round(model_agreement_percentage, 2)
                if model_agreement_percentage is not None
                else None
            ),
            "decision_reason": (
                f"Only {len(scores)} of at least {min_face_frames} "
                "required sampled frames produced usable face scores."
            ),
            "analyzed_frames": analyzed_frames,

            "face_detected_frames":
                face_detected_frames,

            "face_detection_percentage":
                round(
                    face_detection_percentage,
                    2
                ),

            "minimum_required_frames":
                min_face_frames,
            "frame_results": frame_results,
            **media_info
        }

    if len(scores) == 0:

        return {
            "success": False,

            "status": "classification_failed",

            "message": (
                "The system could not extract "
                "usable features from the detected faces."
            ),

            "analyzed_frames": analyzed_frames,

            "face_detected_frames":
                face_detected_frames,

            "face_detection_percentage":
                round(
                    face_detection_percentage,
                    2
                ),
            "frame_results": frame_results,
            **media_info
        }

    # -----------------------------------------------------
    # VIDEO SCORE
    # -----------------------------------------------------

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    median_score = float(
        np.median(scores)
    )

    label, model_label = assessment_from_score(median_score)

    confidence = (
        median_score * 100
        if model_label == "Fake"
        else (1.0 - median_score) * 100
    )
    if label == "Needs Review":
        confidence = None

    real_like_frames = sum(
        frame["assessment"] == "Real"
        for frame in frame_results
    )
    fake_like_frames = sum(
        frame["assessment"] == "Fake"
        for frame in frame_results
    )
    suspicious_frames = sum(
        frame["assessment"] == "Needs Review"
        for frame in frame_results
    )
    real_frames = real_like_frames
    fake_frames = fake_like_frames

    model_agreement_percentage = (
        max(real_frames, fake_frames, suspicious_frames) /
        len(scores) *
        100
    )

    fake_frame_percentage = (
        fake_frames /
        len(scores) *
        100
    )
    real_frame_percentage = (
        real_frames /
        len(scores) *
        100
    )
    suspicious_frame_percentage = (
        suspicious_frames /
        len(scores) *
        100
    )

    return {

        "success": True,

        "status":
            "needs_review"
            if label == "Needs Review"
            else "prediction_complete",

        "label":
            label,

        "model_label":
            model_label,

        "score":
            round(
                median_score,
                4
            ),

        "confidence":
            round(confidence, 2)
            if confidence is not None
            else None,

        "model_score":
            round(
                median_score * 100,
                2
            ),

        "analyzed_frames":
            analyzed_frames,

        "face_detected_frames":
            face_detected_frames,

        "face_detection_percentage":
            round(
                face_detection_percentage,
                2
            ),

        "fake_frames":
            fake_frames,

        "real_like_frames":
            real_like_frames,

        "fake_like_frames":
            fake_like_frames,

        "suspicious_frames":
            suspicious_frames,

        "real_frames":
            real_frames,

        "fake_frame_percentage":
            round(
                fake_frame_percentage,
                2
            ),

        "real_frame_percentage":
            round(
                real_frame_percentage,
                2
            ),

        "suspicious_frame_percentage":
            round(
                suspicious_frame_percentage,
                2
            ),

        "median_score":
            round(
                median_score,
                4
            ),

        "model_agreement_percentage":
            round(
                model_agreement_percentage,
                2
            ),

        "decision_reason": (
            "The model score is close to the 0.50 decision boundary."
            if label == "Needs Review"
            else None
        ),

        "frame_results":
            frame_results,

        **media_info

    }