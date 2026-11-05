import cv2
import numpy as np
import pandas as pd
from pathlib import Path


# =========================
# Configuration
# =========================

IMAGE_SIZE = 224
FRAMES_PER_VIDEO = 10

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REAL_VIDEO_DIR = (
    PROJECT_ROOT
    / "dataset_raw"
    / "original_sequences"
    / "youtube"
    / "c40"
    / "videos"
)

FAKE_VIDEO_DIR = (
    PROJECT_ROOT
    / "dataset_raw"
    / "manipulated_sequences"
    / "Deepfakes"
    / "c40"
    / "videos"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "processed"
METADATA_FILE = PROJECT_ROOT / "results" / "processed_metadata.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Face detector
# =========================

FACE_DETECTOR = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


# =========================
# Extract frames
# =========================

def get_frame_indices(total_frames):
    """
    Select evenly spaced frames from a video.
    """

    if total_frames <= 0:
        return []

    if total_frames <= FRAMES_PER_VIDEO:
        return list(range(total_frames))

    return np.linspace(
        0,
        total_frames - 1,
        FRAMES_PER_VIDEO,
        dtype=int
    ).tolist()


# =========================
# Process one frame
# =========================

def process_frame(frame):

    if frame is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = FACE_DETECTOR.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return None

    # Select the largest detected face
    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    face = frame[y:y + h, x:x + w]

    if face.size == 0:
        return None

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    face = cv2.resize(
        face,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    face = face.astype(np.float32) / 255.0

    return face


# =========================
# Process one video
# =========================

def process_video(video_path, label, output_prefix):

    print(f"\nProcessing: {video_path.name}")

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print(f"Could not open video: {video_path}")
        return []

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    frame_indices = get_frame_indices(total_frames)

    processed_records = []

    for frame_number in frame_indices:

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        success, frame = capture.read()

        if not success:
            continue

        processed_face = process_frame(frame)

        if processed_face is None:
            continue

        output_name = (
            f"{output_prefix}_frame_{frame_number}.npy"
        )

        output_path = OUTPUT_DIR / output_name

        np.save(
            output_path,
            processed_face
        )

        processed_records.append({
            "file": output_name,
            "label": label,
            "video": video_path.name,
            "frame": frame_number
        })

    capture.release()

    print(
        f"  Saved {len(processed_records)} "
        f"face frames"
    )

    return processed_records


# =========================
# Main
# =========================

def main():

    all_records = []

    # -------------------------
    # Real videos
    # -------------------------

    real_videos = sorted(
        list(REAL_VIDEO_DIR.glob("*.mp4"))
    )

    print("=" * 60)
    print("REAL VIDEOS")
    print("=" * 60)
    print(f"Found {len(real_videos)} real videos")

    for video_path in real_videos:

        records = process_video(
            video_path,
            label="Real",
            output_prefix=f"real_{video_path.stem}"
        )

        all_records.extend(records)

    # -------------------------
    # Fake videos
    # -------------------------

    fake_videos = sorted(
        list(FAKE_VIDEO_DIR.glob("*.mp4"))
    )

    print("\n" + "=" * 60)
    print("FAKE VIDEOS")
    print("=" * 60)
    print(f"Found {len(fake_videos)} fake videos")

    for video_path in fake_videos:

        records = process_video(
            video_path,
            label="Fake",
            output_prefix=f"fake_{video_path.stem}"
        )

        all_records.extend(records)

    # -------------------------
    # Save metadata
    # -------------------------

    metadata = pd.DataFrame(all_records)

    metadata.to_csv(
        METADATA_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(f"Processed frames: {len(metadata)}")
    print(f"Metadata file: {METADATA_FILE}")

    if len(metadata) > 0:
        print("\nClass distribution:")
        print(metadata["label"].value_counts())


if __name__ == "__main__":
    main()