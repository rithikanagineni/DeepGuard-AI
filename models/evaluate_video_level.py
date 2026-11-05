import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "deepfake_cnn_rigorous_best.keras"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "results",
    "processed_metadata.csv"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "results",
    "processed"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING TRAINED CNN MODEL")
print("=" * 60)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")
print()


# ============================================================
# LOAD METADATA
# ============================================================

metadata = pd.read_csv(
    METADATA_PATH
)

print("=" * 60)
print("DATA INFORMATION")
print("=" * 60)

print(
    f"Total processed frames : {len(metadata)}"
)

print(
    f"Unique videos          : {metadata['video'].nunique()}"
)

print()


# ============================================================
# RECREATE SOURCE GROUPS
# ============================================================

def extract_source_ids(video_name):

    name = os.path.splitext(
        os.path.basename(
            str(video_name)
        )
    )[0]

    parts = name.split("_")

    ids = []

    for part in parts:

        if part.isdigit():
            ids.append(part)

    if not ids:
        ids.append(name)

    return sorted(set(ids))


video_sources = {}

for video in metadata["video"].unique():

    video_sources[video] = extract_source_ids(
        video
    )


# Union-Find

parent = {}


def find(x):

    if parent[x] != x:
        parent[x] = find(parent[x])

    return parent[x]


def union(a, b):

    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        parent[root_b] = root_a


all_source_ids = set()

for sources in video_sources.values():

    all_source_ids.update(
        sources
    )


for source_id in all_source_ids:

    parent[source_id] = source_id


for sources in video_sources.values():

    if len(sources) > 1:

        first = sources[0]

        for source in sources[1:]:

            union(
                first,
                source
            )


video_groups = {}

for video, sources in video_sources.items():

    video_groups[video] = find(
        sources[0]
    )


metadata["source_group"] = metadata[
    "video"
].map(video_groups)


# ============================================================
# RECREATE THE SAME TEST GROUPS
# ============================================================
#
# IMPORTANT:
# The original training script used seed 42.
# We recreate the same shuffled source groups so that the
# video-level evaluation uses the same held-out test set.
# ============================================================

SEED = 42

rng = np.random.default_rng(
    SEED
)

groups = list(
    metadata["source_group"].unique()
)

rng.shuffle(groups)

n_groups = len(groups)

train_end = int(
    n_groups * 0.60
)

val_end = int(
    n_groups * 0.80
)

test_groups = set(
    groups[val_end:]
)


test_metadata = metadata[
    metadata["source_group"].isin(
        test_groups
    )
].copy()


print("=" * 60)
print("HELD-OUT TEST SET")
print("=" * 60)

print(
    f"Test source groups : {len(test_groups)}"
)

print(
    f"Test videos        : "
    f"{test_metadata['video'].nunique()}"
)

print(
    f"Test frames        : "
    f"{len(test_metadata)}"
)

print()


# ============================================================
# VIDEO-LEVEL PREDICTION
# ============================================================

video_results = []

unique_videos = (
    test_metadata["video"]
    .unique()
)


print("=" * 60)
print("RUNNING VIDEO-LEVEL PREDICTIONS")
print("=" * 60)

print()


for index, video_name in enumerate(
    unique_videos,
    start=1
):

    video_df = test_metadata[
        test_metadata["video"] == video_name
    ].copy()


    images = []
    valid_rows = []


    # --------------------------------------------------------
    # Load all available frames for this video
    # --------------------------------------------------------

    for _, row in video_df.iterrows():

        frame_path = os.path.join(
            PROCESSED_DIR,
            row["file"]
        )

        if not os.path.exists(
            frame_path
        ):
            continue


        image = np.load(
            frame_path
        )


        if image.shape != (
            224,
            224,
            3
        ):
            continue


        # MobileNetV2 preprocessing
        image = (
            image * 2.0
        ) - 1.0


        images.append(
            image
        )

        valid_rows.append(
            row
        )


    if len(images) == 0:

        continue


    images = np.array(
        images,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Predict all frames
    # --------------------------------------------------------

    probabilities = model.predict(
        images,
        batch_size=16,
        verbose=0
    ).ravel()


    # --------------------------------------------------------
    # Aggregate frame predictions
    # --------------------------------------------------------
    #
    # Mean probability is the primary video-level score.
    #
    # Example:
    #
    # Frame probabilities:
    # 0.72, 0.81, 0.43, 0.76
    #
    # Mean = 0.68
    #
    # Final = Fake
    # --------------------------------------------------------

    mean_probability = float(
        np.mean(
            probabilities
        )
    )


    median_probability = float(
        np.median(
            probabilities
        )
    )


    frame_predictions = (
        probabilities >= 0.5
    ).astype(int)


    fake_frames = int(
        np.sum(
            frame_predictions == 1
        )
    )


    real_frames = int(
        np.sum(
            frame_predictions == 0
        )
    )


    # Mean probability determines final result

    predicted_label = (
        1
        if mean_probability >= 0.5
        else 0
    )


    actual_label = (
        1
        if str(
            video_df.iloc[0]["label"]
        ).lower() == "fake"
        else 0
    )


    actual_text = (
        "Fake"
        if actual_label == 1
        else "Real"
    )


    predicted_text = (
        "Fake"
        if predicted_label == 1
        else "Real"
    )


    video_results.append(
        {
            "video": video_name,

            "actual_label":
                actual_text,

            "predicted_label":
                predicted_text,

            "mean_probability":
                mean_probability,

            "median_probability":
                median_probability,

            "total_frames":
                len(probabilities),

            "fake_frames":
                fake_frames,

            "real_frames":
                real_frames,

            "fake_frame_percentage":
                (
                    fake_frames /
                    len(probabilities)
                ) * 100
        }
    )


    print(
        f"[{index:02d}/"
        f"{len(unique_videos):02d}] "
        f"{video_name:<20} "
        f"Actual: {actual_text:<5} "
        f"Predicted: {predicted_text:<5} "
        f"Confidence: "
        f"{mean_probability:.3f}"
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    video_results
)


if len(results_df) == 0:

    raise RuntimeError(
        "No videos were successfully evaluated."
    )


# ============================================================
# CONVERT LABELS
# ============================================================

y_true = (
    results_df[
        "actual_label"
    ]
    .map(
        {
            "Real": 0,
            "Fake": 1
        }
    )
    .values
)


y_pred = (
    results_df[
        "predicted_label"
    ]
    .map(
        {
            "Real": 0,
            "Fake": 1
        }
    )
    .values
)


# ============================================================
# VIDEO-LEVEL METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 60)
print("VIDEO-LEVEL CNN EVALUATION")
print("=" * 60)

print(
    f"Videos evaluated: "
    f"{len(results_df)}"
)

print()

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print()


print("Classification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Real",
            "Fake"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


print("Confusion Matrix:")

print(cm)


# ============================================================
# SAVE VIDEO PREDICTIONS
# ============================================================

predictions_path = os.path.join(
    RESULTS_DIR,
    "video_level_predictions.csv"
)


results_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        "Value": [
            accuracy,
            precision,
            recall,
            f1
        ]
    }
)


metrics_path = os.path.join(
    RESULTS_DIR,
    "video_level_evaluation_metrics.csv"
)


metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,

    index=[
        "Actual Real",
        "Actual Fake"
    ],

    columns=[
        "Predicted Real",
        "Predicted Fake"
    ]
)


cm_path = os.path.join(
    RESULTS_DIR,
    "video_level_confusion_matrix.csv"
)


cm_df.to_csv(
    cm_path
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("VIDEO-LEVEL EVALUATION COMPLETE")
print("=" * 60)

print()

print(
    "Predictions:"
)

print(
    predictions_path
)

print()

print(
    "Metrics:"
)

print(
    metrics_path
)

print()

print(
    "Confusion matrix:"
)

print(
    cm_path
)

print()

print(
    "Each video's frame predictions were aggregated"
)

print(
    "using the mean CNN probability."
)

print()