import os
import numpy as np
import pandas as pd
import joblib

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

SEED = 42

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "deepfake_cnn_svm_rigorous.pkl"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "results",
    "features",
    "cnn_features.npz"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "results",
    "processed_metadata.csv"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING CNN + SVM MODEL")
print("=" * 60)

model = joblib.load(
    MODEL_PATH
)

print("CNN + SVM model loaded successfully.")
print()


# ============================================================
# LOAD FEATURES
# ============================================================

print("=" * 60)
print("LOADING CNN FEATURES")
print("=" * 60)

data = np.load(
    FEATURES_PATH
)

X = data["features"]
y = data["labels"]

print(
    f"Feature shape : {X.shape}"
)

print(
    f"Label shape   : {y.shape}"
)

print()


# ============================================================
# LOAD METADATA
# ============================================================

metadata = pd.read_csv(
    METADATA_PATH
)

if len(metadata) != len(X):

    raise ValueError(
        "Feature count and metadata count do not match."
    )


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

    return sorted(
        set(ids)
    )


video_sources = {}

for video in metadata["video"].unique():

    video_sources[video] = extract_source_ids(
        video
    )


# ============================================================
# UNION-FIND
# ============================================================

parent = {}


def find(x):

    if parent[x] != x:

        parent[x] = find(
            parent[x]
        )

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
# RECREATE SAME TEST GROUPS
# ============================================================

rng = np.random.default_rng(
    SEED
)

groups = list(
    metadata["source_group"].unique()
)

rng.shuffle(
    groups
)

n_groups = len(groups)

val_end = int(
    n_groups * 0.80
)

test_groups = set(
    groups[val_end:]
)


test_mask = metadata[
    "source_group"
].isin(
    test_groups
).values


test_metadata = metadata[
    test_mask
].copy()

X_test = X[
    test_mask
]

y_test = y[
    test_mask
]


print("=" * 60)
print("HELD-OUT TEST SET")
print("=" * 60)

print(
    f"Test source groups : "
    f"{len(test_groups)}"
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
# VERIFY TEST SET
# ============================================================

expected_videos = 20

if test_metadata["video"].nunique() != expected_videos:

    print(
        "WARNING: The reconstructed test set does not "
        f"contain exactly {expected_videos} videos."
    )

    print(
        "Actual videos:",
        test_metadata["video"].nunique()
    )

    print()


# ============================================================
# FRAME-LEVEL SVM PROBABILITIES
# ============================================================

print("=" * 60)
print("GENERATING FRAME-LEVEL SVM PREDICTIONS")
print("=" * 60)

# SVC was trained without probability=True.
# decision_function gives the signed SVM score.
#
# We convert it to a smooth 0-1 score only for aggregation.
#
# IMPORTANT:
# This is a relative confidence score, not a calibrated
# probability.

decision_scores = model.decision_function(
    X_test
)

# Sigmoid transformation

frame_scores = (
    1.0 /
    (
        1.0 +
        np.exp(
            -decision_scores
        )
    )
)


frame_predictions = (
    frame_scores >= 0.5
).astype(int)


# ============================================================
# FRAME-LEVEL CHECK
# ============================================================

frame_accuracy = accuracy_score(
    y_test,
    frame_predictions
)

frame_precision = precision_score(
    y_test,
    frame_predictions,
    zero_division=0
)

frame_recall = recall_score(
    y_test,
    frame_predictions,
    zero_division=0
)

frame_f1 = f1_score(
    y_test,
    frame_predictions,
    zero_division=0
)


print()
print(
    "Frame-level check:"
)

print(
    f"Accuracy : {frame_accuracy:.4f}"
)

print(
    f"Precision: {frame_precision:.4f}"
)

print(
    f"Recall   : {frame_recall:.4f}"
)

print(
    f"F1 Score : {frame_f1:.4f}"
)

print()


# ============================================================
# VIDEO-LEVEL AGGREGATION
# ============================================================

print("=" * 60)
print("RUNNING VIDEO-LEVEL PREDICTIONS")
print("=" * 60)

video_results = []


# Preserve exact row ordering between X_test and metadata

test_metadata = test_metadata.reset_index(
    drop=True
)


for video_name in test_metadata[
    "video"
].unique():

    video_mask = (
        test_metadata["video"]
        == video_name
    ).values


    video_scores = frame_scores[
        video_mask
    ]

    video_frame_predictions = (
        frame_predictions[
            video_mask
        ]
    )

    video_labels = y_test[
        video_mask
    ]


    # Actual video label

    actual_label = int(
        round(
            np.mean(
                video_labels
            )
        )
    )


    # Mean score

    mean_score = float(
        np.mean(
            video_scores
        )
    )


    # Median score

    median_score = float(
        np.median(
            video_scores
        )
    )


    # Percentage of frames classified as fake

    fake_frame_count = int(
        np.sum(
            video_frame_predictions == 1
        )
    )

    total_frames = len(
        video_frame_predictions
    )


    fake_percentage = (
        fake_frame_count /
        total_frames
    ) * 100


    # Final video decision
    #
    # Mean score >= 0.5 → Fake

    predicted_label = int(
        mean_score >= 0.5
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
            "video":
                video_name,

            "actual_label":
                actual_text,

            "predicted_label":
                predicted_text,

            "mean_score":
                mean_score,

            "median_score":
                median_score,

            "total_frames":
                total_frames,

            "fake_frames":
                fake_frame_count,

            "real_frames":
                total_frames -
                fake_frame_count,

            "fake_frame_percentage":
                fake_percentage
        }
    )


    print(
        f"{video_name:<20} "
        f"Actual: {actual_text:<5} "
        f"Predicted: {predicted_text:<5} "
        f"Score: {mean_score:.3f}"
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    video_results
)


if len(results_df) == 0:

    raise RuntimeError(
        "No videos were evaluated."
    )


# ============================================================
# VIDEO-LEVEL LABELS
# ============================================================

y_video_true = (
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


y_video_pred = (
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
    y_video_true,
    y_video_pred
)

precision = precision_score(
    y_video_true,
    y_video_pred,
    zero_division=0
)

recall = recall_score(
    y_video_true,
    y_video_pred,
    zero_division=0
)

f1 = f1_score(
    y_video_true,
    y_video_pred,
    zero_division=0
)


# ============================================================
# DISPLAY FINAL RESULTS
# ============================================================

print()
print("=" * 60)
print("VIDEO-LEVEL CNN + SVM EVALUATION")
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
        y_video_true,
        y_video_pred,
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
    y_video_true,
    y_video_pred
)


print("Confusion Matrix:")

print(cm)


# ============================================================
# SAVE VIDEO PREDICTIONS
# ============================================================

predictions_path = os.path.join(
    RESULTS_DIR,
    "svm_video_level_predictions.csv"
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
    "svm_video_level_evaluation_metrics.csv"
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
    "svm_video_level_confusion_matrix.csv"
)


cm_df.to_csv(
    cm_path
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("CNN + SVM VIDEO-LEVEL EVALUATION COMPLETE")
print("=" * 60)

print()

print("Predictions:")
print(predictions_path)

print()

print("Metrics:")
print(metrics_path)

print()

print("Confusion matrix:")
print(cm_path)

print()

print(
    "Video predictions were generated by aggregating "
    "CNN-feature + SVM frame scores."
)

print()

print(
    "Note: SVM scores were converted using a sigmoid "
    "only for aggregation and should not be interpreted "
    "as calibrated probabilities."
)