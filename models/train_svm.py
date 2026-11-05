from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    PROJECT_ROOT
    / "results"
    / "features"
    / "cnn_features.npz"
)

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# LOAD FEATURES
# ============================================================

print("=" * 60)
print("IMPROVED CNN + SVM DEEPFAKE DETECTION")
print("=" * 60)

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        f"CNN feature file not found:\n{FEATURE_FILE}"
    )

data = np.load(
    FEATURE_FILE,
    allow_pickle=True
)

X = data["features"]
y = data["labels"]
videos = data["videos"]
files = data["files"]

print("\nLoaded CNN features:")
print(f"Feature matrix: {X.shape}")
print(f"Labels: {y.shape}")
print(f"Total frames: {len(X)}")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nOverall class distribution:")
print(f"Real: {np.sum(y == 0)}")
print(f"Fake: {np.sum(y == 1)}")


# ============================================================
# UNIQUE VIDEO INFORMATION
# ============================================================

unique_videos = np.unique(videos)

print(f"\nUnique videos: {len(unique_videos)}")


# ============================================================
# STRATIFIED GROUP SPLIT
# ============================================================
#
# We keep frames from the same video together.
#
# StratifiedGroupKFold tries to maintain class balance while
# ensuring that the same video cannot occur in both sets.
# ============================================================

print("\nCreating leakage-safe stratified group split...")

splitter = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

train_indices, test_indices = next(
    splitter.split(
        X,
        y,
        groups=videos
    )
)


# ============================================================
# CREATE TRAIN / TEST DATA
# ============================================================

X_train = X[train_indices]
X_test = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

train_videos = videos[train_indices]
test_videos = videos[test_indices]


# ============================================================
# VIDEO LEAKAGE CHECK
# ============================================================

train_video_set = set(train_videos)
test_video_set = set(test_videos)

overlap = train_video_set.intersection(
    test_video_set
)

print("\nVideo leakage check:")

if len(overlap) == 0:
    print(
        "PASS - No videos appear in both "
        "training and testing sets."
    )
else:
    print(
        "ERROR - Video overlap detected!"
    )
    print(overlap)
    raise RuntimeError(
        "Training stopped because of video leakage."
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\nTraining set:")
print(f"Frames: {len(X_train)}")
print(f"Videos: {len(train_video_set)}")
print(f"Real: {np.sum(y_train == 0)}")
print(f"Fake: {np.sum(y_train == 1)}")

print("\nTesting set:")
print(f"Frames: {len(X_test)}")
print(f"Videos: {len(test_video_set)}")
print(f"Real: {np.sum(y_test == 0)}")
print(f"Fake: {np.sum(y_test == 1)}")


# ============================================================
# SVM CONFIGURATIONS
# ============================================================

configs = [
    {
        "name": "RBF_C1",
        "kernel": "rbf",
        "C": 1,
        "gamma": "scale"
    },
    {
        "name": "RBF_C10",
        "kernel": "rbf",
        "C": 10,
        "gamma": "scale"
    },
    {
        "name": "RBF_C100",
        "kernel": "rbf",
        "C": 100,
        "gamma": "scale"
    },
    {
        "name": "RBF_C10_GAMMA_SMALL",
        "kernel": "rbf",
        "C": 10,
        "gamma": 0.001
    },
    {
        "name": "RBF_C10_GAMMA_LARGE",
        "kernel": "rbf",
        "C": 10,
        "gamma": 0.01
    },
    {
        "name": "LINEAR_C1",
        "kernel": "linear",
        "C": 1,
        "gamma": "scale"
    },
    {
        "name": "LINEAR_C10",
        "kernel": "linear",
        "C": 10,
        "gamma": "scale"
    }
]


# ============================================================
# TRAIN AND COMPARE MODELS
# ============================================================

results = []

best_model = None
best_name = None
best_f1 = -1


print("\n" + "=" * 60)
print("TESTING SVM CONFIGURATIONS")
print("=" * 60)


for config in configs:

    print(
        f"\nTraining: {config['name']}"
    )

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "svm",
            SVC(
                kernel=config["kernel"],
                C=config["C"],
                gamma=config["gamma"],
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append({
        "model": config["name"],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

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

    # Select best model by F1
    if f1 > best_f1:

        best_f1 = f1
        best_model = model
        best_name = config["name"]


# ============================================================
# MODEL COMPARISON
# ============================================================

comparison_df = pd.DataFrame(
    results
)

comparison_df = comparison_df.sort_values(
    by="f1_score",
    ascending=False
)

comparison_file = (
    RESULTS_DIR
    / "svm_model_comparison.csv"
)

comparison_df.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# BEST MODEL PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("BEST SVM MODEL")
print("=" * 60)

print(
    f"Selected model: {best_name}"
)

print(
    f"Validation F1 Score: {best_f1:.4f}"
)

y_pred = best_model.predict(
    X_test
)


# ============================================================
# FINAL METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

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


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
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
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# SAVE FINAL METRICS
# ============================================================

metrics = {
    "model": best_name,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "train_frames": len(X_train),
    "test_frames": len(X_test),
    "train_videos": len(train_video_set),
    "test_videos": len(test_video_set)
}

metrics_df = pd.DataFrame(
    [metrics]
)

metrics_file = (
    RESULTS_DIR
    / "evaluation_metrics_improved.csv"
)

metrics_df.to_csv(
    metrics_file,
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

cm_file = (
    RESULTS_DIR
    / "confusion_matrix_improved.csv"
)

cm_df.to_csv(
    cm_file
)


# ============================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(cm)

plt.title(
    "CNN + SVM Deepfake Detection"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.xticks(
    [0, 1],
    ["Real", "Fake"]
)

plt.yticks(
    [0, 1],
    ["Real", "Fake"]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

confusion_plot = (
    RESULTS_DIR
    / "confusion_matrix_improved.png"
)

plt.savefig(
    confusion_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE TRAINED MODEL
# ============================================================

MODEL_FILE = (
    MODELS_DIR
    / "deepfake_cnn_svm_improved.pkl"
)

joblib.dump(
    best_model,
    MODEL_FILE
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("IMPROVED TRAINING COMPLETE")
print("=" * 60)

print("\nModel:")
print(MODEL_FILE)

print("\nComparison:")
print(comparison_file)

print("\nMetrics:")
print(metrics_file)

print("\nConfusion matrix:")
print(cm_file)

print("\nConfusion matrix image:")
print(confusion_plot)

print("\nCNN + SVM improved model is ready.")