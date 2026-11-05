import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
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

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


# ============================================================
# LOAD CNN FEATURES
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
# RECREATE SAME TRAIN / VALIDATION / TEST GROUPS
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

train_end = int(
    n_groups * 0.60
)

val_end = int(
    n_groups * 0.80
)

train_groups = set(
    groups[:train_end]
)

val_groups = set(
    groups[
        train_end:val_end
    ]
)

test_groups = set(
    groups[val_end:]
)


# ============================================================
# GET ROW INDICES
# ============================================================

train_mask = metadata[
    "source_group"
].isin(
    train_groups
).values


val_mask = metadata[
    "source_group"
].isin(
    val_groups
).values


test_mask = metadata[
    "source_group"
].isin(
    test_groups
).values


X_train = X[
    train_mask
]

y_train = y[
    train_mask
]

X_val = X[
    val_mask
]

y_val = y[
    val_mask
]

X_test = X[
    test_mask
]

y_test = y[
    test_mask
]


# ============================================================
# VERIFY SPLIT
# ============================================================

print("=" * 60)
print("RIGOROUS SOURCE-GROUP SPLIT")
print("=" * 60)

print(
    f"Training groups   : {len(train_groups)}"
)

print(
    f"Validation groups : {len(val_groups)}"
)

print(
    f"Testing groups    : {len(test_groups)}"
)

print()

print(
    f"Training samples   : {len(X_train)}"
)

print(
    f"Validation samples : {len(X_val)}"
)

print(
    f"Testing samples    : {len(X_test)}"
)

print()

print(
    "Train/Test group overlap:",
    len(
        train_groups.intersection(
            test_groups
        )
    )
)

print()


# ============================================================
# CHECK CLASS DISTRIBUTION
# ============================================================

print("Training class distribution:")

print(
    pd.Series(
        y_train
    ).value_counts()
)

print()

print("Validation class distribution:")

print(
    pd.Series(
        y_val
    ).value_counts()
)

print()

print("Testing class distribution:")

print(
    pd.Series(
        y_test
    ).value_counts()
)

print()


# ============================================================
# SVM CONFIGURATIONS
# ============================================================

experiments = {

    "RBF_C1": {

        "C": 1,
        "kernel": "rbf",
        "gamma": "scale"
    },

    "RBF_C10": {

        "C": 10,
        "kernel": "rbf",
        "gamma": "scale"
    },

    "RBF_C100": {

        "C": 100,
        "kernel": "rbf",
        "gamma": "scale"
    },

    "RBF_C10_GAMMA_SMALL": {

        "C": 10,
        "kernel": "rbf",
        "gamma": 0.001
    },

    "RBF_C10_GAMMA_LARGE": {

        "C": 10,
        "kernel": "rbf",
        "gamma": 0.01
    },

    "LINEAR_C1": {

        "C": 1,
        "kernel": "linear"
    },

    "LINEAR_C10": {

        "C": 10,
        "kernel": "linear"
    }
}


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

results = []

best_model = None
best_name = None
best_f1 = -1


print("=" * 60)
print("SVM EXPERIMENTS")
print("=" * 60)


for name, params in experiments.items():

    print()
    print("-" * 60)
    print(
        f"Training: {name}"
    )
    print("-" * 60)


    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "svm",
                SVC(
                    **params,
                    class_weight="balanced"
                )
            )
        ]
    )


    model.fit(
        X_train,
        y_train
    )


    # Validation predictions
    val_predictions = model.predict(
        X_val
    )


    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    val_precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    val_recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    val_f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )


    print(
        f"Validation Accuracy : "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Validation Precision: "
        f"{val_precision:.4f}"
    )

    print(
        f"Validation Recall   : "
        f"{val_recall:.4f}"
    )

    print(
        f"Validation F1       : "
        f"{val_f1:.4f}"
    )


    results.append(
        {
            "Model": name,

            "Validation_Accuracy":
                val_accuracy,

            "Validation_Precision":
                val_precision,

            "Validation_Recall":
                val_recall,

            "Validation_F1":
                val_f1
        }
    )


    # Select based ONLY on validation F1.
    #
    # The test set is not used for model selection.

    if val_f1 > best_f1:

        best_f1 = val_f1

        best_model = model

        best_name = name


# ============================================================
# SAVE EXPERIMENT RESULTS
# ============================================================

comparison_df = pd.DataFrame(
    results
)

comparison_path = os.path.join(
    RESULTS_DIR,
    "svm_rigorous_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)


print()
print("=" * 60)
print("BEST SVM MODEL")
print("=" * 60)

print(
    f"Selected model: {best_name}"
)

print(
    f"Validation F1 : {best_f1:.4f}"
)

print()


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("=" * 60)
print("FINAL SVM TEST EVALUATION")
print("=" * 60)


test_predictions = best_model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)


print()
print(
    f"Model    : {best_name}"
)

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
        y_test,
        test_predictions,
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
    test_predictions
)


print("Confusion Matrix:")

print(cm)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

import joblib


model_path = os.path.join(
    MODELS_DIR,
    "deepfake_cnn_svm_rigorous.pkl"
)


joblib.dump(
    best_model,
    model_path
)


# ============================================================
# SAVE FINAL METRICS
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
    "svm_rigorous_evaluation_metrics.csv"
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
    "svm_rigorous_confusion_matrix.csv"
)


cm_df.to_csv(
    cm_path
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("RIGOROUS CNN + SVM EXPERIMENT COMPLETE")
print("=" * 60)

print()

print(
    "Model:"
)

print(
    model_path
)

print()

print(
    "Comparison:"
)

print(
    comparison_path
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