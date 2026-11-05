import os
import re
import random
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

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

metadata = pd.read_csv(METADATA_PATH)

print(f"Total processed frames: {len(metadata)}")
print()

print("Class distribution:")
print(metadata["label"].value_counts())
print()


# ============================================================
# SOURCE GROUP CREATION
# ============================================================
#
# Important:
# FaceForensics-style manipulated videos can be derived from
# original source videos.
#
# Example:
#
#   033.mp4
#   033_097.mp4
#   097_033.mp4
#
# These should not be separated across train/validation/test.
#
# We create source-based groups using the identities appearing
# in each video name.
# ============================================================

def extract_source_ids(video_name):

    name = os.path.splitext(os.path.basename(str(video_name)))[0]

    parts = name.split("_")

    ids = []

    for part in parts:
        if part.isdigit():
            ids.append(part)

    if not ids:
        ids.append(name)

    return sorted(set(ids))


# Build graph of connected source identities

video_sources = {}

for video in metadata["video"].unique():

    source_ids = extract_source_ids(video)

    video_sources[video] = source_ids


# Union-Find structure

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


# Initialize identities

all_source_ids = set()

for sources in video_sources.values():
    all_source_ids.update(sources)

for source_id in all_source_ids:
    parent[source_id] = source_id


# Connect identities that occur in the same manipulated video

for sources in video_sources.values():

    if len(sources) > 1:

        first = sources[0]

        for source in sources[1:]:
            union(first, source)


# Assign connected component group

video_groups = {}

for video, sources in video_sources.items():

    root = find(sources[0])

    video_groups[video] = root


metadata["source_group"] = metadata["video"].map(video_groups)


print("=" * 60)
print("SOURCE GROUP ANALYSIS")
print("=" * 60)

print(
    f"Unique videos : {metadata['video'].nunique()}"
)

print(
    f"Unique source groups : {metadata['source_group'].nunique()}"
)

print()


group_summary = (
    metadata
    .groupby("source_group")
    .agg(
        videos=("video", "nunique"),
        frames=("video", "count"),
        real_frames=("label", lambda x: (x == "Real").sum()),
        fake_frames=("label", lambda x: (x == "Fake").sum())
    )
)

print(group_summary)

print()


# ============================================================
# CREATE GROUP-WISE TRAIN / VALIDATION / TEST SPLIT
# ============================================================

groups = metadata["source_group"].unique()

rng = np.random.default_rng(SEED)

groups = list(groups)

rng.shuffle(groups)


# We aim for approximately:
#
# 60% training
# 20% validation
# 20% testing

n_groups = len(groups)

if n_groups < 5:

    raise ValueError(
        "Too few independent source groups for a reliable "
        "train/validation/test split."
    )


train_end = int(n_groups * 0.60)
val_end = int(n_groups * 0.80)

train_groups = set(groups[:train_end])
val_groups = set(groups[train_end:val_end])
test_groups = set(groups[val_end:])


train_df = metadata[
    metadata["source_group"].isin(train_groups)
].copy()

val_df = metadata[
    metadata["source_group"].isin(val_groups)
].copy()

test_df = metadata[
    metadata["source_group"].isin(test_groups)
].copy()


# ============================================================
# VERIFY NO SOURCE GROUP OVERLAP
# ============================================================

train_overlap_val = train_groups.intersection(val_groups)
train_overlap_test = train_groups.intersection(test_groups)
val_overlap_test = val_groups.intersection(test_groups)

if train_overlap_val:
    raise ValueError("Train/validation source leakage detected.")

if train_overlap_test:
    raise ValueError("Train/test source leakage detected.")

if val_overlap_test:
    raise ValueError("Validation/test source leakage detected.")


print("=" * 60)
print("DATA SPLIT")
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
    f"Training frames   : {len(train_df)}"
)

print(
    f"Validation frames : {len(val_df)}"
)

print(
    f"Testing frames    : {len(test_df)}"
)

print()

print("Training classes:")
print(train_df["label"].value_counts())

print()

print("Validation classes:")
print(val_df["label"].value_counts())

print()

print("Testing classes:")
print(test_df["label"].value_counts())

print()


# ============================================================
# LOAD IMAGE ARRAYS
# ============================================================

def load_images(dataframe):

    images = []
    labels = []

    for _, row in dataframe.iterrows():

        path = os.path.join(
            PROCESSED_DIR,
            row["file"]
        )

        if not os.path.exists(path):
            continue

        image = np.load(path)

        if image.shape != (224, 224, 3):
            continue

        images.append(image)

        # Real = 0
        # Fake = 1

        labels.append(
            1 if row["label"] == "Fake" else 0
        )

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.float32)
    )


print("=" * 60)
print("LOADING TRAINING DATA")
print("=" * 60)

X_train, y_train = load_images(train_df)

print(
    f"X_train shape: {X_train.shape}"
)

print(
    f"y_train shape: {y_train.shape}"
)

print()


print("=" * 60)
print("LOADING VALIDATION DATA")
print("=" * 60)

X_val, y_val = load_images(val_df)

print(
    f"X_val shape: {X_val.shape}"
)

print(
    f"y_val shape: {y_val.shape}"
)

print()


print("=" * 60)
print("LOADING TEST DATA")
print("=" * 60)

X_test, y_test = load_images(test_df)

print(
    f"X_test shape: {X_test.shape}"
)

print(
    f"y_test shape: {y_test.shape}"
)

print()


# ============================================================
# MOBILE NET V2 PREPROCESSING
# ============================================================

# Existing processed images are 0-1.
# MobileNetV2 expects approximately -1 to +1.

X_train = (X_train * 2.0) - 1.0
X_val = (X_val * 2.0) - 1.0
X_test = (X_test * 2.0) - 1.0


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation"
)


# ============================================================
# BUILD CNN
# ============================================================

print("=" * 60)
print("BUILDING CNN")
print("=" * 60)

base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Initially freeze the pretrained network

base_model.trainable = False


inputs = layers.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.40)(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)


model = models.Model(
    inputs,
    outputs
)


model.compile(
    optimizer=Adam(
        learning_rate=1e-4
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

best_model_path = os.path.join(
    MODELS_DIR,
    "deepfake_cnn_rigorous_best.keras"
)

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),

    ModelCheckpoint(
        best_model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]


# ============================================================
# PHASE 1 — TRAIN CLASSIFICATION HEAD
# ============================================================

print()
print("=" * 60)
print("PHASE 1 — TRAINING CLASSIFICATION HEAD")
print("=" * 60)

history1 = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=15,

    batch_size=16,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# PHASE 2 — FINE-TUNE LAST CNN LAYERS
# ============================================================

print()
print("=" * 60)
print("PHASE 2 — FINE-TUNING MOBILENETV2")
print("=" * 60)


base_model.trainable = True


# Freeze earlier layers.
# Only the last 30 layers are fine-tuned.

for layer in base_model.layers[:-30]:

    layer.trainable = False


model.compile(

    optimizer=Adam(
        learning_rate=1e-5
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)


history2 = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=10,

    batch_size=16,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

if os.path.exists(best_model_path):

    model = tf.keras.models.load_model(
        best_model_path
    )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print()
print("=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

probabilities = model.predict(
    X_test,
    batch_size=16,
    verbose=1
).ravel()


predictions = (
    probabilities >= 0.5
).astype(int)


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


print()
print("=" * 60)
print("RIGOROUS CNN MODEL EVALUATION")
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

print()


print("Classification Report:")

print(
    classification_report(
        y_test,
        predictions,
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
    predictions
)


print("Confusion Matrix:")

print(cm)


# ============================================================
# SAVE RESULTS
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
    "cnn_rigorous_evaluation_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


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
    "cnn_rigorous_confusion_matrix.csv"
)

cm_df.to_csv(
    cm_path
)


# Save final model

final_model_path = os.path.join(
    MODELS_DIR,
    "deepfake_cnn_rigorous_final.keras"
)

model.save(
    final_model_path
)


print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print()
print("Best model:")
print(best_model_path)

print()
print("Final model:")
print(final_model_path)

print()
print("Metrics:")
print(metrics_path)

print()
print("Confusion matrix:")
print(cm_path)

print()
print("The model was evaluated on a completely")
print("held-out source-group test set.")