from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "results" / "processed"
METADATA_FILE = PROJECT_ROOT / "results" / "processed_metadata.csv"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
RANDOM_STATE = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 60)
print("CNN DEEPFAKE DETECTION")
print("=" * 60)

metadata = pd.read_csv(METADATA_FILE)

print(f"\nTotal processed frames: {len(metadata)}")


# ============================================================
# CREATE VIDEO-LEVEL SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(
        metadata,
        metadata["label"],
        groups=metadata["video"]
    )
)

train_df = metadata.iloc[train_idx].copy()
test_df = metadata.iloc[test_idx].copy()

train_videos = set(train_df["video"])
test_videos = set(test_df["video"])

overlap = train_videos.intersection(test_videos)

if overlap:
    raise RuntimeError(
        f"Video leakage detected: {overlap}"
    )

print("\nVideo-level split:")
print(f"Training videos: {len(train_videos)}")
print(f"Testing videos : {len(test_videos)}")

print(f"\nTraining frames: {len(train_df)}")
print(f"Testing frames : {len(test_df)}")

print("\nTraining class distribution:")
print(train_df["label"].value_counts())

print("\nTesting class distribution:")
print(test_df["label"].value_counts())


# ============================================================
# IMAGE LOADER
# ============================================================

def load_images(dataframe):

    images = []
    labels = []

    for _, row in dataframe.iterrows():

        path = PROCESSED_DIR / row["file"]

        image = np.load(path).astype(np.float32)

        if image.shape != (224, 224, 3):
            continue

        images.append(image)

        if row["label"] == "Real":
            labels.append(0)
        else:
            labels.append(1)

    return np.array(images), np.array(labels)


print("\nLoading training images...")

X_train, y_train = load_images(train_df)

print(
    f"Training image array: {X_train.shape}"
)

print("\nLoading testing images...")

X_test, y_test = load_images(test_df)

print(
    f"Testing image array: {X_test.shape}"
)


# ============================================================
# MOBILE NET V2 EXPECTS -1 TO +1
# ============================================================

X_train = X_train * 2.0 - 1.0
X_test = X_test * 2.0 - 1.0


# ============================================================
# BUILD CNN
# ============================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Initially freeze the convolutional base
base_model.trainable = False


model = models.Sequential([
    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.30),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.30),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy"
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nCNN architecture:")

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

model_file = (
    MODELS_DIR
    / "deepfake_cnn_best.keras"
)

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        model_file,
        monitor="val_loss",
        save_best_only=True
    )
]


# ============================================================
# TRAIN CLASSIFIER HEAD
# ============================================================

print("\n" + "=" * 60)
print("PHASE 1: TRAINING CNN CLASSIFIER")
print("=" * 60)

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_test,
        y_test
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# FINE-TUNE LAST CNN LAYERS
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2: FINE-TUNING CNN")
print("=" * 60)

base_model.trainable = True

# Freeze most layers and fine-tune only the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy"
    ]
)

fine_tune_history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_test,
        y_test
    ),
    epochs=10,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
).ravel()

y_pred = (
    probabilities >= 0.5
).astype(int)


# ============================================================
# METRICS
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
print("CNN MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


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
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame([{
    "model": "MobileNetV2 Fine-Tuned CNN",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "train_frames": len(X_train),
    "test_frames": len(X_test),
    "train_videos": len(train_videos),
    "test_videos": len(test_videos)
}])

metrics_file = (
    RESULTS_DIR
    / "cnn_evaluation_metrics.csv"
)

metrics.to_csv(
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
    / "cnn_confusion_matrix.csv"
)

cm_df.to_csv(
    cm_file
)


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(cm)

plt.title(
    "Fine-Tuned CNN Deepfake Detection"
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

plot_file = (
    RESULTS_DIR
    / "cnn_confusion_matrix.png"
)

plt.savefig(
    plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_file = (
    MODELS_DIR
    / "deepfake_cnn_final.keras"
)

model.save(
    final_model_file
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("CNN TRAINING COMPLETE")
print("=" * 60)

print("\nModel:")
print(final_model_file)

print("\nMetrics:")
print(metrics_file)

print("\nConfusion matrix:")
print(cm_file)

print("\nConfusion matrix image:")
print(plot_file)

print("\nCNN deepfake detector is ready.")