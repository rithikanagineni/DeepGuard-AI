from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "results" / "processed"
METADATA_FILE = PROJECT_ROOT / "results" / "processed_metadata.csv"

FEATURES_DIR = PROJECT_ROOT / "results" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = FEATURES_DIR / "cnn_features.npz"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 60)
print("CNN FEATURE EXTRACTION")
print("=" * 60)

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"Metadata file not found:\n{METADATA_FILE}"
    )

metadata = pd.read_csv(METADATA_FILE)

print(f"Total processed frames: {len(metadata)}")


# ============================================================
# LOAD PRETRAINED CNN
# ============================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
    pooling="avg"
)

base_model.trainable = False

feature_model = Model(
    inputs=base_model.input,
    outputs=base_model.output
)

print("MobileNetV2 loaded successfully.")
print(f"Feature vector size: {feature_model.output_shape[-1]}")


# ============================================================
# EXTRACT FEATURES
# ============================================================

features = []
labels = []
videos = []
files = []

print("\nExtracting CNN features...")


for _, row in tqdm(
    metadata.iterrows(),
    total=len(metadata),
    desc="Processing frames"
):

    image_path = PROCESSED_DIR / row["file"]

    if not image_path.exists():
        print(f"\nWarning: File not found: {image_path}")
        continue

    try:
        # ----------------------------------------------------
        # LOAD NUMPY IMAGE
        # ----------------------------------------------------
        image_array = np.load(image_path)

        # ----------------------------------------------------
        # CHECK IMAGE SHAPE
        # ----------------------------------------------------
        if image_array.shape != (224, 224, 3):
            print(
                f"\nSkipping unexpected shape "
                f"{image_array.shape}: {image_path}"
            )
            continue

        # ----------------------------------------------------
        # CONVERT TO FLOAT32
        # ----------------------------------------------------
        image_array = image_array.astype(np.float32)

        # ----------------------------------------------------
        # IMPORTANT:
        # Preprocessing script already normalized pixels
        # to the range 0-1.
        #
        # MobileNetV2 expects values approximately -1 to 1.
        # Therefore convert 0-1 -> -1 to 1.
        # ----------------------------------------------------
        image_array = image_array * 2.0 - 1.0

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # ----------------------------------------------------
        # CNN FEATURE EXTRACTION
        # ----------------------------------------------------
        feature = feature_model.predict(
            image_array,
            verbose=0
        )[0]

        features.append(feature)

        # Real = 0
        # Fake = 1
        if row["label"] == "Real":
            labels.append(0)
        else:
            labels.append(1)

        videos.append(row["video"])
        files.append(row["file"])

    except Exception as e:

        print(
            f"\nError processing {image_path}: {e}"
        )


# ============================================================
# CONVERT TO NUMPY ARRAYS
# ============================================================

features = np.array(features, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)
videos = np.array(videos)
files = np.array(files)


# ============================================================
# CHECK RESULTS
# ============================================================

if len(features) == 0:
    raise RuntimeError(
        "\nNo CNN features were extracted.\n"
        "Please check the processed .npy files."
    )


# ============================================================
# SAVE FEATURES
# ============================================================

np.savez_compressed(
    OUTPUT_FILE,
    features=features,
    labels=labels,
    videos=videos,
    files=files
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CNN FEATURE EXTRACTION COMPLETE")
print("=" * 60)

print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")

print(f"Real samples: {np.sum(labels == 0)}")
print(f"Fake samples: {np.sum(labels == 1)}")

print("\nFeature file saved to:")
print(OUTPUT_FILE)

print("\nFeature extraction completed successfully.")