"""
prepare_typhoformer_data.py
Process typhoon CSV data and text embeddings into
TyphoFormer-compatible input format.

Output structure:
data/
 ├── train/
 │     ├── AL142024.npy
 │     ├── AL152024.npy
 │     ├── ...
 ├── val/
 │     ├── ...
 └── test/
       ├── ...

Each .npy file can be loaded as:
data = np.load(path, allow_pickle=True).item()
X = data["input"]
Y = data["target"]
--------------------------------------------------
Author: Lincan Li
Date: 2025-06-02
"""

import os
import numpy as np
import pandas as pd
import glob
import json
from sklearn.model_selection import train_test_split


# Configuration
CSV_PATH = "HURDAT_2new_3000_with_descriptions_clean.csv" #"HURDAT_2new_3000_with_descriptions_clean.csv"is the generated csv by running generate_text_description_new.py
EMB_DIR = "embedding_chunks"
OUTPUT_DIR = "data"

INPUT_LEN = 12      # number of historical timesteps used as input
PRED_LEN = 1        # number of timesteps to predict
VAL_RATIO = 0.10
TEST_RATIO = 0.15

# Step 1. Load and validate CSV
print(f"Loading CSV data: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)


# numerical feature columns that we want to use
required_cols = [
    "typhoon_id", "date", "time",
    "latitude", "longitude",
    "max_wind", "min_pressure",
    "wind_radii_34_NE", "wind_radii_34_SE", "wind_radii_34_SW", "wind_radii_34_NW",
    "wind_radii_50_NE", "wind_radii_50_SE", "wind_radii_50_SW", "wind_radii_50_NW",
    "wind_radii_64_NE", "wind_radii_64_SE", "wind_radii_64_SW", "wind_radii_64_NW"
]


for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column '{col}' in CSV!")

print(f"Loaded {len(df)} total records.")

# Step 2. Merge text embeddings (from chunks)
print(f"Loading embeddings from chunks in: {EMB_DIR}")
chunk_files = sorted(glob.glob(os.path.join(EMB_DIR, "emb_chunk_*.npy")))
if len(chunk_files) == 0:
    raise ValueError("No embedding chunks found! Make sure to run generate_text_embeddings.py first.")

embeddings = np.concatenate([np.load(f) for f in chunk_files], axis=0)

if len(embeddings) != len(df):
    raise ValueError(f"Embedding count mismatch: {len(embeddings)} embeddings vs {len(df)} CSV rows")

df["embedding"] = list(embeddings)
print(f"Embeddings merged: shape = {embeddings.shape}")


# Step 3. Sort and group by typhoon ID
groups = df.groupby("typhoon_id")

typhoon_ids = list(groups.groups.keys())
print(f"Found {len(typhoon_ids)} unique typhoons.")


# Step 4. Split train/val/test (by typhoon, not by record)
train_ids, temp_ids = train_test_split(typhoon_ids, test_size=VAL_RATIO + TEST_RATIO, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO), random_state=42)

splits = {"train": train_ids, "val": val_ids, "test": test_ids}
for split in splits:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

print(f"Data split: {len(train_ids)} train | {len(val_ids)} val | {len(test_ids)} test typhoons")


# Step 5. Process each typhoon trajectory
def process_storm(typhoon_df, typhoon_id, save_dir):
    """
    Process one typhoon trajectory into sliding window samples.
    Each output .npy file contains a dict with:
        {"id": typhoon_id, "input": X_seq, "target": Y_seq}
    """
    typhoon_df = typhoon_df.sort_values(by=["date", "time"]).reset_index(drop=True)

    # ---- Convert latitude/longitude from string like "12.7N" -> float
    def convert_latlon(val):
        if isinstance(val, str):
            val = val.strip()
            if val.endswith("N"):
                return float(val[:-1])
            elif val.endswith("S"):
                return -float(val[:-1])
            elif val.endswith("E"):
                return float(val[:-1])
            elif val.endswith("W"):
                return -float(val[:-1])
        return float(val)

    typhoon_df["latitude"] = typhoon_df["latitude"].apply(convert_latlon)
    typhoon_df["longitude"] = typhoon_df["longitude"].apply(convert_latlon)

    # ---- Select numeric meteorological features (exclude lat/lon and text)
    feature_cols = [
        "max_wind", "min_pressure",
        "wind_radii_34_NE", "wind_radii_34_SE", "wind_radii_34_SW", "wind_radii_34_NW",
        "wind_radii_50_NE", "wind_radii_50_SE", "wind_radii_50_SW", "wind_radii_50_NW",
        "wind_radii_64_NE", "wind_radii_64_SE", "wind_radii_64_SW", "wind_radii_64_NW"
    ]

    # ---- Build numerical + semantic feature arrays
    X_num = typhoon_df[feature_cols].to_numpy(dtype=np.float32)
    X_embed = np.stack(typhoon_df["embedding"].to_numpy())
    X_full = np.concatenate([X_num, X_embed], axis=1)  # shape = (T, F_num + F_emb)

    # ---- Prediction targets (lat/lon)
    coords = typhoon_df[["latitude", "longitude"]].to_numpy(dtype=np.float32)

    num_seq = len(X_full)
    if num_seq <= INPUT_LEN + PRED_LEN:
        return 0  # skip too short typhoon tracks

    samples = 0
    
    #Construct the sliding Window for the dataset
    for i in range(num_seq - INPUT_LEN - PRED_LEN + 1):
        X_seq = X_full[i : i + INPUT_LEN]
        Y_seq = coords[i + INPUT_LEN : i + INPUT_LEN + PRED_LEN]

        sample = {
            "id": typhoon_id,
            "input": X_seq,
            "target": Y_seq
        }

        save_path = os.path.join(save_dir, f"{typhoon_id}_{i:03d}.npy")
        np.save(save_path, sample)
        samples += 1

    return samples


# Step 6. Generate TyphoFormer-ready data
total_samples = 0
for split, ids in splits.items():
    split_dir = os.path.join(OUTPUT_DIR, split)
    print(f"\nProcessing {split} set ({len(ids)} typhoons)...")

    count = 0
    for sid in ids:
        typhoon_df = groups.get_group(sid)
        count += process_storm(typhoon_df, sid, split_dir)
    
    print(f"{split} set completed: {count} samples saved.")
    total_samples += count

print(f"\nData preparation finished! Total {total_samples} samples saved across all splits.")

# Step 7. Save metadata
metadata = {
    "input_csv": CSV_PATH,
    "embedding_dir": EMB_DIR,
    "input_len": INPUT_LEN,
    "pred_len": PRED_LEN,
    "train_storms": len(train_ids),
    "val_storms": len(val_ids),
    "test_storms": len(test_ids),
    "total_samples": total_samples,
    "feature_dim": embeddings.shape[1],
}

with open(os.path.join(OUTPUT_DIR, "dataset_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print(f"\nMetadata saved to: {os.path.join(OUTPUT_DIR, 'dataset_metadata.json')}")
print("TyphoFormer-ready dataset successfully generated!")
