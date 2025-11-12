"""
generate_text_embeddings.py
--------------------------------------------------
Step 2 of TyphoFormer Implementation:
Convert GPT-generated language descriptions into dense text embeddings
using Sentence-Transformers model `all-MiniLM-L6-v2`.

Author: Lincan Li
Date: 2025-06-02
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


# Configuration
INPUT_CSV = "HURDAT_2new_3000_with_descriptions.csv"
OUTPUT_DIR = "embedding_chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE = 500  # Number of rows per embedding chunk
OUTPUT_METADATA = os.path.join(OUTPUT_DIR, "embedding_metadata.json")
MODEL_NAME = "all-MiniLM-L6-v2"  # Recommended model


# Load Dataset
print(f"Loading dataset: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

if "description" not in df.columns:
    raise ValueError("The input CSV must contain a 'description' column with text descriptions.")

descriptions = df["description"].astype(str).tolist()
total_records = len(descriptions)
print(f"Loaded {total_records} rows with descriptions.")


# Load Model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print(f"Loading SentenceTransformer model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, device=device)


# ============================================================
# Generate and Save Embeddings in Chunks
# ============================================================
start_time = time.time()
print(f"\n Encoding and saving embeddings in chunks of {CHUNK_SIZE}...")

for start_idx in range(0, total_records, CHUNK_SIZE):
    end_idx = min(start_idx + CHUNK_SIZE, total_records)
    chunk_id = start_idx // CHUNK_SIZE
    chunk_path = os.path.join(OUTPUT_DIR, f"emb_chunk_{chunk_id:03d}.npy")

    # Skip if chunk already exists (resume mode)
    if os.path.exists(chunk_path):
        print(f"Skipping existing {chunk_path}")
        continue

    # Extract current chunk of text
    chunk_texts = descriptions[start_idx:end_idx]
    print(f"\n Processing chunk {chunk_id} ({start_idx}–{end_idx}) ...")

    # Encode embeddings for this chunk
    chunk_embeddings = model.encode(
        chunk_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Save chunk
    np.save(chunk_path, chunk_embeddings)
    print(f"Saved chunk {chunk_id} → {chunk_path} ({chunk_embeddings.shape})")

duration = time.time() - start_time
print(f"\n Finished all chunks in {duration/60:.2f} minutes.")


# Save Metadata
metadata = {
    "model": MODEL_NAME,
    "input_file": INPUT_CSV,
    "num_records": total_records,
    "embedding_dim": 384,
    "chunk_size": CHUNK_SIZE,
    "num_chunks": (total_records + CHUNK_SIZE - 1) // CHUNK_SIZE,
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "device_used": device,
}

with open(OUTPUT_METADATA, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Metadata saved to: {OUTPUT_METADATA}")


# ============================================================
# Optional: Merge all chunks into one big .npy file
# ============================================================
MERGE = False  # Set True if you want to merge immediately

if MERGE:
    print("\nMerging all chunks into one big embedding matrix...")
    chunk_files = sorted([os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.endswith(".npy")])
    all_chunks = [np.load(f) for f in chunk_files]
    embeddings_full = np.concatenate(all_chunks, axis=0)
    np.save(os.path.join(OUTPUT_DIR, "description_embeddings_full.npy"), embeddings_full)
    print(f"Merged embeddings saved to {os.path.join(OUTPUT_DIR, 'description_embeddings_full.npy')}")
    print(f"Final shape: {embeddings_full.shape}")


print("\nStep 2 completed successfully!")
print("These embeddings can now be used for PGF fusion in TyphoFormer.")
