"""
eval_typhoformer.py
-----------------------------------------------
Evaluate trained TyphoFormer model on test set.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model.TyphoFormer import TyphoFormer
from train_typhoformer import haversine_loss

# =========================
# Configuration
# =========================
DATA_DIR = "data/test"
CHECKPOINT = "checkpoints/best_model.pt"
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_LEN = 12
PRED_LEN = 1


# =========================
# Dataset Loader
# =========================
class TyphoonDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
        self.data_dir = data_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_dir, self.files[idx]), allow_pickle=True).item()
        return torch.tensor(data["input"], dtype=torch.float32), torch.tensor(data["target"], dtype=torch.float32)


# =========================
# Evaluation
# =========================
def evaluate(model, loader):
    model.eval()
    total_mse, total_hav = 0.0, 0.0
    count = 0
    with torch.no_grad():
        for X, Y in tqdm(loader, desc="Evaluating"):
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            pred = model(X)

            mse = torch.mean((pred - Y) ** 2).item()
            hav = haversine_loss(pred, Y).item()

            total_mse += mse * X.size(0)
            total_hav += hav * X.size(0)
            count += X.size(0)

    print(f"\nTest MSE: {total_mse / count:.6f}")
    print(f"Mean Haversine Distance: {total_hav / count:.3f} km")



# Main
def main():
    print(f"Evaluating TyphoFormer on {DEVICE}")

    dataset = TyphoonDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TyphoFormer(input_dim=dataset[0][0].shape[-1],
                        input_len=INPUT_LEN,
                        pred_len=PRED_LEN).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    print(f"Loaded model from {CHECKPOINT}")

    evaluate(model, loader)


if __name__ == "__main__":
    main()
