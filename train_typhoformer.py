"""
train_typhoformer.py
--------------------------------------------------
Training pipeline for TyphoFormer model (built upon STAEformer)
Author: Lincan Li
Date: 2025-06-02
"""

import os
import glob
import json
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.TyphoFormer import TyphoFormer  # import your model


# Configuration
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
SAVE_DIR = "checkpoints"

BATCH_SIZE = 1
NUM_EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_LEN = 12
PRED_LEN = 1

# 需要根据prepare_typhoformer_data.py的输出定义这些维度
D_NUM = 14 #数值特征维度（可根据CSV中数值列数量调整）
D_TEXT = 384 #language embedding维度（all-MiniLM-L6-v2）

os.makedirs(SAVE_DIR, exist_ok=True)


# Dataset Definition
class TyphoonDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        assert len(self.files) > 0, f"No .npy samples found in {data_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        X = torch.tensor(data["input"], dtype=torch.float32)
        Y = torch.tensor(data["target"], dtype=torch.float32)
        return X, Y


# =========================================================
# Loss Function (MSE or Haversine)
def haversine_loss(pred, target):
    """
    Compute mean Haversine distance (km) between predicted and target coordinates.
    pred, target: (B, T, 2)
    """
    R = 6371.0  # Earth radius in km
    lat1, lon1 = torch.deg2rad(pred[..., 0]), torch.deg2rad(pred[..., 1])
    lat2, lon2 = torch.deg2rad(target[..., 0]), torch.deg2rad(target[..., 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    return (R * c).mean()


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X, Y in tqdm(loader, desc="Training", leave=False):
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        optimizer.zero_grad()

        # 拆分数值特征和语言embedding
        X_num = X[..., :D_NUM]     # 数值部分
        X_text = X[..., D_NUM:]    # 语言embedding部分
        y_last = Y[:, 0, :]        # 初始坐标（上一真实点）

        #前向传播
        pred = model(X_num, X_text, y_last, pred_steps=Y.shape[1])

        #损失计算 & 反向传播
        loss = criterion(pred, Y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, Y in tqdm(loader, desc="Validation", leave=False):
            X, Y = X.to(DEVICE), Y.to(DEVICE)

            X_num = X[..., :D_NUM]
            X_text = X[..., D_NUM:]
            y_last = Y[:, 0, :]

            pred = model(X_num, X_text, y_last, pred_steps=Y.shape[1])
            loss = criterion(pred, Y)
            total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)



def main():
    print(f"Training TyphoFormer on {DEVICE}")

    train_ds = TyphoonDataset(TRAIN_DIR)
    val_ds = TyphoonDataset(VAL_DIR)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 配置参数
    cfg = {
        "model": {
            "input_dim": D_NUM, #数值特征维度
            "text_dim": D_TEXT, #语言embedding维度(all-MiniLM-L6-v2输出维度)
            "embed_dim": 128, #融合后隐层维度
            "output_dim": 2, #output coordinate [lat, lon]
            "input_len": INPUT_LEN, #输出时间步
            "pred_len": PRED_LEN #预测时间步
        }
    }

    model = TyphoFormer(cfg).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()  # 可切换为 haversine_loss

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
            print(f"Saved best model at epoch {epoch+1} (Val Loss={val_loss:.6f})")

    print("Training completed!")


if __name__ == "__main__":
    main()
