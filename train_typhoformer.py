"""
train_typhoformer.py
--------------------------------------------------
Training pipeline for TyphoFormer model
Author: Lincan Li
Date: 2025-06-02
"""

import os
import glob
import json
import time
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.TyphoFormer import TyphoFormer  # import your model
from utils import *



DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
SAVE_DIR = "checkpoints"

BATCH_SIZE = 8
NUM_EPOCHS = 80
LR = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_LEN = 12
PRED_LEN = 1

# 需要根据prepare_typhoformer_data.py的输出定义这些维度
D_NUM = 14 #dimension of numerical features（You can adjust `D_NUM` according to the exact numerical feature in your .CSV file）
D_TEXT = 384 #dimesion of the language embedding（all-MiniLM-L6-v2）

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================================================
# Dataset Definition
# =========================================================
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
# =========================================================
def haversine_loss(pred, target):
    """
    Compute mean Haversine distance (km) between predicted and target coordinates.
    pred, target: (B, T, 2)
    """
    R = 6371.0  # Earth radius in kilometers(km)
    lat1, lon1 = torch.deg2rad(pred[..., 0]), torch.deg2rad(pred[..., 1])
    lat2, lon2 = torch.deg2rad(target[..., 0]), torch.deg2rad(target[..., 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    return (R * c).mean()


# Training Function
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    for X, Y in tqdm(loader, desc="Training", leave=False):
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        optimizer.zero_grad()

        # 拆分数值特征和语言embedding
        X_num = X[..., :D_NUM]     # 数值部分
        X_text = X[..., D_NUM:]    # 语言embedding部分
        y_last = Y[:, 0, :]        # 初始坐标（上一真实点）

        # 前向传播
        output, gate = model(X_num, X_text, y_last, pred_steps=Y.shape[1])
        if isinstance(output, tuple):
            pred, _ = output
        else:
            pred = output

        # Loss Calculation & Backpropagation
        lambda_gate = 0.1
        gate_penalty = torch.mean((0.6 - gate).clamp(min=0) ** 2) #penalize `gate` that has too small value
        #incorporate a penalty term in the total loss function 'lambda_gate * gate_penalty'
        loss = criterion(pred, Y) + lambda_gate * gate_penalty
        mae = torch.mean(torch.abs(pred-Y))

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total_mae += mae.item() * X.size(0)
        avg_loss = total_loss / len(loader.dataset)
        avg_mae = total_mae / len(loader.dataset)
    return avg_loss, avg_mae


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_mae = 0.0, 0.0
    with torch.no_grad():
        for X, Y in tqdm(loader, desc="Validation", leave=False):
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            X_num, X_text = X[..., :D_NUM], X[..., D_NUM:]
            y_last = Y[:, 0, :]
            #pred = model(X_num, X_text, y_last, pred_steps=Y.shape[1])
            output, gate = model(X_num, X_text, y_last, pred_steps=Y.shape[1])
            if isinstance(output, tuple):
                pred, _ = output
            else:
                pred = output

            loss = criterion(pred, Y)
            mae = torch.mean(torch.abs(pred - Y))

            total_loss += loss.item() * X.size(0)
            total_mae += mae.item() * X.size(0)

    avg_loss = total_loss / len(loader.dataset)
    avg_mae = total_mae / len(loader.dataset)
    return avg_loss, avg_mae


# Main Training Loop
def main():
    print(f"Training TyphoFormer on {DEVICE}")
    start_time = time.time()
    train_ds = TyphoonDataset(TRAIN_DIR)
    val_ds = TyphoonDataset(VAL_DIR)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    cfg = {
    "model": {
        # === 输入结构 ===
        "input_dim": 14,                        # 数值特征维度 (D_NUM)
        "text_dim": 384,                        # 语言 embedding 维度 (D_TEXT)
        "embed_dim": 256,                       # 融合后隐层维度 (PGF 输出前隐层)
        "output_dim": 2,                        # 纬度: [lat, lon]

        # === 时序预测结构 ===
        "input_len": INPUT_LEN,                 # 输入时间步
        "pred_len": PRED_LEN,                   # 预测时间步

        # === Transformer 架构参数 ===
        "d_model": 256,                         # 模型维度 (PGF 输出维度)
        "d_ff": 1024,                           # 前馈层隐藏维度
        "num_heads": 4,                         # 多头注意力头数
        "num_layers": 3,                        # ST-Transformer 层数
        "dropout": 0.1,                         # Dropout 比例
        }
    }

    model = TyphoFormer(cfg, debug = False).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()  # 可切换为 haversine_loss
    best_val_loss = float("inf")
    # ========Debug Begins========
    # for i, (X, Y) in enumerate(train_loader):
    #     print("\n[Dataset] batch shapes:")
    #     print_shape("X", X)
    #     print_shape("Y", Y)
    #     break
    # # ========Debug Ends===========
 
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_mae = evaluate(model, val_loader, criterion)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}" 
              f"| Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
              f"| Train MAE: {train_mae:.6f} | Val MAE: {val_mae:.6f}"
              f"| LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
              )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
            print(f"Saved best model at epoch {epoch+1} (Val Loss={val_loss:.6f})")
    
    #total training time
    total_time = (time.time() - start_time) / 60
    print(f"Training completed in {total_time:.2f} min.")


if __name__ == "__main__":
    main()
