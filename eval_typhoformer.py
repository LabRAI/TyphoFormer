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
BATCH_SIZE = 8
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
    total_mae, total_hav = 0.0, 0.0
    count = 0

    with torch.no_grad():
        for X, Y in tqdm(loader, desc="Evaluating"):
            X, Y = X.to(DEVICE), Y.to(DEVICE)

            # === 拆分输入 ===
            X_num = X[..., :14]        # 数值特征部分
            X_text = X[..., 14:]       # 文本 embedding 部分
            y_last = Y[:, 0, :]        # 上一步真实坐标

            # === 模型推理 ===
            print(torch.cuda.get_device_name(torch.cuda.current_device())) #which GPU I'm currently use
            output, gate = model(X_num, X_text, y_last, pred_steps=Y.shape[1])
            if isinstance(output, tuple):
                pred, _ = output
            else:
                pred = output

            # === 计算误差 ===
            mae = torch.mean(torch.abs(pred - Y)).item()
            hav = haversine_loss(pred, Y).item()

            total_mae += mae * X.size(0)
            total_hav += hav * X.size(0)
            count += X.size(0)

    print(f"\nTest MAE: {total_mae / count:.6f}")
    print(f"Mean Haversine Distance: {total_hav / count:.3f} km")


# Main
def main():
    print(f"Evaluating TyphoFormer on {DEVICE}")

    dataset = TyphoonDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

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
        "num_layers": 3,                        # STAEformer 层数
        "dropout": 0.1,                         # Dropout 比例
        }
    }

    model = TyphoFormer(cfg, debug=False).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    print(f"Loaded model from {CHECKPOINT}")

    evaluate(model, loader)


if __name__ == "__main__":
    main()
