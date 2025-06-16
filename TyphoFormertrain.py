import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from TyphoFormer import TyphoFormer
from data import MyData

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

def train(model_name, load_pretrained=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 5000
    num_epochs = 20
    max_epochs = 3000

    setup_seed(10086)
    train_dataset = MyData(data_path='/data/TrainData.json',l=4,frac=1)
    val_dataset = MyData(data_path='/data/TestData.json',l=4)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TyphoFormer().to(device)

    ckpt_path = f'checkpoints/{model_name}/best.pth'
    os.makedirs(f'checkpoints/{model_name}', exist_ok=True)

    if load_pretrained and os.path.exists(ckpt_path):
        print(f"Loading pretrained weights from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))
    else:
        print("Training from scratch")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_rmse = float('inf')
    history = []

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            ts_input, target, text_embed, _ = [x.to(device) for x in batch]
            optimizer.zero_grad()
            output = model(ts_input, text_embed, pred_len=1)
            loss = criterion(output.squeeze(1), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # 验证阶段
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                ts_input, target, text_embed, _ = [x.to(device) for x in batch]
                output = model(ts_input, text_embed, pred_len=1)
                rmse = torch.sqrt(torch.mean((output.squeeze(1) - target) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    print(f"New best RMSE: {rmse.item():.4f}, saving model...")
                    torch.save(model.state_dict(), ckpt_path)

        history.append((epoch, avg_train_loss, rmse.item()))
        scheduler.step()

        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val RMSE: {rmse.item():.4f}")

    # 保存训练日志
    df = pd.DataFrame(history, columns=['Epoch', 'Loss', 'Val_RMSE'])
    df.to_csv(f'checkpoints/{model_name}/history.csv', index=False)

def test(model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = MyData(data_path='./data/TestData.json', l=4, frac=1)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)

    model = TyphoFormer().to(device)
    model.load_state_dict(torch.load(f'checkpoints/{model_name}/best.pth'))
    model.eval()

    all_preds = []
    all_gts = []
    with torch.no_grad():
        for batch in test_loader:
            ts_input, target, text_embed, _ = [x.to(device) for x in batch]
            pred = model(ts_input, text_embed, pred_len=1)
            all_preds.append(pred.cpu())
            all_gts.append(target.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    gts = torch.cat(all_gts, dim=0).numpy()
    se = np.sum((preds - gts) ** 2, axis=1)
    rmse = np.sqrt(np.mean(se))
    print(f'Test RMSE: {rmse:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch', action='store_true', help='Train from scratch (ignore checkpoints)')
    args = parser.parse_args()

    model_name = 'TyphoFormer'
    train(model_name, load_pretrained=not args.scratch)
    test(model_name)
    print('Training and Testing Complete!')
