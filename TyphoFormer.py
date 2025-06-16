import torch
import torch.nn as nn
import os
from PGF_Fusion import TimeSeriesProjector, PGFModule

# Transformer Encoder Block
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)

    def forward(self, x):
        return self.layer(x)

# 自回归 GRU Decoder
class GRUDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoder_output, pred_len=1):
        # 取 Transformer 最后时间步的上下文向量作为解码初始隐状态
        h_0 = encoder_output[:, -1:, :]  # shape [B, 1, H]
        decoder_input = h_0  # 初始输入
        outputs = []

        for _ in range(pred_len):
            out, h_0 = self.gru(decoder_input, h_0.transpose(0, 1))  # h_0: [1, B, H]
            pred = self.fc(out[:, -1, :])  # 预测输出 [B, output_dim]
            outputs.append(pred.unsqueeze(1))  # [B, 1, output_dim]
            decoder_input = out  # 下一步继续使用当前输出

        return torch.cat(outputs, dim=1)  # [B, pred_len, output_dim]

# 整体 TyphoFormer 模型
class TyphoFormer(nn.Module):
    def __init__(self, input_dim=4, embedding_dim=768, nhead=8, num_layers=4, output_dim=2):
        super().__init__()
        self.embedding_dim = embedding_dim

        # 初步模块
        self.ts_projector = TimeSeriesProjector(input_dim, embedding_dim)
        self.pgf = PGFModule(embedding_dim)

        # Transformer Encoder
        encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embedding_dim, nhead) for _ in range(num_layers)
        ])
        self.transformer_encoder = nn.Sequential(*encoder_layers)

        # GRU Decoder
        self.decoder = GRUDecoder(input_dim=embedding_dim, hidden_dim=embedding_dim, output_dim=output_dim)

    def forward(self, ts_input, text_embedding, pred_len=1):
        """
        ts_input: [B, T, 4]
        text_embedding: [B, 768]
        """
        ts_embedded = self.ts_projector(ts_input)  # [B, T, 768]
        fused = torch.stack([self.pgf(ts_embedded[i], text_embedding[i]) for i in range(ts_input.size(0))])
        encoded = self.transformer_encoder(fused)  # [B, T, 768]
        prediction = self.decoder(encoded, pred_len=pred_len)  # [B, pred_len, 2]
        return prediction

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
