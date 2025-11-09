"""
TyphoFormer: Language-Enhanced Spatio-Temporal Transformer for Typhoon Track Forecasting
-----------------------------------------------------------------------------------------
基于STAEformer的结构进行改造,最终的模型架构如下所示:
1. PGF多模态融合模块
2. 与STAEformer完全一致的Encoder部分架构
3. Autoregressive Decoder Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ST-Transformer import SpatialTemporalFormer  # 复用STAEformer的编码器
from model.PGF_module import PromptGatingFusion  # Step3中实现的PGF模块


# ============================================================
# Autoregressive Decoder Head
class TyphoDecoder(nn.Module):
    """
    自回归式解码器 (Autoregressive Regression Head)
    用于逐步预测未来台风轨迹坐标。
    """

    def __init__(self, hidden_dim, output_dim=2):
        """
        Args:
            hidden_dim: 编码器输出维度
            output_dim: 预测维度（默认为2 -> [lat, lon]）
        """
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim + output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, h_enc, y_prev, pred_steps):
        """
        Args:
            h_enc: [B, hidden_dim] 最后时间步的编码器输出
            y_prev: [B, output_dim] 初始输入（最后一个真实坐标）
            pred_steps: 要预测的时间步数

        Returns:
            preds: [B, pred_steps, output_dim]
        """
        preds = []
        y_t = y_prev
        for _ in range(pred_steps):
            z_t = torch.cat([h_enc, y_t], dim=-1)
            y_t = self.fc2(F.relu(self.fc1(z_t)))
            preds.append(y_t)
        preds = torch.stack(preds, dim=1)
        return preds


# class TyphoFormer(nn.Module):
#     def __init__(self, cfg):
#         """
#         Args:
#             cfg: 从配置文件读取的参数 (可以复用 STAEformer.yaml)
#         """
#         super().__init__()
#         self.d_num = cfg['model']['input_dim']          # 数值特征维度
#         self.d_text = cfg['model']['text_dim']          # 语言嵌入维度
#         self.d_model = cfg['model']['embed_dim']        # 模型隐藏维度
#         self.output_dim = cfg['model']['output_dim']    # 输出纬度（通常2）

#         # PGF 模块
#         self.pgf = PromptGatingFusion(self.d_num, self.d_text, self.d_model)

#         # STAEformer 编码器
#         self.encoder = SpatioTemporalEncoder(cfg)

#         # 3自回归解码器
#         self.decoder = TyphoDecoder(hidden_dim=self.d_model, output_dim=self.output_dim)

#     def forward(self, x_num, x_text, y_last, pred_steps=6):
#         """
#         Args:
#             x_num: [B, T, d_num] 数值输入
#             x_text: [B, T, d_text] 语言嵌入输入
#             y_last: [B, 2] 上一步真实坐标
#             pred_steps: 要预测的时间步数
#         Returns:
#             preds: [B, pred_steps, 2]
#         """
#         # Step 1: PGF 融合
#         fused = self.pgf(x_num, x_text)

#         # Step 2: STAEformer Encoder
#         h_enc_seq = self.encoder(fused)  # [B, T, d_model]
#         h_last = h_enc_seq[:, -1, :]     # 取最后时间步隐藏状态

#         # Step 3: Autoregressive Decoder Head
#         preds = self.decoder(h_last, y_last, pred_steps)
#         return preds

class TyphoFormer(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg: 从配置文件读取的参数 (可以复用 STAEformer.yaml)
        """
        super().__init__()
        self.d_num = cfg['model']['input_dim']          # 数值特征维度
        self.d_text = cfg['model']['text_dim']          # 语言嵌入维度
        self.d_model = cfg['model']['embed_dim']        # 模型隐藏维度
        self.output_dim = cfg['model']['output_dim']    # 输出纬度（通常2）

        # PGF 模块
        self.pgf = PromptGatingFusion(self.d_num, self.d_text, self.d_model)

        # STAEformer 编码器
        self.encoder = SpatialTemporalFormer(
            num_nodes=1,
            in_steps=cfg['model']['input_len'],
            out_steps=cfg['model']['pred_len'],
            input_dim=self.d_model,
            output_dim=self.d_model,
        )

        # 自回归解码器
        self.decoder = TyphoDecoder(hidden_dim=self.d_model, output_dim=self.output_dim)

    def forward(self, x_num, x_text, y_last, pred_steps=6):
        """
        Args:
            x_num: [B, T, d_num] 数值输入
            x_text: [B, T, d_text] 语言嵌入输入
            y_last: [B, 2] 上一步真实坐标
            pred_steps: 要预测的时间步数
        Returns:
            preds: [B, pred_steps, 2]
        """
        # Step 1: PGF 融合
        fused = self.pgf(x_num, x_text)       # [B, T, d_model]

        # Step 2: STAEformer Encoder
        fused = fused.unsqueeze(2)            # [B, T, 1, d_model]
        h_enc_seq = self.encoder(fused)       # [B, out_steps, 1, d_model]
        h_last = h_enc_seq[:, -1, 0, :]       # [B, d_model]

        # Step 3: Autoregressive Decoder Head
        preds = self.decoder(h_last, y_last, pred_steps)
        return preds


 
