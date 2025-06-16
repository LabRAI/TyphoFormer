import torch
import torch.nn as nn
import json
from typing import List
from bert_embedder import BERTEmbedder

# 参数设置
input_dim = 4             # 时间序列的每条记录长度
embedding_dim = 768       # 与 BERT 输出维度保持一致
window_size = 4           # 滑动窗口长度 l

# 初始化 BERT 文本编码器
bert = BERTEmbedder(model_name='bert-base-uncased')

# 加载时间序列 JSON 数据
with open("timeseries_typhoon_data.json", "r") as f:
    typhoon_data = json.load(f)  #三级数组结构

# 加载与每条时间序列对应的文本描述
with open("text_descriptions.json", "r") as f:
    text_descriptions = json.load(f)  # 形如 [[t1, t2, ..., tN], [...], ...]

# 时间序列线性映射模块
class TimeSeriesProjector(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        return self.linear(x)  # [T, input_dim] → [T, embedding_dim]

# PGF模块
class PGFModule(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.gate = nn.Linear(2 * embedding_dim, embedding_dim)

    def forward(self, ts_embeddings, text_embedding):
        T = ts_embeddings.size(0)
        p_mean = text_embedding.unsqueeze(0).expand(T, -1)  # shape [T, d]
        concat = torch.cat([ts_embeddings, p_mean], dim=1)
        gate = torch.sigmoid(self.gate(concat))
        fused = gate * ts_embeddings + (1 - gate) * p_mean
        return fused

# 初始化模块
ts_encoder = TimeSeriesProjector(input_dim, embedding_dim)
pgf = PGFModule(embedding_dim)

# 示例：处理第一个台风的第一个滑动窗口
typhoon = typhoon_data[0]              # 第一个台风
ts_window = typhoon[:window_size]      # l=4 条时间序列
ts_tensor = torch.tensor(ts_window, dtype=torch.float32)  # [4, 7]

# 获取对应的4条文本描述，拼成一句话用于 embedding（你也可以取平均）
text_seqs = text_descriptions[0][:window_size]
joined_text = " ".join(text_seqs)
text_embedding_np = bert.encode(joined_text)              # shape: (768,)
text_embedding = torch.tensor(text_embedding_np, dtype=torch.float32)

# 编码时间序列
ts_embedded = ts_encoder(ts_tensor)    # [4, 768]

# 融合文本与时间序列特征
fused_input = pgf(ts_embedded, text_embedding)  # shape: [4, 768]

print(fused_input.shape)
