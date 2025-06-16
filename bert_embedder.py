from transformers import BertTokenizer, BertModel
import torch

class BERTEmbedder:
    def __init__(self, model_name='bert-base-uncased', device=None):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def encode(self, text):
        """
        给定一段文本，返回其 BERT 的 [CLS] embedding 向量（768维）。
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, 768)
            return cls_embedding.squeeze(0).cpu().numpy()  # shape: (768,)
