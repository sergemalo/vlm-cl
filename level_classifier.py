from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

class LevelClassifier(nn.Module):
    def __init__(self, encoder_name="sentence-transformers/all-MiniLM-L6-v2", num_classes=7):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size  # 384 for MiniLM
        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def mean_pool(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (token_embeddings * mask).sum(1) / mask.sum(1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(out, attention_mask)
        return self.head(pooled)
    
    