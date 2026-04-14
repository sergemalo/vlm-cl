from transformers import (
    AutoModel,
    PreTrainedModel,
    PretrainedConfig,
    )
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
from typing import Optional

CONFIGS = [
    "L1_single",
    "L2_objects",
    "L3_2D_spatial",
#    "L4_occ",
    "L4_pose",
    "L5_6d_spatial",
#    "L5_collision",
]
LABEL2ID = {c: i for i, c in enumerate(CONFIGS)}
ID2LABEL = {i: c for i, c in enumerate(CONFIGS)}

# Optional: if you want 5-level routing instead of 7-config routing,
# use this mapping after prediction.
CONFIG_TO_LEVEL = {
    "L1_single": 1,
    "L2_objects": 2,
    "L3_2D_spatial": 3,
#    "L4_occ": 4,
    "L4_pose": 4,
    "L5_6d_spatial": 5,
#    "L5_collision": 5,
}

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 64

class LevelClassifierConfig(PretrainedConfig):
    model_type = "level_classifier"

    def __init__(
        self,
        encoder_name: str = ENCODER_NAME,
        num_labels: int = len(CONFIGS),
        hidden_size: int = 384,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        self.encoder_name = encoder_name
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.id2label = ID2LABEL
        self.label2id = LABEL2ID


class LevelClassifier(PreTrainedModel):
    """
    Thin classification head on top of a frozen/trainable sentence encoder.

    The `pooled` representation (before the head) is intentionally exposed
    via `return_pooled=True` so it can later be used as a prefix token for
    MoE routing in Qwen2-VL layers.
    """

    config_class = LevelClassifierConfig

    def __init__(self, config: LevelClassifierConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.num_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def mean_pool(self, model_output, attention_mask):
        token_emb = model_output.last_hidden_state          # (B, T, H)
        mask = attention_mask.unsqueeze(-1).float()         # (B, T, 1)
        return (token_emb * mask).sum(1) / mask.sum(1)     # (B, H)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_pooled: bool = False,
    ) -> SequenceClassifierOutput:
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(enc_out, attention_mask)    # (B, H)
        logits = self.head(pooled)                          # (B, num_labels)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        # pooled is stored in hidden_states[0] for easy retrieval downstream
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=(pooled,) if return_pooled else None,
        )
    
    def predict(self, question: str) -> int:
        """
        Given a question string, returns the predicted level id (int).
        Runs on whichever device the model is on.
        """
        device = next(self.parameters()).device
        
        enc = self.processing_class(
            question,
            truncation=True,
            max_length=64,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)
        
        return output.logits.argmax(dim=-1).item()    