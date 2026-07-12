import torch
import torch.nn as nn

class PermutationClassifier(nn.Module):
    def __init__(self, emb_dim=512, hidden=128, n_heads=4, n_layers=1, n_classes=24):
        super().__init__()
        self.proj = nn.Linear(emb_dim, hidden)
        self.text_proj = nn.Linear(emb_dim, hidden)
        self.slot_pos_emb = nn.Parameter(torch.randn(4, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, image_emb, text_emb):
        # image_emb: (B, 4, 512), text_emb: (B, 512)
        frames = self.proj(image_emb) + self.slot_pos_emb          # (B, 4, hidden)
        text_tok = self.text_proj(text_emb).unsqueeze(1)           # (B, 1, hidden)
        tokens = torch.cat([frames, text_tok], dim=1)               # (B, 5, hidden)
        encoded = self.encoder(tokens)
        pooled = encoded[:, :4, :].mean(dim=1)                      # pool the 4 frame slots
        return self.head(pooled)                                    # (B, 24) logits
