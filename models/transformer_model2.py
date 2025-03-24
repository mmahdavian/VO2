import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer_Model2(nn.Module):
    def __init__(self, input_dim=1, static_dim=4, seq_len=64, embed_dim=64, num_heads=4, num_layers=2):
        super(Transformer_Model2,self).__init__()
        self.embed = nn.Linear(input_dim, embed_dim)  # Shared embedding layer
        
        # Store Positional Encoding as a Non-Trainable Buffer
        self.register_buffer("pos_embed", self.positional_encoding(seq_len, embed_dim))

        # Shared Transformer Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=256, dropout=0.1, batch_first=True),
            num_layers
        )

        # Cross-Attention Fusion
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)

        # Static Feature Fusion
        self.static_proj = nn.Linear(static_dim, embed_dim)

        # Regression MLP
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim * 3 + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )


    def positional_encoding(self, seq_len, embed_dim):
        pos = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe.unsqueeze(0)  # (1, seq_len, embed_dim)

    def forward(self, times, speed, HR, x_static):
        """
        x_temporal: List of 3 tensors [(bs, seq_len, input_dim), (bs, seq_len, input_dim), (bs, seq_len, input_dim)]
        x_static: Tensor of shape (bs, static_dim)
        """
        x_temporal = [times.permute(0,2,1), speed.permute(0,2,1), HR.permute(0,2,1)]

        batch_size, seq_len, _ = x_temporal[0].shape

        # Embed each temporal modality & Add Positional Encoding
        temporal_features = [self.embed(x) + self.pos_embed[:, :seq_len, :] for x in x_temporal]

        # Pass each modality through the shared Transformer Encoder
        temporal_features = [self.encoder(x) for x in temporal_features]

        # Stack modalities and apply cross-attention
        x = torch.stack(temporal_features, dim=1)  # (bs, 3, seq_len, embed_dim)
        x = x.mean(dim=2)  # Pool over sequence dimension -> (bs, 3, embed_dim)
        x = x.view(batch_size, 3, -1)  # (bs, 3, embed_dim)

        x_fused, _ = self.cross_attn(x, x, x)  # Cross-attention fusion
    #    x_fused = x

        # Process static features
        x_static = self.static_proj(x_static).unsqueeze(1)  # (bs, 1, embed_dim)

        # Concatenate static and fused temporal features
        x = torch.cat([x_fused.flatten(1), x_static.flatten(1)], dim=1)  # (bs, 3*embed_dim + embed_dim)

        # Regression MLP
        return self.mlp_head(x).squeeze(-1)
