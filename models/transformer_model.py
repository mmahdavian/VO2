import torch
import torch.nn as nn
import numpy as np

class Transformer_Model(nn.Module):
    def __init__(self, embed_dim=32, num_classes=1, dropout=0.3):
        super(Transformer_Model, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Embedding layers for each modality
        self.embed1 = nn.Linear(1, embed_dim)
        self.embed2 = nn.Linear(1, embed_dim)
        self.embed3 = nn.Linear(1, embed_dim)

        # Transformer encoder layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=96, nhead=4, dropout=0.2),
            num_layers=4
        )

        self._pos_encoder = PositionEncodings1D(num_pos_feats=embed_dim,temperature=1000,alpha=1)
        self.encoder_pos_encodings = self._pos_encoder(64).view(1, 64, embed_dim).to(self.device)
        self._encoder_pos_encodings = nn.Parameter(self.encoder_pos_encodings, requires_grad=False)

        # Fully connected layer for constant data
        self.fc_const = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Dropout(0.1)
        )

        # Final regression layer
        self.fc_out = nn.Sequential(
            nn.Linear(160, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, times, speed, HR, const):
        x1 = times
        x2 = speed
        x3 = HR
        # Embedding each modality
        x1 = self.embed1(x1.permute(0,2,1))
        x2 = self.embed2(x2.permute(0,2,1))
        x3 = self.embed3(x3.permute(0,2,1))

        x1 = x1+self.encoder_pos_encodings
        x2 = x2+self.encoder_pos_encodings
        x3 = x3+self.encoder_pos_encodings

        # Concatenate temporal data along the sequence dimension
        x = torch.cat((x1, x2, x3), dim=2)

        # Transformer encoder
        x = self.transformer(x)
        x = torch.mean(x, dim=1)  # Pooling

        # Constant data embedding
        const_feat = self.fc_const(const)

        # Concatenate temporal and constant features
        x = torch.cat((x, const_feat), dim=1)
        output = self.fc_out(x)
        return output.squeeze(1)



class PositionEncodings1D(object):
  def __init__(self, num_pos_feats=512, temperature=10000, alpha=1):
    self._num_pos_feats = num_pos_feats
    self._temperature = temperature
    self._alpha = alpha

  def __call__(self, seq_length):
    angle_rads = self.get_angles(
        np.arange(seq_length)[:, np.newaxis],
        np.arange(self._num_pos_feats)[np.newaxis, :]
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    pos_encoding = pos_encoding.astype(np.float32)

    return torch.from_numpy(pos_encoding)

  def get_angles(self, pos, i):
    angle_rates = 1 / np.power(
        self._temperature, (2 * (i//2)) / np.float32(self._num_pos_feats))
    return self._alpha*pos * angle_rates

