import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out

class TCN(nn.Module):
    def __init__(self, input_channels=1, const_dim=4, num_layers=4, kernel_size=5):
        super(TCN, self).__init__()
        
        # TCN for temporal data
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i  # Exponential dilation
            in_channels = input_channels if i == 0 else 64
            layers.append(TemporalBlock(in_channels, 64, kernel_size, dilation))
        self.tcn = nn.Sequential(*layers)
        
        # Fully connected layers for constant features
        self.const_fc = nn.Sequential(
            nn.Linear(const_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Final fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Predicting a single output
        )

    def forward(self, x, const):
        # x shape: (batch_size, seq_len)
        # const shape: (batch_size, const_dim)
        
        # Process temporal data using TCN
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len)
        x = self.tcn(x)
        x = x.mean(dim=-1)  # Global Average Pooling to (batch_size, 64)
        
        # Process constant features
        const = self.const_fc(const)  # (batch_size, 16)
        
        # Concatenate and predict
        combined = torch.cat((x, const), dim=1)  # (batch_size, 64 + 16)
        output = self.fc(combined)  # Final prediction
        return output
