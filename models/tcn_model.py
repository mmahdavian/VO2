import torch
import torch.nn as nn

# class TemporalBlock(nn.Module):
    # def __init__(self, in_channels, out_channels, kernel_size, dilation):
    #     super(TemporalBlock, self).__init__()
    #     self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation)
    #     self.relu = nn.ReLU()
    #     self.bn1 = nn.BatchNorm1d(out_channels)
    #     self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation)
    #     self.bn2 = nn.BatchNorm1d(out_channels)

    # def forward(self, x):
    #     out = self.relu(self.bn1(self.conv1(x)))
    #     out = self.relu(self.bn2(self.conv2(out)))
    #     return out

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_len,dilation):
        super(TemporalBlock, self).__init__()
        self.kernel_len = kernel_len
        self.dilation = dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_len, stride=1, padding=self.kernel_len // 2 * self.dilation, dilation=self.dilation)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=self.kernel_len, stride=1, padding=self.kernel_len // 2 * self.dilation, dilation=self.dilation)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.proj2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=self.kernel_len, stride=1, padding=self.kernel_len // 2 * self.dilation, dilation=self.dilation)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.proj3 = nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.conv4 = nn.Conv1d(out_channels, out_channels, kernel_size=self.kernel_len, stride=1, padding=self.kernel_len // 2 * self.dilation, dilation=self.dilation)
        self.relu4 = nn.ReLU()
        self.proj4 = nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.constants = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

      #  self.global_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
      #  self.mixer = nn.Linear(24,1)
        self.mixer = nn.Sequential(
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(out_channels+128, 128),        
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x2 = self.pool1(self.relu1(self.conv1(x)+x))
        x2 = self.pool2(self.relu2(self.conv2(x2)+self.proj2(x2)))
        x2 = self.pool3(self.relu3(self.conv3(x2)+self.proj3(x2)))
        x2 = self.relu4(self.conv4(x2)+self.proj4(x2))
        return x2

class TCN(nn.Module):
    def __init__(self, input_channels=1, const_dim=4, num_layers=4, kernel_size=3):
        super(TCN, self).__init__()
        
        # TCN for temporal data
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i  # Exponential dilation
            in_channels = input_channels if i == 0 else 256
            layers.append(TemporalBlock(in_channels, 256, kernel_size, dilation))

        self.tcn = nn.Sequential(*layers)
        
        # Fully connected layers for constant features
        self.const_fc = nn.Sequential(
            nn.Linear(const_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Final fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Predicting a single output
        )

    def forward(self, times,speed,HR, const):
        # x shape: (batch_size, seq_len)
        # const shape: (batch_size, const_dim)
        
        # Process temporal data using TCN
        x_time = self.tcn(times)
        x_time = x_time.mean(dim=-1)  # Global Average Pooling to (batch_size, 64)
        
        # Process temporal data using TCN
        x_speed = self.tcn(speed)
        x_speed = x_speed.mean(dim=-1)  # Global Average Pooling to (batch_size, 64)
        
        # Process temporal data using TCN
        x_HR = self.tcn(HR)
        x_HR = x_HR.mean(dim=-1)  # Global Average Pooling to (batch_size, 64)
        

        x = torch.cat((x_time,x_speed,x_HR), dim=2)

        # Process constant features
        const = self.const_fc(const)  # (batch_size, 16)
        
        # Concatenate and predict
        combined = torch.cat((x, const), dim=1)  # (batch_size, 64 + 16)
        output = self.fc(combined)  # Final prediction
        return output
