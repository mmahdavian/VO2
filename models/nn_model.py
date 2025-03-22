
import torch
import torch.nn as nn


class NN_Model(nn.Module):
    def __init__(self,kernel_len):
        super(NN_Model, self).__init__()
        self.kernel_len = kernel_len
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=self.kernel_len, stride=1, padding=self.kernel_len // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 64, kernel_size=self.kernel_len, stride=1, padding=self.kernel_len // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 64, kernel_size=self.kernel_len, stride=1, padding=self.kernel_len // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 64, kernel_size=self.kernel_len, stride=1, padding=self.kernel_len // 2),
            nn.ReLU(),
        #    nn.MaxPool1d(kernel_size=2, stride=2)
        )

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
            nn.Linear(128+128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    #    self.fc = nn.Linear(128+128, 1)

    def forward(self, time, speed, HR, general):
        general = self.constants(general)
   #     x = torch.cat((time,speed,HR), dim=1)
        x_time = self.temporal_conv(time)
        x_speed = self.temporal_conv(speed)
        x_HR = self.temporal_conv(HR)
        x = torch.cat((x_time,x_speed,x_HR), dim=2)


     #   x = torch.mean(x, dim=-1, keepdim=True)  # Global average pooling
        x = self.mixer(x).squeeze(-1)  # Shape: (batch_size, 64, 1)
        x = torch.cat((x, general), dim=1) # Shape: (batch_size, 64+64)
        x = self.fc(x)  # Shape: (batch_size, output_size)
        return x.squeeze(1)  # Shape: (batch_size)

