
import torch
import torch.nn as nn


class NN_Model(nn.Module):
    def __init__(self,kernel_len,dilation):
        super(NN_Model, self).__init__()
        self.kernel_len = kernel_len
        self.dilation = dilation
        self.conv1 = nn.Conv1d(1, 32, kernel_size=self.kernel_len, stride=1, padding=self.kernel_len // 2 * self.dilation, dilation=self.dilation)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.proj_lin1 = nn.Linear(64, 8)
        self.proj1 = nn.Conv1d(32, 1, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=self.kernel_len, stride=1, padding=self.kernel_len // 2 * self.dilation, dilation=self.dilation)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.proj2 = nn.Conv1d(64, 1, kernel_size=1, stride=1, padding=0)
        self.proj_lin2 = nn.Linear(32, 8)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=self.kernel_len, stride=1, padding=self.kernel_len // 2 * self.dilation, dilation=self.dilation)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.proj3 = nn.Conv1d(128, 1, kernel_size=1, stride=1, padding=0)
        self.proj_lin3 = nn.Linear(16, 8)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=self.kernel_len, stride=1, padding=self.kernel_len // 2 * self.dilation, dilation=self.dilation)
        self.relu4 = nn.ReLU()
        self.proj4 = nn.Conv1d(128, 256, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm1d(256)

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
            nn.Linear(480+128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    #    self.fc = nn.Linear(128+128, 1)

    def forward(self, time, speed, HR, general):
        general = self.constants(general)
        ## time
        x_time = self.relu1(self.bn1(self.conv1(time)+time))
        x_time2 = self.pool1(x_time)
        x_time2 = self.relu2(self.bn2(self.conv2(x_time2)))
        x_time3 = self.pool2(x_time2)
        x_time3 = self.relu3(self.bn3(self.conv3(x_time3)))
        x_time4 = self.pool3(x_time3)
        x_time4 = self.relu4(self.bn4(self.conv4(x_time4)))
        x_time_proj1 = self.proj_lin1(x_time)
        x_time_proj2 = self.proj_lin2(x_time2)
        x_time_proj3 = self.proj_lin3(x_time3)
#        x_time4 = x_time4 + self.proj1(x_time_proj1) + self.proj2(x_time_proj2) + self.proj3(x_time_proj3)
        x_time4 = torch.cat((x_time4,x_time_proj1,x_time_proj2,x_time_proj3),dim=1)

        ## speed
        x_speed = self.relu1(self.bn1(self.conv1(speed) + speed))
        x_speed2 = self.pool1(x_speed)
        x_speed2 = self.relu2(self.bn2(self.conv2(x_speed2)))
        x_speed3 = self.pool2(x_speed2)
        x_speed3 = self.relu3(self.bn3(self.conv3(x_speed3)))
        x_speed4 = self.pool3(x_speed3)
        x_speed4 = self.relu4(self.bn4(self.conv4(x_speed4)))
        x_speed_proj1 = self.proj_lin1(x_speed)
        x_speed_proj2 = self.proj_lin2(x_speed2)
        x_speed_proj3 = self.proj_lin3(x_speed3)
 #       x_speed4 = x_speed4 + self.proj1(x_speed_proj1) + self.proj2(x_speed_proj2) + self.proj3(x_speed_proj3)
        x_speed4 = torch.cat((x_speed4, x_speed_proj1, x_speed_proj2, x_speed_proj3), dim=1)

        ## HR
        x_HR = self.relu1(self.bn1(self.conv1(HR) + HR))
        x_HR2 = self.pool1(x_HR)
        x_HR2 = self.relu2(self.bn2(self.conv2(x_HR2)))
        x_HR3 = self.pool2(x_HR2)
        x_HR3 = self.relu3(self.bn3(self.conv3(x_HR3)))
        x_HR4 = self.pool3(x_HR3)
        x_HR4 = self.relu4(self.bn4(self.conv4(x_HR4)))
        x_HR_proj1 = self.proj_lin1(x_HR)
        x_HR_proj2 = self.proj_lin2(x_HR2)
        x_HR_proj3 = self.proj_lin3(x_HR3)
        #x_HR4 = x_HR4 + self.proj1(x_HR_proj1) + self.proj2(x_HR_proj2) + self.proj3(x_HR_proj3)
        x_HR4 = torch.cat((x_HR4, x_HR_proj1, x_HR_proj2, x_HR_proj3), dim=1)

        #x_time = self.temporal_conv(time)
        #x_speed = self.temporal_conv(speed)
        #x_HR = self.temporal_conv(HR)
        x = torch.cat((x_time4,x_speed4,x_HR4), dim=2)

     #   x = torch.mean(x, dim=-1, keepdim=True)  # Global average pooling
        x = self.mixer(x).squeeze(-1)  # Shape: (batch_size, 64, 1)
        x = torch.cat((x, general), dim=1) # Shape: (batch_size, 64+64)
        x = self.fc(x)  # Shape: (batch_size, output_size)
        return x.squeeze(1)  # Shape: (batch_size)

