
import torch
import torch.nn as nn


class NN_Model(nn.Module):
    def __init__(self,kernel_len,dilation):
        super(NN_Model, self).__init__()
        self.kernel_len = kernel_len
        self.dilation1 = dilation
        self.dilation2 = dilation+4
        self.conv1 = nn.Conv1d(1, 64, kernel_size=self.kernel_len, stride=1, padding=self.kernel_len // 2 * self.dilation1, dilation=self.dilation1)
        self.long_conv1 = nn.Conv1d(1, 64, kernel_size=self.kernel_len*5, stride=1, padding=self.kernel_len*5 // 2 * self.dilation2, dilation=self.dilation2)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.ln1 = nn.LayerNorm(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=self.kernel_len, stride=1, padding=self.kernel_len // 2 * self.dilation1, dilation=self.dilation1)
        self.long_conv2 = nn.Conv1d(64, 128, kernel_size=self.kernel_len*5, stride=1, padding=self.kernel_len*5 // 2 * self.dilation2, dilation=self.dilation2)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.proj2 = nn.Conv1d(64, 1, kernel_size=1, stride=1, padding=0)
        self.ln2 = nn.LayerNorm(32)

        self.drop = nn.Dropout(p=0.1)
        # self.conv3 = nn.Conv1d(64, 128, kernel_size=self.kernel_len, stride=1, padding=self.kernel_len // 2 * self.dilation1, dilation=self.dilation1)
        # self.long_conv3 = nn.Conv1d(64, 128, kernel_size=self.kernel_len*5, stride=1, padding=self.kernel_len*5 // 2 * self.dilation2, dilation=self.dilation2)
        # self.relu3 = nn.LeakyReLU()
        # self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.proj3 = nn.Conv1d(64, 1, kernel_size=1, stride=1, padding=0)
        # self.bn3 = nn.BatchNorm1d(128)

        # self.conv4 = nn.Conv1d(128, 256, kernel_size=self.kernel_len, stride=1, padding=self.kernel_len // 2 * self.dilation1, dilation=self.dilation1)
        # self.long_conv4 = nn.Conv1d(128, 256, kernel_size=self.kernel_len*5, stride=1, padding=self.kernel_len*5 // 2 * self.dilation2, dilation=self.dilation2)
        # self.relu4 = nn.LeakyReLU()
        # self.proj4 = nn.Conv1d(128, 1, kernel_size=1, stride=1, padding=0)
        # self.bn4 = nn.BatchNorm1d(256)

        self.constants = nn.Sequential(
            nn.Linear(4, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
       #     nn.LeakyReLU(),
       #     nn.Linear(64, 128),
       #     nn.LeakyReLU()
        )

      #  self.global_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
      #  self.mixer = nn.Linear(24,1)
        self.mixer = nn.Sequential(
            nn.Linear(48, 24),
            nn.LeakyReLU(),
            nn.Linear(24, 1),
      #      nn.LeakyReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(144, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
    #    self.fc = nn.Linear(128+128, 1)

    def forward(self, time, speed, HR, general):
        general = self.constants(general)
        ## time
        x_time = self.drop(self.pool1(self.relu1(self.ln1(self.conv1(time)+time))))
        x_time = self.pool2(self.relu2(self.ln2(self.conv2(x_time)+self.proj2(x_time))))

        x_time_long = self.drop(self.pool1(self.relu1(self.ln1(self.long_conv1(time)+time))))
        x_time_long = self.pool2(self.relu2(self.ln2(self.long_conv2(x_time_long)+self.proj2(x_time_long))))

#        x_time = self.pool3(self.relu3(self.bn3(self.conv3(x_time)+self.proj3(x_time))))
#        x_time = self.relu4(self.bn4(self.conv4(x_time)+self.proj4(x_time)))
        
        ## speed
        x_speed = self.drop(self.pool1(self.relu1(self.ln1(self.conv1(speed)+speed))))
        x_speed = self.pool2(self.relu2(self.ln2(self.conv2(x_speed)+self.proj2(x_speed))))

        x_speed_long = self.drop(self.pool1(self.relu1(self.ln1(self.long_conv1(speed)+speed))))
        x_speed_long = self.pool2(self.relu2(self.ln2(self.long_conv2(x_speed_long)+self.proj2(x_speed_long))))

#        x_speed = self.pool3(self.relu3(self.bn3(self.conv3(x_speed)+self.proj3(x_speed))))
#        x_speed = self.relu4(self.bn4(self.conv4(x_speed)+self.proj4(x_speed)))

        ## HR
        x_HR = self.drop(self.pool1(self.relu1(self.ln1(self.conv1(HR)+HR))))
        x_HR = self.pool2(self.relu2(self.ln2(self.conv2(x_HR)+self.proj2(x_HR))))

        x_HR_long = self.drop(self.pool1(self.relu1(self.ln1(self.long_conv1(HR)+HR))))
        x_HR_long = self.pool2(self.relu2(self.ln2(self.long_conv2(x_HR_long)+self.proj2(x_HR_long))))

 #       x_HR = self.pool3(self.relu3(self.bn3(self.conv3(x_HR)+self.proj3(x_HR))))
 #       x_HR = self.relu4(self.bn4(self.conv4(x_HR)+self.proj4(x_HR)))

        #x_time = self.temporal_conv(time)
        #x_speed = self.temporal_conv(speed)
        #x_HR = self.temporal_conv(HR)
        x = torch.cat((x_time,x_time_long,x_speed,x_speed_long,x_HR,x_HR_long), dim=2)
        x = x.mean(dim=-1)

     #   x = torch.mean(x, dim=-1, keepdim=True)  # Global average pooling
  #      x = self.mixer(x).squeeze(-1)  # Shape: (batch_size, 64, 1)
        x = torch.cat((x, general), dim=1) # Shape: (batch_size, 64+64)
        x = self.fc(x)  # Shape: (batch_size, output_size)
        return x.squeeze(1)  # Shape: (batch_size)

