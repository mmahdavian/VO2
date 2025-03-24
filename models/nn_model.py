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

        self.drop = nn.Dropout(p=0.2)

        self.constants = nn.Sequential(
            nn.Linear(4, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
        )

        self.mixer = nn.Sequential(
            nn.Linear(48, 24),
            nn.LeakyReLU(),
            nn.Linear(24, 1),
        )
        self.fc = nn.Sequential(
            nn.Linear(192, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, time, speed, HR, general):
        general = self.constants(general)
        ## time
        x_time = self.drop(self.pool1(self.relu1(self.ln1(self.conv1(time)+time))))
        x_time = self.pool2(self.relu2(self.ln2(self.conv2(x_time)+self.proj2(x_time))))

        x_time_long = self.drop(self.pool1(self.relu1(self.ln1(self.long_conv1(time)+time))))
        x_time_long = self.pool2(self.relu2(self.ln2(self.long_conv2(x_time_long)+self.proj2(x_time_long))))

        ## speed
        x_speed = self.drop(self.pool1(self.relu1(self.ln1(self.conv1(speed)+speed))))
        x_speed = self.pool2(self.relu2(self.ln2(self.conv2(x_speed)+self.proj2(x_speed))))

        x_speed_long = self.drop(self.pool1(self.relu1(self.ln1(self.long_conv1(speed)+speed))))
        x_speed_long = self.pool2(self.relu2(self.ln2(self.long_conv2(x_speed_long)+self.proj2(x_speed_long))))


        ## HR
        x_HR = self.drop(self.pool1(self.relu1(self.ln1(self.conv1(HR)+HR))))
        x_HR = self.pool2(self.relu2(self.ln2(self.conv2(x_HR)+self.proj2(x_HR))))

        x_HR_long = self.drop(self.pool1(self.relu1(self.ln1(self.long_conv1(HR)+HR))))
        x_HR_long = self.pool2(self.relu2(self.ln2(self.long_conv2(x_HR_long)+self.proj2(x_HR_long))))

        x = self.drop(torch.cat((x_time,x_time_long,x_speed,x_speed_long,x_HR,x_HR_long), dim=2))
        x = x.mean(dim=-1)

        x = torch.cat((x, general), dim=1)
        x = self.fc(x)
        return x.squeeze(1)
