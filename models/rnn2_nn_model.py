
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN2_Model(nn.Module):
    def __init__(self,kernel_len,dilation):
        super(RNN2_Model, self).__init__()
        self.kernel_len = kernel_len

        # Conv layers for modality 1
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Conv layers for modality 2
        self.conv2_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Conv layers for modality 3
        self.conv3_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=64 * 3, hidden_size=128, batch_first=True, dropout=0.2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 + 4, 64)  # Adding 4 constant values
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, time, speed, HR, general):
        x1 = time
        x2 = speed
        x3 = HR

        x1 = F.relu(self.conv1_1(x1))
        x1 = F.relu(self.conv1_2(x1))
    #    x1 = torch.flatten(x1, start_dim=1)
        
        # Modality 2
        x2 = F.relu(self.conv2_1(x2))
        x2 = F.relu(self.conv2_2(x2))
     #   x2 = torch.flatten(x2, start_dim=1)
        
        # Modality 3
        x3 = F.relu(self.conv3_1(x3))
        x3 = F.relu(self.conv3_2(x3))
    #    x3 = torch.flatten(x3, start_dim=1)
        
        # Concatenate the flattened features
        merged = torch.cat((x1, x2, x3), dim=1) # Adding time dimension for LSTM
        
        # LSTM layer
        merged = merged.permute(0, 2, 1)  # Change the shape to (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(merged)
        lstm_out = lstm_out[:, -1, :]  # Take the last time step
        
        # Concatenate constant values
        merged = torch.cat((lstm_out, general), dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(merged))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        return output.squeeze(1)  # Shape: (batch_size)

