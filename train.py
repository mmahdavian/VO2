import random, os, glob, math, pathlib, csv, zipfile, warnings

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score

import pathlib
import torch
import torch.nn as nn
import torch.optim as optim

import time
#import wandb

from data_loader import Feeder
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='train ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test ratio')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--model_name', default='NN', type=str)
    parser.add_argument('--past_data', default=16, type=int)
    parser.add_argument('--future_data', default=16, type=int)
    parser.add_argument('--interval', default=1, type=int)
    parser.add_argument('--time_interval', default=1, type=int)
    parser.add_argument('--wandb', default=False, type=bool)
    parser.add_argument('--wandb_name', default='transfuser', type=str)
    parser.add_argument('--kernel_len', default=3, type=int)

    return parser.parse_args()


class Trainer:
    def __init__(self,args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_data()
        self.train()

    def load_data(self):
        ## We have both train and test ratio as a general format. They might not sumup to 1
        train_dataset = Feeder(args=self.args, split='train')
        test_dataset = Feeder(args=self.args, split='test')
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)

    def train(self):
        model = NN_Model(self.args.kernel_len).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        # Training loop
        model.train()
        for epoch in range(self.args.epochs):
            epoch_loss = 0.0
            for (time,speed,HR,input_general,targets) in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}"):
                time = time.float().unsqueeze(1).to(self.device)
                speed = speed.float().unsqueeze(1).to(self.device)
                HR = HR.float().unsqueeze(1).to(self.device)
                input_general = input_general.float().to(self.device)
                targets = targets.float().to(self.device)

                # Forward pass
                outputs = model(time,speed,HR,input_general)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.args.epochs}], Loss: {avg_loss:.4f}")

            # Evaluate on train and test sets
      #      train_metrics = self.evaluate(model, self.train_loader)
      #      test_metrics = self.evaluate(model, self.test_loader)

     #       print(f"Train Metrics: MAE: {train_metrics['mae']:.4f}, MSE: {train_metrics['mse']:.4f}, "
     #               f"Pearson: {train_metrics['pearson']:.4f}, R2: {train_metrics['r2']:.4f}")
     #       print(f"Test Metrics: MAE: {test_metrics['mae']:.4f}, MSE: {test_metrics['mse']:.4f}, "
     #               f"Pearson: {test_metrics['pearson']:.4f}, R2: {test_metrics['r2']:.4f}")

    def evaluate(self, model, data_loader):
        model.eval()
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for batch in tqdm(data_loader):
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = model(inputs)
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        all_targets = np.concatenate(all_targets, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)

        mae = np.mean(np.abs(all_outputs - all_targets))
        mse = np.mean((all_outputs - all_targets) ** 2)
        pearson = np.corrcoef(all_outputs.flatten(), all_targets.flatten())[0, 1]
        r2 = r2_score(all_targets, all_outputs)

        return {"mae": mae, "mse": mse, "pearson": pearson, "r2": r2}




class NN_Model(nn.Module):
    def __init__(self,kernel_len):
        super(NN_Model, self).__init__()
        self.kernel_len = kernel_len
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=self.kernel_len, stride=1, padding=self.kernel_len // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=self.kernel_len, stride=1, padding=self.kernel_len // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=self.kernel_len, stride=1, padding=self.kernel_len // 2),
            nn.ReLU()
       #    ,nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.constants = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

      #  self.global_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.mixer = nn.Linear(24,1)
        self.fc = nn.Linear(128, 1)

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


def main():
    args = get_parser()
    trainer = Trainer(args)
 #   if args.wandb:
 #       wandb.init(project=args.wandb_name,  entity="transfuser", name = args.model_name)    
    trainer.train()

if __name__ == '__main__':
    main()