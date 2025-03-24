import random, os, glob, math, pathlib, csv, zipfile, warnings

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score

import pathlib
import torch
import torch.nn as nn
import torch.optim as optim

#import time
import wandb

from data_loader import Feeder
from models.nn_model import NN_Model
from models.rnn_model import RNN_Model
from models.transformer_model2 import Transformer_Model2
import argparse
from collections import OrderedDict


def get_parser():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='train ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test ratio')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_update_interval', type=int, default=1, help='learning rate update interval')
    parser.add_argument('--weight_decay', default=1e-3, type=float) # 1e-4
    parser.add_argument('--past_data', default=32, type=int)
    parser.add_argument('--future_data', default=32, type=int)
    parser.add_argument('--interval', default=1, type=int)
    parser.add_argument('--time_interval', default=1, type=int)
    parser.add_argument('--model_name', default='Transformer2', type=str)
    parser.add_argument('--wandb', default=False, type=bool)
    parser.add_argument('--wandb_name', default='Zepp', type=str)
    parser.add_argument('--kernel_len', default=3, type=int)
    parser.add_argument('--dilation', default=3, type=int)

    return parser.parse_args()

class Trainer:
    def __init__(self,args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_data()
        self.load_logger()

    def load_data(self):
        ## We have both train and test ratio as a general format. They might not sumup to 1
        train_dataset = Feeder(args=self.args, split='train')
        test_dataset = Feeder(args=self.args, split='test')
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=8)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=8)

    def load_logger(self):
        self.log = OrderedDict([
                        ('epoch', []),
                        ('MSE_loss_total', []),
                        ('MAE_train', []),
                        ('MSE_train', []),
                        ('RMSE_train', []),
                        ('R2_train', []),
                        ('Pearson_train',[]),
                        ('lrate', []),
                      #  ('elapsed_time_train', []),
                      #  ('test_loss_total', []),
                        ('MAE_test', []),
                        ('MSE_test', []),
                        ('RMSE_test', []),
                        ('R2_test', []),
                        ('Pearson_test',[])
                      #  ('test_accuracy', []),
                     #   ('elapsed_time_val', []),
                ])

    # def initialize_weights(self,model):
    #     for layer in model.modules():
    #         if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
    #             # Kaiming He Initialization for Conv layers
    #             nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
    #             if layer.bias is not None:
    #                 nn.init.zeros_(layer.bias)
    #         elif isinstance(layer, nn.Linear):
    #             # Kaiming He Initialization for Linear layers with ReLU
    #             nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
    #             if layer.bias is not None:
    #                 nn.init.zeros_(layer.bias)
    #         elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
    #             # Initialize BatchNorm to be identity transformation
    #             nn.init.ones_(layer.weight)
    #             nn.init.zeros_(layer.bias)

    def initialize_weights(self, model):
        for layer in model.modules():
            if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # Kaiming He Initialization for Conv layers
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                # Kaiming He Initialization for Linear layers with ReLU
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Initialize BatchNorm to be identity transformation
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Embedding):
                # Initialize Embedding layers
                nn.init.normal_(layer.weight, mean=0, std=0.01)
            elif isinstance(layer, nn.MultiheadAttention):
                # Initialize MultiheadAttention layers
                nn.init.xavier_uniform_(layer.in_proj_weight)
                if layer.in_proj_bias is not None:
                    nn.init.zeros_(layer.in_proj_bias)
                nn.init.xavier_uniform_(layer.out_proj.weight)
                if layer.out_proj.bias is not None:
                    nn.init.zeros_(layer.out_proj.bias)
            elif isinstance(layer, nn.LayerNorm):
                # Initialize LayerNorm layers
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def train(self):
    #    self.model = RNN_Model(self.args.kernel_len,self.args.dilation).to(self.device)
    #    self.model = NN_Model(self.args.kernel_len,self.args.dilation).to(self.device)
        self.model = Transformer_Model2().to(self.device)
        
        self.initialize_weights(self.model)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=self.args.lr/20)

        # Training loop
        for epoch in range(self.args.epochs):
            self.model.train()
            epoch_loss = 0.0
            for (times,speed,HR,input_general,targets,stats) in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}"):
                times = times.float().unsqueeze(1).to(self.device)
                speed = speed.float().unsqueeze(1).to(self.device)
                HR = HR.float().unsqueeze(1).to(self.device)
                input_general = input_general.float().to(self.device)
                targets = targets.float().to(self.device)

                # Forward pass
                outputs = self.model(times,speed,HR,input_general)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            
            if epoch>20:
                print(outputs,targets)

            avg_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.args.epochs}], Loss: {avg_loss:.4f}")

            if epoch % self.args.lr_update_interval == 0:
                self.scheduler.step()

            self.log['MSE_loss_total'].append(avg_loss)
            self.log['epoch'].append(epoch)
            self.log['lrate'].append(optimizer.param_groups[0]['lr'])

            # Evaluate on train and test sets
            train_metrics = self.evaluate(self.train_loader)
            test_metrics = self.evaluate(self.test_loader)

            print(f"Train Metrics: MAE: {train_metrics['mae']:.4f}, MSE: {train_metrics['mse']:.4f}, RMSE: {train_metrics['rmse']:.4f}, "
                    f"Pearson: {train_metrics['pearson']:.4f}, R2: {train_metrics['r2']:.4f}")
            print(f"Test Metrics: MAE: {test_metrics['mae']:.4f}, MSE: {test_metrics['mse']:.4f}, RMSE: {test_metrics['rmse']:.4f}, "
                    f"Pearson: {test_metrics['pearson']:.4f}, R2: {test_metrics['r2']:.4f}")

            self.log['MAE_train'].append(train_metrics['mae'])
            self.log['MSE_train'].append(train_metrics['mse'])
            self.log['RMSE_train'].append(train_metrics['rmse'])
            self.log['R2_train'].append(train_metrics['r2'])
            self.log['Pearson_train'].append(train_metrics['pearson'])

            self.log['MAE_test'].append(test_metrics['mae'])
            self.log['MSE_test'].append(test_metrics['mse'])
            self.log['RMSE_test'].append(test_metrics['rmse'])
            self.log['R2_test'].append(test_metrics['r2'])
            self.log['Pearson_test'].append(test_metrics['pearson'])


            if self.args.wandb:
                    dic = {x: v[-1] for x,v in self.log.items() if v }
                    wandb.log(dic)
            
            self.save_model(epoch)

    def save_model(self, epoch):
        # Create directory based on model name if it doesn't exist
        model_dir = os.path.join('./saved_models', self.args.model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Save the model
        model_path = os.path.join(model_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved at {model_path}")

    # def plot_model_output(self, data_loader, num_samples=100):
    #     self.model.eval()
    #     all_targets = []
    #     all_outputs = []
    #     all_stats = []

    #     with torch.no_grad():
    #         for times, speed, HR, input_general, targets, stats in tqdm(data_loader):
    #             times = times.float().unsqueeze(1).to(self.device)
    #             speed = speed.float().unsqueeze(1).to(self.device)
    #             HR = HR.float().unsqueeze(1).to(self.device)
    #             input_general = input_general.float().to(self.device)
    #             targets = targets.float().to(self.device)

    #             # Forward pass
    #             outputs = self.model(times, speed, HR, input_general)

    #             all_targets.append(targets.cpu().numpy())
    #             all_outputs.append(outputs.cpu().numpy())

    #     all_targets = np.concatenate(all_targets, axis=0)
    #     all_outputs = np.concatenate(all_outputs, axis=0)

    #     # Destandardize using VO2 mean and std from stats
    #     vo2_mean = stats['VO2']['mean']  
    #     vo2_std = stats['VO2']['std']  

    #     destandardized_targets = all_targets * vo2_std + vo2_mean
    #     destandardized_outputs = all_outputs * vo2_std + vo2_mean

    #     # Select a subset of samples to plot
    #     indices = np.random.choice(len(destandardized_targets), num_samples, replace=False)
    #     sampled_targets = destandardized_targets[indices]
    #     sampled_outputs = destandardized_outputs[indices]

    #     # Plot the results
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(sampled_targets, label="True Values", marker='o', linestyle='dashed', alpha=0.7)
    #     plt.plot(sampled_outputs, label="Model Predictions", marker='x', linestyle='dashed', alpha=0.7)
    #     plt.xlabel("Sample Index")
    #     plt.ylabel("VO2 Value")
    #     plt.title("Destandardized Model Output vs True Values")
    #     plt.legend()
    #     plt.grid()
    #     plt.show()


    def evaluate(self, data_loader):
        self.model.eval()
        all_targets = []
        all_outputs = []
        all_targets_ds = []
        all_outputs_ds = []

        with torch.no_grad():
            for times,speed,HR,input_general,targets,stats in tqdm(data_loader):
                times = times.float().unsqueeze(1).to(self.device)
                speed = speed.float().unsqueeze(1).to(self.device)
                HR = HR.float().unsqueeze(1).to(self.device)
                input_general = input_general.float().to(self.device)
                targets = targets.float().to(self.device)

                # Forward pass
                outputs = self.model(times,speed,HR,input_general)
            
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

                targets_ds = targets*stats['VO2']['std'][0]+stats['VO2']['mean'][0]
                outputs_ds = outputs*stats['VO2']['std'][0]+stats['VO2']['mean'][0]

                all_targets_ds.append(targets_ds.cpu().numpy())
                all_outputs_ds.append(outputs_ds.cpu().numpy())               
                

        all_targets = np.concatenate(all_targets, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets_ds = np.concatenate(all_targets_ds, axis=0)
        all_outputs_ds = np.concatenate(all_outputs_ds, axis=0)

        mae = np.mean(np.abs(all_outputs - all_targets))
        mse = np.mean((all_outputs - all_targets) ** 2)
        rmse = np.sqrt(mse)
        pearson = np.corrcoef(all_outputs.flatten(), all_targets.flatten())[0, 1]
        r2 = r2_score(all_targets, all_outputs)

        return {"mae": mae, "mse": mse, "rmse": rmse, "pearson": pearson, "r2": r2}

def main():
    args = get_parser()
    trainer = Trainer(args)
    if args.wandb:
        wandb.init(project=args.wandb_name,  entity="transfuser", name = args.model_name)    
    trainer.train()

if __name__ == '__main__':
    main()
