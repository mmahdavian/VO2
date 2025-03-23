
import torch
from data_loader import Feeder
from models.nn_model import NN_Model
import argparse
import os 
import numpy as np
import matplotlib.pyplot as plt



def get_parser():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='train ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test ratio')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--past_data', default=32, type=int)
    parser.add_argument('--future_data', default=32, type=int)
    parser.add_argument('--interval', default=1, type=int)
    parser.add_argument('--time_interval', default=1, type=int)
    parser.add_argument('--model_name', default='NN2_normalized_newID', type=str)
    parser.add_argument('--kernel_len', default=5, type=int)
    parser.add_argument('--model_path', default='./saved_models', type=str)
    return parser.parse_args()

class Tester:
    def __init__(self,args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_data()

    def load_data(self):
        test_dataset = Feeder(args=self.args, split='test')
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)

    def test(self):
        model = NN_Model(kernel_len=self.args.kernel_len).to(self.device)
        model_path = os.path.join(self.args.model_path,self.args.model_name,'model_epoch_4.pth')
        model.load_state_dict(torch.load(model_path))
        model.eval()
        all_targets = []
        all_outputs = []
        all_targets_ds = []
        all_outputs_ds = []

        with torch.no_grad():
            for batch_idx, (times, speed, HR, general, targets, stats) in enumerate(self.test_loader):
                times, speed, HR, general, targets = times.to(self.device), speed.to(self.device), HR.to(self.device), general.to(self.device), targets.to(self.device)
                if times.dim()==2:
                    times = times.unsqueeze(1)
                    speed = speed.unsqueeze(1)
                    HR = HR.unsqueeze(1)
                    
                outputs = model(times, speed, HR, general)
            #    print(outputs)
            
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

        start_data_num = 500
        end_data_num = 1500
        plt.figure(figsize=(10, 6))
        plt.plot(all_outputs_ds[start_data_num:end_data_num], label='Predicted', color='blue')
        plt.plot(all_targets_ds[start_data_num:end_data_num], label='Actual', color='orange')
        plt.xlabel('Sample Index')
        plt.ylabel('VO2')
        plt.title('Comparison of Predicted and Actual VO2')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print()



def main():
    args = get_parser()
    tester = Tester(args)
    tester.test()

if __name__ == '__main__':
    main()
