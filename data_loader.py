import random, os, glob, math, pathlib, csv, zipfile, warnings

import numpy as np

import pathlib
import pandas as pd


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

class Feeder:
    def __init__(self,args, split='train'):
        self.args = args
        self.train_ratio = self.args.train_ratio
        self.test_ratio = self.args.test_ratio
        self.past_data = self.args.past_data
        self.future_data = self.args.future_data
        self.interval = self.args.interval
        self.time_interval = self.args.time_interval
        self.split = split
        self.preprocess()
    
    def preprocess(self):
        base_path = pathlib.Path(__file__).parent.absolute()
        sub_df = pd.read_csv("./subject-info.csv", low_memory=False)
        test_measure_df = pd.read_csv("./test_measure.csv", low_memory=False)
        df = pd.merge(sub_df, test_measure_df, on='ID_test')
        df.dropna(subset=['VO2'], inplace=True)
        df.drop(['VCO2','RR','VE','ID_y','Humidity','Temperature'], axis=1, inplace=True) 
        df_cleaned = df
        df_cleaned['VO2'] = df_cleaned['VO2']/df_cleaned['Weight']
        
        ## Standardize the Age, weight and height
        # check again
        # Save mean and std for denormalization
        self.stats = {
            'Age': {'mean': df_cleaned['Age'].mean(), 'std': df_cleaned['Age'].std()},
            'Weight': {'mean': df_cleaned['Weight'].mean(), 'std': df_cleaned['Weight'].std()},
            'Height': {'mean': df_cleaned['Height'].mean(), 'std': df_cleaned['Height'].std()},
            'time': {'mean': df_cleaned['time'].mean(), 'std': df_cleaned['time'].std()},
            'Speed': {'mean': df_cleaned['Speed'].mean(), 'std': df_cleaned['Speed'].std()},
            'HR': {'mean': df_cleaned['HR'].mean(), 'std': df_cleaned['HR'].std()},
            'VO2': {'mean': df_cleaned['VO2'].mean(), 'std': df_cleaned['VO2'].std()},
       #     'VO2_weight': {'mean': df_cleaned['VO2_weight'].mean(), 'std': df_cleaned['VO2_weight'].std()}
        }

        # Standardize the values
        df_cleaned['Age'] = (df_cleaned['Age'] - self.stats['Age']['mean']) / self.stats['Age']['std']
        df_cleaned['Weight'] = (df_cleaned['Weight'] - self.stats['Weight']['mean']) / self.stats['Weight']['std']
        df_cleaned['Height'] = (df_cleaned['Height'] - self.stats['Height']['mean']) / self.stats['Height']['std']
        df_cleaned['time'] = (df_cleaned['time'] - self.stats['time']['mean']) / self.stats['time']['std']
        df_cleaned['Speed'] = (df_cleaned['Speed'] - self.stats['Speed']['mean']) / self.stats['Speed']['std']
        df_cleaned['HR'] = (df_cleaned['HR'] - self.stats['HR']['mean']) / self.stats['HR']['std']
        df_cleaned['VO2'] = (df_cleaned['VO2'] - self.stats['VO2']['mean']) / self.stats['VO2']['std']
       # df_cleaned['VO2_Weight'] = (df_cleaned['VO2_Weight'] - self.stats['VO2_weight']['mean']) / self.stats['VO2_weight']['std']

        self.target_variable = df_cleaned['VO2']
        #df_cleaned.drop(['VO2_Weight'], axis=1, inplace=True) 
        df_cleaned.drop(['VO2'], axis=1, inplace=True)

        unique_ids = df_cleaned['ID_x'].unique()
        train_size = int(len(unique_ids) * self.train_ratio)
        train_ids = unique_ids[:train_size]
        test_ids = unique_ids[train_size:]

        groups = dict(tuple(df_cleaned.groupby('ID_test')))
        
        input_temporal_sequence_data_train = []
        input_general_data_train = []
        output_train = []

        input_temporal_sequence_data_test = []
        input_general_data_test = []
        output_test = []

        for id_value, group in groups.items():
            group.drop(['ID_test'], axis=1, inplace=True) 
            if groups[id_value]['ID_x'].values[0] in train_ids:
                cur_split='train'
            else:
                cur_split='test'
            group.drop(['ID_x'], axis=1, inplace=True) 
            cur_df = groups[id_value]
            imputer = SimpleImputer(strategy='mean')
            cur_df_imputed = imputer.fit_transform(cur_df)
            categorical_features = ['Sex']
            categorical_indices = [cur_df.columns.get_loc(col) for col in categorical_features]
            one_hot_encoder = OneHotEncoder(categories='auto', sparse_output=False)
            df_encoded_categorical = one_hot_encoder.fit_transform(cur_df_imputed[:, categorical_indices])
            numerical_features = [col for col in cur_df.columns if col not in categorical_features]
            df_numerical = cur_df_imputed[:, [cur_df.columns.get_loc(col) for col in numerical_features]]

            if cur_split=='train' and self.split=='train':    
                for i in range(0, len(df_numerical) - (self.past_data + self.future_data) * self.time_interval, self.interval):
                    past_sequence = df_numerical[i:i + self.past_data * self.time_interval:self.time_interval, 3:]
                    future_sequence = df_numerical[i + self.past_data * self.time_interval:i + (self.past_data + self.future_data) * self.time_interval:self.time_interval, 3:]
                    combined_sequence = np.concatenate((past_sequence, future_sequence), axis=0)
                    input_temporal_sequence_data_train.append(combined_sequence)
                    ## non temporal data
                    input_general_data_train.append([cur_df_imputed[i,:4]])
                    output_train.append(self.target_variable[i+ self.past_data * self.time_interval])

            elif cur_split=='test' and self.split=='test':
                for i in range(0, len(df_numerical) - (self.past_data + self.future_data) * self.time_interval, self.interval):
                    past_sequence = df_numerical[i:i + self.past_data * self.time_interval:self.time_interval, 3:]
                    future_sequence = df_numerical[i + self.past_data * self.time_interval:i + (self.past_data + self.future_data) * self.time_interval:self.time_interval, 3:]
                    combined_sequence = np.concatenate((past_sequence, future_sequence), axis=0)
                    input_temporal_sequence_data_test.append(combined_sequence)
                    ## non temporal data
                    input_general_data_test.append([cur_df_imputed[i,:4]])
                    output_test.append(self.target_variable[i+ self.past_data * self.time_interval])

            # if split=='train':
            #     num_train_samples = int(len(general_data) * self.train_ratio)
            #     self.input_temporal_sequence_data = temporal_sequence_data[:num_train_samples]
            #     self.input_general_data = general_data[:num_train_samples]
            #     self.output = output[:num_train_samples]
            # else:
            #     num_train_samples = int(len(general_data) * self.train_ratio)
            #     self.input_temporal_sequence_data = temporal_sequence_data[num_train_samples:]
            #     self.input_general_data = general_data[num_train_samples:]
            #     self.output = output[num_train_samples:]

        if self.split=='train':
            self.input_general_data = np.array(input_general_data_train).squeeze(1).astype(np.float32)
            self.input_temporal_sequence_data = np.array(input_temporal_sequence_data_train)
            self.output = np.array(output_train).astype(np.float32)
        else:
            self.input_general_data = np.array(input_general_data_test).squeeze(1).astype(np.float32)
            self.input_temporal_sequence_data = np.array(input_temporal_sequence_data_test)
            self.output = np.array(output_test).astype(np.float32)

        self.input_temporal_sequence_data = np.array(self.input_temporal_sequence_data)
        self.time_data = self.input_temporal_sequence_data[:,:,0].astype(np.float32)
        self.walking_speed = self.input_temporal_sequence_data[:,:,1].astype(np.float32)
        self.HR = self.input_temporal_sequence_data[:,:,2].astype(np.float32)
        
    def __len__(self):
        return len(self.input_general_data)
    
    def __getitem__(self, idx):
        return self.time_data[idx], self.walking_speed[idx], self.HR[idx], self.input_general_data[idx], self.output[idx]    