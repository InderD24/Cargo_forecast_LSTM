import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class DataPreprocessor:
    def __init__(self, file_path, n_steps_in, n_steps_out, target_column='TEU'):
        self.file_path = file_path
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.target_column = target_column
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path, parse_dates=[
                              'trans_date'], index_col='trans_date')
        self.df.index = pd.to_datetime(self.df.index)


    def clean_data(self):
        Q1 = self.df[self.target_column].quantile(0.25)
        Q3 = self.df[self.target_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.df[(self.df[self.target_column] < lower_bound) | (self.df[self.target_column] > upper_bound)]
        outliers_mean = outliers[self.target_column].mean()
        self.df[self.target_column] = self.df[self.target_column].mask(
            (self.df[self.target_column] < lower_bound) | (self.df[self.target_column] > upper_bound), outliers_mean)


    def scale_data(self):
        self.df['scaled_' + self.target_column] = self.scaler.fit_transform(
            self.df[[self.target_column]])

    def create_sequences(self):
        data = self.df[['scaled_' + self.target_column]]
        X, y = [], []
        for i in range(len(data)):
            end_idx = i + self.n_steps_in
            out_end_idx = end_idx + self.n_steps_out - 1
            if out_end_idx > len(data):
                break
            X.append(data.iloc[i:end_idx, :])
            y.append(data.iloc[end_idx - 1:out_end_idx,
                     data.columns.get_loc('scaled_' + self.target_column)])
        return np.array(X), np.array(y)
    
    
