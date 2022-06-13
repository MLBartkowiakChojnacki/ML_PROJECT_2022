import pandas as pd
import numpy as np
import pickle as pkl
import os


def set_abs_path(folder_name: str) -> str:
    full_dir = os.path.join(os.getcwd(), folder_name)
    return full_dir


def load_csv(directory: str, df_name: str) -> pd.DataFrame:
    """loading dataframe for the first time"""
    df = pd.read_csv('{}/{}.csv'.format(directory, df_name), header=None)
    if df.shape[1] > 1:
        df.columns = [ 'feature_' + str(i) for i in range(len(df.columns)) ]
    elif df.shape[1] == 1:
        df.columns = ['target']
    return pd.DataFrame(df)


def save_df_as_pkl(df: pd.DataFrame, directory: str, filename: str):
    """saving dataframe as pickle"""
    df.to_pickle('{}/{}.pkl'.format(directory, filename))
    return 


def load_df_from_pkl(directory: str, filename: str) -> pd.DataFrame:
    """loading prepared and pickled dataframe"""
    df = pd.read_pickle('{}/{}.pkl'.format(directory, filename))
    return pd.DataFrame(df)


df_name = 'train_labels'
filename = 'y'
full_dir = set_abs_path('data\\raw')
y = load_csv(full_dir, df_name)
print(y.head())
save_df_as_pkl(y, full_dir, filename)
df = load_df_from_pkl(full_dir, filename)
print(df.head())