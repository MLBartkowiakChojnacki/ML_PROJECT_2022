import pandas as pd
import numpy as np
import pickle as pkl

data_directory_vsc = 'C:/Users/Marta/Desktop/Studia/CDV/IV semestr 2022L/Wykorzystanie Pythona w uczeniu maszynowym/ml_project/project/ML_PROJECT_2022/data'
data_directory_colab = '/content/project/data'

directory = 'C:/Users/Marta/Desktop/project'
df_name = 'train_labels'


# loading dataframe for the first time
def load_csv(directory: str, df_name: str) -> pd.DataFrame:
    df = pd.read_csv('{}/{}.csv'.format(directory, df_name), header=None)
    if df.shape[1] > 1:
        df.columns = [ 'feature_' + str(i) for i in range(len(df.columns)) ]
    elif df.shape[1] == 1:
        df.columns = ['target']
    return df


# saving dataframe as pickle
def save_df_as_pkl(df: pd.DataFrame, directory: str, filename: str) -> str:
    df.to_pickle('{}/{}.pkl'.format(directory, filename))
    return print("Dataframe '{}.pkl' saved as a pickle.".format(filename))


# loading prepared and pickled dataframe
def load_df_from_pkl(directory: str, filename: str) -> pd.DataFrame:
    df = pd.read_pickle('{}/{}.pkl'.format(directory, filename))
    return df


filename = 'y'
y = load_csv(directory, df_name)
print(y.head())
save_df_as_pkl(y, directory, filename)
df = load_df_from_pkl(directory, filename)
print(df.head())