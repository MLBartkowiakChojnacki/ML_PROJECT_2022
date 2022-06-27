import pandas as pd
import numpy as np
import os
import instructions
import handling_outliers
import standardization


path = os.path.split(os.getcwd())
#data_directory = os.path.join(path[0], 'data\\raw')
data_directory = os.path.join(os.getcwd(), 'data\\raw')


def main():
    """loading data"""
    X = instructions.load_csv(data_directory, 'train_data')

    """rescaling data"""
    X_std = standardization.standardize(X)

    """removing outliers"""
    outliers = handling_outliers.removing_iqr(X_std)
    X_mask = handling_outliers.mask_outliers(pd.DataFrame(X_std), outliers)
    X_prep = handling_outliers.replace_missing_values(X_mask, 5)
    print(X_prep)

    """saving preprocessed data"""
    X_prep.to_csv('X_prep.csv')



if __name__ == '__main__':
    main()
