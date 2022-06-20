import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy.stats import zscore, median_abs_deviation


def removing_iqr(dataset: pd.DataFrame) -> pd.DataFrame:
    for column in dataset:
        q1, q3 = np.percentile(dataset[column], [25, 75])
        iqr = q3 - q1
        dataset[column] = (dataset[column] < (q1 - 1.5 * iqr)) | (dataset[column] > (q3 + 1.5 * iqr))
    return pd.DataFrame(dataset)


def removing_percentiles(dataset: pd.DataFrame) -> pd.DataFrame:
    for column in dataset:
        low, high = np.percentile(dataset[column], [10, 90])
        dataset[column] = (dataset[column] < low) | (dataset[column] > high)
    return pd.DataFrame(dataset)


def zscore_outlier(dataset: pd.DataFrame) -> pd.DataFrame:
    z_score = np.abs(zscore(dataset))
    treshold = 3
    return z_score > treshold
    

def modified_z_score_outlier(dataset: pd.DataFrame) -> pd.DataFrame:
    mad_column = median_abs_deviation(dataset)
    median = np.median(dataset, axis = 0)
    mad_score = np.abs(0.6745 * (dataset - median) / mad_column)
    return mad_score > 3.5


def count_outliers(dataset: pd.DataFrame) -> pd.DataFrame:
    outliers_count = dataset.sum(axis=1)
    percentage = []
    for i in outliers_count:
        percentage.append(i/10000)
    sorted_percentages = pd.DataFrame(percentage).sort_values(by=[0], ascending=False)
    return outliers_count, sorted_percentages


def mask_outliers(dataset: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
    dataset_copy = dataset.copy()
    dataset_copy = dataset_copy.where(outliers == False, np.nan)
    return pd.DataFrame(dataset_copy)


def replace_missing_values(dataset: pd.DataFrame, n_neighbors: int) -> pd.DataFrame:
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_df = knn_imputer.fit_transform(dataset)
    return pd.DataFrame(imputed_df)

