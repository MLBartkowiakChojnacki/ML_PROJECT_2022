from sklearn.preprocessing import StandardScaler
import pandas as pd

def standardize(X):
    std = StandardScaler()

    X_std = std.fit_transform(X)
    return pd.DataFrame(X_std)
