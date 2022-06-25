# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 10:25:56 2022

@author: krzys
"""

# ladownaie danych
# podzial danych
# trenowanie danych
# przepuszczenie modelu testowego

import pandas as pd
import numpy as np
from sklearn.preprocessing import binarize
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from os import chdir
from sklearn.decomposition import PCA

path = r"C:\Users\krzys\Desktop\data science\IV semestr\machine_learning_2\projekt"
# data_directory = os.path.join(path[0], 'data\\raw')
chdir(path)


def loading_files(path):
    '''
    

    Parameters
    ----------
    path : our working directory

    Returns
    -------
    X : DataFrame
    
    y : DataFrame

    '''
    X = pd.read_csv("train_data.csv", header=None)
    y = pd.read_csv("train_labels.csv", header=None)
    return X, y


def binarization(y):
    '''
    

    Parameters
    ----------
    y : DataFrame
        Target variable

    Returns
    -------
    y : DataFrame
        Target variable after binarization.

    '''
    y_binar = binarize(y)
    y = pd.DataFrame(np.ravel(y_binar))
    return y


def principal_component_analysis(X, n_components=0.95):
    '''
    

    Parameters
    ----------
    X : DataFrame
        Labels which will undergo PCA.
    n_components : float
        DESCRIPTION. The default is 0.95.

    Returns
    -------
    X_pca : DataFrame
        After dimensions reduction. By default 95% of variance.

    '''
    pca = PCA(n_components=n_components, whiten=True)
    X_pca = pca.fit_transform(X)
    X_pca = pd.DataFrame(X_pca)
    return X_pca


def splitting_data(X, y):
    '''
    

    Parameters
    ----------
    X : DataFrame
        Labels.
    y : DataFrame
        Target.

    Returns
    -------
    X_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.

    '''
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, stratify=y
    )
    return X_train, X_test, y_train, y_test


def oversampling_data(X_train, y_train):
    sm = SMOTE()
    X_sm, y_sm = sm.fit_resample(X_train, y_train)
    return X_sm, y_sm


def machine_learning_model_KNN(n_neighbors=9, weights="uniform", algorithm="auto"):
    clf = KNeighborsClassifier(
        n_neighbors=n_neighbors, weights=weights, algorithm=algorithm
    )
    return clf


def learning_model(X_sm, y_sm, clf):
    clf = clf.fit(X_sm, y_sm.values.flatten()) 
    return clf


def using_model(clf, X_test):
    y_pred = clf.predict(X_test)
    return y_pred


def metric(y_test, y_pred):
    score = balanced_accuracy_score(y_test, y_pred)
    return score


def conf_matrix(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    return conf_matrix


if __name__ == "__main__":
    X, y = loading_files(path)
    y_binar = binarization(y)
    X_pca = principal_component_analysis(X, n_components=0.95)
    X_train, X_test, y_train, y_test = splitting_data(X=X_pca, y=y_binar)
    X_sm, y_sm = oversampling_data(X_train, y_train)
    clf = machine_learning_model_KNN()
    clf_learned = learning_model(X_sm, y_sm, clf)
    y_pred = using_model(clf=clf_learned, X_test=X_test)
    score = metric(y_test, y_pred)
    conf_matrix = conf_matrix(y_test, y_pred)
