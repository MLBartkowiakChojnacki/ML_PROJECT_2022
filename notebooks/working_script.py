# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 10:25:56 2022

@author: krzys
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import binarize
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn.decomposition import KernelPCA
import joblib

def loading_files(X_name, y_name):
    '''
    Loading labels and target for model.

    Parameters
    ----------
    X_name : DataFrame
        Labels used for learning model.
    y_name : DataFrame
        Target used for learning model.

    Returns
    -------
    X : DataFrame
        Labela.
    y : DataFrame
        Target.

    '''
    X = pd.read_csv(f"{X_name}.csv", header=None)
    y = pd.read_csv(f"{y_name}.csv", header=None)
    return X, y


def binarization(y):
    """
    Fucntion used to represent categorical data as numbers.

    Parameters
    ----------
    y : DataFrame
        Target variable

    Returns
    -------
    y : DataFrame
        Target variable after binarization.

    """
    y_binar = binarize(y)
    y = pd.DataFrame(np.ravel(y_binar))
    return y


def kernel_principal_component_analysis(X, n_components=1012):
    global kpca
    """
    Non-linear dimensionality reduction through the use of kernels 

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

    """
    kpca = KernelPCA(n_components=n_components)
    X_pca = kpca.fit_transform(X)
    X_pca = pd.DataFrame(X_pca)
    return X_pca


def splitting_data(X, y):
    """
    Splitting data into training and testing sets.

    Parameters
    ----------
    X : DataFrame
        Labels.
    y : DataFrame
        Target.

    Returns
    -------
    X_train : DataFrame
        Training labels.
    X_test : DataFrame
        Testing labels.
    y_train : DataFrame
        Training target.
    y_test : DataFrame
        Testing target.

    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


def oversampling_data(X_train, y_train):
    """
    Function to perform over-sampling using SMOTE.

    Parameters
    ----------
    X_train : DataFrame
        Training labels.
    y_train : DataFrame
        training target.

    Returns
    -------
    X_sm : DataFrame
        Training labels after oversampling.
    y_sm : DataFrame
        Training target after oversampling.

    """
    sm = SMOTE()
    X_sm, y_sm = sm.fit_resample(X_train, y_train)
    return X_sm, y_sm


def machine_learning_model_KNN(n_neighbors=9, weights="uniform", algorithm="auto"):
    """
    Function used to definie KNN algorithm.

    Parameters
    ----------
    n_neighbors : int,
        Number of neighbors algorithm. The default is 9.
    weights : str,
        Weight function used in prediction. The default is "uniform".
    algorithm : str,
        Algorithm used to compute the nearest neighbors. The default is "auto".

    Returns
    -------
    clf : classifier
        Model used for classification.

    """
    clf = KNeighborsClassifier(
        n_neighbors=n_neighbors, weights=weights, algorithm=algorithm
    )
    return clf


def learning_model(X_sm, y_sm, clf):
    """
    Function used to train model.

    Parameters
    ----------
    X_sm : DataFrame
        Labels after oversampling.
    y_sm : DataFrame
        Target after oversampling.
    clf : classifier
        Classifier.

    Returns
    -------
    clf : classifier
        Trained classifier.

    """
    clf = clf.fit(X_sm, y_sm.values.flatten())
    return clf


def using_model(clf, X_test):
    """
    Function used to predict targets for training set to measure its scores.

    Parameters
    ----------
    clf : classifier
        Classifier used in our model.
    X_test : DataFrame
        Test target used for validation.

    Returns
    -------
    y_pred : array of int64
        Predicted target for X_test labels.

    """
    y_pred = clf.predict(X_test)
    return y_pred


def metric(y_test, y_pred):
    """
    Function used to measure score. Here we have balanced accuracy score.

    Parameters
    ----------
    y_test : DataFrame
        Target as it is.
    y_pred : array of int64
        Target as it is predicted.

    Returns
    -------
    score : float
        Here is used balanced accuracy score.
        On a slale 0-1 it describes how good our predictions are.

    """
    score = balanced_accuracy_score(y_test, y_pred)
    return score


def conf_matrix(y_test, y_pred):
    """
    Function used to generate confusion matrix.

    Parameters
    ----------
    y_test : DataFrame
        Test target.
    y_pred : array of int64
        Predicted target.

    Returns
    -------
    conf_matrix : array of int64
        Specific table layout that allows visualization of the performance
        of an algorithm .

    """
    conf_matrix = confusion_matrix(y_test, y_pred)
    np.savetxt("confusion_matrix.csv", conf_matrix, delimiter=",")
    return conf_matrix


def load_test_data():
    """
    Loading test data

    Returns
    -------
    test_data : DataFrame

    """
    test_data = pd.read_csv("test_data.csv", header=None)
    return test_data


def predict_target_for_test_data(test_data, clf, kpca):
    """
    Function used to predict target for test data set.

    Parameters
    ----------
    test_data : DataFrame
        Data for which labels we want to predict.
    clf : classfier
        Classifier used for prediction.
    kpca : KernelPCA
        kPCA object used to transform input data.

    Returns
    -------
    y_test_pred : DataFrame
        Predicted target for input data.

    """
    y_test_pred = clf.predict(kpca.transform(test_data))
    y_test_pred = pd.DataFrame(y_test_pred)
    return y_test_pred


def y_test_pred_to_csv(y_test_pred, name):
    """
    Function used to save predicted target as csv.

    Parameters
    ----------
    y_test_pred : DataFrame
        Predicted target for test data set..

    Returns
    -------
    None.

    """
    y_test_pred.to_csv(f"{name}.csv", index = False, header=None)


if __name__ == "__main__":
    os.chdir(os.path.join(os.getcwd(),"projekt"))
    X, y = loading_files(X_name = "train_data", y_name = "train_labels")
    y_binar = binarization(y)
    X_pca = kernel_principal_component_analysis(X, n_components=1012)
    X_train, X_test, y_train, y_test = splitting_data(X_pca, y_binar)
    X_sm, y_sm = oversampling_data(X_train, y_train)
    clf = machine_learning_model_KNN(n_neighbors=9, weights="uniform", algorithm="auto")
    clf_learned = learning_model(X_sm, y_sm, clf)
    y_pred = using_model(clf=clf_learned, X_test=X_test)
    score = metric(y_test, y_pred)
    conf_matrix = conf_matrix(y_test, y_pred)
    file = "KNN.sav"
    joblib.dump(clf_learned, filename=file)
    test_data = load_test_data()
    y_test_pred = predict_target_for_test_data(test_data, clf=clf_learned, kpca=kpca)
    y_test_pred_to_csv(y_test_pred, name="predicted_test_target")
