"""
Utility Functions for Linear Regression

Contains helper functions used across implementations.

"""

import numpy as np


def add_bias_feature(X):

    """
    Adds x0 = 1 column to feature matrix.
    """

    ones = np.ones((X.shape[0], 1))

    return np.concatenate((ones, X), axis=1)


def mean_squared_error(y_true, y_pred):

    """
    Computes Mean Squared Error.
    """

    return np.mean((y_true - y_pred) ** 2)


def train_test_split(X, y, test_size=0.2):

    """
    Splits dataset into train and test sets.
    """

    m = len(X)

    split_index = int(m * (1 - test_size))

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test


def feature_normalize(X):

    """
    Performs feature normalization.

    Formula:

    x = (x - mean) / std
    """

    mean = np.mean(X, axis=0)

    std = np.std(X, axis=0)

    normalized_X = (X - mean) / std

    return normalized_X, mean, std
