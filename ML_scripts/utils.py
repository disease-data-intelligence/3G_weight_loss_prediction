#!/usr/bin/python3

# Import packages
seed = 42
import numpy as np
np.random.seed(seed) # Set numpy random seed
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, KFold
import scipy

#########################
#   Helping functions
#########################
# Calculate performance metrics
def performance_metrics(y_true, y_pred, threshold=0.5): 
    """
    y_true: True labels
    y_pred: Predicted labels
    """
    # Estimate roc_auc for current k-fold
    auc_score = roc_auc_score(y_true=y_true, y_score=y_pred)

    # Calculate confusion matrix, sensitivity and specificity on test
    y_pred_labels = [1 if v >= threshold else 0 for v in y_pred]
    con = confusion_matrix(y_true=y_true, y_pred=y_pred_labels)
    tn, fp, fn, tp = con.ravel()
    sensitivity = float(tp / (tp+fn))
    specificity = float(tn / (fp+tn))
    mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred_labels)

    return auc_score, sensitivity, specificity, mcc, con

# Get intersection between two lists
def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 


########################
#   Shuffle target 
########################
# Shuffle target
def shuffle_target(y_target, seed=None, verbose=False):
    """
    y_train: Training labels, input as numpy array. Must have same N as X_train. 
    seed: Random seed, to make reproducable results. 
    verbose: if True, prints results. 
    """
    # Shuffle target labels to check for random forest over-fitting
    shuffled = y_target.copy()
    np.random.RandomState(seed=seed).shuffle(shuffled)
    if verbose: 
        print(shuffled.head(), y_target.head())
    return shuffled

# Plot shuffle
def plot_shuffle(auc_shuffle, auc_fit, path, yerr=False, title=False, xlim=False, ylim=False, loc='upper left'): 
    # Plot distribution for shuffled scores
    fig = plt.figure(figsize=(7,6))
    if title: 
        plt.title(title)
    plt.xlabel('AUC score')
    plt.ylabel('Number of models')
    plt.hist(x=auc_shuffle, label='permuted', alpha=0.7)
    if xlim: 
        plt.xlim(xlim)
    if ylim: 
        plt.ylim(ylim)
    if not yerr: 
        yerr = 5
    plt.axvline(np.mean(auc_fit, axis=0), color='red', linestyle='-', linewidth=1, label='mean_model')
    plt.errorbar(x=np.mean(auc_fit, axis=0), y=yerr, xerr=np.std(auc_fit, axis=0), 
                 color='red', linewidth=1, capsize=5)
    plt.axvline(0.5, color='black', linestyle='dashed', linewidth=1, label='random')
    plt.hist(auc_fit, label='model_fit', alpha=0.7)
    plt.legend(loc=loc)
    plt.grid(which='major')
    plt.tight_layout()
    plt.savefig(path, dpi=500, transparent=True)
    plt.close(fig)


#####################
#     OTHER
#####################
# Normalize by training set mean and stdv
def normalize_by_train(X_train, X_test): 
    """
    X_train: Training data, input as a numpy array. 
    X_test: Test data, input as a numpy array. Must have same M as X_train. 
    """
    print('NaNs in X data:', np.sum(np.isnan(X_train)), np.sum(np.isnan(X_test)))

    # Get normalization parameters
    mean = np.nanmean(X_train, axis=0)
    stdv = np.nanstd(X_train, axis=0)

    # Normalize data
    X_train = (X_train-mean)/stdv
    X_test = (X_test-mean)/stdv

    print('NaNs in X normalized data:', np.sum(np.isnan(X_train)), np.sum(np.isnan(X_test)))

    return X_train, X_test

# Downsample (Only works for binary classification)
def downsample_major_class(X_train, y_train, major_class, seed=seed): 
    """
    X_train: Training data, input as a numpy array. 
    y_train: Training labels, input as numpy array. Must have same N as X_train. 
    major_class: integer, indicating the name of the major class in the y_train column. 
    seed: Random seed, to make reproducable results. 
    """
    # Indicies of each class' observations
    i_major = np.where(y_train == major_class)[0]
    minor_class = [i for i in [0, 1] if i != major_class][0]
    i_minor = np.where(y_train == minor_class)[0]

    # Number of observations in each class
    n_major = len(i_major)
    n_minor = len(i_minor)

    # For every observation of class 0, randomly sample from class 1 without replacement
    i_major_downsampled = np.random.RandomState(seed=seed).choice(i_major, size=n_minor, replace=False)

    # Join together class 0's target vector with the downsampled class 1's target vector
    y_train = np.concatenate((y_train[i_minor,], y_train[i_major_downsampled,]))

    # Concatenate the new training set 
    X_train = np.concatenate((X_train[i_minor,:], X_train[i_major_downsampled,:]))

    return X_train, y_train
