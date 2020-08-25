#!/usr/bin/python3

# Import packages
seed = 42
import numpy as np
np.random.seed(seed) # Set numpy random seed
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
import scipy
import utils as u

########################
#   Sklearn models
########################
# Function to train and test sklearn estimator
def sklearn_model(X_data, y_data, model, seed=seed, cv_splits=5, stratified=True, permute=False, verbose=False):
    """
    Parameter optimization can be built on top of this function. 

    X_data: Model data, input as pandas dataframe. 
    y_data: True labels, input as numpy 1d array. 
    model: Sklearn class classifier object with function ".predict_proba()". 
    seed: Random seed, to make reproducable results. 
    cv_splits: Number of crossvalidation splits, max = N. 
    stratified: Whether or not to make k-fold splits stratified by target classes. Only used when cv_splits < N. 
    verbose: if True, prints results. 
    """
    # Check IDs are in same order for X and y
    intersect = X_data.index.intersection(y_data.index)
    X_data, y_data = X_data.loc[intersect], y_data.loc[intersect]
    print(X_data.shape, y_data.shape)

    # Data shapes 
    N, M = X_data.shape

    # If permutation test is enabled
    if permute: 
        y_shuffle = y_data['y_shuffle']
        y_data = y_data['y']

    # Set cross-validation (partition vector)
    if cv_splits == N: 
        part_vec = np.array(list(range(N)))
    else: 
        if stratified: 
            CV = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)   # Split stratified by target value
        else: 
            CV = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)             # Random split
        part_vec, k = np.zeros(N), 0
        for train_index, test_index in CV.split(X_data, y_data): # y must here be made a 1d array if error occurs
            part_vec[test_index] = k
            k += 1

    # Set lists and parameters  
    train_scores, test_scores = [], []
    train_pred, test_pred = [], []
    y_target_train, y_target_test = [], []
    ids_test, model_list = [], []
    X, y = X_data.values, y_data.values.ravel()
    if permute: 
        ys = y_shuffle.values.ravel()

    # Run through cross-validation
    for k in range(cv_splits): 
        if verbose: 
            print('\n# K-fold {0}/{1}'.format(k+1, cv_splits))

        # Get training and test data
        train_index, test_index = np.where(part_vec != k), np.where(part_vec == k)
        X_train, X_test = X[train_index], X[test_index]
        print(X_train.shape, X_test.shape)
        if permute: 
            y_train, y_test = y[train_index], ys[test_index]
        else: 
            y_train, y_test = y[train_index], y[test_index]

        # Get the IDs of the test group
        test_ids = X_data.index[test_index]
        ids_test.append(test_ids)

        # Classbalance
        unique, counts = np.unique(y_train, return_counts=True)
        print('\tClassbalance in training is:\t', dict(zip(unique, counts)))
        unique, counts = np.unique(y_test, return_counts=True)
        print('\tClassbalance in test is:\t', dict(zip(unique, counts)), '\n')

        # Train model 
        model.fit(X_train, y_train)

        # Calculate metrics
        if cv_splits == N: 
            # Get prediction on training and test set 
            pred_train = model.predict_proba(X_train)[:,1]
            pred_test = model.predict_proba(X_test)[:,1]

        else: 
            # Get prediction on training and test set 
            pred_train = model.predict_proba(X_train)[:,1]
            pred_test = model.predict_proba(X_test)[:,1]

            #auc.append(roc_auc_score(y_test, y_test_pred))
            auc_train, sens_train, spec_train, mcc_train, con_train = u.performance_metrics(y_true=y_train, y_pred=pred_train)
            auc_test, sens_test, spec_test, mcc_test, con_test = u.performance_metrics(y_true=y_test, y_pred=pred_test)

            # Append metrics to lists
            if verbose: 
                print('AUC train = {0}, AUC test = {1}'.format(auc_train, auc_test))
            train_scores.append([auc_train, sens_train, spec_train, mcc_train])
            test_scores.append([auc_test, sens_test, spec_test, mcc_test])

        # Append labels and predictions
        #model_list.append(model)
        model_list.append(model.feature_importances_.tolist()) # save feature importances instead
        train_pred.append(pred_train)
        test_pred.append(pred_test)
        y_target_train.append(y_train)
        y_target_test.append(y_test)
        k += 1

    # convert to long np array:
    y_train_pred=np.array(train_pred)
    y_test_pred=np.array(test_pred)
    y_target_train=np.array(y_target_train)
    y_target_test=np.array(y_target_test)
    ids_test=np.array(ids_test)
    
    # Flatten arrays
    y_train_pred=np.concatenate(y_train_pred).flatten()
    y_test_pred=np.concatenate(y_test_pred).flatten()
    y_target_train=np.concatenate(y_target_train).flatten()
    y_target_test=np.concatenate(y_target_test).flatten()
    ids_test=np.concatenate(ids_test).flatten()

    # Make dataframe for metrics
    auc_train, sens_train, spec_train, mcc_train, con_train = u.performance_metrics(y_true=y_target_train, y_pred=y_train_pred)
    auc_test, sens_test, spec_test, mcc_test, con_test = u.performance_metrics(y_true=y_target_test, y_pred=y_test_pred)
    metrics_train = pd.DataFrame(data=np.array([[auc_train], [sens_train], [spec_train], [mcc_train]]).T, columns=['auc', 'sens', 'spec', 'mcc'], index=[seed])
    metrics_test = pd.DataFrame(data=np.array([[auc_test], [sens_test], [spec_test], [mcc_test]]).T, columns=['auc', 'sens', 'spec', 'mcc'], index=[seed])

    metrics = (metrics_train, metrics_test)
    
    if verbose: 
        print("\n# Train AUC: " + str(round(auc_train,4)))
        print("# Test AUC: " + str(round(auc_test,4)))

    # calculate false positive and true positive rate at different thresholds (ROC curve):
    fpr,tpr,thresholds = roc_curve(y_target_test, y_test_pred, drop_intermediate=False)

    return metrics, (fpr, tpr), ids_test, (y_test_pred, y_target_test), thresholds, model_list


# Function to train and test sklearn estimator
def sklearn_predict(X_data, y_data, model, seed=seed, cv_splits=5, stratified=True, verbose=False, valid_set=False):
    """
    Parameter optimization can be built on top of this function. 

    X_data: Model data, input as pandas dataframe. 
    y_data: True labels, input as numpy 1d array. 
    model: Sklearn class classifier object with function ".predict_proba()". 
    seed: Random seed, to make reproducable results. 
    cv_splits: Number of crossvalidation splits, max = N. 
    stratified: Whether or not to make k-fold splits stratified by target classes. Only used when cv_splits < N. 
    verbose: if True, prints results. 
    """
    # Check IDs are in same order for X and y
    intersect = X_data.index.intersection(y_data.index)

    # Check if any validation set samples are in the training data 
    if valid_set != False: 
        X_val = X_data.loc[valid_set]
        y_val = y_data.loc[valid_set]

        intersect2 = [ID for ID in intersect if ID not in valid_set]
        X_data, y_data = X_data.loc[intersect2], y_data.loc[intersect2]
    else: 
        X_data, y_data = X_data.loc[intersect], y_data.loc[intersect]
        
    print('Sklearn shape of data:', X_data.shape, y_data.shape)

    # Data shapes 
    N, M = X_data.shape

    # Set cross-validation (partition vector)
    if cv_splits == N: 
        part_vec = np.array(list(range(N)))
    else: 
        if stratified: 
            CV = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)   # Split stratified by target value
        else: 
            CV = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)             # Random split
        part_vec, k = np.zeros(N), 0
        for train_index, test_index in CV.split(X_data, y_data): # y must here be made a 1d array if error occurs
            part_vec[test_index] = k
            k += 1

    # Set lists and parameters  
    train_scores, test_scores = [], []
    train_pred, test_pred = [], []
    y_target_train, y_target_test = [], []
    ids_test, model_list = [], []
    X, y = X_data.values, y_data.values.ravel()

    # Run through cross-validation
    for k in range(cv_splits): 
        if verbose: 
            print('\n# K-fold {0}/{1}'.format(k+1, cv_splits))

        # Get training and test data
        train_index, test_index = np.where(part_vec != k), np.where(part_vec == k)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Get the IDs of the test group
        test_ids = X_data.index[test_index]
        ids_test.append(test_ids)
        
        # Classbalance
        if verbose: 
            unique, counts = np.unique(y_train, return_counts=True)
            print('\tClassbalance in training is:\t', dict(zip(unique, counts)))
            unique, counts = np.unique(y_test, return_counts=True)
            print('\tClassbalance in test is:\t', dict(zip(unique, counts)), '\n')

        # Train model 
        model.fit(X_train, y_train)

        # Calculate metrics
        if cv_splits == N: 
            # Get prediction on training and test set 
            pred_train = model.predict_proba(X_train)[:,1]
            pred_test = model.predict_proba(X_test)[:,1]

        else: 
            # Get prediction on training and test set 
            pred_train = model.predict_proba(X_train)[:,1]
            pred_test = model.predict_proba(X_test)[:,1]

            #auc.append(roc_auc_score(y_test, y_test_pred))
            auc_train, sens_train, spec_train, mcc_train, con_train = u.performance_metrics(y_true=y_train, y_pred=pred_train)
            auc_test, sens_test, spec_test, mcc_test, con_test = u.performance_metrics(y_true=y_test, y_pred=pred_test)

            # Append metrics to lists
            if verbose: 
                print('AUC train = {0}, AUC test = {1}'.format(auc_train, auc_test))
                
            train_scores.append([auc_train, sens_train, spec_train, mcc_train])
            test_scores.append([auc_test, sens_test, spec_test, mcc_test])

        # Append labels and predictions
        #model_list.append(model)
        model_list.append(model.feature_importances_.tolist()) # save feature importances instead
        train_pred.append(pred_train)
        test_pred.append(pred_test)
        y_target_train.append(y_train)
        y_target_test.append(y_test)
        k += 1

    # convert to long np array:
    y_train_pred=np.array(train_pred)
    y_test_pred=np.array(test_pred)
    y_target_train=np.array(y_target_train)
    y_target_test=np.array(y_target_test)
    ids_test=np.array(ids_test)
    
    # Flatten arrays
    y_train_pred=np.concatenate(y_train_pred).flatten()
    y_test_pred=np.concatenate(y_test_pred).flatten()
    y_target_train=np.concatenate(y_target_train).flatten()
    y_target_test=np.concatenate(y_target_test).flatten()
    ids_test=np.concatenate(ids_test).flatten()

    # Make dataframe for metrics
    auc_train, sens_train, spec_train, mcc_train, con_train = u.performance_metrics(y_true=y_target_train, y_pred=y_train_pred)
    auc_test, sens_test, spec_test, mcc_test, con_test = u.performance_metrics(y_true=y_target_test, y_pred=y_test_pred)
    metrics_train = pd.DataFrame(data=np.array([[auc_train], [sens_train], [spec_train], [mcc_train]]).T, columns=['auc', 'sens', 'spec', 'mcc'],
                                index=[seed])
    metrics_test = pd.DataFrame(data=np.array([[auc_test], [sens_test], [spec_test], [mcc_test]]).T, columns=['auc', 'sens', 'spec', 'mcc'],
                                index=[seed])

    metrics = (metrics_train, metrics_test)

    # Print results
    if verbose: 
        print("\n# Train AUC: " + str(round(auc_train,4)))
        print("# Test AUC: " + str(round(auc_test,4)))
    
    # If validation set is specified
    if valid_set != False: 
        # Class balance on validation set
        unique, counts = np.unique(y_val, return_counts=True)
        print('\tClassbalance in validation is:\t', dict(zip(unique, counts)))

        # Predict on validation set
        pred_val = model.predict_proba(X_val)[:,1]
        auc_val, sens_val, spec_val, mcc_val, con_val = u.performance_metrics(y_true=y_val, y_pred=pred_val)

        # Calculate false positive and true positive rate at different thresholds (ROC curve):
        fpr,tpr,thresholds = roc_curve(y_val, pred_val, drop_intermediate=False)

        validation = [(X_val, y_val, pred_val), (fpr, tpr), [auc_val, sens_val, spec_val, mcc_val]]

        return validation, metrics, (fpr, tpr), ids_test, (y_test_pred, y_target_test), thresholds, model_list
    else: 
        # Calculate false positive and true positive rate at different thresholds (ROC curve):
        fpr,tpr,thresholds = roc_curve(y_target_test, y_test_pred, drop_intermediate=False)

        return metrics, (fpr, tpr), ids_test, (y_test_pred, y_target_test), thresholds, model_list


############################################
#   Extra functions
############################################
# Function to print mean performances
def print_means(train, test, to_file=False, d=4, output=True): 
    mtrain = train.mean()
    mtest = test.mean()
    strain = train.std()
    stest = test.std()

    n_features = []
    for val in train[['(samples|features)']].values: 
        n_features.append(val[0][1])
    n_samples = val[0][0]

    if output: 
        # Data shape
        print('\tShape of data: n_samples = {0}, n_features = {1}+-{2}'.format(n_samples, sum(n_features)/train.shape[0], round(np.std(n_features), d)))
        #print('\tShape of data: {}'.format('|'.join(str(train[['(samples|features)']].iloc[0].values[0]).split(', '))))

        # AUC
        print('\tTrain AUC = {0}+-{1}, Test AUC = {2}+-{3}'.format(round(mtrain.auc,d), round(strain.auc,d), 
                                                                round(mtest.auc,d), round(stest.auc,d)))
        # Sensitivity
        print('\tTrain sens = {0}+-{1}, Test sens = {2}+-{3}'.format(round(mtrain.sens,d), round(strain.sens,d), 
                                                                   round(mtest.sens,d), round(stest.sens,d)))
        # Specificity
        print('\tTrain spec = {0}+-{1}, Test spec = {2}+-{3}'.format(round(mtrain.spec,d), round(strain.spec,d), 
                                                                   round(mtest.spec,d), round(stest.spec,d)))
        # MCC
        print('\tTrain mcc = {0}+-{1}, Test mcc = {2}+-{3}\n'.format(round(mtrain.mcc,d), round(strain.mcc,d), 
                                                                   round(mtest.mcc,d), round(stest.mcc,d)))
    
    if to_file: 
        n_features = '({0}|{1})'.format(n_samples, int(sum(n_features)/train.shape[0]))
        #n_features = '{}'.format('|'.join(str(train[['(samples|features)']].iloc[0].values[0]).split(', ')))
        auc_train = '{0}+-{1}'.format(round(mtrain.auc,d), round(strain.auc,d))
        sens_train = '{0}+-{1}'.format(round(mtrain.sens,d), round(strain.sens,d))
        spec_train = '{0}+-{1}'.format(round(mtrain.spec,d), round(strain.spec,d))
        mcc_train = '{0}+-{1}'.format(round(mtrain.mcc,d), round(strain.mcc,d))
        
        auc_test = '{0}+-{1}'.format(round(mtest.auc,d), round(stest.auc,d))
        sens_test = '{0}+-{1}'.format(round(mtest.sens,d), round(stest.sens,d))
        spec_test = '{0}+-{1}'.format(round(mtest.spec,d), round(stest.spec,d))
        mcc_test = '{0}+-{1}'.format(round(mtest.mcc,d), round(stest.mcc,d))
        
        return (n_features, auc_train, sens_train, spec_train, mcc_train), (n_features, auc_test, sens_test, spec_test, mcc_test)
