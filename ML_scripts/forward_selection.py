#!/usr/bin/python3

# Import packages
seed = 42
import numpy as np
np.random.seed(seed) # Set numpy random seed
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
import random
random.seed(seed)
import utils as u
import model as mod

# Add feature to model
def add_and_run(X, y, col, added, rf, cv=5, seed=seed, verbose=False): 
    """Function to add a feature and run this combination of features. This function 
    was made in order to be able to run different combinations in parallel."""
    
    if verbose: 
        print('\n# Adding {0} to {1}'.format(col, added))
        print('\tNumber of features in model is: ', len([col]+added))

    # Add variable
    X_crop = X[added+[col]]

    # Make model
    metrics, roc_data, ids_test, preds_labels, thresholds, models = mod.sklearn_model(X_data=X_crop, 
                                                                                 y_data=y, 
                                                                                 model=rf, seed=seed, 
                                                                                 cv_splits=cv, 
                                                                                 stratified=True, 
                                                                                 verbose=False)

    # Performance
    return [(X_crop, y), metrics, roc_data, ids_test, preds_labels, thresholds, models]

# Function to do fast forward selection
def forward_selection2(X_data, y_data, model, max_features=10, selects=None, frac=0.4, step=0.1, seed=None, cv=5): 
    # Features and samples
    N, M = X_data.shape
    all_features = X_data.columns.tolist()
    m = model
    print('Maximum number of selected features: {0}/{1}'.format(max_features, len(all_features)))

    # Base model 
    if selects != None: 
        base = [idx for idx, f in enumerate(all_features) if f not in selects]
        X_crop = X_data.iloc[:,base]
        metrics, roc_data, ids_test, preds_labels, thresholds, models = mod.sklearn_model(X_data=X_crop, 
                                                                                     y_data=y_data, 
                                                                                     model=m, 
                                                                                     seed=seed, 
                                                                                     cv_splits=cv, 
                                                                                     stratified=True, 
                                                                                     verbose=False)
        
        perf_models = [[(X_crop, y_data), metrics, roc_data, ids_test, preds_labels, thresholds, models]]
        added = [x for x in X_crop.columns]
        M = len(added)
        M_best = M
        max_auc = metrics[1].auc.iloc[0]
    else: 
        selects = X_data.columns.tolist()
        perf_models = []
        added = []
        M = 0
        M_best = M
        max_auc = 0
        
    # Run until optimal features is found 
    auc_new = 0
    remove = []
    X = X_data.copy()
    while len(selects) > 0 and len(added) <= max_features:
        m = model

        # Find number of cores available
        n_cpus = multiprocessing.cpu_count()
        if int(n_cpus) >= len(selects): 
            n_jobs = len(selects)
        else: 
            n_jobs = int(n_cpus*0.8)

        # Do parallel computing for each run of n_features
        print('\tParallel jobs:', n_jobs)
        model_data = Parallel(n_jobs=n_jobs)(delayed(add_and_run)(X, y_data, col, added, m, cv=5, seed=seed, verbose=False) for col in selects)
        perf_models.extend(model_data)
        
        # Find highest performing model
        test_aucs = [mod[1][1].auc.iloc[0] for mod in perf_models]
        max_auc = max(test_aucs)
        max_idx = [i for i, j in enumerate(test_aucs) if j == max_auc]

        if len(max_idx) > 1: 
            more_than_one = list(set([item for idx in max_idx for item in perf_models[idx][0][0].columns if item in selects]))
            print('\tMore than one best model present:', max_idx, more_than_one)

            # Test if both features are equally good in combined model
            X_crop = X_data[more_than_one]
            metrics, roc_data, ids_test, preds_labels, thresholds, models = mod.sklearn_model(X_data=X_crop, 
                                                                                         y_data=y_data, 
                                                                                         model=m, 
                                                                                         seed=seed, 
                                                                                         cv_splits=cv, 
                                                                                         stratified=True, 
                                                                                         verbose=False)
            
            if metrics[1].auc.iloc[0] >= max_auc:
                best_model = [(X_crop, y_data), metrics, roc_data, ids_test, preds_labels, thresholds, models]
            else: 
                print('\tNew model was not better')
                
                # Set new best model 
                max_idx = [random.choice(max_idx)]
                best_model = perf_models[max_idx[0]]
        else: 
            # Set new best model 
            best_model = perf_models[max_idx[0]]

        auc_new = max_auc
        X_crop = best_model[0][0]
        if M_best == X_crop.shape[1]: 
            print('\t### Best model was found: AUC = {0} with {1} features\n'.format(auc_new, M_best))
            break

        M_best = X_crop.shape[1]
        added += [x for x in X_crop.columns if x not in added]

        # Set number to remove
        n_remove = int(len(perf_models)*frac)
        if n_remove <= 0: 
            # Remove the smallest performing feature
            min_auc = min(test_aucs)
            idx_remove = [i for i, j in enumerate(test_aucs) if j == min_auc]
            n_remove = 1
        else: 
            # Sort performances and indices
            sorted_perf_idx = [i for p, i in sorted(zip(test_aucs, range(len(test_aucs))))]
            idx_remove = sorted_perf_idx[:n_remove]
        print('\tNumber of features to remove:', n_remove, step, round(frac,2))

        # Remove features with the selected indices
        to_remove = [f for i, pm in enumerate(perf_models) for f in pm[0][0].columns if i in idx_remove and f not in added and f not in remove]
        remove += to_remove

        # Reset possibilities to select from (remove bottom 10% performing features)
        selects = [f for f in all_features if f not in added and f not in remove]
        print('\tNumber of features to select from:', len(selects))
        perf_models = [best_model]
        if round(frac,2) <= 0.1: 
            step = 0.01
            frac -= step
            print('\tUpdating step-size...', round(frac,2))
        else: 
            frac -= step

        M += len(max_idx)
        print('\t# Best performance is AUC = {0} with {1} features\n'.format(auc_new, M_best))
    return best_model # Format of returned: [(X_crop, y), metrics, roc_data, ids_test, preds_labels, thresholds, models]



# # Function to do exhaustive forward selection
def forward_selection(X_data, y_data, model, selects=None, seed=None, extra_its=2, cv=5): 
    # Features and samples
    N, M = X_data.shape
    all_features = X_data.columns.tolist()
    m = model

    # Base model 
    if selects != None: 
        base = [idx for idx, f in enumerate(all_features) if f not in selects]
        X_crop = X_data.iloc[:,base]
        metrics, roc_data, ids_test, preds_labels, thresholds, models = mod.sklearn_model(X_data=X_crop, 
                                                                                     y_data=y_data, 
                                                                                     model=m, 
                                                                                     seed=seed, 
                                                                                     cv_splits=cv, 
                                                                                     stratified=True, 
                                                                                     verbose=False)
        
        perf_models = [[(X_crop, y_data), metrics, roc_data, ids_test, preds_labels, thresholds, models]]
        added = X_crop.columns.tolist()
        M = len(added)
        M_best = M
    else: 
        selects = X_data.columns.tolist()
        perf_models = []
        added = []
        M = 0
        M_best = M
        
    # Run until optimal features is found 
    auc_new = 0
    X = X_data.copy()
    while M_best+extra_its >= M or M < len(all_features):
        m = model
        
        # Find number of cores available
        n_cpus = multiprocessing.cpu_count()

        # Do parallel computing for each run of n_features
        model_data = Parallel(n_jobs=int(n_cpus*0.75))(delayed(add_and_run)(X, y_data, col, added, m, cv=5, seed=seed, verbose=False) for col in selects)
        perf_models.extend(model_data)
        
        # Find highest performing model
        test_aucs = [mod[1][1].auc.iloc[0] for mod in perf_models]
        max_auc = max(test_aucs)
        max_idx = [i for i, j in enumerate(test_aucs) if j == max_auc]

        if len(max_idx) > 1: 
            more_than_one = list(set([item for idx in max_idx for item in perf_models[idx][0][0].columns]))
            print('More than one best model present:', max_idx, more_than_one)

            # Test if both features are equally good in combined model
            X_crop = X_data[more_than_one]
            metrics, roc_data, ids_test, preds_labels, thresholds, models = mod.sklearn_model(X_data=X_crop, 
                                                                                         y_data=y_data, 
                                                                                         model=m, 
                                                                                         seed=seed, 
                                                                                         cv_splits=cv, 
                                                                                         stratified=True, 
                                                                                         verbose=False)
            
            if metrics[1].auc.iloc[0] >= max_auc:
                best_model = [(X_crop, y_data), metrics, roc_data, ids_test, preds_labels, thresholds, models]
                auc_new = max_auc
                M_best = best_model[0][0].shape[1]
                perf_models = [best_model]
                X_crop = best_model[0][0]
                added = X_crop.columns.tolist()
                selects = [f for f in all_features if f not in added]
            else: 
                print('New model was not better')
                
                # Set new best model 
                best_model = perf_models[random.choice(max_idx)]
                auc_new = max_auc
                M_best = best_model[0][0].shape[1]
                perf_models = [best_model]
                X_crop = best_model[0][0]
                added = X_crop.columns.tolist()
                selects = [f for f in all_features if f not in added]


        else: 
            # Set new best model 
            best_model = perf_models[max_idx[0]]
            auc_new = max_auc
            M_best = best_model[0][0].shape[1]
            perf_models = [best_model]
            X_crop = best_model[0][0]
            added = X_crop.columns.tolist()
            selects = [f for f in all_features if f not in added]
        M += 1
        print('# Best performance is AUC = {0} with {1} features'.format(auc_new, M_best))
    return best_model # [(X_crop, y), metrics, roc_data, ids_test, preds_labels, thresholds, models]
