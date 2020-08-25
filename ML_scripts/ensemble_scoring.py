#!/usr/bin/python3

# Import packages
seed = 42
import numpy as np
np.random.seed(seed) # Set numpy random seed
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import utils as u

###############################
#   Ensemble predictions
###############################
# Average prediction scores
def average_scoring(scores_frame, true_labels_col, drop=False, t=0.5): 
    # Extract scores and true labels
    true_labels = scores_frame[true_labels_col]
    if drop: 
        scores = scores_frame.drop(labels=[true_labels_col]+drop, axis=1)
    else: 
        scores = scores_frame.drop(labels=[true_labels_col], axis=1)
    
    # Get mean of each score
    means = np.nanmean(scores, axis=1)
    
    # Get metrics
    auc_score, sensitivity, specificity, mcc, con = u.performance_metrics(y_true=true_labels, y_pred=means, threshold=t)
    
    # Get ROC_AUC parameters
    fpr, tpr, thresholds = roc_curve(y_true=true_labels, y_score=means)
    
    return pd.Series(means, index=scores.index, name='mean'), [auc_score, sensitivity, specificity, mcc], (fpr, tpr)

# Majority voting
def majority_voting(scores_frame, true_labels_col, drop=False, t=0.5): 
    # Extract scores and true labels
    true_labels = scores_frame[true_labels_col]
    if drop: 
        scores = scores_frame.drop(labels=[true_labels_col]+drop, axis=1)
    else: 
        scores = scores_frame.drop(labels=[true_labels_col], axis=1)
    
    # Change scores to predicted class
    preds = scores.copy()
    preds[preds < t] = 0
    preds[preds >= t] = 1
    
    # Get majority class
    majorities = []
    for ID in preds.index: 
        row = preds.loc[ID]
        mean = np.nanmean(row)
        if mean < t: 
            majorities.append(0)
        else: 
            majorities.append(1)
    #majority = pd.DataFrame(majorities, columns=['preds'], index=preds.index)
    
    # Get metrics
    auc_score, sensitivity, specificity, mcc, con = u.performance_metrics(y_true=true_labels, y_pred=majorities, threshold=t)
    
    # Get ROC_AUC parameters
    fpr, tpr, thresholds = roc_curve(y_true=true_labels, y_score=majorities)
    
    return pd.Series(majorities, index=scores.index, name='majority'), [auc_score, sensitivity, specificity, mcc], (fpr, tpr)

# Confident average scores
def confident_average(scores_frame, true_labels_col, confidence=0.6, drop=False, verbose=False, t=0.5):
    # Extract scores and true labels
    true_labels = scores_frame[true_labels_col]
    if drop: 
        scores = scores_frame.drop(labels=[true_labels_col]+drop, axis=1)
    else: 
        scores = scores_frame.drop(labels=[true_labels_col], axis=1)

    # Find confident scores for each sample
    confident_scores, labels, ids = [], [], []
    no_con, no_cons = 0, {0: 0, 1: 0}
    for ID in scores.index: 
        row = scores.loc[ID]
        con = [v for v in row if v >= confidence or v <= 1-confidence]
        if len(con) < 1: 
            #print(row, '\n')
            #con = [0.5]
            no_con += 1
            if true_labels.loc[ID] == 1: 
                no_cons[1] += 1
            else: 
                no_cons[0] += 1
        else: 
            confident_scores.append(np.nanmean(con))
            labels.append(true_labels.loc[ID])
            ids.append(ID)
        #confident_scores.append(np.nanmean(con))

    # Look at samples with no confident scores
    if no_con > 0 and verbose: 
        print('Fraction of samples with no confident scores: {0}/{1}'.format(no_con, len(scores.index)))
        print('Classbalance:', no_cons)
    
    # Get metrics
    auc_score, sensitivity, specificity, mcc, con = u.performance_metrics(y_true=labels, y_pred=confident_scores, threshold=t)
    
    # Get ROC_AUC parameters
    fpr, tpr, thresholds = roc_curve(y_true=labels, y_score=confident_scores)
    
    return pd.Series(confident_scores, index=ids, name='con_scor_'+str(round(confidence, 2))), [auc_score, sensitivity, specificity, mcc], (fpr, tpr)

# Confident scores majority voting
def confident_majority_voting(scores_frame, true_labels_col, confidence=0.6, drop=False, verbose=False, t=0.5):
    # Extract scores and true labels
    true_labels = scores_frame[true_labels_col]
    if drop: 
        scores = scores_frame.drop(labels=[true_labels_col]+drop, axis=1)
    else: 
        scores = scores_frame.drop(labels=[true_labels_col], axis=1)

    # Find confident scores for each sample
    confident_votes, labels, ids = [], [], []
    no_con, no_cons = 0, {0: 0, 1: 0}
    for ID in scores.index: 
        row = scores.loc[ID]
        con = [v for v in row if v >= confidence or v <= 1-confidence]
        if len(con) < 1: 
            #print(row, '\n')
            #con = [0.5]
            no_con += 1
            if true_labels.loc[ID] == 1: 
                no_cons[1] += 1
            else: 
                no_cons[0] += 1
        else: 
            labels.append(true_labels.loc[ID])
            ids.append(ID)

            # Change scores to classes
            classes = [1 if c >= t else 0 for c in con]
            mean = np.nanmean(classes)
            if mean < t: 
                confident_votes.append(0)
            else: 
                confident_votes.append(1)

    # Look at samples with no confident scores
    if no_con > 0 and verbose: 
        print('Fraction of samples with no confident votes: {0}/{1}'.format(no_con, len(scores.index)))
        print('Classbalance:', no_cons)
    
    # Get metrics
    auc_score, sensitivity, specificity, mcc, con = u.performance_metrics(y_true=labels, y_pred=confident_votes, threshold=t)
    
    # Get ROC_AUC parameters
    fpr, tpr, thresholds = roc_curve(y_true=labels, y_score=confident_votes)
    
    return pd.Series(confident_votes, index=ids, name='con_vot_'+str(round(confidence, 2))), [auc_score, sensitivity, specificity, mcc], (fpr, tpr)


#########################################
#   Use all ensemble scoring methods
#########################################
def score_ensemble(scores, true_col='y', drop=False, min_conf=0.6, max_conf=0.99, step=0.05, threshold=0.5, verbose=True): 
    # Mean of scores
    means, [auc_score, sens, spec, mcc], (fpr, tpr) = average_scoring(scores_frame=scores, true_labels_col=true_col, drop=drop, t=threshold)
    if verbose: 
        print('\n# Mean scoring: ', auc_score, sens, spec, mcc)

    # Set up dataframe to hold info
    ens_perf = pd.DataFrame(data=[round(i, 2) for i in [0, auc_score, sens, spec, mcc]], 
                            index=['miss', 'auc', 'sens', 'spec', 'mcc'], columns=['mean'])
    N = len(means)

    # Majority voting
    majority, [auc_score, sens, spec, mcc], (fpr, tpr) = majority_voting(scores_frame=scores, true_labels_col=true_col, drop=drop, t=threshold)
    if verbose: 
        print('# Majority voting: ', auc_score, sens, spec, mcc, '\n')
    ens_perf['majority_voting'] = [round(i, 2) for i in [0, auc_score, sens, spec, mcc]]

    # Combine first two series to dataframe
    ens_scores = pd.concat([means, majority], join='outer', axis=1)

    # Mean and majority of confident scores
    if verbose: 
        print('\n### Scoring with threholds on confidence...')
    confidences = np.arange(start=min_conf, stop=max_conf, step=step)
    for con in confidences: 
        # Mean
        confident_scores, [auc_score, sens, spec, mcc], (fpr, tpr) = confident_average(scores_frame=scores, true_labels_col=true_col, confidence=con, drop=drop, verbose=verbose, t=threshold)
        if verbose:
            print('# Mean of {} confident scores: '.format(con), auc_score, sens, spec, mcc)
            print()
        ens_perf['con_scor_'+str(round(con, 2))] = [round(i, 2) for i in [(N-len(confident_scores))/N, auc_score, sens, spec, mcc]]
        
        # Majority
        confident_votes, [auc_score, sens, spec, mcc], (fpr, tpr) = confident_majority_voting(scores_frame=scores, true_labels_col=true_col, confidence=con, drop=drop, verbose=verbose, t=threshold)
        if verbose: 
            print('# Majority of {} confident votes: '.format(con), auc_score, sens, spec, mcc)
            print()
        ens_perf['con_majority_'+str(round(con, 2))] = [round(i, 2) for i in [(N-len(confident_votes))/N, auc_score, sens, spec, mcc]]

        # Add to scores dataframe 
        ens_scores = pd.concat([ens_scores, confident_scores, confident_votes], join='outer', axis=1)

    ens_scores = pd.concat([scores[true_col], ens_scores], join='outer', axis=1)

    print('Shape of performances and scores:', ens_perf.shape, ens_scores.shape)

    return ens_perf, ens_scores


###################
# Marianne plot
###################
def show_predictions(scores, target='y', threshold=0.5, path_out=False, verbose=True, figsize=(7, 200)): 
    """This function will plot which have been correctly classified. The input is 
    single dict containing labels as keys and information on each model as values 
    in the order [auc_score, ids_test, y_true, y_pred].

    all_ids: List, IDs of all samples as strings. 
    model_dict: Dict, containing model name as key and [auc_score, ids_test, y_true, y_pred] as value. 
    path_out: String, path where to save plot. 
    show: If True, show plot. 
    """
    all_ids = scores.index.tolist()
    N, M = scores.shape
    y_true = scores[target]

    # Set up figure to hold IDs vs model type
    f, id_fig = plt.subplots(figsize=figsize)
    id_fig.margins(0.01, 0.01)
    plt.ylabel('Samples (IDs)', fontsize=14)
    plt.xlabel('Models', fontsize=14)
    plt.title('Correctly classified samples', fontsize=20)
    plt.yticks(range(len(all_ids)), all_ids, fontsize=12)
    plt.grid(which='major', linestyle='dashed', linewidth=0.1)
    plt.rc('axes', axisbelow=True)
    cmap = plt.get_cmap('tab20', M)

    # Coordinates and legend
    counts = [0 for item in all_ids]
    how_many_correct = dict(zip(all_ids, counts))
    all_ids = dict(zip(all_ids, list(range(len(all_ids)))))
    xticks = []
    height = 0
    legend = []

    # Run through each model
    missing_counts = {}
    for col in scores.columns: 
        if col != target: 
            y_pred = scores[col].dropna(how='any')

            # Find correct IDs
            ids_test = []
            pred_labels = [1 if v >= threshold else 0 for v in y_pred]
            for ID, true, pred in zip(y_pred.index, y_true, pred_labels): 
                if true == round(pred): 
                    ids_test.append(ID)

                    # Count item
                    how_many_correct[ID] += 1

            # Get correct classifications
            xticks.append(col)
            y = [all_ids[i] for i in ids_test]
            x = [height]*len(y)

            # Plot correct IDs
            plot_ids = id_fig.scatter(x=x, y=y, s=15, label=col)
            
            # Plot x for missing IDs
            missing = []
            for ID in all_ids: 
                if ID not in missing_counts.keys(): 
                    missing_counts[ID] = 0
                if ID not in y_pred.index: 
                    missing.append(ID)
                    missing_counts[ID] += 1

            if len(missing) > 0: 
                y = [all_ids[i] for i in missing]
                x = [height]*len(y)
                id_fig.scatter(x=x, y=y, marker='x', color='black')
            
            legend.append(height)
            height += 1

    plt.xticks(legend, xticks, fontsize=12, rotation=90)
    plt.tight_layout()
    plt.show()

    if path_out: 
        plt.savefig(path_out, dpi=1000, transparent=True)
    return how_many_correct, missing_counts
    
