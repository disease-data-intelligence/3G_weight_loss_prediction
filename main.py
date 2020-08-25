#!/usr/bin/python3

# Import packages
seed = 42
import numpy as np
np.random.seed(seed) # Set numpy random seed
import os
import pandas as pd
import pickle
import random
from sklearn.ensemble import RandomForestClassifier
import utils as u
import ML_scripts.model as mod
import ML_scripts.forward_selection as forward
import ML_scripts.ensemble_scoring as ens
import argparse

####################
#   Make parser
####################
def get_args():
    parser = argparse.ArgumentParser(description='Module will run random forests for a set of random states and predefined settings based on the input label.')
    parser.add_argument('-i', '--input_labels', required=True, dest='input_labels', nargs='+', help='Input label(s)')
    parser.add_argument('-t', '--target_label', required=False, dest='target_label', help='Target label')
    parser.add_argument('-s', '--states', required=False, type=int, dest='random_states', help='Random state to use.')
    parser.add_argument('-cv', '--cv_splits', required=False, type=int, dest='cv_splits', help='Number of cross validation splits.')
    parser.add_argument('-for', '--forwards', required=False, type=int, dest='forwards', nargs='+', help='Enable forwards selection on label defined as int.')
    parser.add_argument('-dir', '--directory', required=False, dest='directory', help='Directory to output results.')
    parser.add_argument('-per', '--permute', required=False, action='store_true', dest='permute', help='Permute target.')
    parser.add_argument('-samples', '--samples', required=False, dest='samples', help='File with single line of samples to use.')
    parser.add_argument('-w', '--wkdir', required=False, dest='wkdir', help='Working directory where the pickled input data dictionary is located.')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', required=False)
        
    # Set defaults
    parser.set_defaults(input_labels=['diet'], target_label='target', cv_splits=5, forwards=False, samples=False, features=False, directory=os.getcwd(), wkdir=os.getcwd()+'/')
    return parser


#######################
#   Main program
#######################
def main(args): 
    # Set working and output directory
    wkdir = args.wkdir+'/'
    out_dir = args.directory

    # Set random states
    if isinstance(args.random_states, int): 
        random_states = [args.random_states]
    elif isinstance(args.random_states, list): 
        random_states = args.random_states
    else: 
        random_states = [654, 114, 25, 759, 281, 250, 228, 142, 754, 104, 692, 
        758, 913, 558, 89, 604, 432, 32, 30, 95, 223, 238, 517, 616, 27, 574, 
        203, 733, 665, 718, 429, 225, 459, 603, 284, 828, 890, 6, 777, 825, 
        163, 714, 348, 159, 220, 980, 781, 344, 94, 389]

    print('# Random states:', args.random_states, len(random_states))

    ############################
    #       LOAD DATA
    ############################
    # Load pickle saved file
    with open(wkdir+'load_data.pickle', 'rb') as outfile: 
        data = pickle.load(outfile)
    if args.verbose:
        print('# Data loaded:', data.keys())

    # Set X matrix
    if args.input_labels == ['diet']: 
        X_select = data['diet']
    elif len(args.input_labels) == 1: 
        X_select = data[args.input_labels[0]]
    else: 
        # Iterate over labels to make X_select
        X_select = data[args.input_labels[0]]
        for lab in args.input_labels[1:]: 
            rows = X_select.index.intersection(data[lab].index)
            X_select = pd.concat([X_select, data[lab]], sort=True, axis=1)
            X_select = X_select.loc[rows]

    # Select samples
    target_label = args.target_label
    if args.samples != False: 
        with open(args.samples, 'r') as sample_file: 
            samples = [float(s) for s in sample_file.readline().split(',')]
        X_select = X_select.loc[samples]
        y_select = data[target_label].loc[samples]
    else: 
        rows = X_select.index.intersection(data[target_label].index)
        y_select = data[target_label].loc[rows]
        X_select = X_select.loc[rows]

    print('# Shape of selected data:', X_select.shape)


    ##########################
    #   Forward selection
    ##########################
    # set selections for forwards selection 
    if args.forwards: 
        if isinstance(args.forwards, list): 
            select = []
            for i in args.forwards:
                select += data[args.input_labels[i]].columns.tolist()
        else: 
            select = data[args.input_labels[args.forwards]].columns.tolist()

    printFeatures = list()

    # Run through each random state and select model 
    n_state = 1
    prints = list()
    for state in random_states: 
        print('State {0}/{1}'.format(n_state, len(random_states)))
        if args.verbose: 
            print('\n# Random state:', state)

        # If shuffled target is enabled
        if args.permute: 
            y_shuffled = u.shuffle_target(y_target=y_select.values, seed=state, verbose=False)
            y_shuffled = pd.DataFrame(data=y_shuffled, index=y_select.index)

        # Set forest parameters
        params = {'n_estimators': 50, 'random_state': seed, 'min_samples_split': 0.025, 'min_impurity_decrease': 0.01, 'max_features': None}
        rf = RandomForestClassifier(**params)

        # Find best forward selection
        if args.forwards: 
            if args.permute: 
                if args.input_labels[1] == '16s_var250': 
                    max_f = max(7, int(y_select.shape[0]**(1/2)/2))
                else: 
                    max_f = X_select.shape[1]

                best_model = forward.forward_selection2(X_data=X_select, y_data=y_shuffled, model=rf, selects=select, seed=state, cv=5, max_features=max_f)

            else: 
                if args.input_labels[1] == '16s_var250': 
                    max_f = max(7, int(y_select.shape[0]**(1/2)/2))
                else: 
                    max_f = X_select.shape[1]

                best_model = forward.forward_selection2(X_data=X_select, y_data=y_select, model=rf, selects=select, seed=state, cv=args.cv_splits, max_features=max_f)

            # Look into selection
            (X_crop, y), metrics, (fpr,tpr), ids_test, preds_labels, thresholds, models = best_model

            # Save objects
            with open(out_dir+'forward.{0}.rs{1}.model'.format('_'.join(args.input_labels), state), 'wb') as outfile: 
                pickle.dump(best_model, outfile)

            # Save features list 
            printFeatures.append([str(state)]+[c for c in X_crop.columns])

        # Run forest on selected data (no data driven selections)
        else: 
            if args.permute: 
                metrics, (fpr,tpr), ids_test, preds_labels, thresholds, models = mod.sklearn_model(X_data=X_select, y_data=y_shuffled, model=rf, seed=state, cv_splits=args.cv_splits, stratified=True, verbose=args.verbose)
            else: 
                metrics, (fpr,tpr), ids_test, preds_labels, thresholds, models = mod.sklearn_model(X_data=X_select, y_data=y_select, model=rf, seed=state, cv_splits=args.cv_splits, stratified=True, verbose=args.verbose)

            # Save ordinary random forests
            saved_model = (X_select, y_select), metrics, (fpr,tpr), ids_test, preds_labels, thresholds, models
            with open(out_dir+'rf.{0}.rs{1}.model'.format('_'.join(args.input_labels), state), 'wb') as outfile: 
                pickle.dump(saved_model, outfile)

            # Get all non-zero importance features 
            theseCols = X_select.columns
            noZeroImps = set([theseCols[i] for i in range(len(theseCols)) for j in range(args.cv_splits) if models[j][i] > 0])
            printFeatures.append([str(state)]+list(noZeroImps))

        n_state += 1

        # Save the things to print
        prints.append(metrics[1].auc.iloc[0])
    
    # Print metrics of this run
    print(prints, np.mean(prints), np.std(prints))

    # Write features file
    labels = '_'.join(args.input_labels)
    with open('{0}forward.{1}.features'.format(out_dir, labels), 'w') as outfile: 
        for line in printFeatures: 
            outfile.write(','.join(line)+'\n')

    print('# Done!')

#######################
#   Run the program 
#######################
if __name__ == "__main__":
    parser = get_args()
    try:
        args = parser.parse_args()
    except IOError as msg: 
        parser.error(str(msg))
    main(args)      # Main program 
