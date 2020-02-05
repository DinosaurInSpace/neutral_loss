import pandas as pd
import numpy as np
import time

from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.svm import SVC

import xgboost as xgb
#from thundersvm import SVC

"""
nl_03_filter_model_score_reg.py:

Purpose:
This script contains the functions to a) filter, b) split, c) model, and d) score
neutral loss data.  Processing primarily takes place in an interactive Jupyter notebook
to allow for interactive processing and review.  

This script is edited to handle the regression
case, where observations of true/false for neutral losses are first agggregated together on HMDB.

Steps include:
1) Filtering.
2) Aggregation on HMDB ID
3) Splitting with or without groups segregation to  particular set.
4) Direct models: x vs y.
5) Machine learning models: Random Forest, XGboost, SVM.
6) Deep learning models via DeepChem

Previous script ion series is:
"nl_02_join.py"

Next script in series is:
"tbd"

Example command line:

n/a

"""

def sanitize(df):
    # Drops rows with nans and inconsistent types
    df = df.dropna(axis=0)
    return df


def print_report(df, row):
    # Plot histogram for each case tested.
    # Printing reports to file? make sure to print row as header...
    print('Printing reports!')
    pass


def aggregate_on_hmdb(df, row):
    # Aaggregate the large table of HMDB T/F observations to counts with
    # a new y column, range 0-1.  This column will be:
    # [# loss observations f/HMDB] / [# total observations f/HMDB].
    # This converts the task from classification to regression.



    '''
    Future:

    1. Replace confusion, accuracy w/RMSE?  Plot output?
    2. How to handle case of different loss types?  Row will carry parameter up.
    --> Specifically should "n_loss only" be dropped when considering "n_loss_wparent"
    Filtering step.  Can do on FDR is not nan for parent?

    Where to put this?  Split for both cases.  Likely need to carry intensities forward...
    3. Can you plot the intensities (for all cases when you have both intact and intact-H2O detected, record their intensities and plot them in a scatter plot: intensity of intact vs intensity of intact-H2O)? to test that intact-H2O goes beyond LOD?
    4. Normalize: (int of intact-H2O / int of intact)
    5. Maybe makes sense to plot a histogram of a ratio first
    '''

    # Temp column to use to for counting observations (rows in classification df)
    df['n_obs'] = 1

    # Need to generate dict due to n columns input because of variable X input types.
    colname_aggmethod_dict = {}
    cols = list(df)

    print('len(cols)')
    print(len(cols))

    # 'first' is default method since other columns should be the same per every row with
    # matching HMDB
    for c in cols:
        if c is 'hmdb_ids':
            pass
        elif c is 'y' or c is 'w' or c is 'n_obs':
            colname_aggmethod_dict[c] = 'sum'
        else:
            colname_aggmethod_dict[c] = 'first'

    print('colname_aggmethod_dict')
    print(colname_aggmethod_dict)

    # This aggregation on hmdb_ids changes the task from classification to regression.
    df = df.groupby('hmdb_ids').agg(colname_aggmethod_dict)
    df['reg_y'] = df['y'] / df['n_obs']

    # divide weights by n obs to get average weight
    df['avg_w'] = df['w'] / df['n_obs']

    # drop columns not expected by downstream steps re: iloc indexing for n columns in X input
    df = df.drop(columns=['y', 'w', 'n_obs']).copy(deep=True)

    # rename new y/w

    print('Check cols aggregate out')
    print(list(df))

    # print_report(df, row)

    return df


def split(df, single_fold):
    # True: will segregate formulas to one group or another.  Prevents memorization and
    # enhances ability to predict new molecues.
    # False: is memorization a bad thing over 2900+ datasets?  Extinction plot for novel ID's
    # 50:25:25 / Train:Test:Val. Rationale: Chemical space is large.
    # Working!

    if single_fold is True:
        splitter = GroupKFold(n_splits=2)

        X = df
        y = df.y
        groups = df.formula

        for train_ix, tv_ix in splitter.split(X, y, groups):
            X_train, X_tv = X.iloc[train_ix, :].copy(deep=True), X.iloc[tv_ix, :].copy(deep=True)

        X = X_tv
        y = X_tv.y
        groups = X_tv.formula

        for test_ix, val_ix in splitter.split(X, y, groups):
            X_test, X_val = X.iloc[test_ix, :].copy(deep=True), X.iloc[val_ix, :].copy(deep=True)

    else:
        splitter = KFold(n_splits=2)

        X = df
        y = df.y

        for train_ix, tv_ix in splitter.split(X, y):
            X_train, X_tv = X.iloc[train_ix, :].copy(deep=True), X.iloc[tv_ix, :].copy(deep=True)

        X = X_tv
        y = X_tv.y

        for test_ix, val_ix in splitter.split(X, y):
            X_test, X_val = X.iloc[test_ix, :].copy(deep=True), X.iloc[val_ix, :].copy(deep=True)

    X_train = X_train.drop(columns=['formula', 'hmdb_ids', 'ds_id']).copy(deep=True)
    X_test = X_test.drop(columns=['formula', 'hmdb_ids', 'ds_id']).copy(deep=True)
    X_val = X_val.drop(columns=['formula', 'hmdb_ids', 'ds_id']).copy(deep=True)

    return [X_train, X_test, X_val]


def confuse(obs_y, theo_y):
    # Copy from confuse ipynb
    con = confusion_matrix(list(obs_y), list(theo_y))
    if con.shape == (1, 1):
        print('error!')

    elif con.shape == (2, 2):
        tn, fp, fn, tp = con.ravel()
        sens = tpr = tp / (tp + fn)
        spec = tnr = tn / (tn + fp)
        f1 = (2 * tp) / (2 * tp + fp + fn)
        acc = (tp + tn) / (tp + tn + fp + fn)
        # prec = tp / (tp + fp)

        return [acc, {'sens': sens, 'spec': spec, 'f1': f1,
                      'test_n': tn + fp + fn + tp, 'test_true': tp + fp,
                      'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
                      }]

    else:
        print('error!')


def direct_model(model, dfs):
    print('direct_model')
    train_df = dfs[0]
    test_df = dfs[1]

    acc_train = confuse(np.array(train_df.y), np.array(train_df.X))[0]
    result = confuse(np.array(test_df.y), np.array(test_df.X))
    acc_test = result[0]
    result_dict = result[1]
    result_dict['acc_train'] = acc_train
    result_dict['acc_test'] = acc_test

    return result_dict


def ml_model(model, submodel, dfs):
    print(model)
    # Deal with case of bits X = column with 1024 np.array
    # Deal with case Mordred, X = all columns but ['formula', 'hmdb_ids', 'ds_id', row.y, row.w]
    # df.iloc[:,2:]
    # .iloc[:, 2:]
    # ['y', 'w', 'X'], maybe many X's in case of Mordred...
    print('mL_dfs')

    train_df = dfs[0]
    train_y = np.array(train_df.y)
    print('train_y: ' + str(train_y.shape))

    train_w = np.array(train_df.w)
    print('train_w: ' + str(train_w.shape))

    train_X = train_df.iloc[:,2:].copy(deep=True).to_numpy()
    print('train_X: ' + str(train_X.shape))

    test_df = dfs[0]
    test_y = np.array(test_df.y)
    test_w = np.array(test_df.w)
    test_X = test_df.iloc[:,2:].copy(deep=True).to_numpy()

    if submodel is 'random_forest':
        # https://stackoverflow.com/questions/30805192/scikit-learn-random-forest-class-weight-and-sample-weight-parameters
        # https://scikit-learn.org/stable/auto_examples/svm/plot_weighted_samples.html
        clf = rfc(max_features=32, n_estimators=100, random_state=0,
                  class_weight="balanced", n_jobs=-1)  # "balanced"  {0:1,1:5}

    elif submodel is 'XGBoost':
        # https://xgboost.readthedocs.io/en/latest/get_started.html
        # https://xgboost.readthedocs.io/en/latest/parameter.html
        # https://www.datacamp.com/community/tutorials/xgboost-in-python
        # Set parameter max_delta_step to a finite number (say 1) to help convergence
        clf = xgb.XGBClassifier(max_depth=6, objective='binary:hinge', eta=0.2, min_child_weight=1,
                                gamma=0, eval_metric='error')

    elif submodel is 'ThunderSVM':
        # Copied from vanilla scikit SVM parameters
        clf = SVC(kernel='linear', C=10, gamma=1)

    else:
        # No model or wrong model
        return(1)

    model = clf.fit(train_X, train_y, sample_weight=train_w)
    acc_train = model.score(train_X, train_y)
    acc_test = model.score(test_X, test_y)
    predict_y = model.predict(test_X)
    result_dict = confuse(test_y, predict_y)[1]
    result_dict['acc_train'] = acc_train
    result_dict['acc_test'] = acc_test

    '''
    scores = cross_val_score(model, theo_x,
                                 obs_y,
                                 cats,
                                 cv=GroupKFold(n_splits=5)
                                 )
    cross_a = float(scores.mean())
    cross_s = float(scores.std())
    '''

    return result_dict


def dc_model(model, submodel, dfs):
    print(model)
    return None


def is_good_1024(x):
    if type(x) == np.ndarray and x.shape == (1024,):
        return True
    else:
        return False


def filter_split_model_score(filter_param_df, join_df_path, mord_df, bits_df, single_fold_group):
    # Assumes only one target (e.g. water) at a time!
    start_time = time.time()
    dict_dict = {}

    join_df = pd.read_pickle(join_df_path)

    # Temp fix, small number of Trues got in?
    join_df['falses'] = join_df['falses'].astype(bool).replace(True, False)

    for index, row in filter_param_df.iterrows():
        print('start ' + str(index))
        target_fdr = 'fdr_' + row.target
        target_coloc = 'colocalization_' + row.target
        join_df['best_fdr'] = join_df[['fdr', target_fdr]].min(axis=1)

        filtered_df = join_df[(join_df.polarity == row.polarity) &
                             (join_df.best_fdr <= row.fdr) &
                             ((join_df[target_coloc] >= row.coloc) | (join_df[target_coloc] == 0))
                             ]

        # Different cases depending on input X's.
        if row.X is 'bits':
            xyw_df = filtered_df[['formula', 'hmdb_ids', 'ds_id', row.y, row.w]].copy(deep=True)

            # Change column of np.array 1024 bits to discrete columns
            arr_df = bits_df.bits
            arr_2d = np.stack(arr_df.to_numpy())
            df_1024 = pd.DataFrame(arr_2d)
            df_1024.insert(0, 'hmdb_ids', list(bits_df.hmdb_ids))

            # Inner join, hmdb_ids in both only!
            xyw_df = pd.merge(xyw_df, df_1024, how='inner', on='hmdb_ids')

        elif row.X is 'mord_norm':
            #mord_df = mord_df.drop(columns='Molecule')
            xyw_df = filtered_df[['formula', 'hmdb_ids', 'ds_id', row.y, row.w]].copy(deep=True)

            # Inner join, hmdb_ids in both only
            xyw_df = pd.merge(xyw_df, mord_df, how='left', on='hmdb_ids')

            print(list(xyw_df))

        else:
            xyw_df = filtered_df[['formula', 'hmdb_ids', 'ds_id', row.y, row.w, row.X]].copy(deep=True)

        if row.X != row.y:
            xyw_df = xyw_df.rename(columns={row.X: 'X', row.y: 'y', row.w: 'w'}, inplace=False)

        else:
            # Control case with perfect prediction/memorization!
            xyw_df['temp_X'] = filtered_df[row.y]
            xyw_df['temp_y'] = filtered_df[row.y]
            xyw_df = xyw_df[['formula', 'hmdb_ids', 'ds_id', 'temp_y', row.w, 'temp_X']].copy(deep=True)
            xyw_df = xyw_df.rename(columns={'temp_X': 'X', 'temp_y': 'y', row.w: 'w'}, inplace=False)

        xyw_df = sanitize(xyw_df)

        if len(xyw_df.index) > 0:

            print('aggregating ' +str(index))

            xyw_df = aggregate_on_hmdb(xyw_df, row)

            print('splitting ' + str(index))
            split_dfs = split(xyw_df, single_fold_group)

            print('modeling ' + str(index))
            if row.model == 'direct':
                result = direct_model(row.model, split_dfs)
                all_line = {**dict(row), **result}
                dict_dict[index] = all_line

            if row.model == 'ml':
                result = ml_model(row.model, row.submodel, split_dfs)
                all_line = {**dict(row), **result}
                dict_dict[index] = all_line

            if row.model == 'dc':
                result = dc_model(row.model, row.submodel, split_dfs)
                all_line = {**dict(row), **result}
                dict_dict[index] = all_line

        elif len(xyw_df.index) == 0:
            print('Caution: empty filter set!')
            dict_dict[index] = row

    results = pd.DataFrame.from_dict(dict_dict, orient='index')

    elapsed_time = time.time() - start_time
    print('Elapsed time:\n')
    print('\nExecuted without error\n')
    print(elapsed_time)
    return results

####Body####

# None