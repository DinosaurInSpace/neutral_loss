#!/usr/bin/env python

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


import pandas as pd
import numpy as np
import time
from datetime import datetime
from matplotlib import pyplot as plt
import re
from math import sqrt
import textwrap
import glob
import os
import argparse

from sklearn.model_selection import GroupKFold, KFold, GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from scipy.stats import gaussian_kde

import xgboost as xgb
#from thundersvm import SVC


__author__ = "Christopher M Baxter Rath"
__copyright__ = "Copyright 2020"
__credits__ = ["Christopher M Baxter Rath"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Christopher M Baxter Rath"
__email__ = "chrisrath@gmail.com"
__status__ = "Development"


def print_histogram(data, row, y_title, x_title, path):
    # Plot histogram for each case tested.
    print('Printing histogram!')

    title = str(row.to_dict())
    fig, ax = plt.subplots()
    ax.hist(data, 100)

    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_title("\n".join(textwrap.wrap(title, 60)))

    filename = str(datetime.now())
    filename = re.sub('[^0-9a-zA-Z]+', '_', filename)
    filename = path + filename
    print(filename)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return


def aggregate_on_hmdb(df, row):
    # Aaggregate the large table of HMDB T/F observations to counts with
    # a new y column, range 0-1.  This column will be:
    # [# loss observations f/HMDB] / [# total observations f/HMDB].
    # This converts the task from classification to regression.
    # Temp column to use to for counting observations (rows in classification df)
    df['n_obs'] = 1

    df['hmdb_ids'] = df.hmdb_ids + df.adduct
    df = df.drop(columns='adduct').copy(deep=True)

    # Need to generate dict due to n columns input because of variable X input types.
    column_agg_method_dict = {}
    cols = list(df)

    for c in cols:
        if c == 'hmdb_ids':
            pass
        elif c == 'intensity_avg' or c == 'intensity_nl' or c == 'X':
            column_agg_method_dict[c] = 'mean'
        elif c == 'y' or c == 'w' or c == 'n_obs':
            column_agg_method_dict[c] = 'sum'
        else:
            column_agg_method_dict[c] = 'first'

    # This aggregation on hmdb_ids changes the task from classification to regression.
    df = df.groupby('hmdb_ids').agg(column_agg_method_dict).reset_index()

    print('checking aggregation: ')

    print(df[['hmdb_ids', 'y', 'n_obs']].sample(5).round(3))
    print(df.shape)

    df['y'] = df['y'] / df['n_obs']

    # divide weights by n obs to get average weight
    df['w'] = df['w'] / df['n_obs']

    # filter on minimum n_obs.  Sampling issue with predicting probability for 1 or 2...
    df = df[df.n_obs >= row.min_n_obs].copy(deep=True)

    print_histogram(df.n_obs, row, 'Counts by observation', 'Counts n_obs', 'n_obs_plots/')
    print_histogram(df.y, row, 'Counts by observation', 'Probability neutral loss', 'hist_plots/')

    return df


def model_scatter_plt(test_output, predict_output, row):
    # Plot scatter plot for model output
    print('Printing regression actual versus predicted output!')

    title = str(row.to_dict())
    x = test_output
    y = predict_output

    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=10, edgecolor='')
    ax.set_xlabel('Observed test probability water loss')
    ax.set_ylabel('Predicted test probability water loss')
    ax.set_title("\n".join(textwrap.wrap(title, 60)))
    ax.set_xlim(0,1)
    ax.set_ylim(0, 1)

    filename = str(datetime.now())
    filename = re.sub('[^0-9a-zA-Z]+', '_', filename)
    filename = 'model_plots/' + filename
    print(filename)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return(filename)


def filter(row, j):
    # target_fdr
    tf = 'fdr_' + row.target

    # target colocalizatiaon
    tc = 'colocalization_' + row.target
    target_int = 'intensity_avg_' + row.target

    # To support filtering on FDR, replace np.nan with -1
    j[['fdr', tf, ]] = j[['fdr', tf]].fillna(value=-1)
    j[['intensity_avg', target_int]] = j[['intensity_avg', target_int]].fillna(value=0)

    parent = 'n_loss_wparent_' + row.target
    only = 'n_loss_only_' + row.target
    all = 'n_loss_all_' + row.target

    if row.y == parent:
        # Filter polarity, fdr, omit hits for n_loss_only, and
        # rows with NL filtered on coloc while rows w/out NL not filtered
        f_df = j[(j.polarity == row.polarity) &
                 (j.fdr <= row.fdr) &
                 (j.fdr != -1) &
                 (((j[tc] >= row.coloc) & (j[tf] != -1)) |
                  ((j[tc] == 0) & (j[tf] == -1)))].copy(deep=True)

    elif row.y == only:
        # Filter polarity, fdr
        f_df = j[(j.polarity == row.polarity) &
                 ((j.fdr <= row.fdr) | (j.fdr == -1))].copy(deep=True)

    elif row.y == all:
        # Filter polarity, fdr
        f_df = j[(j.polarity == row.polarity) &
                 ((j.fdr <= row.fdr) | (j.fdr == -1))].copy(deep=True)
        f_df[all] = f_df[parent].astype(int) + f_df[only].astype(int)
        check_dict = {'all': f_df[all].astype(int).sum(),
                      'parent': f_df[parent].astype(int).sum(),
                      'only': f_df[only].astype(int).sum()}
        print('All counts: ', str(check_dict))


    else:
        print('Bad target choice!')
        f_df = pd.DataFrame()

    return [f_df, target_int]


def clean_columns(filtered, row):
    # Clean columns used for filtering and any additional columns
    f_df = filtered[0]
    target_int = filtered[1]

    if row.model is 'ml':
        xyw_df = f_df[['formula', 'adduct', 'hmdb_ids', 'intensity_avg',
                       target_int, row.y, row.w]].copy(deep=True)
        xyw_df = xyw_df.rename(columns={target_int: 'intensity_nl',
                                        row.y: 'y', row.w: 'w'}, inplace=False)
    elif row.model is 'dc':
        xyw_df = f_df[['formula', 'adduct', 'hmdb_ids', 'intensity_avg',
                       target_int, row.y, row.w,
                       'Molecule']].copy(deep=True)
        xyw_df = xyw_df.rename(columns={target_int: 'intensity_nl',
                                        row.y: 'y', row.w: 'w',
                                        row.Molecule: 'X'}, inplace=False)
    elif row.model is 'direct':
        xyw_df = f_df[['formula', 'adduct', 'hmdb_ids', 'intensity_avg',
                       target_int, row.y, row.w,
                       row.X]].copy(deep=True)
        if row.X != row.y:
            xyw_df = xyw_df.rename(columns={target_int: 'intensity_nl',
                                            row.y: 'y', row.w: 'w',
                                            row.X: 'X'}, inplace=False)
        else:
            # Control case with perfect prediction/memorization!
            xyw_df['temp_X'] = f_df[row.y]
            xyw_df['temp_y'] = f_df[row.y]
            xyw_df = xyw_df[['formula', 'hmdb_ids', 'intensity_avg',
                             target_int, 'temp_y', row.w,
                             'temp_X']].copy(deep=True)
            xyw_df = xyw_df.rename(columns={target_int: 'intensity_nl',
                                            'temp_y': 'y', row.w: 'w',
                                            'temp_X': 'X'}, inplace=False)
    else:
        print('Bad model choice!')
        xyw_df = pd.DataFrame()

    xyw_df = xyw_df.dropna(axis=0)

    return xyw_df


def hmdb_ion_split(x, row):
    if row.polarity == 1:
        return x.split('+')[0]

    elif row.polarity == -1:
        return x.split('-')[0]

def join_x_data(agg_df, row, mord_df, bits_df):
    # Different cases depending on input X's.
    # Join after aggregate, as >1K rows

    print(list(agg_df))

    agg_df['temp'] = agg_df['hmdb_ids']
    agg_df['hmdb_ids'] = agg_df['hmdb_ids'].apply(lambda x: hmdb_ion_split(x, row))

    if row.X is 'bits':
        # Change column of np.array 1024 bits to discrete columns
        arr_df = bits_df.bits
        arr_2d = np.stack(arr_df.to_numpy())
        df_1024 = pd.DataFrame(arr_2d)
        df_1024.insert(0, 'hmdb_ids', list(bits_df.hmdb_ids))

        # Inner join, hmdb_ids in both only!
        agg_df = pd.merge(agg_df, df_1024, how='inner', on='hmdb_ids')

    elif row.X is 'mord_norm':
        mord_df = mord_df.replace(np.inf, np.nan).copy(deep=True)
        mord_df = mord_df.fillna(0).copy(deep=True)

        # Inner join, hmdb_ids in both only
        agg_df = pd.merge(agg_df, mord_df, how='left', on='hmdb_ids')

    else:
        # X's should already be incorporated!
        pass

    agg_df['hmdb_ids'] = agg_df['temp']
    agg_df = agg_df.drop(columns='temp')

    return agg_df


def intensity_plotter(df, row):
    '''
     Can you plot the intensities (for all cases when you have both intact and intact-H2O detected,
     record their intensities and plot them in a scatter plot: intensity of intact vs intensity of
     intact-H2O)? to test that intact-H2O goes beyond LOD.

     Maybe makes sense to plot a histogram of a ratio first.
     Deal with on scale intensities: int w_loss / int + int w_loss
    '''

    i_par = 'intensity_avg'
    i_nl = 'intensity_nl'

    if row.y == ('n_loss_wparent_' + row.target):
        df = df[df[i_nl] != 0].copy(deep=True)
        df['to_plot'] = df[i_nl].astype(int) / (df[i_nl].astype(int) + df[i_par].astype(int))

        # Plot histogram for each case tested.
        print('Printing intensities parent and NL!')

        title = str(row.to_dict())
        data = df['to_plot']
        fig, ax = plt.subplots()
        ax.hist(data, 50)

        ax.set_xlabel('neutral_loss_int / (neutral_loss_int + parent_int)')
        ax.set_ylabel('Counts by observation')
        ax.set_title("\n".join(textwrap.wrap(title, 60)))

        filename = str(datetime.now())
        filename = re.sub('[^0-9a-zA-Z]+', '_', filename)
        filename = 'int_plots/' + filename
        print(filename)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        return filename

    else:
        print('No intensity plot printed for n_loss_only or n_loss_all')
        return


def test_split(train_df, test_df, row):
    # Test if split correctly segregated on formula, hmdb's, y's, and X's
    known_columns = ['formula', 'hmdb_ids', 'intensity_avg', 'intensity_nl', 'y', 'w', 'n_obs']
    compare_columns = ['formula', 'hmdb_ids', 'y']

    test_split_dict = {}
    for c in compare_columns:
        inter = list(set(train_df[c]) & set(test_df[c]))
        test_split_dict[c] = len(inter)

    X_train_df = train_df.drop(columns=known_columns).copy(deep=True)
    X_test_df = test_df.drop(columns=known_columns).copy(deep=True)

    print('X_df_shapes:')
    print(X_train_df.shape)
    print(X_test_df.shape)

    if row.X == 'mord_norm':
        X_train_sum = list(X_train_df.sum(axis =1))
        X_test_sum = list(X_test_df.sum(axis=1))
        test_split_dict['X'] = len(list(set(X_train_sum) & set(X_test_sum)))

    else:
        X_overlap_df = pd.merge(X_train_df, X_test_df, how='inner', on=list(X_train_df))
        test_split_dict['X'] = X_overlap_df.shape[0]

    print('Testing split, n overlap train/test: ')
    print(test_split_dict)

    return


def split(df, single_fold, row):
    # True: will segregate formulas to one group or another.  Prevents memorization and
    # enhances ability to predict new molecues.
    # False: is memorization a bad thing over 2900+ datasets?
    # 50:25:25 / Train:Test:Val. Rationale: Chemical space is large.

    if single_fold is True:
        # splitter = GroupKFold(n_splits=2)
        splitter = GroupShuffleSplit(n_splits=2, test_size=0.5, train_size=0.5)

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

    test_split(X_train, X_test, row)

    X_train = X_train.drop(columns=['formula', 'hmdb_ids']).copy(deep=True)
    X_test = X_test.drop(columns=['formula', 'hmdb_ids']).copy(deep=True)
    X_val = X_val.drop(columns=['formula', 'hmdb_ids']).copy(deep=True)

    return [X_train, X_test, X_val]


def direct_model(dfs, row):
    print('direct_model')
    train_df = dfs[0]
    test_df = dfs[1]

    train_predicted = train_df.X
    train_observed = train_df.y

    test_predicted = test_df.X
    test_observed = test_df.y

    result_dict = {}
    result_dict['n_test'] = test_observed.shape[0]
    result_dict['plt_path'] = model_scatter_plt(test_observed, test_predicted, row)

    # MSE can take sample weights!
    result_dict['rmse_train'] = sqrt(mse(train_observed, train_predicted))
    result_dict['rmse_test'] = sqrt(mse(test_observed, test_predicted))

    return result_dict


def ml_submodel(row):
    if row.submodel is 'random_forest':
        # https://stackoverflow.com/questions/30805192/scikit-learn-random-forest-class-weight-and-sample-weight-parameters
        # https://scikit-learn.org/stable/auto_examples/svm/plot_weighted_samples.html
        reg = rfr(criterion='mse', max_features='auto', n_estimators=100, random_state=0, n_jobs=-1)
        return reg

    elif row.submodel is 'XGBoost':
        # https://xgboost.readthedocs.io/en/latest/get_started.html
        # https://xgboost.readthedocs.io/en/latest/parameter.html
        # https://www.datacamp.com/community/tutorials/xgboost-in-python
        # Set parameter max_delta_step to a finite number (say 1) to help convergence
        reg = xgb.XGBRegressor(max_depth=6, objective='reg:logistic', eta=0.4, min_child_weight=1,
                                gamma=0, eval_metric='rmse', early_stopping_rounds=10)
        return reg
    # Default amx depth =6

    elif row.submodel is 'SVM':
        # Copied from vanilla scikit SVM parameters
        # Score will give you the R2...
        reg = SVR(kernel='linear', C=10, gamma=1)

        return reg

    else:
        print('No model or wrong model')
        exit(1)


def histogram_10_bin_norm(ys):
    density, bin_edges = np.histogram(ys, density=True)
    weights = 1 / density
    y_bins = np.digitize(ys, bin_edges[1:-1])
    ws = weights[y_bins]
    return ws


def weights(train_X, train_y, train_df, test_X, test_y, test_df, reg, row):
    # Could also weight on intensity and FDR!
    print('Weighting: ', row.w_norm)

    target_columns = ['intensity_avg', 'intensity_nl', 'y', 'w', 'n_obs']
    train_w_df = train_df[target_columns].copy(deep=True).rename(columns={'w': 'w_isobar'})
    test_w_df = test_df[target_columns].copy(deep=True).rename(columns={'w': 'w_isobar'})

    # Apply to dfs:
    train_w_df['w_10_y_bins'] = histogram_10_bin_norm(train_y)
    test_w_df['w_10_y_bins'] = histogram_10_bin_norm(test_y)

    if row.w_norm == False:
        train_w = None
        test_w = None
        model = reg.fit(train_X, train_y)

    else:
        # Samples are weighted proportional to HMDBs obs in each of 10 bins.
        if row.w_norm == '10_y_bins':
            train_w = train_w_df.w_10_y_bins.to_numpy()
            test_w = test_w_df.w_10_y_bins.to_numpy()
        # Samples are weight proportional to number of observed rows.
        elif row.w_norm == 'n_obs':
            train_w = train_w_df.n_obs.to_numpy()
            test_w = test_w_df.n_obs.to_numpy()
        # Weight based on number of isobars identified.
        elif row.w_norm == 'isobar':
            train_w = train_w_df.w_isobar.to_numpy()
            test_w = test_w_df.w_isobar.to_numpy()
        # Binary combos
        elif row.w_norm == '10_y_bins_W_n_obs':
            train_w = (train_w_df.w_10_y_bins * train_w_df.n_obs).to_numpy()
            test_w = (test_w_df.w_10_y_bins * test_w_df.n_obs).to_numpy()
        # Binary combos
        elif row.w_norm == '10_y_bins_W_isobar':
            train_w = (train_w_df.w_10_y_bins * train_w_df.w_isobar).to_numpy()
            test_w = (test_w_df.w_10_y_bins * test_w_df.w_isobar).to_numpy()
        # Binary combos
        elif row.w_norm == 'n_obs_W_isobar':
            train_w = (train_w_df.n_obs * train_w_df.w_isobar).to_numpy()
            test_w = (test_w_df.n_obs * test_w_df.w_isobar).to_numpy()
        # Triple combo
        elif row.w_norm == '10_y_bins_W_n_obs_W_isobar':
            train_w = (train_w_df.w_10_y_bins * train_w_df.n_obs * train_w_df.w_isobar).to_numpy()
            test_w = (test_w_df.w_10_y_bins * test_w_df.n_obs * test_w_df.w_isobar).to_numpy()
        model = reg.fit(train_X, train_y, sample_weight=train_w)

    return [model, train_w, test_w]


def ml_model(dfs, row):
    # ['y', 'w', 'int', 'int_nl', 'n_obs', 'X'], maybe many X's in case of Bits/Mordred...
    not_X = ['y', 'w', 'intensity_avg', 'intensity_nl', 'n_obs']
    train_df = dfs[0]
    train_y = np.array(train_df.y)
    train_w = np.array(train_df.w)
    train_X_df = train_df.drop(columns=not_X).copy(deep=True)
    train_X = train_X_df.fillna(0).copy(deep=True).to_numpy()

    test_df = dfs[1]
    test_y = np.array(test_df.y)
    test_w = np.array(test_df.w)
    test_X_df = test_df.drop(columns=not_X).copy(deep=True)
    test_X = test_X_df.fillna(0).copy(deep=True).to_numpy()

    print(row.model, ': ', row.submodel)
    print('train_y: ' + str(train_y.shape))
    print('train_w: ' + str(train_w.shape))
    print('train_X: ' + str(train_X.shape))

    result_dict = {}

    # Select regression and parameters
    reg = ml_submodel(row)

    # Apply different weighting scheme
    model_weights = weights(train_X, train_y, train_df, test_X, test_y, test_df, reg, row)
    model = model_weights[0]
    train_w = model_weights[1]
    test_w = model_weights[2]

    predict_test_y = model.predict(test_X)
    predict_train_y = model.predict(train_X)
    result_dict['rmse_train'] = sqrt(mse(train_y, predict_train_y))
    result_dict['rmse_test'] = sqrt(mse(test_y, predict_test_y))
    result_dict['r2_test'] = r2_score(test_y, predict_test_y)

    if row.w_norm == False:
        result_dict['rmse_train_w'] = None
        result_dict['rmse_test_w'] = None
        result_dict['r2_test_w'] = None

    else:
        result_dict['rmse_train_w'] = sqrt(mse(train_y, predict_train_y, sample_weight=train_w))
        result_dict['rmse_test_w'] = sqrt(mse(test_y, predict_test_y, sample_weight=test_w))
        result_dict['r2_test_w'] = r2_score(test_y, predict_test_y,sample_weight=test_w)

    result_dict['n_test'] = test_y.shape[0]
    result_dict['plt_path'] = model_scatter_plt(test_y, predict_test_y, row)

    '''
    Need to bring grouping forward to implement!
    scores = cross_val_score(model, theo_x, obs_y, cats, cv=GroupKFold(n_splits=5))
    cross_a = float(scores.mean())
    cross_s = float(scores.std())
    '''

    return result_dict


def dc_model(dfs, row):
    print(row.model)
    return None


def model_selector(split_dfs, row, dict_dict, index):
    if row.model is 'direct':
        result = direct_model(split_dfs, row)
        all_line = {**dict(row), **result}
        dict_dict[index] = all_line

    elif row.model is 'ml':
        result = ml_model(split_dfs, row)
        all_line = {**dict(row), **result}
        dict_dict[index] = all_line

    elif row.model is 'dc':
        result = dc_model(split_dfs, row)
        all_line = {**dict(row), **result}
        dict_dict[index] = all_line

    else:
        print('Caution: Model not known!')
        dict_dict[index] = row

    return dict_dict


def filter_split_model_score(filter_param_df, join_df_path, mord_df, bits_df, single_fold_group, datasets):
    # Assumes only one target (e.g. water) at a time!
    dict_dict = {}
    time_list = []

    for index, row in filter_param_df.iterrows():
        print('start ' + str(index))
        start_time = time.time()
        # Reopen each loop to ensure all data are entered.
        join_df = pd.read_pickle(join_df_path)
        join_df = join_df[join_df.ds_id.isin(datasets)].copy(deep=True)

        # Temp fix, small number of Trues got in?
        join_df['falses'] = join_df['falses'].astype(bool).replace(True, False)

        if row.one_id_only == True:
            join_df = join_df[join_df.weight == 1].copy(deep=True)

        print('filtering ' + str(index))
        filtered = filter(row, join_df)
        xyw_df = clean_columns(filtered, row)

        print('aggregating ' + str(index))
        agg_df = aggregate_on_hmdb(xyw_df, row)

        print('joining X ' + str(index))
        agg_df = join_x_data(agg_df, row, mord_df, bits_df)

        # Generate intensity plots here!
        intensity_plotter(agg_df, row)

        if len(agg_df.index) > 0:
            print('splitting ' + str(index))
            split_dfs = split(agg_df, single_fold_group, row)
            print('modeling ' + str(index))
            model_selector(split_dfs, row, dict_dict, index)

        elif len(agg_df.index) == 0:
            print('Caution: empty filter set!')
            dict_dict[index] = row

        elapsed_time = time.time() - start_time
        time_list.append(int(elapsed_time))

    results = pd.DataFrame.from_dict(dict_dict, orient='index')
    results['time'] = time_list
    print('Elapsed time:\n')
    print('\nExecuted without error\n')
    print(elapsed_time)
    return results


def main():
    ### Main ###
    # filter_param_df, join_df_path, mord_df, bits_df, single_fold_group
    # Typically run from ipynb via import.  Not yet tested.
    # Need to take in pickles of everything then read prior to passing (except join)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--filter", default='filter_param_df', type=str, help="Filter df")
    parser.add_argument("--join", default='join_df', type=str, help="Join df from nl_02")
    parser.add_argument("--mord", default='mord_df', type=str, help="Mord df from nl_02")
    parser.add_argument("--bits", default='bits_df', type=str, help="Bits df from nl_02")
    parser.add_argument("--group", default=True, type=bool, help="Split groups formula")
    parser.add_argument("--ds", default=[], type=list, help="List of datasets to include")
    args = parser.parse_args()

    input_args = glob.glob(os.path.join(args.path, "*"))
    filter_split_model_score(input_args)


if __name__ == "__main__":
    main()