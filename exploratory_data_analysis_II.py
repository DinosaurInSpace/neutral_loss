#!/usr/bin/env python

"""
nl_00_hmdb_pre.py:

Purpose:
This script contains various functions for exploratory data analysis.

This script is intended to be called from Jupyter or the command line
to allow in line plotting of data analysis without various boiler plate.

Example commandline:

Example notebook usage:
http://localhost:8888/notebooks/PycharmProjects/neutral_loss/Exploratory_data_analysis.ipynb

Previous script in series is:

Next script in series is:

"""

import pandas as pd
import numpy as np
import itertools
import argparse
from collections import Counter

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

import rdkit.Chem as Chem
from structures_to_search_dicts_water import target_structures, target_loss_formula

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap

# Surpress umap warnings from umap
import warnings


__author__ = "Christopher M Baxter Rath"
__copyright__ = "Copyright 2020"
__credits__ = ["Christopher M Baxter Rath"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Christopher M Baxter Rath"
__email__ = "chrisrath@gmail.com"
__status__ = "Development"

'''
# to do:

Dataframes:
Use chem_scatter to plot!
    Supports df
    Supports plot
    Supports labeling some points differently
    Fast

Plots:   
    Chemical space x3 ns
    Chemical space x3 cross-cats (3)
    Chemical space obs v all
    Chemical space obs only
    Chemical space obs ds
    Chemical space obs nl

Lachlan:   
    PCA then umap? To drop non-contributing features
    Build models and identify contributing features.  

Easy:
    https://seaborn.pydata.org/examples/pair_grid_with_kde.html
    Drop duplicate datasets
    Prune bad molecules in db (e.g. metals)
    Get clusters back from visualization!
        -https://github.com/mwaskom/seaborn/issues/1207

Hard:
    Chemical space vs chemical space (Mapping mordred onto morgan)
        Calculate pair-wise matrix each then do what?
            Network
            Dendrogram
    Link mord, morg, mm and show tooltip on all three!
        https://community.plot.ly/t/is-it-possible-to-link-selections-across-plotly-plots/7120
        https://github.com/plotly/crosslink-plotly.js/
        https://plot.ly/python/v3/ipython-notebooks/cufflinks/#chart-types
        
'''
# Surpress warnings from umap
def action_with_warnings():
    warnings.warn("should not appear")

# Experimental dataframe specific functions:
def n_loss_only(lir):
    if float(lir) == 1:
        return True
    else:
        return False


def n_loss_wparent(lir):
    if float(lir) > 0 and float(lir) < 1:
        return True
    else:
        return False


def n_loss_none(lir):
    if float(lir) == 0:
        return True
    else:
        return False


def annotate_nl(df, loss_label):
    n_loss_o = 'n_loss_only_' + loss_label
    n_loss_wp = 'n_loss_wparent_' + loss_label
    n_loss_n = 'n_loss_none_' + loss_label
    lir = 'loss_intensity_share_' + loss_label
    df[n_loss_o] = df[lir].apply(lambda x: n_loss_only(x))
    df[n_loss_wp] = df[lir].apply(lambda x: n_loss_wparent(x))
    df[n_loss_n] = df[lir].apply(lambda x: n_loss_none(x))
    return df


def best_df(df, fdr_max, nl_name):
    fdr_nl = 'fdr_' + nl_name
    df = df[(df['fdr'] <= fdr_max) |
            (df[fdr_nl] <= fdr_max)].copy(deep=True)
    return df


# Database specific functions
def add_targets(df):
    # Add rdkit filter objects.
    pd.options.mode.chained_assignment = None  # default='warn'

    target_names = []
    for target, smarts in target_structures.items():
        substruct_object = Chem.MolFromSmarts(smarts)
        target_name = target_loss_formula[target] + '_RDtarget'
        df[target_name] = substruct_object
        target_names.append(target_name)

    pd.options.mode.chained_assignment = 'warn'  # default='warn'
    return df


def target_present(structure, target):
    # Can find arbitrary structures in rd object formats
    search_result = structure.HasSubstructMatch(target)
    return search_result


def cde(c, d, e):
    result = bool(c) or bool(d) or bool(e)
    return result


def num_ids(df, list_col):
    # What if num_ids is 1, is it is a strong?
    df['num_ids'] = df[list_col].apply(lambda x: len(x))
    return df

def weights_n(df, id_col):
    # What if num_ids is 1, is it is a strong?
    out = 'weight_' + id_col
    df[out] = df[id_col].apply(lambda x: 1 / x)
    return df

def targets_present(targeted_df):
    # Score for presence of rd objects in one column as product of two
    # Implemented in hmdb_structure_searcher.ipynb
    pd.options.mode.chained_assignment = None  # default='warn'

    headers = []
    for target, formula in target_loss_formula.items():
        target_name = formula + '_RDtarget'
        headers.append(target_name)

    x_df = targeted_df
    for formula in headers:
        r = formula.split('_')
        res = r[0] + '_Present'
        x_df[res] = x_df.apply(lambda x: target_present(x['Molecule'], x[formula]), axis=1)

    # This will cause in error if water loss is not selected.
    x_df['H2O_Present'] = x_df.apply(lambda x: cde(x.H2Oc_Present, x.H2Od_Present,
                                                   x.H2Oe_Present), axis=1)

    col_names = ['H2Oc_Present', 'H2Od_Present', 'H2Oe_Present']
    x_df = x_df.drop(columns=col_names)

    pd.options.mode.chained_assignment = 'warn'  # default='warn'

    return x_df


# Exploratory data analysis functions
def db_obs_df_merge(db_df, obs_df, out_col, bool_not_n):
    df = obs_df
    df['idx'] = df.index
    if bool_not_n is True:
        df[out_col] = True
    else:
        df[out_col] = df['sum']
        df = df.drop(columns=['sum'])
    db_df = db_df.merge(df, how='left', left_on='hmdb_ids', right_on='idx')
    db_df = db_df.drop(columns=['idx'])
    db_df[out_col] = db_df[out_col].fillna(0)
    return db_df


def db_df_add_nl(db_df, data_df, pol_list, loss_list):
    for l in loss_list:
        for p in pol_list:
            out_col = p + '_' + l
            df = data_df[(data_df['polarity'] == p) & (data_df[l] == True)]
            obs_df = generate_obs_df(df, 'ds_id')
            obs_df['idx'] = obs_df.index

            obs_df['sum'] = obs_df.sum(axis = 1, skipna = True)
            obs_df = obs_df[['idx', 'sum']].copy(deep=True)
            db_df = db_obs_df_merge(db_df, obs_df, out_col, False)
    return db_df


def venn_3_f_df(df, class_A, class_B, class_C, target):
    total = len(set(list(df[target])))
    a = set(list(df[df[class_A] == True][target]))
    b = set(list(df[df[class_B] == True][target]))
    c = set(list(df[df[class_C] == True][target]))

    print('Venn diagram for observed losses out of ' + str(total) + ' unique formulas')
    plt = venn3([a, b, c], set_labels=(class_A, class_B, class_C))
    return plt


def kmeans_after_umap_and_merge_df(df, n, do_df):
    df = km_cluster_after_dim_red(df, list(df)[0], list(df)[1], n)
    do_df = pd.merge(do_df, df, how='left', left_on='hmdb_ids', right_on='idx')
    do_df = do_df.drop(columns=['idx'])
    return do_df


def add_mord_morg_mm_umap_distance(do_df, mord_df, morg_df, n):
    with warnings.catch_warnings(record=True):
        action_with_warnings()
        df = umap_from_df(mord_to_dim_red(mord_df), 'mord')
        df['idx'] = list(mord_df.iloc[:, -1])
        do_df = kmeans_after_umap_and_merge_df(df, n, do_df)

        df = umap_from_df(morg_to_dim_red(morg_df), 'morg')
        df['idx'] = list(morg_df.hmdb_ids)
        do_df = kmeans_after_umap_and_merge_df(df, n, do_df)

        df = umap_from_df(mord_morg_to_dim_red(mord_df, morg_df), 'mm')
        df['idx'] = list(morg_df.hmdb_ids)
        do_df = kmeans_after_umap_and_merge_df(df, n, do_df)
        return do_df



def mord_to_dim_red(mord_df):
    mord_df = mord_df.iloc[:, :-1].copy(deep=True)
    return mord_df


def morg_to_dim_red(morg_df):
    morg_df = morg_df.bits
    morg_df = pd.DataFrame(list(morg_df)).copy(deep=True)
    return morg_df


def mord_morg_to_dim_red(mord_df, morg_df):
    m1_df =  mord_to_dim_red(mord_df).reset_index()
    m2_df = morg_to_dim_red(morg_df).reset_index()
    mm_df = pd.concat([m2_df, m1_df], axis=1).astype(float).reset_index()
    return mm_df


def pca_from_df(df, input_type):
    # https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    x = 'pca_' + str(input_type) + '_x'
    y = 'pca_' + str(input_type) + '_y'
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df.astype(float))
    df = pd.DataFrame(data = principalComponents, columns = [x, y])
    return df


def umap_from_df(df, input_type):
    # https://umap-learn.readthedocs.io/en/latest/basic_usage.html
    x = 'umap_' + str(input_type) + '_x'
    y = 'umap_' + str(input_type) + '_y'

    if input_type is 'morg':
        reducer = umap.UMAP(metric='jaccard')
    else:
        reducer = umap.UMAP()

    embedding = reducer.fit_transform(df)
    df = pd.DataFrame(data=embedding, columns=[x, y])
    return df


def km_cluster_after_dim_red(df, x_label, y_label, n):
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    kmeans = KMeans(n_clusters=n, random_state=0).fit(df[[x_label, y_label]])
    label = x_label.rsplit('_',1)[0] + '_cat'
    df[label] = kmeans.labels_
    return df


def expt_id_in_df(df, expt_ids):
    df['obs_expt'] = df.hmdb_ids.apply(lambda x: expt_or_no(x, expt_ids))
    return df


def unstack_hmdbs_list(df, col):
    expt_ids = []
    for item in itertools.chain.from_iterable(df[col]):
        expt_ids.append(item)
    return expt_ids


def unstack_hmdbs_count(df, col):
    expt_ids = unstack_hmdbs_list(df, col)
    k = Counter(expt_ids).keys()
    v = Counter(expt_ids).values()
    expt_ids = dict(zip(k,v))
    return expt_ids


def generate_obs_df(data_df, sort_catagory):
    # Generates a new dataframe showing DB observations by catagory such as ds_id
    obs_df = pd.DataFrame()
    expt_ids = sorted(list(set(unstack_hmdbs_list(data_df, 'hmdb_ids'))))
    obs_df['exptl_ids'] = expt_ids
    x_list = data_df[sort_catagory].unique()
    for x in x_list:
        df = data_df[data_df[sort_catagory] == x]
        x_dict = unstack_hmdbs_count(df, 'hmdb_ids')
        x_df = pd.DataFrame.from_dict(x_dict, orient='index', columns=[x])
        x_df = x_df.reset_index()
        x_df = x_df.rename(columns={'index': 'hmdb_ids'})
        obs_df = pd.merge(obs_df, x_df, how='left', left_on='exptl_ids', right_on='hmdb_ids')
        obs_df = obs_df.drop(columns=['hmdb_ids'])
    obs_df = obs_df.fillna(0)
    obs_df = obs_df.set_index('exptl_ids')
    return obs_df


def normalize_df(df):
    # Defaults to by column
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df

# Not yet implemented functions
def tooltip(ids, Molecule):
    # Use to show structure on graph
    '''
    This is how I did the tooltip-style information with matplotlib btw:
    https://github.com/alexandrovteam/ims-direct-control/blob/master/remote_control/acquisition.py#L234
    plt.gca().format_coord should be a function that takes an arbitrary position in the
    coordinate frame of the chart. In this case coord_formatter returns a function that
    captures all the source data, and every time format_coord gets called it just finds
    the nearest point and returns a string containing the additional info about that point
    (mainly acquisition index, and z coordinate)

    https://github.com/alexandrovteam/ims-direct-control/blob/master/remote_control/acquisition.py#L234

    '''

    #plt.gca().format_coord = coord_formatter([xy[0] for xy in xys], [xy[1] for xy in xys], [0 for xy in xys])
    pass


def expt_or_no(x, expt_ids):
    if x in expt_ids:
        return 2
    else:
        return 1





def master(dummy):
    pass


def main():
    # Command line input
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-dummy", default='', type=str, help="")
    return master(parser.parse_args().dummy)


if __name__ == "__main__":
    main()