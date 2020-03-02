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
from difflib import SequenceMatcher
from collections import Counter

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

import rdkit.Chem as Chem
from structures_to_search_dicts_water import target_structures, target_loss_formula

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap



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
Clean-up functions below.
Feedback Lachlan on questions
Bring over fields from nl02 and nl03

Dataframes:
   db_df: 
        in expt (do distance calculations +/-)
        no_loss bool, nl_wp_bool, nl_only_bool, 
        no_loss_n, nl_wp_n, nl_only_n, 
        umap distance: 
            mord_x, mord_y, mord_cat
            morg_x, morg_g, morg_cat
            mm_x, mm_y, mm_cat
            Calculate distance metrics first?
        -->join obs and/or exptl df df if not too big...

Plots:   
    Chemical space x3 ns
    Chemical space x3 cross-cats (3)
    Chemical space obs v all
    Chemical space obs only
    Chemical space obs ds
    Chemical space obs nl

Easy:
    Plots to implement: 
        https://seaborn.pydata.org/examples/grouped_violinplots.html
        https://seaborn.pydata.org/examples/pair_grid_with_kde.html

Hard:
    Chemical space vs chemical space (Mapping mordred onto morgan)
        Calculate pair-wise matrix each --> Cluster map
    Link mord, morg, mm and show tooltip on all three!
        https://community.plot.ly/t/is-it-possible-to-link-selections-across-plotly-plots/7120
        https://github.com/plotly/crosslink-plotly.js/
        https://plot.ly/python/v3/ipython-notebooks/cufflinks/#chart-types
        
'''

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
def venn_3_f_df(df, class_A, class_B, class_C, target):
    total = len(set(list(df[target])))
    a = set(list(df[df[class_A] == True][target]))
    b = set(list(df[df[class_B] == True][target]))
    c = set(list(df[df[class_C] == True][target]))

    print('Venn diagram for observed losses out of ' + str(total) + ' unique formulas')
    plt = venn3([a, b, c], set_labels=(class_A, class_B, class_C))
    return plt


def mord_to_dim_red(mord_df):
    mord_df = mord_df.iloc[:, :-2].copy(deep=True)
    return mord_df


def morg_to_dim_red(morg_df):
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


def umap_from_df(df, title, input_type):
    # https://umap-learn.readthedocs.io/en/latest/basic_usage.html
    x = 'pca_' + str(input_type) + '_x'
    y = 'pca_' + str(input_type) + '_y'

    if input_type is 'morg':
        reducer = umap.UMAP(metric='jaccard')
    else:
        reducer = umap.UMAP()

    embedding = reducer.fit_transform(df)
    df = pd.DataFrame(data=embedding, columns=[x, y])
    return df


def km_cluster_after_dim_red(df, x_label, y_label):
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    kmeans = KMeans(n_clusters=6, random_state=0).fit(df[[x_label, y_label]])
    label = SequenceMatcher(None, x_label, y_label).find_longest_match(0, len(x_label), 0, len(y_label))
    label = label + '_cat'
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