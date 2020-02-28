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
import functools
import importlib
import os
import time
import argparse

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap
import seaborn as sns



__author__ = "Christopher M Baxter Rath"
__copyright__ = "Copyright 2020"
__credits__ = ["Christopher M Baxter Rath"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Christopher M Baxter Rath"
__email__ = "chrisrath@gmail.com"
__status__ = "Development"

# to do:
# Debug
# All chemical space versus identified chemical space
# Chemical space vs chemical space
# Filtering via FDR and coloc
# Other plots of chemical space, Tanimoto similiarity?
# Mapping mordred onto morgan?
# https://scanpy.readthedocs.io/en/stable/api/scanpy.plotting.html
# Link mord, morg, mm and show tooltip on all three!
# Visualize nuetral loss data


def db_mm_parser(dfs, mord_bool, morg_bool):
    db_df = dfs[0]
    morg = dfs[1]
    mord = dfs[2]

    if mord_bool is True and morg_bool is False:
        m_df = mord.iloc[:,:-2].copy(deep=True)
        return {'data': m_df, 'ids': db_df.hmdb_ids, 'Molecule': db_df.Molecule,
                'data_type': 'Mordred_feat'}

    elif mord_bool is False and morg_bool is True:
        m_df = pd.DataFrame(list(morg.bits)).copy(deep=True)
        return {'data': m_df, 'ids': db_df.hmdb_ids, 'Molecule': db_df.Molecule,
                'data_type': 'Morgan_fp'}

    elif mord_bool is True and morg_bool is True:
        n_df = pd.DataFrame(list(morg.bits)).copy(deep=True).reset_index()
        m_df = mord.iloc[:, :-2].copy(deep=True).reset_index()
        m_df = pd.concat([mord, morg], axis=1).astype(float)
        return {'data': m_df, 'ids': db_df.hmdb_ids.reset_index(),
                'Molecule': db_df.Molecule.reset_index(),
                'data_type': 'feat_fp'}

    else:
        return {'data': None, 'ids': db_df.hmdb_ids.reset_index(),
                'Molecule': db_df.Molecule.reset_index(),
                'data_type': None}


def dataset_parser(data_df, data_sort):
    # Can look at all data, 4x catagories, group, experiment or
    # Custom for filtering by passing 'ColName'=='value'?

    data_df['4x'] = data_df['analyzer'] + data_df['polarity']
    data_df['color'] = None
    column_agg_method_dict = {'has_no_loss': 'describe',
                              'has_H2O': 'describe',
                              'msm': 'describe',
                              'fdr': 'describe',
                              'off_sample': 'describe',
                              'intensity_avg': 'describe',
                              'msm_H2O': 'describe',
                              'fdr_H2O': 'describe',
                              'off_sample_H2O': 'describe',
                              'intensity_avg_H2O': 'describe',
                              'colocalization_H2O': 'describe',
                              'loss_intensity_ratio_H2O': 'describe',
                              'loss_intensity_share_H2O': 'describe',
                              'FDR10-v4': 'describe'}

    column_join_list = ['formula', 'adduct', 'ds_id', 'hmdb_ids', 'ion',
                        'ion_H2O', 'ion_formula', 'ion_formula_H2O', 'group',
                        'analyzer', 'polarity', '4x']

    if data_sort is 'all':
        return data_df

    elif data_sort is '4x':
        gb_label = '4x'

    elif data_sort is 'group':
        gb_label = 'group'

    elif data_sort is 'experiment':
        gb_label = 'ds_id'

    else:
        # Custom?
        data_df = data_df[data_df[data_sort]].copy(deep=True)

    x = list(column_agg_method_dict.keys()).append(gb_label)
    data_df1 = data_df[x].groupby(gb_label).agg(column_agg_method_dict).reset_index().copy(deep=True)

    # https://stackoverflow.com/questions/27298178/concatenate-strings-from-several-rows-using-pandas-groupby/45925961
    y = column_join_list
    data_df2 = data_df[y].groupby(gb_label).apply(','.join).reset_index().copy(deep=True)

    data_df = data_df1.join(data_df2, how='left', on=gb_label)

    return data_df


def pca_from_df(df, title, binary_bool):
    # https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df)
    pca_df = pd.DataFrame(data = principalComponents, columns = ['x', 'y'])
    return {'df': pca_df, 'title': title, 'x_label': 'pca_1', 'y_label': 'pca_2',
            'color_type': None}


def umap_from_df(df, title, binary_bool):
    # https://umap-learn.readthedocs.io/en/latest/basic_usage.html
    if binary_bool is True:
        reducer = umap.UMAP(metric='jaccard')
    else:
        reducer = umap.UMAP()

    embedding = reducer.fit_transform(df)
    umap_df = pd.DataFrame(data=embedding, columns=['x', 'y'])
    return {'df': umap_df, 'title': title, 'x_label': 'umap_1', 'y_label': 'umap_2',
            'color_type': None}


def km_cluster_after_dim_red(df, n):
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    if n == 0:
        return df

    kmeans = KMeans(n_clusters=n, random_state=0).fit(df['x', 'y'])
    df['color'] = kmeans.labels_
    return df


def scatter_plot(df_dict):
    title = df_dict['title']
    x = df_dict['df'].x
    y = df_dict['df'].y
    c = df_dict['df'].color
    xl = df_dict['x_label']
    yl = df_dict['y_label']
    data = df_dict['x', 'y']

    if df_dict['color_type'] is None:
        # https://seaborn.pydata.org/examples/hexbin_marginals.html
        sns.set(style="ticks")
        plot = sns.jointplot(x=x, y=y, kind="hex", color="#4CB391").set(title=title)

    else:
        # https://seaborn.pydata.org/examples/different_scatter_variables.html
        plot = sns.scatterplot(data=data, x=xl, y=yl, hue_order=c).set(title=title)

    return plot


def scatterplot_matrix(df_dict):
    # {'df': a, 'title': b, 'x_label': c, 'y_label': d, 'color_type': e}
    # https://seaborn.pydata.org/examples/scatterplot_matrix.html
    # https://seaborn.pydata.org/examples/pair_grid_with_kde.html
    pass


def h_cluster(df_dict):
    # {'df': a, 'title': b, 'x_label': c, 'y_label': d, 'color_type': e}
    # https://seaborn.pydata.org/examples/structured_heatmap.html
    pass


def violinplot(df_dict):
    title = df_dict['title']
    x = df_dict['df'].x
    y = df_dict['df'].y
    c = df_dict['df'].color
    xl = df_dict['x_label']
    yl = df_dict['y_label']
    data = df_dict['x', 'y']

    sns.set()
    pal = sns.cubehelix_palette(c, rot=-.5, dark=.3)
    plot = sns.violinplot(data=data, x=xl, y=yl, palette=pal,
                          inner="points").set(title=title)
    return plot



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


def plot_picker(df_dict, plot_type):
    if plot_type is 'scatter':
        plot = scatter_plot(df_dict)

    elif plot_type is 'scatterplot_matrix':
        plot = scatterplot_matrix(df_dict)

    elif plot_type is 'h_cluster':
        plot = h_cluster(df_dict)

    elif plot_type is 'violin_plot':
        plot = violinplot(df_dict)

    elif plot_type is 'none':
        plot = None

    else:
        print('Plot picker selection unknown!')
        exit(10)

    return plot


def expt_or_no(x, expt_ids):
    if x in expt_ids:
        return 2
    else:
        return 1


def show_color(df_dict, expt_ids):
    if df_dict['color_type'] is 'kmeans':
        pass

    elif df_dict['color_type'] is None:
        d = df_dict['data']
        d['color'] = 0
        df_dict['data'] = d

    elif df_dict['color_type'] is 'experimental':
        d = df_dict['data']
        d['color'] = d.hmdb_ids.apply(lambda x: expt_or_no(x, expt_ids))
        df_dict['data'] = d

    else:
        print('Color scheme unknown')
        exit(1)

    return df_dict


def master(data_df, dfs, mord_bool, morg_bool, expt_only,
           dim_red, data_sort, title, n, color, plot_type):

    expt_ids = []
    for item in itertools.chain.from_iterable(data_df.hmdb_ids):
        expt_ids.append(item)
    expt_ids = list(set(expt_ids))

    # Parses theoretical databases
    if expt_only is True:
        proc_df_list = []
        for d in dfs:
            d = d[d.hmdb_ids.isin(expt_ids)].copy(deep=True)
            proc_df_list.append(d)
        db_dfs = db_mm_parser(proc_df_list, mord_bool, morg_bool)

    else:
        # db_dfs = {'data': a, 'ids': b, 'Molecule': c, 'data_type': d}
        db_dfs = db_mm_parser(dfs, mord_bool, morg_bool)


    # Parses experimental data:
    data_df = dataset_parser(data_df, data_sort)

    if db_dfs['data_type'] is 'Morgan_fp':
        binary_bool = True
    else:
        binary_bool = False

    if dim_red is 'umap':
        #{'df': a, 'title': b, 'x_label': c, 'y_label': d, 'color_type': e}
        df_dict = umap_from_df(db_dfs['data'], title, binary_bool)
    elif dim_red is 'pca':
        df_dict = pca_from_df(db_dfs['data'], title, binary_bool)
    else:
        # Plot unclustered data? To-do
        pass

    df_dict['df'] = km_cluster_after_dim_red(df_dict['df'], n)
    df_dict['color_type'] = color

    df_dict = show_color(df_dict, data_df)
    plot = plot_picker(df_dict, plot_type)
    #x = tooltip(db_dfs['ids', db_dfs['Molecule]'])

    return {'df_dict': df_dict, 'data_df': data_df, 'plot': plot}


def main():
    # Command line input
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-data_df", default='', type=str, help="Name of nuetral loss dataframe")
    parser.add_argument("-dfs", default='', type=str, help="List of DB's")
    parser.add_argument("-mord_bool", default=True, type=bool, help="T/F: use FP4 fingerprints")
    parser.add_argument("-morg_bool", default=True, type=bool, help="T/F: use Mordred descriptors")
    parser.add_argument("-expt_onlyl", default=False, type=bool, help="T/F: use expt data only")
    parser.add_argument("-dim_red", default='umap', type=str, help="umap or pca")
    parser.add_argument("-data_sort", default='none', type=str, help="Aggregate data?")
    parser.add_argument("-title", default='', type=str, help="Final title")
    parser.add_argument("-n", default=0, type=int, help="n for kmeans clustering")
    parser.add_argument("-color", default=None, type=bool, help="Label for groups/plotting color")
    parser.add_argument("-plot_type", default='scatter', type=bool, help="Plot type")

    return master(parser.parse_args().data_df,
                    parser.parse_args().dfs,
                    parser.parse_args().mord_bool,
                    parser.parse_args().morg_bool,
                    parser.parse_args().expt_only,
                    parser.parse_args().dim_red,
                    parser.parse_args().data_sort,
                    parser.parse_args().title,
                    parser.parse_args().n,
                    parser.parse_args().color,
                    parser.parse_args().plot_type
                    )


if __name__ == "__main__":
    main()