import pandas as pd
import numpy as np
import rdkit.Chem as Chem
import time
import argparse

# Hardcoded neutral loss parameters.
# Examples -OH/-CO2H SMARTS for H2O
from structures_to_search_dicts_water import target_structures, target_loss_formula

"""
nl_01_preprocess:

Purpose:
This script is designed to perform pre-processing on neutral loss searches from
metaspace prior to module 2 (handle multiple hits and join), and module 3 
filtering/model building/testing.

Steps include:
1) Parse METASPACE search results and export pickle.     
2) Parse hmdb and apply RDkit to molecules in 1) and export pickle.

Previous script in series is:
"nl_00_hdmb_pre.py"

Next script in series is:
"nl_02_join.py"

Example command line:
python nl_01_preprocess.py -m all_public.pickle -p all_public_hmdb_00.pickle -is_H2O True

"""


class PreprocessLoop(object):
    # This class processes METASPACE neutral loss-output data for downstream scripts.

    def __init__(self, target_dict, input_df):
        self.target_dict = target_dict
        self.input_df = input_df


    def check_polarity(self, adduct):
        if adduct == '-H' or adduct == '+Cl':
            return -1
        elif adduct == '+H' or adduct == '+Na' or adduct == '+K':
            return 1
        else:
            return 0


    def n_loss_only(self, lir):
        if float(lir) == 1:
            return True
        else:
            return False


    def n_loss_wparent(self, lir):
        if float(lir) > 0 and float(lir) < 1:
            return True
        else:
            return False


    def join_nl_searches(self):
        input_df = self.input_df

        # Annotate polarity for downstream model building.
        input_df['polarity'] = input_df['adduct'].apply(lambda row: self.check_polarity(row))

        # Annotate multiple ID for next script.
        input_df['num_ids'] = input_df['hmdb_ids'].apply(lambda x: len(x))

        # Cludge to fix water loss as three smarts targets:
        input_df['loss_intensity_share_H2Oc'] = input_df['loss_intensity_share_H2O']
        input_df['loss_intensity_share_H2Od'] = input_df['loss_intensity_share_H2O']
        input_df['loss_intensity_share_H2Oe'] = input_df['loss_intensity_share_H2O']

        # Annotate loss-type for each target group
        for long_name, formula in self.target_dict.items():

            # Fixes issue with water being present 3x for smart search string.
            if formula == 'H2Oc' or formula == 'H2Od' or formula == 'H2Oe':
                formula = 'H2O'

            n_loss_o = 'n_loss_only_' + str(formula)
            n_loss_wp = 'n_loss_wparent_' + str(formula)
            lir = 'loss_intensity_share_' + str(formula)

            input_df[n_loss_o] = input_df[lir].apply(lambda row: self.n_loss_only(row))
            input_df[n_loss_wp] = input_df[lir].apply(lambda row: self.n_loss_wparent(row))

        return input_df


class hmdb_rd(object):
    # This class filters and exports HMDB for observed ID's from METASPACE from the previous class.

    def __init__(self, target_dict):
        self.target_dict = target_dict


    def get_expt_ids(self, input_df):
        hmdb_ids = input_df.hmdb_ids.tolist()

        # Sort for all unique values in list of list hmdb_ids
        # https://stackoverflow.com/questions/38895856/python-pandas-how-to-compile-all-lists-in-a-column-into-one-unique-list
        hmdb_unique = list(set([a for b in hmdb_ids for a in b]))
        return hmdb_unique


    def add_targets(self, sanitized_df):
        # Add rdkit filter objects.
        pd.options.mode.chained_assignment = None  # default='warn'

        target_names = []
        for target, smarts in target_structures.items():
            substruct_object = Chem.MolFromSmarts(smarts)
            target_name = target_loss_formula[target] + '_RDtarget'
            sanitized_df[target_name] = substruct_object
            target_names.append(target_name)

        pd.options.mode.chained_assignment = 'warn'  # default='warn'
        return [sanitized_df, target_names]


    def target_present(self, structure, target):
        # Can find arbitrary structures in rd object formats
        search_result = structure.HasSubstructMatch(target)
        return search_result


    def cde(self, c, d, e):
        result = bool(c) or bool(d) or bool(e)
        return result


    def targets_present(self, targeted_df):
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
            x_df[res] = x_df.apply(lambda x: self.target_present(x['Molecule'],
                                                                 x[formula]), axis=1)

        # This will cause in error if water loss is not selected.
        x_df['H2O_Present'] = x_df.apply(lambda x: self.cde(x.H2Oc_Present,
                                                                        x.H2Od_Present,
                                                                        x.H2Oe_Present),
                                                                        axis=1)

        col_names = ['H2Oc_Present', 'H2Od_Present', 'H2Oe_Present']
        x_df = x_df.drop(columns=col_names)

        pd.options.mode.chained_assignment = 'warn'  # default='warn'

        return x_df


    def cleanup(self, x_df, col_names):
        # Drop rdkit objects in df not needed anymore
        x_df = x_df.drop(columns=col_names)

        return x_df


    def hmdb_rd_loop(self, input_df, hmdb_file):
        # 1) Loads hmdb, 2) finds id's in this expt, 3) is target there?

        hmdb_df = pd.read_pickle(hmdb_file)
        expt_ids = self.get_expt_ids(input_df)

        # Filter hmdb_df for id's observed in this experiment
        hmdb_filtered = hmdb_df[hmdb_df['hmdb_ids'].isin(expt_ids)]

        # Adds SMARTS target as temporary column to search against.
        hmdb_targeted = self.add_targets(hmdb_filtered)

        # Searches SMARTS target against RDKit object via apply to rows.
        hmdb_searched = self.targets_present(hmdb_targeted[0])

        # Clean-up
        hmdb_searched = self.cleanup(hmdb_searched, hmdb_targeted[1])

        return hmdb_searched


def water_cleanup(x_df):
    desired_columns = ['formula', 'hmdb_ids',  'polarity', 'adduct', 'ds_id',
                       'num_ids', 'has_no_loss', 'has_H2O', 'fdr', 'fdr_H2O',
                       'colocalization_H2O', 'loss_intensity_share_H2O',
                       'n_loss_only_H2O', 'n_loss_wparent_H2O',
                       'msm', 'msm_H2O', 'off_sample', 'off_sample_H2O',
                       'intensity_avg', 'intensity_avg_H2O']

    x_df = x_df[desired_columns].copy(deep=True)

    convert_dict = {'polarity': np.int8, 'num_ids': np.uint8,
                    'fdr': np.float32, 'fdr_H2O': np.float32,
                    'loss_intensity_share_H2O': np.float32,
                    'msm': np.float32, 'msm_H2O': np.float32,
                    'intensity_avg': np.float32,
                    'intensity_avg_H2O': np.float32,
                    'off_sample': np.float32, 'off_sample_H2O': np.float32}

    x_df = x_df.astype(convert_dict).copy(deep=True)

    return x_df


def main_loop_1(input_file, hmdb_file, input_dict, out_stub, is_H2O):
    start_time = time.time()

    # Setup main loop classes
    input_df = pd.read_pickle(input_file)
    pre_loop = PreprocessLoop(input_dict, input_df)
    db_loop = hmdb_rd(input_dict)

    # Run main loops
    output_df = pre_loop.join_nl_searches()
    if is_H2O is True:
        output_df = water_cleanup(output_df)

    hmdb_df = db_loop.hmdb_rd_loop(output_df, hmdb_file)

    # Output
    out_file = out_stub + '_output_01.pickle'
    output_df.to_pickle(out_file)
    out_hmdb = out_stub + '_hmdb_01.pickle'
    hmdb_df.to_pickle(out_hmdb)

    elapsed_time = int(time.time() - start_time)
    print('Elapsed time (seconds):\n')
    print(elapsed_time)
    print('\nExecuted without error\n')
    print(out_file)
    print(out_hmdb)

    return [output_df, hmdb_df]


### Body ###
parser = argparse.ArgumentParser(description='')
parser.add_argument("-m", default='all_public_data.pickle', type=str, help="Metaspace results pickle")
parser.add_argument("-p", default='all_public_hmdb.pickle', type=str, help="HMDB database pickle")
parser.add_argument("-is_H2O", default=True, type=bool, help="Is neutral loss type water?")

input_file = parser.parse_args().m
hmdb_file = parser.parse_args().p
is_H2O = parser.parse_args().is_H2O
input_dict = target_loss_formula
out_stub = input_file.split('.')[0]

result = main_loop_1(input_file, hmdb_file, input_dict, out_stub, is_H2O)