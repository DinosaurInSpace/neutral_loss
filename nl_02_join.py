import pandas as pd
import numpy as np
import time
import argparse

"""
nl_02_join.py:

Purpose:
This script is designed to join neutral loss search results with HMDB metadata prior to 
module 3 filtering/model building/testing).

Steps include:
1) Load data from both pickles output by previous script.
2) Make a row for each HMDB ID.     
3) Simple join between both inputs.
4) Export to pickle.

Next script in series is:
"nl_03_filter_model"
Previous script ion series is:
"nl_01_preprocess"

Example command:
python nl_02_join.py --o all_public_output_01.pickle --h all_public_hmdb_01.pickle

"""


def split_data_frame_list(df, target_column):
    # Accepts a column with multiple types and splits list variables to several rows.

    row_accumulator = []

    def split_list_to_rows(row):
        split_row = row[target_column]

        if isinstance(split_row, list):

          for s in split_row:
              new_row = row.to_dict()
              new_row[target_column] = s
              row_accumulator.append(new_row)

          if split_row == []:
              new_row = row.to_dict()
              new_row[target_column] = None
              row_accumulator.append(new_row)

        else:
          new_row = row.to_dict()
          new_row[target_column] = split_row
          row_accumulator.append(new_row)

    df.apply(split_list_to_rows, axis=1)
    new_df = pd.DataFrame(row_accumulator)

    return new_df


def column_clean(x_df):
    desired_columns = ['formula', 'hmdb_ids', 'polarity', 'adduct', 'ds_id',
                       'num_ids', 'has_no_loss', 'H2O_Present', 'fdr', 'fdr_H2O',
                       'colocalization_H2O', 'loss_intensity_share_H2O', 'n_loss_only_H2O',
                       'n_loss_wparent_H2O', 'off_sample', 'off_sample_H2O',
                       'ion', 'ion_H2O', 'ion_formula', 'ion_formula_H2O',
                       'intensity_avg', 'intensity_avg_H2O', 'trues', 'falses', 'rando',
                       'Molecule', 'weight']
    x_df = x_df[desired_columns].copy(deep=True)

    convert_dict = {'polarity': np.int8, 'adduct': str ,'num_ids': np.uint8,
                    'fdr': np.float32, 'fdr_H2O': np.float32,
                    'colocalization_H2O': np.float32,
                    'loss_intensity_share_H2O': np.float32,
                    'ion': str, 'ion_H2O': str, 'ion_formula': str, 'ion_formula_H2O': str,
                    'intensity_avg': np.float32,
                    'intensity_avg_H2O': np.float32,
                    'off_sample': np.float32, 'off_sample_H2O': np.float32,
                    'trues': np.bool_, 'falses': np.bool_, 'rando': np.bool_,
                    'H2O_Present':np.bool_, 'weight': np.float32}
    x_df = x_df.astype(convert_dict).copy(deep=True)

    return x_df


def main_loop_2(input_file, input_hmdb, is_H2O):
    start_time = time.time()

    input_df = pd.read_pickle(input_file)
    hmdb_df = pd.read_pickle(input_hmdb)

    # Split df to one HMDB ID per row
    input_df = split_data_frame_list(input_df, 'hmdb_ids')

    # Merge theoretical and calculated data
    joined_df = pd.merge(input_df, hmdb_df, how='left', on='hmdb_ids')
    joined_df = joined_df.rename(columns={'formula_x': 'formula'})

    # Calculate weights
    joined_df['weight'] = joined_df['num_ids'].apply(lambda x: 1 / x)

    # Final cleanup
    if is_H2O is True:
        joined_df = column_clean(joined_df)

    # Export
    out_stub = input_file.split('_output_01')[0]
    out_file = out_stub + '_output_02.pickle'
    joined_df.to_pickle(out_file)

    elapsed_time = time.time() - start_time
    print('Elapsed time:\n')
    print('\nExecuted without error\n')
    print(elapsed_time)
    print(out_file)

    return joined_df


### Body ###
# Input files and arguements
parser = argparse.ArgumentParser(description='')
parser.add_argument("--o", default='all_public_output_01.pickle', type=str, help="Output df from nl01")
parser.add_argument("--h", default='all_public_hmdb_01.pickle', type=str, help="nl_01_preprocess_hmdb.pickle")
parser.add_argument("--is_H2O", default=True, type=bool, help="Is neutral loss type water?")
args = parser.parse_args()
input_file = args.o
input_hmdb = args.h
is_H2O = args.is_H2O

# Main loop.
result = main_loop_2(input_file, input_hmdb, is_H2O)