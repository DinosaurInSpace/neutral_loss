import pandas as pd
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import rdkit.DataStructs
from mordred import Calculator, descriptors
import time
import argparse

"""
nl_00_hmdb_pre.py:

Purpose:
This script is designed to perform pre-processing on hmdb for fingerprints,
and other features to reduce the overhead from recalculating as datasets are rerun.

Steps include:
1) Perform all calculations that are not specific to NL's under investigation.
2) Export pickles

Example commandline:
python nl_00_hmdb_pre.py -o all_public_hmdb.pickle -f hmdb_out_molecule -bits True -mord True

Previous script in series is:
""

Next script in series is:
"nl_01_preprocess.py"

"""


def load_hmdb_df(hmdb_df_path):
    # input HMDB file is expected to have precalculated rdkit Molecule object.
    hmdb_df = pd.read_pickle(hmdb_df_path)
    hmdb_df = hmdb_df.rename(columns={'id': 'hmdb_ids'})
    hmdb_df = hmdb_df.dropna(axis=0, how='any')
    good_columns = ['formula', 'hmdb_ids', 'mol_name', 'Molecule']

    for col in list(hmdb_df):
        if col not in good_columns:
            hmdb_df = hmdb_df.drop(columns=[col])
        else:
            continue

    return hmdb_df


def molecule_fp_array(molecule):
    fp = Chem.AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=1024)
    arr = np.zeros((1,))
    rdkit.DataStructs.ConvertToNumpyArray(fp, arr)
    arr = arr.astype(np.bool_)  # int, bool same memory usage
    return arr


def bits_loop(hmdb_df, bits_rerun):
    # 1024 bit chemical fingerprint (Morgan/FP4)
    if bits_rerun is True:
        bits_df = hmdb_df[['hmdb_ids', 'Molecule']].copy(deep=True)

        bits_df['bits'] = bits_df['Molecule'].apply(lambda x: molecule_fp_array(x))
        bits_df.to_pickle('bits_hmdb_df.pickle')
        bits_df.drop(columns=['Molecule'])
    else:
        bits_df = pd.read_pickle('bits_hmdb_df.pickle')

    return bits_df


def mordred(molecule_series):
    calc = Calculator(descriptors)
    m_df = calc.pandas(molecule_series)
    m_df = m_df.astype(np.float32)
    return m_df


def mord_norm(df):
    # Use 0-1 norm as sparse for many variables
    for c in list(df.columns):
        df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())

    # Get rid of anything not in range 0-1
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    df = df.mask(df > 1, 0)
    df = df.mask(df < 0, 0)
    df = df.astype(np.float32)
    return df


def mord_loop(hmdb_df, mord_rerun):
    # This calculates Mordred chemical descriptors for all of HMDB.
    # Moriwaki, Hirotomo, et al.
    # "Mordred: a molecular descriptor calculator."
    # Journal of cheminformatics 10.1(2018): 4.
    if mord_rerun is True:
        mord_df = mordred(hmdb_df['Molecule'])
        mord_norm_df = mord_norm(mord_df)
        mord_df['hmdb_ids'] = hmdb_df['hmdb_ids']
        mord_norm_df['hmdb_ids'] = hmdb_df['hmdb_ids']

        mord_df.to_pickle('mord_hmdb_df.pickle')
        mord_norm_df.to_pickle('mord_norm_hmdb_df.pickle')

    else:
        mord_df = pd.read_pickle('mord_hmdb_df.pickle')
        mord_norm_df = pd.read_pickle('mord_norm_hmdb_df.pickle')

    return [mord_df, mord_norm_df]


def main_loop_0(out_name, hmdb_in, bits_rerun, mord_rerun):

    start_time = time.time()

    hmdb_df = load_hmdb_df(hmdb_in)
    bits_df = bits_loop(hmdb_df, bits_rerun)
    mord_norm_df = mord_loop(hmdb_df, mord_rerun)[1]

    hmdb_df['trues'] = True
    hmdb_df['falses'] = False
    hmdb_df['rando'] = np.random.randint(2, size=hmdb_df.shape[0], dtype=np.bool_)

    hmdb_df.to_pickle(out_name)

    elapsed_time = time.time() - start_time
    print('Elapsed time:\n')
    print(elapsed_time)
    print('\nExecuted without error\n')
    print(out_name)


    return [hmdb_df, bits_df, mord_norm_df]

### Body ###

# Command line input
parser = argparse.ArgumentParser(description='')
parser.add_argument("-o", default='all_public_hmdb.pickle', type=str, help="Output HMDB dataframe as pickle")
parser.add_argument("-f", default='hmdb_out_molecule', type=str, help="Input HMDB dataframe as pickle")
parser.add_argument("-bits", default=True, type=bool, help="T/F: Recalculate FP4 fingerprints")
parser.add_argument("-mord", default=True, type=bool, help="T/F: Recalculate Mordred descriptors")

out_name = parser.parse_args().o
hmdb_file = parser.parse_args().f
bits_rerun = parser.parse_args().bits
mord_rerun = parser.parse_args().mord

# Main loop, alternatively can import and call function from Jupyter.
result = main_loop_0(out_name, hmdb_file, bits_rerun, mord_rerun)