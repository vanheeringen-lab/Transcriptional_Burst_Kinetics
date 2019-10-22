"""
Script that calculates the lambda, mu, and nu values of two experimental conditions, and the chance
whether or not those parameters are different.
"""
import sys
import os
import argparse
import multiprocessing as mp
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(f"{os.getcwd()}/."))
from tbk.inference import likelihood_ratio_test


parser = argparse.ArgumentParser(description='Description')
parser.add_argument('file_1', type=str, help='comma separated count input file')
parser.add_argument('file_2', type=str, help='comma separated count input file')
parser.add_argument('--outfile', default='likelihood_ratio_test', type=str, help='Name of the output file(csv)')
parser.add_argument('--nworkers', default=1, type=int, help='Output dir for image')
args = parser.parse_args()

data_1 = pd.read_csv(args.file_1, sep=',', index_col=0)
data_2 = pd.read_csv(args.file_2, sep=',', index_col=0)

assert list(data_1.index) == list(data_2.index), "Files should contain the same genes " \
                                                 "(in the same order)"

values_1 = np.sort(data_1.values, axis=1)
values_2 = np.sort(data_2.values, axis=1)

with mp.Pool(processes=args.nworkers) as pool:
    params = pool.starmap(likelihood_ratio_test, [(vals1, vals2) for vals1, vals2 in zip(values_1, values_2)])

params = [[*sublists[0], *sublists[1], *sublists[2]] for sublists in params]
df = pd.DataFrame(params, index=data_1.index, columns=['1 k_on', '1 k_off', '1 k_syn',
                                                       '2 k_on', '2 k_off', '2 k_syn',
                                                       'p k_on', 'p k_off', 'p k_syn'])

df = df.fillna('---')
df.to_csv(args.outfile)
