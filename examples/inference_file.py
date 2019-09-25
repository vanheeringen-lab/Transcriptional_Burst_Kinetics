"""
Script that calculates the lambda, mu, and nu values for each gene over all samples.
"""
import sys
import os
import argparse
import multiprocessing as mp
import pandas as pd

sys.path.append(os.path.abspath(f"{os.getcwd()}/."))
from tbk.inference import maximum_likelihood

parser = argparse.ArgumentParser(description='Description')
parser.add_argument('file', type=str, help='comma separated count file')
parser.add_argument('--nworkers', default=1, type=int, help='Output dir for image')

args = parser.parse_args()

# load the data
data = pd.read_csv(args.file, sep=',', index_col=0)

# estimate the values
with mp.Pool(processes=args.nworkers) as pool:
    params = pool.starmap(maximum_likelihood, [(products,) for products in data.values])

# save
df = pd.DataFrame(index=data.index, columns=['k_on', 'k_off', 'k_syn'])
for index, rowname in enumerate(df.index):
    df.iloc[index] = params[index]

df = df.fillna('---')
df.to_csv(f'{os.path.splitext(args.file)[0]}_params.csv')
