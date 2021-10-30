from pandas import read_csv, get_dummies
from apply_out_of_k import *

df = k_coding(read_csv('heart.csv'))
print(df)