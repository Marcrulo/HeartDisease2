from pandas import read_csv
from apply_out_of_k import k_coding

df = read_csv('heart.csv')
df = k_coding(df) # apply out-of-k coding

classification_data   = df.loc[:,'age':'cp_3']
classification_target = df.loc[:,'target']

regression_data = df.loc[:,'age':'trestbps'].join(df.loc[:,'fbs':'target'])
regression_target = df.loc[:,'chol']
