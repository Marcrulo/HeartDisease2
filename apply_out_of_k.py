from pandas import read_csv, get_dummies

def k_coding(df):
    
    # create out-of-k coding (columns)
    one_hot = get_dummies(df['cp'], prefix='cp')
    
    # remove original column
    df = df.drop('cp',axis = 1)
    
    # add out-of-k columns
    df = df.join(one_hot)
    
    # move target to right-most column
    df1 = df.pop('target')
    df['target']=df1
    
    return df
