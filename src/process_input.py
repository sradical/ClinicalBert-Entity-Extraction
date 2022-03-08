import pandas as pd
from sklearn import preprocessing

def inputdata(datapath):
    df = pd.read_csv(datapath, encoding="latin-1")
    enc_pos = preprocessing.LabelEncoder()
    enc_tag = preprocessing.LabelEncoder()
    # Convert pos and tag to digits
    df.dropna()
    df.loc[:, 'pos'] = enc_pos.fit_transform(df.loc[:, 'pos'])
    df.loc[:, 'tag'] = enc_tag.fit_transform(df.loc[:, 'tag'])
    # Convert to list of lists grouped by sentences
    sentences = df.groupby('sentence#')['token'].apply(list).values
    pos = df.groupby('sentence#')['pos'].apply(list).values
    tag = df.groupby('sentence#')['tag'].apply(list).values
    return sentences, pos, tag, enc_pos, enc_tag