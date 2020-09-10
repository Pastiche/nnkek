# df.drop_duplicates(['item_id'], inplace=True)
# df.drop_duplicates(['img_id'], inplace=True)
# df.dropna(inplace=True)
# etc.


import pandas as pd
# Train/test/val sets
from sklearn.model_selection import train_test_split


def preprocess(df):
    df.drop_duplicates(['item_id'], inplace=True)
    df.drop_duplicates(['main_im'], inplace=True)
    df.dropna(inplace=True)
    return df


def train_test_val_split(df, frac, random_state=42):
    """No stratification etc, just randomly splits into three sets"""

    _, test_set = train_test_split(df, test_size=frac,
                                   random_state=random_state)

    train_set, val_set = train_test_split(_, test_size=test_set.shape[0],
                                          random_state=random_state)
    val_set['scope'] = 'val'
    train_set['scope'] = 'train'
    test_set['scope'] = 'test'

    df_normed = pd.concat([val_set, train_set, test_set], axis=0)
    df_normed.reset_index(drop=True, inplace=True)

    print(df_normed[df_normed.scope == 'train'].shape)
    print(df_normed[df_normed.scope == 'val'].shape)
    print(df_normed[df_normed.scope == 'test'].shape)

    return df_normed
