import pandas as pd
import numpy as np


def want_features(dataset, features):
    dataset2 = pd.DataFrame()
    for i in features:
        dataset2[i] = dataset[i]
    return dataset2





df_train = pd.read_csv('train_new.csv')
df_test = pd.read_csv('test_new.csv')

new_df = want_features(df_train, ['Dalc','failures','Fedu','higher','school','studytime'])

print(new_df.head())

