#%%

from sklearn.datasets import load_boston

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV


def want_features(dataset, features):
    dataset2 = pd.DataFrame()
    for i in features:
        dataset2[i] = dataset[i]
    return dataset2



#%%
df_train = pd.read_csv('train_new.csv')

df_train_x = df_train.drop(['grade','ID'],axis=1)
X = np.array(want_features(df_train_x, ['Dalc','failures','higher','school','studytime','famsize','internet','age']))

a = want_features(df_train_x, ['freetime','Walc'])
b = np.array(a.mean(axis=1))
# c = np.reshape(b,(1,len(b)))
c = np.reshape(b,(len(b),1))
X = np.concatenate((X, c),axis=1)

a = want_features(df_train_x, ['Fedu','Medu'])
b = np.array(a.mean(axis=1))
c = np.reshape(b,(len(b),1))
X = np.concatenate((X, c),axis=1)

y = np.array(df_train['grade'])

# For test dataset



df_test = pd.read_csv('test_new.csv')
df_test_x = df_test.drop(['ID'],axis=1)

X_test = np.array(want_features(df_test_x, ['Dalc','failures','higher','school','studytime','famsize','internet','age']))

a_test = want_features(df_test_x, ['freetime','Walc'])
b_test = np.array(a_test.mean(axis=1))
# c = np.reshape(b,(1,len(b)))
c_test = np.reshape(b_test,(len(b_test),1))
X_test = np.concatenate((X_test, c_test),axis=1)

a_test = want_features(df_test_x, ['Fedu','Medu'])
b_test = np.array(a_test.mean(axis=1))
c_test = np.reshape(b_test,(len(b_test),1))
X_test = np.concatenate((X_test, c_test),axis=1)


# dataset.columns = df.feature_names
# dataset['Price'] = df.target

ridge = Ridge(alpha = 5)
ridge.fit(X,y)

print(ridge.score)

test_predict_ridge = ridge.predict(X_test)
test_predict_ridge = np.around(test_predict_ridge)
test_predict_ridge = test_predict_ridge.astype(int)

lasso = Lasso(alpha = 5)
lasso.fit(X,y)

print(lasso.score)

test_predict_lasso = lasso.predict(X_test)
test_predict_lasso = np.around(test_predict_lasso)
test_predict_lasso = test_predict_lasso.astype(int)

# %% Create submission.csv

test_ids = np.array(df_test['ID'])

matrix_test_score = np.vstack([test_ids, test_predict_ridge])
matrix_test_score = np.transpose(matrix_test_score)
df_test_score = pd.DataFrame(matrix_test_score, columns = ['ID', 'grade'])

df_test_score.to_csv('submission.csv', index=False)
# %%
