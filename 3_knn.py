from sklearn.datasets import load_boston

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV


df = load_boston()

def want_features(dataset, features):
    dataset2 = pd.DataFrame()
    for i in features:
        dataset2[i] = dataset[i]
    return dataset2


df_train = pd.read_csv('train_new.csv')

# DROP ID
df_train_x = df_train.drop(['grade','ID'],axis=1)
# X = np.array(want_features(df_train_x, ['Dalc','failures','Fedu','higher','school','studytime']))
X = np.array(want_features(df_train_x, ['Dalc','failures','Fedu','higher','school','studytime', 'Fjob','reason','goout']))

y = np.array(df_train['grade'])

df_test = pd.read_csv('test_new.csv')

# dataset.columns = df.feature_names
# dataset['Price'] = df.target

ridge = Ridge()
parameters = {'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor = GridSearchCV( ridge, parameters, scoring='neg_root_mean_squared_error',cv=5)
ridge_regressor.fit(X,y)

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


#%%

lasso = Lasso()
parameters = {'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor = GridSearchCV( lasso, parameters, scoring='neg_root_mean_squared_error',cv=5)
lasso_regressor.fit(X,y)

print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


#%%


# elastic = ElasticNet()
# parameters = {'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100], \
#                 'l1_ratio': np.arange(0,1,0.01)}
# elastic_regressor = GridSearchCV( elastic, parameters, scoring='neg_root_mean_squared_error',cv=5)
# elastic_regressor.fit(X,y)

# print(elastic_regressor.best_params_)
# print(elastic_regressor.best_score_)

