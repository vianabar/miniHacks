import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


df_train = pd.read_csv('train.csv').drop(['grade'], axis=1)
df_test = pd.read_csv('test.csv')

corrMatrix = df_train.corr()
plt.figure(figsize=(15, 10), dpi=200)
sn.heatmap(corrMatrix, annot=True)

plt.savefig('correlation_matrix')