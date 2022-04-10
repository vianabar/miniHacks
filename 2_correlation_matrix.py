import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


df_train = pd.read_csv('train_new.csv').drop(['grade'], axis=1)
df_test = pd.read_csv('test_new.csv')

corrMatrix = df_train.corr()
plt.figure(figsize=(25, 20), dpi=200)
sn.heatmap(corrMatrix, annot=True)

plt.savefig('correlation_matrix_2')



train = pd.read_csv('train_new.csv')

plt.figure()
plt.hist(train['grade'])
plt.grid()
plt.savefig('grade_distribution')