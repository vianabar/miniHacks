# transform 
import pandas as pd

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')



# Yes or No binary ones : 1 or 0
# schoolsup, famsup, paid, activities, nursery, higher, internet, romantic (17 to 24)
df_train = df_train.replace('yes', 1)
df_train = df_train.replace('no', 0)
df_test = df_test.replace('yes', 1)
df_test = df_test.replace('no', 0)

# sex: 'F' to 1 and 'M' to 0
df_train = df_train.replace('F', 1)
df_train = df_train.replace('M', 0)
df_test = df_test.replace('F', 1)
df_test = df_test.replace('M', 0)

# address: 'U' to 1 and 'R' to 0
df_train = df_train.replace('U', 1)
df_train = df_train.replace('R', 0)
df_test = df_test.replace('U', 1)
df_test = df_test.replace('R', 0)

# famsize: 'LE3' to 1 and 'GT3' to 0
df_train = df_train.replace('LE3', 1)
df_train = df_train.replace('GT3', 0)
df_test = df_test.replace('LE3', 1)
df_test = df_test.replace('GT3', 0)

# Pstatus: 'T' to 1 and 'A' to 0
df_train = df_train.replace('T', 1)
df_train = df_train.replace('A', 0)
df_test = df_test.replace('T', 1)
df_test = df_test.replace('A', 0)

# school: 'GP' to 1 and 'MS' to 0
df_train = df_train.replace('GP', 1)
df_train = df_train.replace('MS', 0)
df_test = df_test.replace('GP', 1)
df_test = df_test.replace('MS', 0)

df_train.to_csv('train_new.csv')  
df_test.to_csv('test_new.csv')  