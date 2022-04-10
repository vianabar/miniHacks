# transform 
import pandas as pd

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

datasets = [df_train, df_test]
for i in range(len(datasets)):

    # Yes or No binary ones : 1 or 0
    # schoolsup, famsup, paid, activities, nursery, higher, internet, romantic (17 to 24)
    datasets[i]= datasets[i].replace('yes', 1)
    datasets[i] = datasets[i].replace('no', 0)

    # school: 'GP' to 1 and 'MS' to 0
    datasets[i] = datasets[i].replace('GP', 1)
    datasets[i] = datasets[i].replace('MS', 0)

    # sex: 'F' to 1 and 'M' to 0
    datasets[i] = datasets[i].replace('F', 1)
    datasets[i] = datasets[i].replace('M', 0)

    # address: 'U' to 1 and 'R' to 0
    datasets[i] = datasets[i].replace('U', 1)
    datasets[i] = datasets[i].replace('R', 0)

    # famsize: 'LE3' to 1 and 'GT3' to 0
    datasets[i] = datasets[i].replace('LE3', 1)
    datasets[i] = datasets[i].replace('GT3', 0)

    # Pstatus: 'T' to 1 and 'A' to 0
    datasets[i] = datasets[i].replace('T', 1)
    datasets[i] = datasets[i].replace('A', 0)

    #% Nominal
    # Mjob/Fjob: 'teacher', 'health', 'services', 'at_home', 'other' to 0-4
    datasets[i] = datasets[i].replace('teacher', 0)
    datasets[i] = datasets[i].replace('health', 1)
    datasets[i] = datasets[i].replace('services', 2)
    datasets[i] = datasets[i].replace('at_home', 3)
    datasets[i] = datasets[i].replace( value = 4, to_replace = {'Mjob':'other','Fjob':'other'})

    # reason: 'home', 'reputation', 'course', 'other' to 0-3
    datasets[i] = datasets[i].replace('home', 0)
    datasets[i] = datasets[i].replace('reputation', 1)
    datasets[i] = datasets[i].replace('course', 2)
    datasets[i] = datasets[i].replace( value=3, to_replace={'reason':'other'})

    # guardian: 'mother', 'father', 'other' to 0-2
    datasets[i] = datasets[i].replace('mother', 0)
    datasets[i] = datasets[i].replace('father', 1)
    datasets[i] = datasets[i].replace( value=2, to_replace={'guardian':'other'})



datasets[0].to_csv('train_new.csv')  
datasets[1].to_csv('test_new.csv')  