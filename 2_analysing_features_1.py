#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_numeric(dataset, feature):

    sns.set_theme(style="whitegrid")
    
    # ax = sns.boxplot(x=feature, y="grade", data=dataset)
    # ax.title = "Boxplot of " + feature + " vs. grade"
    # png_title = "boxplot_" + feature + "_vs_grade"
    # ax.savefig("/boxplot_figs/" + png_title)\

    plt.figure()
    sns.boxplot(x=feature, y="grade", data=dataset)
    plt.title("Boxplot of " + feature + " vs. grade")
    png_title = "boxplot_" + feature + "_vs_grade"
    plt.savefig(png_title)
    print(feature,end='\r')


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

    
# %%
for feature in train:
    plot_numeric(train, feature)


