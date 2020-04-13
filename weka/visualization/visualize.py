import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_columns():
    df = pd.read_csv('weka/clean/clean.csv')
    return df.columns

def pair_plot():
    df = pd.read_csv('weka/clean/clean.csv')
    sns_plot = sns.pairplot(df, height=2.5)
    if (os.path.isfile('weka/static/img/pairplot1.png')):
        os.remove('weka/static/img/pairplot1.png')
    sns_plot.savefig('weka/static/img/pairplot1.png')
    return True

def xy_plot(col1, col2):
    df = pd.read_csv('weka/clean/clean.csv')
    plt.scatter(df[col1], df[col2])
    if (os.path.isfile('weka/static/img/fig.png')):
        os.remove('weka/static/img/fig.png')
    plt.savefig('weka/static/img/fig.png')
    return df