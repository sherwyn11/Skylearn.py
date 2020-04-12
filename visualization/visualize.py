import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def pair_plot():

    df = pd.read_csv('clean/clean1.csv')
    sns_plot = sns.pairplot(df, height=2.5)
    sns_plot.savefig('static/img/pairplot1.png')
    return True