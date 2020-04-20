import pandas as pd
import matplotlib
import asyncio
import time
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
<<<<<<< Updated upstream
import csv


def get_columns():
    df = pd.read_csv("weka/clean/clean.csv")
    return df.columns

def pair_plot():
    df = pd.read_csv("weka/clean/clean.csv")
    start=time.time()
    sns_plot = sns.pairplot(df, height=2.5)
    mid=time.time()
    sns_plot.savefig("weka/static/img/pairplot1.png")
    end=time.time()
    print(f'{mid-start} {end-mid} {sns_plot}')
    return True


def xy_plot(col1, col2):
    df = pd.read_csv("weka/clean/clean.csv")
    return df

def hist_plot(df,feature_x):
    # df=df.sort_values([feature_x], axis=0, ascending=True, inplace=True) 
    x= df[feature_x]
    x.to_csv("weka/visualization/col.csv",mode="w", index=False,header=['price'])
    with open("weka/visualization/col.csv", 'r') as filehandle:
        lines = filehandle.readlines()
        lines[-1]=lines[-1].strip()
    with open("weka/visualization/col.csv", 'w') as csvfile:
        for i in lines:
            csvfile.write(i)  
    return True
=======

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
>>>>>>> Stashed changes
