import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

brainFile = './Data/brainsize.csv'
brainFrame = pd.read_csv(brainFile)

print(brainFrame)


menDf = brainFrame[(brainFrame.Gender == 'Male')]
womenDf = brainFrame[(brainFrame.Gender == 'Female')]

menMeanSmarts = menDf[["PIQ", "FSIQ", "VIQ"]].mean(axis=1)
plt.scatter(menMeanSmarts, menDf["MRI_Count"])
plt.show()

womenMeanSmarts = womenDf[["PIQ", "FSIQ", "VIQ"]].mean(axis=1)
plt.scatter(womenMeanSmarts, womenDf["MRI_Count"])

plt.show()

print(brainFrame.corr(method='pearson'))

print(womenDf.corr(method='pearson') )

print(menDf.corr(method='pearson') )

wcorr = womenDf.corr()
sns.heatmap(wcorr)
plt.show()

mcorr = womenDf.corr()
sns.heatmap(mcorr)
plt.show()
