import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

data = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/MECHINE LEARNING/KMEANS CLUSTER/Country clusters standardized.csv', index_col='Country')
print(data)

x_scaled = data.copy()  
x_scaled = x_scaled.drop(['Language'], axis=1) 
print(x_scaled)

sns.clustermap(x_scaled, cmap='mako')
plt.show()