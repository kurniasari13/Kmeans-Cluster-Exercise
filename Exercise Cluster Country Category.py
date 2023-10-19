import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

raw_data = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/MECHINE LEARNING/KMEANS CLUSTER/Categorical.csv')
data_mapped = raw_data.copy()
#print(data_mapped.describe(include='all'))
print(data_mapped['continent'].unique())

data_mapped['continent'] = data_mapped['continent'].map({'North America':0,'Europe':1,'Asia':2, 'Africa':3,'South America':4,'Oceania':5,'Seven seas (open ocean)':6, 'Antarctica':7})
print(data_mapped)

x = data_mapped.iloc[:,3:4]
kmeans = KMeans(8)
kmeans.fit(x)

identified_clusters = kmeans.fit_predict(x)
data_with_cluster = data_mapped.copy()
data_with_cluster['Cluster'] = identified_clusters
print(data_with_cluster)

plt.scatter(data_with_cluster['Longitude'], data_with_cluster['Latitude'], c= data_with_cluster['Cluster'], cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()

