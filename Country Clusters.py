import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

raw_data = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/MECHINE LEARNING/KMEANS CLUSTER/3.01.Country clusters.csv')
data = raw_data.copy()
#print(data)

#CLUSTER BERDASARKAN GEOGRAFI
plt.scatter(data['Longitude'],data['Latitude'])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()

#pilih variabel x (disini variabel x nya adalah geografi)
x = data.iloc[:,1:3]
#print(x)

kmeans = KMeans(3)
kmeans.fit(x)

#KLUSTER RESULTS (PREDICTED)
identified_clusters = kmeans.fit_predict(x)
data_with_cluster = data.copy()
data_with_cluster['Cluster'] = identified_clusters
print(data_with_cluster)

plt.scatter(data_with_cluster['Longitude'], data_with_cluster['Latitude'], c=data_with_cluster['Cluster'],cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()


