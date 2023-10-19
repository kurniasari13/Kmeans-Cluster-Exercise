import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

raw_data = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/MECHINE LEARNING/KMEANS CLUSTER/3.01.Country clusters.csv')
data_mapped = raw_data.copy()
data_mapped['Language'] = data_mapped['Language'].map({'English':0,'French':1,'German':2})
#print(data_mapped)

x = data_mapped.iloc[:,1:4]
#print(x)
kmeans = KMeans(2)
kmeans.fit(x)

identified_clusters = kmeans.fit_predict(x)
data_with_cluster = data_mapped.copy()
data_with_cluster['Cluster_Language'] = identified_clusters
print(data_with_cluster)

plt.scatter(data_with_cluster['Longitude'], data_with_cluster['Latitude'], c=data_with_cluster['Cluster_Language'], cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()

# WCSS RIGHT NOW
print(kmeans.inertia_)

# GRAFIK ELBOW WCSS
wcss=[]

for i in range(1,7):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

print(wcss)

number_clusters = range(1,7)
plt.plot(number_clusters,wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Cluster')
plt.ylabel('Within-cluster Sum of Square')
plt.show()