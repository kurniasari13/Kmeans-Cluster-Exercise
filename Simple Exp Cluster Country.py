import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

raw_data = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/MECHINE LEARNING/KMEANS CLUSTER/Countries exercise.csv')
#print(raw_data)
data = raw_data.copy()
#print(data)

plt.scatter(data['Longitude'],data['Latitude'])
plt.xlim(-180,180)
plt.ylim(-90, 90)
plt.show()

x = data.iloc[:,1:3]
print(x)

kmenas = KMeans(3)
kmenas.fit(x)

identified_clusters = kmenas.fit_predict(x)
data_with_clusters = data.copy()
data_with_clusters['Cluster'] = identified_clusters

pd.options.display.max_rows = 999
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print(data_with_clusters.sort_values(by=['Cluster']))

plt.scatter(data['Longitude'],data['Latitude'],c=data_with_clusters['Cluster'],cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90, 90)
plt.show()

print(kmenas.inertia_)

wcss =[]
cluster_lenth = 11
for i in range(1,cluster_lenth):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

print(wcss)

number_clusters = range(1,cluster_lenth)
plt.plot(number_clusters,wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Cluster')
plt.ylabel('WCSS')
plt.show()