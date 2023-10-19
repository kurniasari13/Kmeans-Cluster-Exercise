import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set()
from sklearn.cluster import KMeans 

data = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/MECHINE LEARNING/KMEANS CLUSTER/iris dataset.csv')
#print(data)

plt.scatter(data['sepal_length'], data['sepal_width'])
plt.xlabel('Length of Sepal')
plt.ylabel('Width of Sepal')
#plt.show()

x = data.copy() 
kmeans = KMeans(2)
kmeans.fit(x)

clusters = data.copy() 
clusters['cluster_pred'] = kmeans.fit_predict(x)

plt.scatter(clusters['sepal_length'], clusters['sepal_width'], c=clusters['cluster_pred'],cmap='rainbow')
plt.xlabel('Length of Sepal')
plt.ylabel('Width of Sepal')
#plt.show()

from sklearn import preprocessing 
x_scaled = preprocessing.scale(x) 
print(x_scaled)

kmeans_scaled = KMeans(3)
kmeans_scaled.fit(x_scaled)
clusters_scaled = data.copy()
clusters_scaled['cluster_pred'] = kmeans_scaled.fit_predict(x_scaled) 

plt.scatter(clusters_scaled['sepal_length'], clusters_scaled['sepal_width'], c=clusters_scaled['cluster_pred'],cmap='rainbow')
plt.xlabel('Length of Sepal')
plt.ylabel('Width of Sepal')
plt.show()

wcss =[]
cl_num = 10 
for i in range(1,cl_num):
    kmeans= KMeans(i)
    kmeans.fit(x_scaled)
    wcss_iter = kmeans.inertia_ 
    wcss.append(wcss_iter) 

print(wcss) 

number_clusters = range(1,cl_num)
plt.plot(number_clusters, wcss) 
plt.title('The Elbow Method')
plt.xlabel('Number Of Clusters') 
plt.ylabel('Within Cluster Sum of Square')
plt.show()

#2 ATAU 3 ATAU 5 CLUSTER SOLUTION BASE ON ELBOW METHOD

kmeans_new = KMeans(3)
kmeans_new.fit(x_scaled)
clusters_new = data.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled) 

plt.scatter(clusters_new['sepal_length'], clusters_new['sepal_width'], c=clusters_new['cluster_pred'],cmap='rainbow')
plt.xlabel('Length of Sepal')
plt.ylabel('Width of Sepal')
plt.show()
#SEPERTINYA CLUSTER YANG PALING TEPAT BERJUMLAH 3 CLUSTER

#PERBANDINGAN DENGAN REAL DATA
real_data = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/MECHINE LEARNING/KMEANS CLUSTER/iris with answers.csv')
#print(real_data)

print(real_data['species'].unique())
real_data['species'] = real_data['species'].map({'setosa':0, 'versicolor':1, 'virginica':2}) 
#print(real_data)

#SEPAL
plt.scatter(real_data['sepal_length'], real_data['sepal_width'], c=real_data['species'],cmap='rainbow')
plt.xlabel('Length of Sepal')
plt.ylabel('Width of Sepal')
plt.show()

#PETAL
plt.scatter(real_data['petal_length'], real_data['petal_width'], c=real_data['species'],cmap='rainbow')
plt.xlabel('Length of Petal')
plt.ylabel('Width of Petal')
plt.show()

#KMEANS SEPAL VS PETAL
plt.scatter(clusters_new['sepal_length'], clusters_new['sepal_width'], c=clusters_new['cluster_pred'],cmap='rainbow')
plt.xlabel('Length of Sepal')
plt.ylabel('Width of Sepal')
plt.show()

plt.scatter(clusters_new['petal_length'], clusters_new['petal_width'], c=clusters_new['cluster_pred'],cmap='rainbow')
plt.xlabel('Length of Petal')
plt.ylabel('Width of Petal')
plt.show()

#NOTES: ELBOW METHOD TIDAK SELALU TEPAT, KMEANS SANGAT BERGUNA SAAT KITA SUDAH TAU JUMLAH CLUSTER, DAN DATA BIOLOGI KURANG BAGUS DENGAN KMEANS