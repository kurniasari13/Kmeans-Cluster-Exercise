import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set() 
from sklearn.cluster import KMeans 

data = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/MECHINE LEARNING/KMEANS CLUSTER/3.12.Example.csv')
#print(data)

#Eksplorasi Awal
plt.scatter(data['Satisfaction'],data['Loyalty'])
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()

#Kmeans tanpa standarisasi
x = data.copy()
kmeans = KMeans(3)
kmeans.fit(x)

clusters = x.copy() 
clusters['clusters_predict'] = kmeans.fit_predict(x)

#Gafik kmeans tanpa standarisasi
plt.scatter(clusters['Satisfaction'],clusters['Loyalty'], c=clusters['clusters_predict'], cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()
#berdasarkan plot, hanya varabel satisfaction yang diperhitungkan karena weightnya besar, sehingga harus distandarisasi

#STANDARISASI
from sklearn import preprocessing
x_scaled = preprocessing.scale(x)
print(x_scaled)

# CARI JUMLAH CLUSTER DENGAN WCSS
wcss = []
for i in range(1,10):
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

print(wcss)

#PLOT OF WCSS
plt.plot(range(1,10), wcss)
plt.xlabel('Numbers of Clusters')
plt.ylabel('WCSS')
plt.show()
#hasil wcss ada 2,3,4,5 solusi

#MEMBUAT KMEANS ULANG DENGAN VARIABEL X STANDARISASI DAN WCSS SOLUSI
kmeans_new = KMeans(4)
kmeans_new.fit(x_scaled)
clusters_new = x.copy() 
clusters_new['Clusters_predict'] = kmeans_new.fit_predict(x_scaled)
print(clusters_new)

plt.scatter(clusters_new['Satisfaction'], clusters_new['Loyalty'], c=clusters_new['Clusters_predict'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()
#stranger, supporter, roamers, fans (solusinya ada 4 cluster)