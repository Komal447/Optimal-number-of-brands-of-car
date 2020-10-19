#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


# Importing the dataset
dataset = pd.read_csv(r'C:\Users\komal soni\Downloads\cars.csv')


# In[3]:


dataset


# In[4]:


#Take all the columns
X = dataset.iloc[:, :].values


# In[6]:


print(X)


# In[7]:


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans


# In[8]:


wcss = []
for i in range(1, 11):
    k_means = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    k_means.fit(X)
    wcss.append(k_means.inertia_)


# In[9]:


wcss


# In[15]:


plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# From figure we can say optimal number of cluster is 3.

# In[17]:


# Applying k-means to the cars dataset
model = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
model_predict = model.fit_predict(X)
print(model_predict )


# In[ ]:




