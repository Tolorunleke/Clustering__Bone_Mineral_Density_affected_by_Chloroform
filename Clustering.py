#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy import stats
get_ipython().run_line_magic('matplotlib', ' inline')
import sklearn

import plotly.express as px
import scipy.cluster.hierarchy as sch
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score as ss
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from scipy.stats import shapiro
import scipy.stats as stats

import os
os.environ["OMP_NUM_THREADS"]= '1'

print("pandas version: {}".format(pd.__version__))
print("numpy version: {}".format(np.__version__))
print("seaborn version: {}".format(sns.__version__))


# In[2]:


bone = pd.read_excel('Jun_Liang.xlsx')
bone.head(4)


# In[3]:


bone.tail(5)


# In[4]:


#check for information about data set
bone.info()


# In[5]:


bone.describe()


# In[6]:


bone.isna().values.any()


# In[7]:


#columns with null values in our data
bone.columns[bone.isna().any()].tolist()


# In[8]:


#ldl column has total of 1313 rows with null values and the total dataset is about 2553
bone['LDL'].isna().sum()


# In[9]:


bone = bone.drop(columns =['LDL'], axis = 1)
#has exess null values and wont be useful for the analysis its better off dropping


# In[10]:


#using simple inputer from sklearn library


# In[11]:


#sorting missing values
impute = SimpleImputer(missing_values=np.nan, strategy= 'mean')
impute = impute.fit(bone[[
    'Ratio of family income to poverty','Weight','Height','Waist',
 'Systolic Pressure','Diastolic Pressure','BMI','Fasting Blood Glucose',
 'Glycosylated Haemoglobin','Albumin',
 'ALT', 'AST','calcium','Cholesterol','Triglyceride',
 'Creatinine','Glucose','Serum Iron','Serum Phosphorus',
 'Total Bilirubin','Uric Aacid','Sodium','Potassium']])

bone[[
 'Ratio of family income to poverty','Weight','Height','Waist',
 'Systolic Pressure','Diastolic Pressure','BMI','Fasting Blood Glucose',
 'Glycosylated Haemoglobin','Albumin',
 'ALT', 'AST','calcium','Cholesterol','Triglyceride',
 'Creatinine','Glucose','Serum Iron','Serum Phosphorus',
 'Total Bilirubin','Uric Aacid','Sodium','Potassium']] = impute.transform(bone[[
 'Ratio of family income to poverty','Weight','Height','Waist',
 'Systolic Pressure','Diastolic Pressure','BMI','Fasting Blood Glucose',
 'Glycosylated Haemoglobin','Albumin',
 'ALT', 'AST','calcium','Cholesterol','Triglyceride',
 'Creatinine','Glucose','Serum Iron','Serum Phosphorus',
 'Total Bilirubin','Uric Aacid','Sodium','Potassium']])


# In[12]:


#since education is a categorical data will be using the most feature in the class
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
imputer = imputer.fit(bone[['Educational level']])
bone['Educational level']= imputer.transform(bone[['Educational level']])

#cross checking if our missing values has been sorted
bone.isna().values.any()


# In[13]:


#having clean our data we will go further to select most relevant features as we will not be able to use all 62 features left
bone_scan = bone[['Chloroform','TotalFemurBMD','FemoralNeckBMD','TrochanterBMD','IntertrochanterBMD',
 'WardsTriangleBMD','TotalSpineBMD']]


# In[14]:


#visualisation of the age distribution
bone['Sex'].value_counts().plot.pie(explode= [0.1,0], autopct='%3.1f%%'
                                   ,shadow = True, legend=True, startangle = 45)
plt.title('Gender Distribution', size =14)
plt.show()


# In[15]:


#correlation plot
plt.figure(figsize=(10,5))
sns.heatmap(bone_scan.corr(), annot = True, cmap = 'Oranges', fmt =".2f")
plt.title('Correlation of selected columns')
plt.show


# In[16]:


#relationship between TotalFemur and Sex
sns.histplot(data = bone, x = 'TotalFemurBMD', hue = 'Sex', kde = True, palette = "Spectral")


# In[17]:


#check for outliers
plt.figure(figsize = (15,10))
sns.boxplot(bone_scan)


# In[18]:


#shapiru test of normality
def test_normality(column_name, data):
    column_data = data[column_name]
    stat, p_value = shapiro(column_data)
    alpha = 0.05  # Significance level

    if p_value > alpha:
        print(f"The data in column '{column_name}' is normally distributed")
    else:
        print(f"The data in column '{column_name}' is not normally distributed")
        
#test_normality for column 'Chloroform'
test_normality('Chloroform', bone)


# In[52]:


#using the stat function from scipy
stats.probplot(bone['Chloroform'], plot = plt)
plt.show()

#this is to treat the ouliers using Zcore as this its only chloroform that is not normally distributed, non gussian 
# In[20]:


#standardize features by removing the mean and scaling to unit variance, mean o stard deviation 1 with standard scaler

# Using zscore to remove outliers since its just a column that is not normally distributed which is the chloroform column
z_scores = pd.DataFrame(StandardScaler().fit_transform(bone_scan), columns=bone_scan.columns)
data_no_outliers_z = bone_scan[(z_scores < 3).all(axis=1)]


# In[21]:


plt.figure(figsize = (15,10))
sns.boxplot(data = data_no_outliers_z)


# In[22]:


sns.pairplot(data_no_outliers_z)


# In[23]:


plt.figure(figsize = (70, 30))
data_no_outliers_z.hist()
plt.show()


# In[24]:


#after removing outliers from the data set
sns.scatterplot(data = data_no_outliers_z, x= 'TotalFemurBMD', y = 'Chloroform')
#the higher the presence of chlorofoam the lesser the patient with with healthy bone and patient with good bone density above 1.3 and 1.4 has zero or very low chloroform


# In[25]:


#using the variance threshold to to see if there is any feature that needs removing from the preselected dataset
selector = VarianceThreshold(threshold = 0)
X = selector.fit_transform(data_no_outliers_z)

print(f'{data_no_outliers_z.shape[1] - X.shape[1]} Number of Features Removed {X.shape[1]} Features left')


# In[26]:


#selecting the most important feature for measuring bone density in human body from the data set, we would be using 6 features
#since the feature selector used has not been able to reduce features in our data set would be using just the below selected features

Bknn = data_no_outliers_z[['Chloroform','TotalFemurBMD']]


# In[27]:


# using the standard scaler to set our mean of zero and standard deviation of 1
scaler = StandardScaler()
X = scaler.fit_transform(Bknn)


# In[28]:


#using the kmeans function to determine the k and building clusters afterwards

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init= 'k-means++',n_init = 5, random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.title("The elbow Method")
plt.xlabel('The number of Cluters')
plt.ylabel('WCSS')
plt.plot(range(1,11), wcss, marker ='o')


# In[53]:


#we would be using 5 clusters even if it cant be vividly seen but its portraing 4 and 5
#with k =5
kmeans = KMeans(n_clusters = 3, init = 'k-means++', n_init = 3, random_state = 42)
y_kmeans = kmeans.fit_predict(X)

#see clusters of each patients to to their bone type scan and chloroform measur
Bknn["k-means_cluster"] = y_kmeans


# In[30]:


# test efficiency of the cluster using silhouette
score = ss(X, y_kmeans)
score


# In[31]:


# Assuming X is your data, and y_kmeans is the cluster labels

# Create a DataFrame with your data and cluster labels
df = pd.DataFrame({'X1': X[:, 0], 'X2': X[:, 1], 'Cluster': y_kmeans})

# Set the size of the plot
plt.figure(figsize=(8, 8))

# Use Seaborn's scatterplot to create the scatter plot
sns.scatterplot(data=df, x='X1', y='X2', hue='Cluster', palette= 'viridis', s=100)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker= 'X', color = 'red')
# Add labels and legend
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('K-Means Clustering')
plt.legend(title='Clusters', loc='upper right')

# Show the plot
plt.show()


# In[32]:





# # Hierarchical Clustering

# In[33]:


#use the hierarchy function renamed as sch to draw a dendrogram
plt.figure(figsize = (15,3))
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Patients')
plt.ylabel('Euclidean distance')
plt.show()


# In[34]:


#agglomerative clustering
hc = AgglomerativeClustering(n_clusters = 3, metric = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)


# In[35]:


#check for number of patient belonging to each clusters
clusters = pd.DataFrame(y_hc)
clusters.value_counts()


# In[36]:


df = pd.DataFrame({'X1': X[:, 0], 'X2': X[:, 1], 'Cluster': y_hc})

# Set the size of the plot
plt.figure(figsize=(8, 8))

sns.scatterplot(data=df, x='X1', y='X2', hue='Cluster', palette= 'viridis', s=100)

plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Hierarchical Clustering')
plt.legend(title='Clusters', loc='upper right')
sns.scatterplot(data=df, x='X1', y='X2', hue='Cluster', palette= 'viridis', s=100)


# In[37]:


#running silhouette test to check the score of the 
score = ss(X, y_hc)
score


# ## add all clusters to the Bknn dataframe

# In[38]:


Bknn['K_Means_cluster'] = y_kmeans
Bknn['Agglo_cluster'] = y_hc
#viewed the cluseters after making the evaluation
Bknn


# In[39]:


#grouped data for k-means
grouped_data_kmeans = Bknn.groupby('K_Means_cluster')
grouped_kmeans_cluster = grouped_data_kmeans.mean().drop(columns = 'Agglo_cluster', axis = 1)
grouped_kmeans_cluster


# ## DBScaN

# In[40]:


#using the Nearest neighbors function to determine our epsilon
neighbours = NearestNeighbors(n_neighbors = 2)
distance, indices = neighbours.fit(X).kneighbors(X)

distance = distance[:,1]
distance = np.sort(distance, axis =0)
plt.plot(distance)


# In[41]:


#using DBScan

dbscan = DBSCAN(eps = 0.13, min_samples =20)
y_dbscan = dbscan.fit_predict(X)


# In[42]:


plt.figure(figsize=(8,8))
plt.scatter(X[y_dbscan == 0, 0], X[y_dbscan == 0,1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_dbscan == 1, 0], X[y_dbscan == 1,1], s = 100, c = 'yellow', label = 'Cluster 2')
plt.scatter(X[y_dbscan == 2, 0], X[y_dbscan == 2,1], s = 100, c = 'black', label = 'Cluster 3')
plt.scatter(X[y_dbscan == 3, 0], X[y_dbscan == 3,1], s = 100, c = 'green',label = 'Cluster 4')
plt.scatter(X[y_dbscan == 4, 0], X[y_dbscan == 4,1], s = 100, c = 'cyan', label = 'Cluster 5')
plt.scatter(X[y_dbscan == 5, 0], X[y_dbscan == 5,1], s = 100, c = 'brown', label = 'cluster 6')
plt.scatter(X[y_dbscan == 6,0], X[y_dbscan ==6,1], s= 100, c = 'magenta', label = 'Noise')


# In[43]:


#view clusters and add to the main
Bknn_copy = Bknn.copy()
Bknn_copy.loc[:, 'DB_Clusters'] = y_dbscan
Bknn_copy['DB_Clusters'].value_counts()


# In[44]:


score = ss(X, y_dbscan)
score


# # clustering of multiple class

# In[45]:


#using the whole selected bonescan and chloroform. we already initialized our scaler
#define X
scaler = StandardScaler()
X = scaler.fit_transform(data_no_outliers_z)


# In[46]:


wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init= 'k-means++',n_init = 5, random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.title("The elbow Method")
plt.xlabel('The number of Cluters')
plt.ylabel('WCSS')
plt.plot(range(1,11), wcss, marker = 'o')


# In[47]:


k_values = range(2, 11)
silhouette_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init = 10, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = ss(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Find the optimal k value
optimal_k = k_values[np.argmax(silhouette_scores)]

# Print the silhouette scores and the optimal k value
for k, score in zip(k_values, silhouette_scores):
    print(f"Silhouette score for k={k}: {score:.4f}")

print("Optimal k value:", optimal_k)


# In[48]:


#we have determine our k to be 2 from silhouette
kmeans = KMeans(n_clusters = 2, init = 'k-means++', n_init = 3, random_state = 42)
y_kmeans = kmeans.fit_predict(X)


# In[49]:


import plotly.express as px
plot = pd.DataFrame({'X1': X[:, 0], 'X2': X[:, 1], 'Cluster': y_kmeans})

# Create an interactive scatter plot using Plotly Express with the Viridis color palette
fig = px.scatter(plot, x='X1', y='X2', color='Cluster', size_max=100, color_continuous_scale='Viridis')

# Update layout
fig.update_layout(
    title='K-Means Clustering',
    xaxis_title='X-axis Label',
    yaxis_title='Y-axis Label',
    legend_title='Clusters',
    showlegend=True
)

# Show the plot
fig.show()


# In[50]:


data_no_outliers_z['K-Clusters'] = y_kmeans
grouped_kmeans_ = data_no_outliers_z.groupby('K-Clusters')
grouped_kmeans_ = grouped_kmeans_.mean()


# In[51]:


grouped_kmeans_


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




