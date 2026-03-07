import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting

# Load dataset
df = pd.read_csv('data/mall_customers.csv')

# Select features for clustering 
X = df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']]

# KMeans with the 3 features
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# 2. Setup for 3D Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 3. Plot the clusters
# We assign X, Y, and Z to the three features
ax.scatter(
    df['Annual Income (k$)'], 
    df['Spending Score (1-100)'], 
    df['Age'], 
    c=df['Cluster'], 
    cmap='viridis',
    s=60
)

# 4. Plot the centroids
# Centroids will now have 3 coordinates: [Income, Spending, Age]
centroids = kmeans.cluster_centers_
ax.scatter(
    centroids[:, 0], 
    centroids[:, 1], 
    centroids[:, 2], 
    marker='X', 
    s=100, 
    c='red', 
    label='Centroids'
)

# Labeling the axes
ax.set_title("3D Customer Segmentation using K-means")
ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Spending Score (1-100)')
ax.set_zlabel('Age')

plt.legend()
plt.show()