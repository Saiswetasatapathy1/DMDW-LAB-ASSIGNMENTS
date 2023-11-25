import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read the dataset
data = pd.read_csv('/content/iris.csv')  # Replace 'your_dataset.csv' with the actual path to your CSV file

# Extract features for clustering
features = data[['sepal_length', 'sepal_width']]

# Specify the number of clusters (change as needed)
k = 3

# Apply K-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
data['cluster'] = kmeans.fit_predict(features)

# Visualize the clusters
plt.figure(figsize=(10, 6))

# Define colors for each cluster
colors = plt.cm.nipy_spectral(data['cluster'].astype(float) / k)

# Plot the clusters
plt.scatter(data['sepal_length'], data['sepal_width'], c=colors, edgecolor='k', label=f'Cluster {cluster + 1}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, color='red', label='Centroids')

# Additional plot configurations
plt.title('K-means Clustering')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()
