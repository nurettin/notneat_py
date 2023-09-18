import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score

def calculate_davies_bouldin(X, labels):
    """
    Calculate the Davies-Bouldin index for a clustering solution.
    
    Parameters:
    - X: Data matrix (n_samples, n_features).
    - labels: Cluster labels for each data point.

    Returns:
    - Davies-Bouldin index for the clustering solution.
    """
    n_clusters = len(np.unique(labels))
    cluster_centers = [X[labels == i].mean(axis=0) for i in range(n_clusters)]
    distances = pairwise_distances(X, cluster_centers, metric='euclidean')
    max_similarity = np.zeros(n_clusters)

    for i in range(n_clusters):
        other_clusters = np.arange(n_clusters)[np.arange(n_clusters) != i]
        avg_distance = np.mean(distances[:, other_clusters])
        max_similarity[i] = 1 / avg_distance

    return np.mean(max_similarity)

def iterative_kmeans(X, max_clusters):
    """
    Perform iterative k-means clustering with Davies-Bouldin index evaluation.

    Parameters:
    - X: Data matrix (n_samples, n_features).
    - max_clusters: Maximum number of clusters to consider.

    Returns:
    - Optimal number of clusters based on Davies-Bouldin index.
    """
    best_davies_bouldin = float('inf')
    optimal_k = 2  # Start with k=2
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(X)
        db_index = davies_bouldin_score(X, labels)

        if db_index < best_davies_bouldin:
            best_davies_bouldin = db_index
            optimal_k = k

    return optimal_k

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate synthetic data for demonstration purposes
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Set the maximum number of clusters you want to consider
max_clusters = 10

# Find the optimal number of clusters using Davies-Bouldin index and get the best KMeans model
optimal_k, best_kmeans = iterative_kmeans(X, max_clusters)

print("Optimal number of clusters:", optimal_k)

# Fit the best KMeans model on the data
best_kmeans.fit(X)
cluster_labels = best_kmeans.labels_

# Plot the data points with cluster assignments
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
plt.scatter(best_kmeans.cluster_centers_[:, 0], best_kmeans.cluster_centers_[:, 1], s=200, marker='X', c='red', label='Cluster Centers')
plt.legend()
plt.title("K-Means Clustering")
plt.show()

