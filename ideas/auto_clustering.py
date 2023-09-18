import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.signal import find_peaks
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage, fcluster


def create_distance_matrix(objects, distance):
    return squareform(pdist(objects, distance))


def generate_coordinates(distance_matrix):
    mds = MDS(dissimilarity="precomputed", n_components=1, normalized_stress="auto")
    return mds.fit_transform(distance_matrix).flatten()


def density_line(coordinates, bins=100):
    return np.histogram(coordinates, bins=bins, density=True)[0]


def count_peaks(density):
    peaks, _ = find_peaks(density)
    return len(peaks)


def cluster(objects, num_clusters, distance_matrix):
    Z = linkage(distance_matrix, method='complete')
    labels = fcluster(Z, num_clusters, criterion='maxclust')
    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(objects[i])
    return clusters


if __name__ == "__main__":
    # Example usage
    objects = np.array([[1, 2], [5, 5], [1, 1], [5, 6], [10, 10]])  # Replace with your objects


    def distance(x, y):
        return np.linalg.norm(x - y)


    distance_matrix = create_distance_matrix(objects, distance)
    coordinates = generate_coordinates(distance_matrix)
    density = density_line(coordinates)
    number_of_clusters = count_peaks(density)
    clusters = cluster(objects, number_of_clusters, distance_matrix)
    print("Distance matrix:", distance_matrix)
    print("Coordinates:", coordinates)
    print("Density:", density)
    print("Number of clusters:", number_of_clusters)
    print("Object Clusters:", clusters)
