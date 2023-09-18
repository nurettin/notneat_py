from collections import defaultdict


def calculate_distance(cluster1, cluster2, distance_function):
    # Calculate the distance between two clusters using the provided distance function
    min_distance = float('inf')
    for point1 in cluster1:
        for point2 in cluster2:
            distance = distance_function(point1, point2)
            if distance < min_distance:
                min_distance = distance
    return min_distance


def agglomerative_hierarchical_clustering(data, distance_function):
    clusters = [[x] for x in data]  # Start with each integer as a single-point cluster

    while len(clusters) > 1:
        # Calculate pairwise distances between clusters
        distances = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = calculate_distance(clusters[i], clusters[j], distance_function)
                distances.append((i, j, distance))

        # Find the two closest clusters
        min_dist = min(distances, key=lambda x: x[2])
        cluster1_idx, cluster2_idx, min_distance = min_dist

        # Merge the two closest clusters
        merged_cluster = clusters[cluster1_idx] + clusters[cluster2_idx]
        del clusters[cluster2_idx]
        clusters[cluster1_idx] = merged_cluster

    # Return the final clustering result
    return clusters


def hierarchical_clustering(objects, distance):
    # Initialize each object as its own cluster
    clusters = [[obj] for obj in objects]
    distances = defaultdict(dict)

    # Compute the distances between all pairs of objects
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            distances[i][j] = distance(objects[i], objects[j])

    # Repeatedly merge the two closest clusters
    while len(clusters) > 1:
        min_distance = float('inf')
        merge_i, merge_j = -1, -1

        # Find the two clusters with the smallest distance between them
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster_i, cluster_j = clusters[i], clusters[j]
                dist = min(distances[obj_i][obj_j] for obj_i in cluster_i for obj_j in cluster_j)
                if dist < min_distance:
                    min_distance = dist
                    merge_i, merge_j = i, j

        # Merge the two clusters
        clusters[merge_i].extend(clusters[merge_j])
        del clusters[merge_j]

    return clusters[0]
