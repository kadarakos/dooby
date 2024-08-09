from collections import deque
from scipy.spatial import KDTree
from structures import Clustering


def find_clusters(X, *, radius=0.5, min_samples=5) -> Clustering:
    kdtree = KDTree(X)
    clustering = Clustering(X)
    processed = set()
    for i in range(kdtree.n):
        if i in processed:
            continue
        focus_point = kdtree.data[i]
        focus_ball = kdtree.query_ball_point(focus_point, radius)
        if len(focus_ball) < min_samples:
            continue
        else:
            # Perform breadth-first search the core-point
            cluster_id = len(clustering)
            to_visit = deque(focus_ball)
            while to_visit:
                idx = to_visit.popleft()
                point = kdtree.data[idx]
                neighbors = kdtree.query_ball_point(point, radius)
                if idx not in processed:
                    processed.add(idx)
                    clustering.assign(idx, cluster_id)
                    # Core-point
                    if len(neighbors) >= min_samples:
                        for n in neighbors:
                            to_visit.append(n)
    return clustering
