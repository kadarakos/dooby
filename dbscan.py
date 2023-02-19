"""https://www.reneshbedre.com/blog/dbscan-python.html"""
from scipy.spatial import KDTree
from typing import Tuple
from collections import deque
from structures import Cluster, Clustering


def find_clusters(X, *, eps=0.5, min_samples=5) -> Tuple[Clustering, Cluster]:
    """
    Runs the DBSCAN algorithm to find a clustering of X.
    Returns a Clustering, which is a List[Set[int]] of
    all clusters and a Cluster which is a Set[int] of points
    identified as outliers (not belonging to any Cluster).
    """
    kdtree = KDTree(X)
    clusters = []
    outliers = set()
    processed = set()
    outliers = set(range(kdtree.n))
    for i in range(kdtree.n):
        if i in processed:
            continue
        focus_point = kdtree.data[i]
        focus_ball = kdtree.query_ball_point(focus_point, eps)
        # Maybe an outlier, maybe a core-point
        if len(focus_ball) < min_samples:
            continue
        # Found a core-point (neighbors >= min_samples)
        else:
            # Perform breadth-first search from the core-point
            core_points = set()
            boundary_points = set()
            to_visit = deque(focus_ball)
            while to_visit:
                idx = to_visit.popleft()
                point = kdtree.data[idx]
                neighbors = kdtree.query_ball_point(point, eps)
                # Add core-points to cluster and for further exploration.
                if idx not in processed:
                    processed.add(idx)
                    outliers.remove(idx)
                    if len(neighbors) >= min_samples:
                        core_points.add(idx)
                        for n in neighbors:
                            to_visit.append(n)
                    # Add to boundary points otherwise.
                    else:
                        boundary_points.add(idx)
            # Add all boundary points to the cluster.
            cluster = Cluster(core_points, boundary_points)
            clusters.append(cluster)
            processed.union(cluster.points)
    return Clustering(clusters, outliers)
