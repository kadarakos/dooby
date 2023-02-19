import numpy as np

from dataclasses import dataclass
from typing import Set, List


@dataclass
class Cluster:
    internal_points: Set[int]
    boundary_points: Set[int]

    def __post_init__(self):
        self.points = set(self.internal_points).union(self.boundary_points)

    def __len__(self):
        return len(self.internal_points) + len(self.boundary_points)

    def __contains__(self, element):
        internal = element in self.internal_points
        boundary = element in self.boundary_points
        return internal or boundary


@dataclass
class Clustering:
    clusters: List[Cluster]
    outliers: Set[int]

    def __post_init__(self):
        self.points = set()
        for cluster in self.clusters:
            self.points = self.points.union(cluster.points)
        self.points = self.points.union(self.outliers)

    @property
    def has_outliers(self) -> bool:
        return bool(self.outliers)

    @property
    def n_clusters(self) -> int:
        return len(self.clusters) + int(self.has_outliers)

    @property
    def n_points(self) -> int:
        return len(self.points)

    def to_labels(self) -> List[int]:
        labels = np.empty(self.n_points)
        for label, cluster in enumerate(self.clusters):
            for idx in cluster.points:
                labels[idx] = label
        if self.has_outliers:
            label += 1
            for idx in self.outliers:
                labels[idx] = label
        return labels
