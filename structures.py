from dataclasses import dataclass
from typing import Union, Sequence
import numpy as np


@dataclass
class Clustering:
    X: np.ndarray
    assignment: np.ndarray

    def __init__(self, X: np.ndarray, outlier=-99):
        """
        X: 2D array of points.
        outliers: whether clustering contains outlier points.
        """
        if X.ndim != 2:
            raise ValueError("X has to be a 2D array")
        self.X = X
        self.assignment = np.zeros(X.shape[0])
        self.outlier = outlier

    @property
    def clusters(self):
        return np.unique(self.assignment)

    @property
    def n_clusters(self):
        return len(self.clusters)

    def assign(self, idx: Union[int, Sequence[int]], cluster: int) -> None:
        self.assignment[idx] = cluster

    def add_outlier(self, idx: Union[int, Sequence[int]]) -> None:
        self.assignment[idx] = self.outlier

    def get_idxs(self, cluster: int) -> np.ndarray:
        return np.where(self.assignment == cluster)[0]

    def get_points(self, cluster: int) -> np.ndarray:
        return self.X[self.get_idxs(cluster)]
