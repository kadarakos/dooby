from dataclasses import dataclass
from typing import Union, Sequence, Optional
from scipy.spatial import distance_matrix
import numpy as np


@dataclass
class Clustering:
    X: np.ndarray
    _assignment: np.ndarray

    def __init__(
        self,
        X: np.ndarray,
        assignment: Optional[np.ndarray] = None,
        centroids: Optional[np.ndarray] = None,
    ):
        """
        X: 2D array of points.

        Assings each point to a cluster, the unassigned ones have
        value -1.
        """
        if X.ndim != 2:
            raise ValueError("X has to be a 2D array")
        self.X = X
        if assignment is None:
            self._assignment = np.zeros(X.shape[0]) - 1
        else:
            self._assignment = assignment
        if centroids is not None:
            self.centroids = centroids

    @property
    def clusters(self):
        return np.unique(self.assignment)

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, items):
        return self.clusters[items]

    @property
    def assignment(self):
        return self._assignment

    def assign(self, idx: Union[int, Sequence[int]], cluster: int) -> None:
        self._assignment[idx] = cluster

    def get_idxs(self, cluster: int) -> np.ndarray:
        return np.where(self._assignment == cluster)[0]

    def get_points(self, cluster: int) -> np.ndarray:
        return self.X[self.get_idxs(cluster)]

    def to_disk(self, path: str):
        np.savez_compressed(path, X=self.X, assignment=self.assignment)

    @staticmethod
    def from_disk(path: str):
        data = np.load(path)
        X = data["X"]
        assignment = data["assignment"]
        clustering = Clustering(X, assignment)
        return clustering


@dataclass
class Centroids:
    C: np.ndarray

    def __init__(self, C: np.ndarray):
        self.C = C

    def __len__(self):
        return self.C.shape[0]

    @property
    def d(self):
        return self.C.shape[1]

    def __getitem__(self, items):
        return self.C[items]

    def assign(self, X: np.ndarray) -> np.ndarray:
        distances = distance_matrix(X, self.C)
        return np.argmin(distances, axis=1)
