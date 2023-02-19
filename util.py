from sklearn import datasets
import numpy as np


def make_dbscan_data():
    """
    Example from:
        https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
    """
    moons, _ = datasets.make_moons(n_samples=50, noise=0.05)
    blobs, _ = datasets.make_blobs(
        n_samples=50, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25
    )
    data = np.vstack([moons, blobs])
    return data
