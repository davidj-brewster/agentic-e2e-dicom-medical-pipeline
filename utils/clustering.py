"""
Clustering analysis utilities.
Implements various clustering methods for anomaly detection.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ClusteringResult:
    """Container for clustering results"""
    def __init__(
        self,
        labels: np.ndarray,
        centers: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        self.labels = labels
        self.centers = centers
        self.scores = scores
        self.metrics = metrics or {}


def prepare_data(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    features: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare image data for clustering by extracting features.
    
    Args:
        image: Input image data
        mask: Optional binary mask
        features: List of features to extract
        
    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    if mask is not None:
        valid_voxels = mask > 0
    else:
        valid_voxels = np.ones_like(image, dtype=bool)
    
    features = features or ["intensity"]
    feature_list = []
    
    # Basic intensity
    if "intensity" in features:
        feature_list.append(image[valid_voxels])
    
    # Local statistics
    if "local_mean" in features:
        local_mean = ndimage.uniform_filter(image, size=3)
        feature_list.append(local_mean[valid_voxels])
    
    if "local_std" in features:
        local_std = ndimage.generic_filter(image, np.std, size=3)
        feature_list.append(local_std[valid_voxels])
    
    # Gradient magnitude
    if "gradient" in features:
        gx = ndimage.sobel(image, axis=0)
        gy = ndimage.sobel(image, axis=1)
        gz = ndimage.sobel(image, axis=2)
        gradient = np.sqrt(gx**2 + gy**2 + gz**2)
        feature_list.append(gradient[valid_voxels])
    
    # Laplacian
    if "laplacian" in features:
        laplacian = ndimage.laplace(image)
        feature_list.append(laplacian[valid_voxels])
    
    # Create feature matrix
    X = np.column_stack(feature_list)
    
    return X, features


def kmeans_clustering(
    data: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42
) -> ClusteringResult:
    """
    Perform K-means clustering.
    
    Args:
        data: Feature matrix
        n_clusters: Number of clusters
        random_state: Random seed
        
    Returns:
        Clustering result
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    # Fit K-means
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state
    )
    labels = kmeans.fit_predict(X_scaled)
    
    # Get cluster centers in original space
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Calculate silhouette scores
    from sklearn.metrics import silhouette_samples
    scores = silhouette_samples(X_scaled, labels)
    
    # Calculate metrics
    metrics = {
        "inertia": float(kmeans.inertia_),
        "silhouette_avg": float(scores.mean())
    }
    
    return ClusteringResult(labels, centers, scores, metrics)


def gmm_clustering(
    data: np.ndarray,
    n_components: int = 3,
    random_state: int = 42
) -> ClusteringResult:
    """
    Perform Gaussian Mixture Model clustering.
    
    Args:
        data: Feature matrix
        n_components: Number of components
        random_state: Random seed
        
    Returns:
        Clustering result
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    # Fit GMM
    gmm = GaussianMixture(
        n_components=n_components,
        random_state=random_state
    )
    labels = gmm.fit_predict(X_scaled)
    
    # Get component means in original space
    centers = scaler.inverse_transform(gmm.means_)
    
    # Calculate probabilities
    probs = gmm.predict_proba(X_scaled)
    scores = np.max(probs, axis=1)  # Use highest probability as score
    
    # Calculate metrics
    metrics = {
        "bic": float(gmm.bic(X_scaled)),
        "aic": float(gmm.aic(X_scaled))
    }
    
    return ClusteringResult(labels, centers, scores, metrics)


def dbscan_clustering(
    data: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5
) -> ClusteringResult:
    """
    Perform DBSCAN clustering.
    
    Args:
        data: Feature matrix
        eps: Maximum distance between samples
        min_samples: Minimum number of samples in neighborhood
        
    Returns:
        Clustering result
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    # Fit DBSCAN
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )
    labels = dbscan.fit_predict(X_scaled)
    
    # Calculate metrics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    metrics = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": float(n_noise) / len(labels)
    }
    
    return ClusteringResult(labels, None, None, metrics)


def detect_anomalies(
    image_path: Path,
    mask_path: Optional[Path] = None,
    method: str = "gmm",
    features: Optional[List[str]] = None,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Detect anomalies in image using clustering.
    
    Args:
        image_path: Path to input image
        mask_path: Optional path to mask image
        method: Clustering method ('kmeans', 'gmm', or 'dbscan')
        features: List of features to use
        **kwargs: Additional arguments for clustering method
        
    Returns:
        Tuple of (anomaly_mask, metrics)
    """
    try:
        # Load images
        image = nib.load(str(image_path))
        image_data = image.get_fdata()
        
        if mask_path:
            mask = nib.load(str(mask_path))
            mask_data = mask.get_fdata() > 0
        else:
            mask_data = None
        
        # Prepare feature matrix
        X, feature_names = prepare_data(
            image_data,
            mask_data,
            features
        )
        
        # Perform clustering
        if method == "kmeans":
            result = kmeans_clustering(X, **kwargs)
        elif method == "gmm":
            result = gmm_clustering(X, **kwargs)
        elif method == "dbscan":
            result = dbscan_clustering(X, **kwargs)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Create anomaly mask
        anomaly_mask = np.zeros_like(image_data)
        if mask_data is not None:
            valid_voxels = mask_data > 0
        else:
            valid_voxels = np.ones_like(image_data, dtype=bool)
        
        if method == "dbscan":
            # Mark noise points as anomalies
            anomaly_mask[valid_voxels] = (result.labels == -1)
        else:
            # Use score threshold for other methods
            threshold = np.percentile(result.scores, 95)  # Top 5% as anomalies
            anomaly_mask[valid_voxels] = (result.scores < threshold)
        
        # Calculate additional metrics
        metrics = result.metrics.copy()
        metrics.update({
            "anomaly_ratio": float(np.sum(anomaly_mask)) / np.sum(valid_voxels),
            "n_features": len(feature_names)
        })
        
        return anomaly_mask, metrics
        
    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        raise