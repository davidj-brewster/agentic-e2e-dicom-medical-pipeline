"""
Utilities for neuroimaging analysis operations.
Provides async wrappers for FreeSurfer and FSL analysis tools.
"""
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
from scipy.ndimage import label

from utils.clustering import (
    ClusteringResult,
    detect_anomalies,
    gmm_clustering,
    kmeans_clustering,
    prepare_data
)
from utils.neuroimaging import run_command, run_freesurfer_command, run_fsl_command

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """Results from FreeSurfer/FSL segmentation"""
    region_name: str
    mask_path: Path
    volume: float
    center_of_mass: Tuple[float, ...]
    voxel_count: int
    label_value: int


@dataclass
class ClusterMetrics:
    """Metrics for identified clusters"""
    cluster_id: int
    size: int
    mean_intensity: float
    std_intensity: float
    center: Tuple[float, ...]
    bounding_box: Tuple[Tuple[int, ...], Tuple[int, ...]]
    outlier_score: float


async def run_first_segmentation(
    t1_path: Path,
    output_dir: Path,
    regions_of_interest: Dict[str, int]
) -> Tuple[bool, Optional[str], Dict[str, SegmentationResult]]:
    """
    Run FSL FIRST segmentation.
    
    Args:
        t1_path: Path to T1 image
        output_dir: Output directory
        regions_of_interest: Dictionary of region names and label values
        
    Returns:
        Tuple of (success, error_message, results)
    """
    try:
        # Run FIRST segmentation
        success, stdout, stderr = await run_fsl_command(
            "run_first_all",
            [
                "-i", str(t1_path),
                "-o", str(output_dir / "first"),
                "-d",  # debug mode
                "-v"   # verbose output
            ]
        )
        
        if not success:
            return False, f"FIRST segmentation failed: {stderr}", {}
            
        # Process results
        results = {}
        for region_name, label_value in regions_of_interest.items():
            # Load segmentation result
            seg_path = output_dir / f"first_all_{region_name}.nii.gz"
            if not seg_path.exists():
                logger.warning(f"Segmentation not found for region: {region_name}")
                continue
            
            img = nib.load(str(seg_path))
            data = img.get_fdata()
            
            # Calculate metrics
            voxel_count = np.sum(data > 0)
            volume = voxel_count * np.prod(img.header.get_zooms())
            com = np.mean(np.where(data > 0), axis=1)
            
            results[region_name] = SegmentationResult(
                region_name=region_name,
                mask_path=seg_path,
                volume=float(volume),
                center_of_mass=tuple(float(x) for x in com),
                voxel_count=int(voxel_count),
                label_value=label_value
            )
        
        return True, None, results
        
    except Exception as e:
        return False, f"Error during FIRST segmentation: {str(e)}", {}


async def generate_binary_masks(
    segmentation_results: Dict[str, SegmentationResult],
    output_dir: Path
) -> Tuple[bool, Optional[str], Dict[str, Path]]:
    """
    Generate binary masks for segmented regions.
    
    Args:
        segmentation_results: Dictionary of segmentation results
        output_dir: Output directory
        
    Returns:
        Tuple of (success, error_message, mask_paths)
    """
    try:
        binary_masks = {}
        
        for region_name, result in segmentation_results.items():
            mask_path = output_dir / f"{region_name}_mask.nii.gz"
            
            success, stdout, stderr = await run_fsl_command(
                "fslmaths",
                [
                    str(result.mask_path),
                    "-bin",
                    str(mask_path)
                ]
            )
            
            if not success:
                return False, f"Binary mask generation failed for {region_name}: {stderr}", {}
            
            binary_masks[region_name] = mask_path
        
        return True, None, binary_masks
        
    except Exception as e:
        return False, f"Error generating binary masks: {str(e)}", {}


async def analyze_intensity_clusters(
    flair_path: Path,
    mask_path: Path,
    region_name: str,
    method: str = "gmm",
    features: Optional[List[str]] = None,
    **kwargs
) -> Tuple[bool, Optional[str], List[ClusterMetrics]]:
    """
    Perform intensity-based clustering analysis within masked region.
    
    Args:
        flair_path: Path to FLAIR image
        mask_path: Path to binary mask
        region_name: Name of the region
        method: Clustering method ('kmeans', 'gmm', or 'dbscan')
        features: List of features to use for clustering
        **kwargs: Additional arguments for clustering method
        
    Returns:
        Tuple of (success, error_message, cluster_metrics)
    """
    try:
        # Load images
        flair_img = nib.load(str(flair_path))
        mask_img = nib.load(str(mask_path))
        
        flair_data = flair_img.get_fdata()
        mask_data = mask_img.get_fdata() > 0
        
        # Prepare feature matrix
        X, feature_names = prepare_data(
            flair_data,
            mask_data,
            features or ["intensity", "local_mean", "gradient"]
        )
        
        if len(X) == 0:
            return True, None, []
        
        # Perform clustering
        if method == "kmeans":
            result = kmeans_clustering(X, **kwargs)
        elif method == "gmm":
            result = gmm_clustering(X, **kwargs)
        else:  # dbscan
            from utils.clustering import dbscan_clustering
            result = dbscan_clustering(X, **kwargs)
        
        # Convert to cluster metrics
        cluster_metrics = []
        unique_labels = np.unique(result.labels)
        coordinates = np.array(np.where(mask_data)).T
        
        for label in unique_labels:
            if label == -1:  # Skip noise points for DBSCAN
                continue
            
            # Get cluster points
            cluster_mask = result.labels == label
            cluster_intensities = flair_data[mask_data][cluster_mask]
            cluster_coords = coordinates[cluster_mask]
            
            # Calculate metrics
            mean_intensity = float(np.mean(cluster_intensities))
            std_intensity = float(np.std(cluster_intensities))
            center = tuple(float(x) for x in np.mean(cluster_coords, axis=0))
            
            # Calculate bounding box
            mins = tuple(int(x) for x in np.min(cluster_coords, axis=0))
            maxs = tuple(int(x) for x in np.max(cluster_coords, axis=0))
            
            # Use clustering scores if available
            outlier_score = float(
                1.0 - result.scores[cluster_mask].mean()
                if result.scores is not None else 0.0
            )
            
            metrics = ClusterMetrics(
                cluster_id=int(label),
                size=int(len(cluster_intensities)),
                mean_intensity=mean_intensity,
                std_intensity=std_intensity,
                center=center,
                bounding_box=(mins, maxs),
                outlier_score=outlier_score
            )
            
            cluster_metrics.append(metrics)
        
        return True, None, cluster_metrics
        
    except Exception as e:
        return False, f"Error during clustering analysis: {str(e)}", []


def identify_anomalies(
    cluster_metrics: List[ClusterMetrics],
    threshold: float = 2.0
) -> List[ClusterMetrics]:
    """
    Identify potentially anomalous clusters.
    
    Args:
        cluster_metrics: List of cluster metrics
        threshold: Outlier score threshold
        
    Returns:
        List of anomalous clusters
    """
    # Filter clusters based on outlier score
    anomalies = [
        cluster for cluster in cluster_metrics
        if cluster.outlier_score > threshold
    ]
    
    return sorted(
        anomalies,
        key=lambda x: x.outlier_score,
        reverse=True
    )


async def validate_segmentation(
    segmentation_path: Path,
    reference_path: Path,
    min_volume: float = 100.0,
    max_volume: float = 100000.0
) -> Tuple[bool, Optional[str], Dict[str, float]]:
    """
    Validate segmentation results.
    
    Args:
        segmentation_path: Path to segmentation
        reference_path: Path to reference image
        min_volume: Minimum expected volume in mm³
        max_volume: Maximum expected volume in mm³
        
    Returns:
        Tuple of (valid, error_message, metrics)
    """
    try:
        # Load images
        seg_img = nib.load(str(segmentation_path))
        ref_img = nib.load(str(reference_path))
        
        # Check dimensions match
        if seg_img.shape != ref_img.shape:
            return False, "Segmentation dimensions do not match reference", {}
        
        # Calculate metrics
        seg_data = seg_img.get_fdata()
        voxel_volume = np.prod(seg_img.header.get_zooms())
        volume = np.sum(seg_data > 0) * voxel_volume
        
        # Validate volume
        if volume < min_volume:
            return False, f"Segmentation volume ({volume:.1f} mm³) below minimum ({min_volume} mm³)", {}
        if volume > max_volume:
            return False, f"Segmentation volume ({volume:.1f} mm³) above maximum ({max_volume} mm³)", {}
        
        metrics = {
            "volume_mm3": float(volume),
            "voxel_count": int(np.sum(seg_data > 0)),
            "max_value": float(np.max(seg_data)),
            "mean_value": float(np.mean(seg_data[seg_data > 0]))
        }
        
        return True, None, metrics
        
    except Exception as e:
        return False, f"Error validating segmentation: {str(e)}", {}