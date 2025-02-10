"""
Test data generation script for neuroimaging analysis pipeline.
Creates synthetic T1 and T2-FLAIR volumes with simulated anomalies.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


def create_synthetic_brain(
    shape: Tuple[int, ...] = (256, 256, 256),
    voxel_size: Tuple[float, ...] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """Create a synthetic brain volume with basic structures"""
    # Create empty volume
    volume = np.zeros(shape)
    center = np.array(shape) // 2
    
    # Create skull (ellipsoid)
    x, y, z = np.ogrid[
        :shape[0],
        :shape[1],
        :shape[2]
    ]
    
    # Outer skull
    skull_radii = (shape[0]//3, shape[1]//2.5, shape[2]//3)
    skull = (
        ((x - center[0])/skull_radii[0])**2 +
        ((y - center[1])/skull_radii[1])**2 +
        ((z - center[2])/skull_radii[2])**2
    ) <= 1
    volume[skull] = 0.3
    
    # Brain matter (smaller ellipsoid)
    brain_radii = (shape[0]//3.5, shape[1]//3, shape[2]//3.5)
    brain = (
        ((x - center[0])/brain_radii[0])**2 +
        ((y - center[1])/brain_radii[1])**2 +
        ((z - center[2])/brain_radii[2])**2
    ) <= 1
    volume[brain] = 0.8
    
    # Add ventricles
    ventricle_center = center + np.array([0, 0, 10])
    ventricle_radii = (shape[0]//20, shape[1]//10, shape[2]//8)
    ventricles = (
        ((x - ventricle_center[0])/ventricle_radii[0])**2 +
        ((y - ventricle_center[1])/ventricle_radii[1])**2 +
        ((z - ventricle_center[2])/ventricle_radii[2])**2
    ) <= 1
    volume[ventricles] = 0.1
    
    # Add some subcortical structures
    for offset in [(20, 0, 0), (-20, 0, 0)]:  # Bilateral structures
        struct_center = center + np.array(offset)
        struct_radii = (shape[0]//15, shape[1]//15, shape[2]//15)
        structure = (
            ((x - struct_center[0])/struct_radii[0])**2 +
            ((y - struct_center[1])/struct_radii[1])**2 +
            ((z - struct_center[2])/struct_radii[2])**2
        ) <= 1
        volume[structure] = 0.6
    
    # Apply smoothing
    volume = gaussian_filter(volume, sigma=2.0)
    
    return volume


def add_anomalies(
    volume: np.ndarray,
    num_anomalies: int = 3,
    intensity_range: Tuple[float, float] = (0.7, 1.0),
    size_range: Tuple[int, int] = (5, 15)
) -> Tuple[np.ndarray, List[Dict[str, any]]]:
    """Add synthetic anomalies to the volume"""
    anomaly_info = []
    modified_volume = volume.copy()
    
    # Find valid locations (within brain matter)
    brain_mask = (volume > 0.4) & (volume < 0.9)
    valid_coords = np.array(np.where(brain_mask)).T
    
    if len(valid_coords) < num_anomalies:
        raise ValueError("Not enough valid locations for anomalies")
    
    # Add anomalies
    for _ in range(num_anomalies):
        # Random location within brain matter
        loc_idx = np.random.randint(len(valid_coords))
        center = valid_coords[loc_idx]
        
        # Random size and intensity
        size = np.random.randint(size_range[0], size_range[1])
        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        
        # Create anomaly mask
        x, y, z = np.ogrid[
            :volume.shape[0],
            :volume.shape[1],
            :volume.shape[2]
        ]
        anomaly = (
            ((x - center[0])/size)**2 +
            ((y - center[1])/size)**2 +
            ((z - center[2])/size)**2
        ) <= 1
        
        # Apply anomaly
        modified_volume[anomaly] = intensity
        
        # Store anomaly information
        anomaly_info.append({
            "center": center.tolist(),
            "size": size,
            "intensity": intensity
        })
    
    return modified_volume, anomaly_info


def create_test_subject(
    output_dir: Path,
    subject_id: str,
    shape: Tuple[int, ...] = (256, 256, 256),
    voxel_size: Tuple[float, ...] = (1.0, 1.0, 1.0)
) -> Dict[str, Path]:
    """Create test data for a single subject"""
    subject_dir = output_dir / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    
    # Create affine matrix
    affine = np.eye(4)
    affine[:3, :3] = np.diag(voxel_size)
    
    # Generate base brain volume
    base_volume = create_synthetic_brain(shape, voxel_size)
    
    # Create T1
    t1_volume = base_volume.copy()
    t1_path = subject_dir / f"{subject_id}_T1.nii.gz"
    nib.save(nib.Nifti1Image(t1_volume, affine), str(t1_path))
    
    # Create T2-FLAIR with anomalies
    flair_base = 1.0 - base_volume  # Invert contrast
    flair_volume, anomalies = add_anomalies(flair_base)
    flair_path = subject_dir / f"{subject_id}_T2_FLAIR.nii.gz"
    nib.save(nib.Nifti1Image(flair_volume, affine), str(flair_path))
    
    # Save anomaly information
    import json
    anomaly_path = subject_dir / "anomalies.json"
    with open(anomaly_path, 'w') as f:
        json.dump(anomalies, f, indent=2)
    
    return {
        "T1": t1_path,
        "T2_FLAIR": flair_path,
        "anomalies": anomaly_path
    }


def create_test_dataset(
    num_subjects: int = 3,
    output_dir: Path = Path("test_data"),
    shape: Tuple[int, ...] = (256, 256, 256),
    voxel_size: Tuple[float, ...] = (1.0, 1.0, 1.0)
) -> Dict[str, Dict[str, Path]]:
    """Create a test dataset with multiple subjects"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = {}
    for i in range(num_subjects):
        subject_id = f"sub-{i+1:03d}"
        dataset[subject_id] = create_test_subject(
            output_dir,
            subject_id,
            shape,
            voxel_size
        )
    
    return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate synthetic neuroimaging test data"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_data"),
        help="Output directory for test data"
    )
    
    parser.add_argument(
        "--num-subjects",
        type=int,
        default=3,
        help="Number of subjects to generate"
    )
    
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        default=[256, 256, 256],
        help="Volume dimensions (x, y, z)"
    )
    
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Voxel size in mm (x, y, z)"
    )
    
    args = parser.parse_args()
    
    print(f"Generating test dataset with {args.num_subjects} subjects...")
    dataset = create_test_dataset(
        num_subjects=args.num_subjects,
        output_dir=args.output_dir,
        shape=tuple(args.shape),
        voxel_size=tuple(args.voxel_size)
    )
    
    print("\nGenerated files:")
    for subject_id, files in dataset.items():
        print(f"\n{subject_id}:")
        for modality, path in files.items():
            print(f"  {modality}: {path}")