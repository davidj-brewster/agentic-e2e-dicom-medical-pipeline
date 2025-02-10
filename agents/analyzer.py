"""
Analyzer Agent for segmentation and clustering analysis.
Handles FreeSurfer segmentation, mask generation, and anomaly detection.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import datetime 
import nibabel as nib
import numpy as np
from pydantic import BaseModel
from scipy.ndimage import label
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from .coordinator import AgentMessage, MessageType, Priority, TaskStatus


@dataclass
class SegmentationResult:
    """Results from FreeSurfer segmentation"""
    region_name: str
    mask_path: Path
    volume: float
    center_of_mass: Tuple[float, ...]
    voxel_count: int


class ClusterMetrics(BaseModel):
    """Metrics for identified clusters"""
    cluster_id: int
    size: int
    mean_intensity: float
    std_intensity: float
    center: Tuple[float, ...]
    bounding_box: Tuple[Tuple[int, ...], Tuple[int, ...]]
    outlier_score: float


class AnalyzerAgent:
    """
    Agent responsible for image segmentation and anomaly detection.
    Handles FreeSurfer segmentation, mask generation, and clustering analysis.
    """

    def __init__(self, coordinator_id: str):
        self.agent_id = "analyzer"
        self.coordinator_id = coordinator_id
        self.current_subject: Optional[str] = None
        self.working_dir: Optional[Path] = None
        
        # FreeSurfer region definitions
        self.regions_of_interest = {
            "L_Thal": 10,    # Left Thalamus
            "R_Thal": 49,    # Right Thalamus
            "L_Caud": 11,    # Left Caudate
            "R_Caud": 50,    # Right Caudate
            "L_Puta": 12,    # Left Putamen
            "R_Puta": 51,    # Right Putamen
            "L_Pall": 13,    # Left Pallidum
            "R_Pall": 52,    # Right Pallidum
            "L_Hipp": 17,    # Left Hippocampus
            "R_Hipp": 53,    # Right Hippocampus
            "L_Amyg": 18,    # Left Amygdala
            "R_Amyg": 54,    # Right Amygdala
        }

    async def initialize_subject(self, subject_id: str, working_dir: Path) -> None:
        """Initialize analysis for a new subject"""
        self.current_subject = subject_id
        self.working_dir = working_dir
        
        # Create subject-specific directories
        subject_dir = working_dir / subject_id
        for subdir in ["seg", "masks", "clusters"]:
            (subject_dir / subdir).mkdir(parents=True, exist_ok=True)

    async def run_first_segmentation(self, t1_path: Path) -> Dict[str, SegmentationResult]:
        """Run FSL FIRST segmentation"""
        try:
            output_dir = self.working_dir / self.current_subject / "seg"
            
            # Run FIRST segmentation
            cmd = (
                f"run_first_all -i {t1_path} -o {output_dir}/first "
                f"-d -v"  # debug mode and verbose output
            )
            os.system(cmd)  # In real implementation, use subprocess with proper error handling
            
            # Process results
            results = {}
            for region_name, label_value in self.regions_of_interest.items():
                # Load segmentation result
                seg_path = output_dir / f"first_all_{region_name}.nii.gz"
                if not seg_path.exists():
                    await self._send_error(f"Segmentation failed for region: {region_name}")
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
                    voxel_count=int(voxel_count)
                )
            
            return results
            
        except Exception as e:
            await self._send_error(f"FIRST segmentation failed: {str(e)}")
            return {}

    async def generate_binary_masks(
        self, 
        segmentation_results: Dict[str, SegmentationResult]
    ) -> Dict[str, Path]:
        """Generate binary masks for each segmented region"""
        try:
            masks_dir = self.working_dir / self.current_subject / "masks"
            binary_masks = {}
            
            for region_name, result in segmentation_results.items():
                mask_path = masks_dir / f"{region_name}_mask.nii.gz"
                
                # Create binary mask using fslmaths
                cmd = f"fslmaths {result.mask_path} -bin {mask_path}"
                os.system(cmd)
                
                binary_masks[region_name] = mask_path
            
            return binary_masks
            
        except Exception as e:
            await self._send_error(f"Binary mask generation failed: {str(e)}")
            return {}

    async def analyze_intensity_clusters(
        self,
        flair_path: Path,
        mask_path: Path,
        region_name: str
    ) -> List[ClusterMetrics]:
        """Perform intensity-based clustering analysis within masked region"""
        try:
            # Load images
            flair_img = nib.load(str(flair_path))
            mask_img = nib.load(str(mask_path))
            
            flair_data = flair_img.get_fdata()
            mask_data = mask_img.get_fdata() > 0
            
            # Extract masked intensities
            masked_intensities = flair_data[mask_data]
            masked_coordinates = np.array(np.where(mask_data)).T
            
            if len(masked_intensities) == 0:
                return []
            
            # Normalize intensities
            scaler = StandardScaler()
            normalized_intensities = scaler.fit_transform(masked_intensities.reshape(-1, 1))
            
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = dbscan.fit_predict(normalized_intensities)
            
            # Analyze clusters
            cluster_metrics = []
            unique_clusters = set(clusters)
            
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise points
                    continue
                    
                # Get cluster points
                cluster_mask = clusters == cluster_id
                cluster_intensities = masked_intensities[cluster_mask]
                cluster_coords = masked_coordinates[cluster_mask]
                
                # Calculate metrics
                mean_intensity = float(np.mean(cluster_intensities))
                std_intensity = float(np.std(cluster_intensities))
                center = tuple(float(x) for x in np.mean(cluster_coords, axis=0))
                
                # Calculate bounding box
                mins = tuple(int(x) for x in np.min(cluster_coords, axis=0))
                maxs = tuple(int(x) for x in np.max(cluster_coords, axis=0))
                
                # Calculate outlier score based on intensity distribution
                z_scores = np.abs(scaler.transform(cluster_intensities.reshape(-1, 1)))
                outlier_score = float(np.mean(z_scores))
                
                metrics = ClusterMetrics(
                    cluster_id=int(cluster_id),
                    size=int(len(cluster_intensities)),
                    mean_intensity=mean_intensity,
                    std_intensity=std_intensity,
                    center=center,
                    bounding_box=(mins, maxs),
                    outlier_score=outlier_score
                )
                
                cluster_metrics.append(metrics)
            
            return cluster_metrics
            
        except Exception as e:
            await self._send_error(f"Clustering analysis failed for {region_name}: {str(e)}")
            return []

    async def identify_anomalies(
        self,
        cluster_metrics: List[ClusterMetrics],
        threshold: float = 2.0
    ) -> List[ClusterMetrics]:
        """Identify potentially anomalous clusters"""
        try:
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
            
        except Exception as e:
            await self._send_error(f"Anomaly identification failed: {str(e)}")
            return []

    async def run_analysis(
        self,
        subject_id: str,
        t1_path: Path,
        flair_path: Path,
        working_dir: Path
    ) -> None:
        """Execute the complete analysis pipeline"""
        try:
            # Initialize subject
            await self.initialize_subject(subject_id, working_dir)
            
            # Run segmentation
            segmentation_results = await self.run_first_segmentation(t1_path)
            if not segmentation_results:
                return
            
            # Generate binary masks
            binary_masks = await self.generate_binary_masks(segmentation_results)
            if not binary_masks:
                return
            
            # Analyze each region
            analysis_results = {}
            for region_name, mask_path in binary_masks.items():
                # Perform clustering analysis
                clusters = await self.analyze_intensity_clusters(
                    flair_path,
                    mask_path,
                    region_name
                )
                
                # Identify anomalies
                anomalies = await self.identify_anomalies(clusters)
                
                analysis_results[region_name] = {
                    "segmentation": segmentation_results[region_name],
                    "clusters": [c.dict() for c in clusters],
                    "anomalies": [a.dict() for a in anomalies]
                }
            
            # Send success message with results
            await self._send_message(
                MessageType.RESULT,
                {
                    "subject_id": subject_id,
                    "analysis_results": analysis_results
                }
            )
            
        except Exception as e:
            await self._send_error(f"Analysis pipeline failed: {str(e)}")

    async def _send_message(self, message_type: MessageType, payload: Dict[str, Any]) -> None:
        """Send message to coordinator"""
        message = AgentMessage(
            sender=self.agent_id,
            recipient=self.coordinator_id,
            message_type=message_type,
            payload=payload,
            priority=Priority.NORMAL
        )
        # In real implementation, this would use proper message passing
        self.logger(f"Sending message: {message}")  # Placeholder for actual message sending

    async def _send_error(self, error_message: str) -> None:
        """Send error message to coordinator"""
        await self._send_message(
            MessageType.ERROR,
            {
                "error": error_message,
                "subject_id": self.current_subject,
                "timestamp": datetime.utcnow().isoformat()
            }
        )