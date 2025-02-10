"""
Visualizer Agent for generating neuroimaging visualizations.
Handles multi-planar views, 3D rendering, and anomaly visualization.
"""
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from nilearn import plotting
from pydantic import BaseModel

from .analyzer import ClusterMetrics
from .coordinator import AgentMessage, MessageType, Priority


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters"""
    output_dpi: int = 300
    slice_spacing: int = 5
    overlay_alpha: float = 0.35
    anomaly_cmap: str = "hot"
    background_cmap: str = "gray"
    freeview_options: str = "-viewport 3d"


class SliceSelection(BaseModel):
    """Selected slices for multi-planar visualization"""
    sagittal: List[int]
    coronal: List[int]
    axial: List[int]


class VisualizerAgent:
    """
    Agent responsible for generating visualizations of neuroimaging data and analysis results.
    Handles 2D multi-planar views and 3D rendering using FreeSurfer's freeview.
    """

    def __init__(self, coordinator_id: str):
        self.agent_id = "visualizer"
        self.coordinator_id = coordinator_id
        self.current_subject: Optional[str] = None
        self.working_dir: Optional[Path] = None
        self.config = VisualizationConfig()

    async def initialize_subject(self, subject_id: str, working_dir: Path) -> None:
        """Initialize visualization for a new subject"""
        self.current_subject = subject_id
        self.working_dir = working_dir
        
        # Create visualization directory
        vis_dir = working_dir / subject_id / "vis"
        for subdir in ["mpr", "3d", "reports"]:
            (vis_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _select_informative_slices(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        n_slices: int = 5
    ) -> SliceSelection:
        """Select informative slices containing the masked region"""
        try:
            # Find slices containing masked data
            x_indices, y_indices, z_indices = np.where(mask > 0)
            
            def select_slices(indices: np.ndarray) -> List[int]:
                if len(indices) == 0:
                    return []
                
                min_idx = np.min(indices)
                max_idx = np.max(indices)
                
                if max_idx - min_idx < n_slices:
                    return list(range(min_idx, max_idx + 1))
                
                # Select evenly spaced slices
                return list(np.linspace(min_idx, max_idx, n_slices, dtype=int))
            
            return SliceSelection(
                sagittal=select_slices(x_indices),
                coronal=select_slices(y_indices),
                axial=select_slices(z_indices)
            )
            
        except Exception as e:
            print(f"Error selecting slices: {str(e)}")
            return SliceSelection(sagittal=[], coronal=[], axial=[])

    async def generate_multiplanar_views(
        self,
        t1_path: Path,
        flair_path: Path,
        mask_path: Path,
        anomaly_clusters: List[ClusterMetrics]
    ) -> Path:
        """Generate multi-planar views with overlays"""
        try:
            # Load images
            t1_img = nib.load(str(t1_path))
            flair_img = nib.load(str(flair_path))
            mask_img = nib.load(str(mask_path))
            
            t1_data = t1_img.get_fdata()
            flair_data = flair_img.get_fdata()
            mask_data = mask_img.get_fdata()
            
            # Create anomaly overlay
            anomaly_overlay = np.zeros_like(mask_data)
            for cluster in anomaly_clusters:
                mins, maxs = cluster.bounding_box
                slice_indices = tuple(slice(min_, max_+1) for min_, max_ in zip(mins, maxs))
                anomaly_overlay[slice_indices] = cluster.outlier_score
            
            # Select informative slices
            slices = self._select_informative_slices(flair_data, mask_data)
            
            # Create figure
            n_rows = 3  # One row each for sagittal, coronal, axial
            n_cols = len(slices.sagittal)
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(4*n_cols, 4*n_rows),
                dpi=self.config.output_dpi
            )
            
            if n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            # Create custom colormap for anomalies
            anomaly_cmap = plt.get_cmap(self.config.anomaly_cmap)
            
            # Plot slices
            views = [
                ("Sagittal", slices.sagittal, (1, 2)),
                ("Coronal", slices.coronal, (0, 2)),
                ("Axial", slices.axial, (0, 1))
            ]
            
            for row, (view_name, slice_indices, display_axes) in enumerate(views):
                for col, slice_idx in enumerate(slice_indices):
                    ax = axes[row, col]
                    
                    # Extract slice data
                    if row == 0:  # Sagittal
                        t1_slice = t1_data[slice_idx, :, :]
                        flair_slice = flair_data[slice_idx, :, :]
                        mask_slice = mask_data[slice_idx, :, :]
                        anomaly_slice = anomaly_overlay[slice_idx, :, :]
                    elif row == 1:  # Coronal
                        t1_slice = t1_data[:, slice_idx, :]
                        flair_slice = flair_data[:, slice_idx, :]
                        mask_slice = mask_data[:, slice_idx, :]
                        anomaly_slice = anomaly_overlay[:, slice_idx, :]
                    else:  # Axial
                        t1_slice = t1_data[:, :, slice_idx]
                        flair_slice = flair_data[:, :, slice_idx]
                        mask_slice = mask_data[:, :, slice_idx]
                        anomaly_slice = anomaly_overlay[:, :, slice_idx]
                    
                    # Normalize slice data
                    t1_slice = np.clip((t1_slice - t1_slice.min()) / 
                                     (t1_slice.max() - t1_slice.min()), 0, 1)
                    flair_slice = np.clip((flair_slice - flair_slice.min()) / 
                                        (flair_slice.max() - flair_slice.min()), 0, 1)
                    
                    # Plot base image
                    ax.imshow(t1_slice, cmap=self.config.background_cmap)
                    
                    # Overlay FLAIR
                    ax.imshow(flair_slice, cmap="Blues", alpha=0.3)
                    
                    # Overlay mask
                    masked_anomalies = anomaly_slice * mask_slice
                    if np.any(masked_anomalies > 0):
                        ax.imshow(
                            masked_anomalies,
                            cmap=anomaly_cmap,
                            alpha=self.config.overlay_alpha
                        )
                    
                    # Add slice information
                    ax.set_title(f"{view_name} - Slice {slice_idx}")
                    ax.axis('off')
            
            # Save figure
            output_path = (self.working_dir / self.current_subject / "vis" / "mpr" /
                          f"multiplanar_view.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=self.config.output_dpi)
            plt.close()
            
            return output_path
            
        except Exception as e:
            await self._send_error(f"Multi-planar visualization failed: {str(e)}")
            return None

    async def generate_3d_visualization(
        self,
        t1_path: Path,
        flair_path: Path,
        mask_path: Path,
        anomaly_clusters: List[ClusterMetrics]
    ) -> None:
        """Generate 3D visualization using freeview"""
        try:
            # Prepare freeview command
            cmd_parts = [
                "freeview",
                f"-v {t1_path}:grayscale=0,100",  # Load T1 as base
                f"{flair_path}:colormap=heat:opacity=0.4",  # Overlay FLAIR
                f"{mask_path}:colormap=lut:opacity=0.6"  # Overlay segmentation
            ]
            
            # Add overlay for each anomaly cluster
            clusters_dir = self.working_dir / self.current_subject / "vis" / "3d" / "clusters"
            clusters_dir.mkdir(exist_ok=True)
            
            for i, cluster in enumerate(anomaly_clusters):
                # Create volume for cluster
                cluster_vol = np.zeros_like(nib.load(str(mask_path)).get_fdata())
                mins, maxs = cluster.bounding_box
                slice_indices = tuple(slice(min_, max_+1) for min_, max_ in zip(mins, maxs))
                cluster_vol[slice_indices] = cluster.outlier_score
                
                # Save cluster volume
                cluster_path = clusters_dir / f"cluster_{i}.nii.gz"
                nib.save(nib.Nifti1Image(cluster_vol, nib.load(str(mask_path)).affine),
                        str(cluster_path))
                
                # Add to freeview command
                cmd_parts.append(
                    f"{cluster_path}:colormap=heat:opacity=0.8"
                )
            
            # Add viewport options
            cmd_parts.append(self.config.freeview_options)
            
            # Execute freeview
            cmd = " ".join(cmd_parts)
            os.system(cmd)  # In real implementation, use subprocess with proper error handling
            
        except Exception as e:
            await self._send_error(f"3D visualization failed: {str(e)}")

    async def generate_report(
        self,
        analysis_results: Dict[str, Any],
        visualization_paths: Dict[str, Path]
    ) -> Path:
        """Generate HTML report summarizing analysis and visualizations"""
        try:
            report_path = (self.working_dir / self.current_subject / "vis" / "reports" /
                          "analysis_report.html")
            
            # Create HTML report
            with open(report_path, 'w') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Analysis Report - Subject {self.current_subject}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        .section {{ margin-bottom: 30px; }}
                        .visualization {{ margin: 20px 0; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                    </style>
                </head>
                <body>
                    <h1>Analysis Report - Subject {self.current_subject}</h1>
                    
                    <div class="section">
                        <h2>Analysis Summary</h2>
                        <table>
                            <tr>
                                <th>Region</th>
                                <th>Volume (mm³)</th>
                                <th>Anomalous Clusters</th>
                                <th>Max Outlier Score</th>
                            </tr>
                """)
                
                # Add analysis results
                for region, results in analysis_results.items():
                    seg_result = results["segmentation"]
                    anomalies = results["anomalies"]
                    max_score = max([a["outlier_score"] for a in anomalies]) if anomalies else 0
                    
                    f.write(f"""
                            <tr>
                                <td>{region}</td>
                                <td>{seg_result.volume:.2f}</td>
                                <td>{len(anomalies)}</td>
                                <td>{max_score:.2f}</td>
                            </tr>
                    """)
                
                f.write("""
                        </table>
                    </div>
                    
                    <div class="section">
                        <h2>Visualizations</h2>
                """)
                
                # Add visualizations
                for vis_type, path in visualization_paths.items():
                    f.write(f"""
                        <div class="visualization">
                            <h3>{vis_type}</h3>
                            <img src="{path}" alt="{vis_type}" style="max-width: 100%;">
                        </div>
                    """)
                
                f.write("""
                    </div>
                </body>
                </html>
                """)
            
            return report_path
            
        except Exception as e:
            await self._send_error(f"Report generation failed: {str(e)}")
            return None

    async def run_visualization(
        self,
        subject_id: str,
        t1_path: Path,
        flair_path: Path,
        analysis_results: Dict[str, Any],
        working_dir: Path
    ) -> None:
        """Execute the complete visualization pipeline"""
        try:
            # Initialize subject
            await self.initialize_subject(subject_id, working_dir)
            
            visualization_paths = {}
            
            # Process each region
            for region_name, results in analysis_results.items():
                mask_path = results["segmentation"].mask_path
                anomaly_clusters = [
                    ClusterMetrics(**cluster_dict)
                    for cluster_dict in results["anomalies"]
                ]
                
                # Generate multi-planar views
                mpr_path = await self.generate_multiplanar_views(
                    t1_path,
                    flair_path,
                    mask_path,
                    anomaly_clusters
                )
                if mpr_path:
                    visualization_paths[f"{region_name}_mpr"] = mpr_path
                
                # Generate 3D visualization
                await self.generate_3d_visualization(
                    t1_path,
                    flair_path,
                    mask_path,
                    anomaly_clusters
                )
            
            # Generate report
            report_path = await self.generate_report(
                analysis_results,
                visualization_paths
            )
            
            if report_path:
                # Send success message with results
                await self._send_message(
                    MessageType.RESULT,
                    {
                        "subject_id": subject_id,
                        "visualization_paths": {
                            str(k): str(v)
                            for k, v in visualization_paths.items()
                        },
                        "report_path": str(report_path)
                    }
                )
            
        except Exception as e:
            await self._send_error(f"Visualization pipeline failed: {str(e)}")

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
        print(f"Sending message: {message}")  # Placeholder for actual message sending

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