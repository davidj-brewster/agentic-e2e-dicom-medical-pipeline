"""
Visualizer Agent for generating neuroimaging visualizations.
Handles multi-planar views, 3D rendering, and anomaly visualization.
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from pydantic import BaseModel, Field

from core.config import VisualizationConfig
from core.messages import (
    Message,
    MessageType,
    Priority,
    create_command,
    create_data_message,
    create_error,
    create_status_update
)
from core.workflow import ResourceRequirements, WorkflowState
from utils.visualization import (
    ColorMap,
    SliceSelection,
    ViewType,
    ViewportConfig,
    create_3d_rendering,
    create_multiplanar_figure,
    save_figure,
    save_vtk_screenshot,
    select_informative_slices
)
from .base import AgentConfig, BaseAgent


class VisualizerAgent(BaseAgent):
    """
    Agent responsible for generating visualizations of neuroimaging data and analysis results.
    Handles 2D multi-planar views and 3D rendering using multiple backends.
    """

    def __init__(
        self,
        config: AgentConfig,
        visualization_config: VisualizationConfig,
        message_queue: Optional[Any] = None
    ):
        super().__init__(config, message_queue)
        self.visualization_config = visualization_config
        self.current_subject: Optional[str] = None
        self.subject_dir: Optional[Path] = None

    async def _initialize(self) -> None:
        """Initialize agent resources"""
        await super()._initialize()
        self.logger.info("Initializing visualizer agent")

    async def _handle_command(self, message: Message) -> None:
        """Handle command messages"""
        command = message.payload.command
        params = message.payload.parameters
        
        try:
            if command == "visualize_subject":
                await self._visualize_subject(
                    subject_id=params["subject_id"],
                    t1_path=Path(params["t1_path"]),
                    flair_path=Path(params["flair_path"]),
                    analysis_results=params["analysis_results"]
                )
            else:
                await self._handle_error(
                    f"Unknown command: {command}",
                    message
                )
                
        except Exception as e:
            await self._handle_error(str(e), message)

    async def _visualize_subject(
        self,
        subject_id: str,
        t1_path: Path,
        flair_path: Path,
        analysis_results: Dict[str, Any]
    ) -> None:
        """Execute visualization pipeline for a subject"""
        try:
            self.current_subject = subject_id
            self.subject_dir = self.config.working_dir / subject_id
            
            # Create visualization directories
            vis_dir = self.subject_dir / "vis"
            for subdir in ["mpr", "3d", "reports"]:
                (vis_dir / subdir).mkdir(parents=True, exist_ok=True)
            
            # Send initial status
            await self._send_message(
                create_status_update(
                    sender=self.config.name,
                    recipient="coordinator",
                    state="visualizing",
                    progress=0.0,
                    message="Starting visualization"
                )
            )
            
            visualization_paths = {}
            total_regions = len(analysis_results)
            
            # Process each region
            for i, (region_name, results) in enumerate(analysis_results.items(), 1):
                # Load data
                t1_img = nib.load(str(t1_path))
                flair_img = nib.load(str(flair_path))
                mask_img = nib.load(str(results["segmentation"]["mask_path"]))
                
                t1_data = t1_img.get_fdata()
                flair_data = flair_img.get_fdata()
                mask_data = mask_img.get_fdata()
                
                # Create anomaly overlay
                anomaly_overlay = np.zeros_like(mask_data)
                for anomaly in results["anomalies"]:
                    mins, maxs = anomaly["bounding_box"]
                    slice_indices = tuple(slice(min_, max_+1) for min_, max_ in zip(mins, maxs))
                    anomaly_overlay[slice_indices] = anomaly["outlier_score"]
                
                # Generate multi-planar views
                success, error, fig = await create_multiplanar_figure(
                    background=t1_data,
                    overlay=anomaly_overlay,
                    mask=mask_data,
                    background_cmap=ColorMap(
                        name=self.visualization_config.background_cmap,
                        alpha=1.0
                    ),
                    overlay_cmap=ColorMap(
                        name=self.visualization_config.anomaly_cmap,
                        alpha=self.visualization_config.overlay_alpha
                    ),
                    dpi=self.visualization_config.output_dpi,
                    title=f"{region_name} Analysis"
                )
                
                if not success:
                    self.logger.error(f"Failed to create multi-planar view for {region_name}: {error}")
                    continue
                
                # Save multi-planar figure
                mpr_path = vis_dir / "mpr" / f"{region_name}_mpr.png"
                success, error = await save_figure(
                    fig,
                    mpr_path,
                    dpi=self.visualization_config.output_dpi
                )
                
                if success:
                    visualization_paths[f"{region_name}_mpr"] = mpr_path
                else:
                    self.logger.error(f"Failed to save multi-planar view for {region_name}: {error}")
                
                # Generate 3D visualization
                success, error, renderer = await create_3d_rendering(
                    volume=t1_data,
                    affine=t1_img.affine,
                    config=ViewportConfig(
                        width=800,
                        height=600,
                        background_color=(0.1, 0.1, 0.1)
                    ),
                    overlay=anomaly_overlay,
                    mask=mask_data,
                    volume_cmap=ColorMap(
                        name=self.visualization_config.background_cmap,
                        alpha=1.0
                    ),
                    overlay_cmap=ColorMap(
                        name=self.visualization_config.anomaly_cmap,
                        alpha=self.visualization_config.overlay_alpha
                    )
                )
                
                if success:
                    # Save 3D visualization
                    render_path = vis_dir / "3d" / f"{region_name}_3d.png"
                    success, error = await save_vtk_screenshot(
                        renderer,
                        render_path,
                        width=800,
                        height=600
                    )
                    
                    if success:
                        visualization_paths[f"{region_name}_3d"] = render_path
                    else:
                        self.logger.error(f"Failed to save 3D view for {region_name}: {error}")
                else:
                    self.logger.error(f"Failed to create 3D view for {region_name}: {error}")
                
                # Update progress
                progress = i / total_regions
                await self._send_message(
                    create_status_update(
                        sender=self.config.name,
                        recipient="coordinator",
                        state="visualizing",
                        progress=progress,
                        message=f"Processed region {i}/{total_regions}"
                    )
                )
            
            # Generate report
            report_path = await self._generate_report(
                analysis_results,
                visualization_paths
            )
            
            if report_path:
                # Send success message
                await self._send_message(
                    create_data_message(
                        sender=self.config.name,
                        recipient="coordinator",
                        data_type="visualization_results",
                        content={
                            "subject_id": subject_id,
                            "visualization_paths": {
                                str(k): str(v)
                                for k, v in visualization_paths.items()
                            },
                            "report_path": str(report_path)
                        }
                    )
                )
                
                await self._send_message(
                    create_status_update(
                        sender=self.config.name,
                        recipient="coordinator",
                        state="completed",
                        progress=1.0,
                        message="Visualization complete"
                    )
                )
            
        except Exception as e:
            await self._handle_error(f"Visualization failed: {str(e)}")

    async def _generate_report(
        self,
        analysis_results: Dict[str, Any],
        visualization_paths: Dict[str, Path]
    ) -> Optional[Path]:
        """Generate HTML report summarizing analysis and visualizations"""
        try:
            report_path = (
                self.subject_dir / "vis" / "reports" / "analysis_report.html"
            )
            
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
                                <td>{seg_result["volume"]:.2f}</td>
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
                            <img src="{path.relative_to(self.subject_dir)}"
                                 alt="{vis_type}"
                                 style="max-width: 100%;">
                        </div>
                    """)
                
                f.write("""
                    </div>
                </body>
                </html>
                """)
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            return None

    async def _cleanup(self) -> None:
        """Cleanup agent resources"""
        await super()._cleanup()
        self.logger.info("Cleaning up visualizer agent")
        # Additional cleanup if needed