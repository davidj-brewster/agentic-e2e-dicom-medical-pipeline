"""
Analyzer Agent for segmentation and clustering analysis.
Handles FreeSurfer segmentation, mask generation, and anomaly detection.
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import nibabel as nib
import numpy as np
from pydantic import BaseModel, Field

from core.config import (
    ClusteringConfig,
    ProcessingConfig,
    SegmentationConfig
)
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
from utils.analysis import (
    ClusterMetrics,
    SegmentationResult,
    analyze_intensity_clusters,
    generate_binary_masks,
    identify_anomalies,
    run_first_segmentation,
    validate_segmentation
)
from .base import AgentConfig, BaseAgent


class AnalyzerConfig(BaseModel):
    """Analyzer agent configuration"""
    segmentation: SegmentationConfig
    clustering: ClusteringConfig
    processing: ProcessingConfig


class AnalyzerAgent(BaseAgent):
    """
    Agent responsible for image segmentation and anomaly detection.
    Handles FreeSurfer segmentation, mask generation, and clustering analysis.
    """

    def __init__(
        self,
        config: AgentConfig,
        analyzer_config: AnalyzerConfig,
        message_queue: Optional[Any] = None
    ):
        super().__init__(config, message_queue)
        self.analyzer_config = analyzer_config
        self.current_subject: Optional[str] = None
        self.subject_dir: Optional[Path] = None
        self.env_setup_verified = False

    async def _initialize(self) -> None:
        """Initialize agent resources"""
        await super()._initialize()
        self.logger.info("Initializing analyzer agent")
        
        # Verify environment setup
        if not await self._verify_environment():
            raise RuntimeError("Environment verification failed")

    async def _verify_environment(self) -> bool:
        """Verify FSL and FreeSurfer environment setup"""
        try:
            # Check FSL setup
            fsl_dir = self.analyzer_config.processing.fsl_dir
            if not fsl_dir.exists():
                await self._handle_error(f"FSL directory not found: {fsl_dir}")
                return False
            
            # Check FreeSurfer setup
            fs_home = self.analyzer_config.processing.freesurfer_home
            if not fs_home.exists():
                await self._handle_error(f"FreeSurfer directory not found: {fs_home}")
                return False
            
            # Verify FSL commands
            fsl_cmds = ["first", "run_first_all", "fslmaths"]
            for cmd in fsl_cmds:
                cmd_path = fsl_dir / "bin" / cmd
                if not cmd_path.exists():
                    await self._handle_error(f"FSL command not found: {cmd}")
                    return False
            
            self.env_setup_verified = True
            return True
            
        except Exception as e:
            await self._handle_error(f"Environment verification failed: {str(e)}")
            return False

    async def _handle_command(self, message: Message) -> None:
        """Handle command messages"""
        command = message.payload.command
        params = message.payload.parameters
        
        try:
            if command == "analyze_subject":
                await self._analyze_subject(
                    subject_id=params["subject_id"],
                    t1_path=Path(params["t1_path"]),
                    flair_path=Path(params["flair_path"])
                )
            else:
                await self._handle_error(
                    f"Unknown command: {command}",
                    message
                )
                
        except Exception as e:
            await self._handle_error(str(e), message)

    async def _analyze_subject(
        self,
        subject_id: str,
        t1_path: Path,
        flair_path: Path
    ) -> None:
        """Execute analysis pipeline for a subject"""
        try:
            self.current_subject = subject_id
            self.subject_dir = self.config.working_dir / subject_id
            
            # Create subject directories
            for subdir in ["seg", "masks", "clusters"]:
                (self.subject_dir / subdir).mkdir(parents=True, exist_ok=True)
            
            # Send initial status
            await self._send_message(
                create_status_update(
                    sender=self.config.name,
                    recipient="coordinator",
                    state="analyzing",
                    progress=0.0,
                    message="Starting analysis"
                )
            )
            
            # Run segmentation
            success, error, segmentation_results = await run_first_segmentation(
                t1_path,
                self.subject_dir / "seg",
                self.analyzer_config.segmentation.regions_of_interest
            )
            
            if not success:
                await self._handle_error(error or "Segmentation failed")
                return
            
            await self._send_message(
                create_status_update(
                    sender=self.config.name,
                    recipient="coordinator",
                    state="analyzing",
                    progress=0.3,
                    message="Segmentation complete"
                )
            )
            
            # Generate binary masks
            success, error, binary_masks = await generate_binary_masks(
                segmentation_results,
                self.subject_dir / "masks"
            )
            
            if not success:
                await self._handle_error(error or "Mask generation failed")
                return
            
            await self._send_message(
                create_status_update(
                    sender=self.config.name,
                    recipient="coordinator",
                    state="analyzing",
                    progress=0.5,
                    message="Mask generation complete"
                )
            )
            
            # Analyze each region
            analysis_results = {}
            total_regions = len(binary_masks)
            
            for i, (region_name, mask_path) in enumerate(binary_masks.items(), 1):
                # Validate segmentation
                valid, error, metrics = await validate_segmentation(
                    mask_path,
                    t1_path,
                    min_volume=100.0,  # TODO: Configure per region
                    max_volume=100000.0
                )
                
                if not valid:
                    self.logger.warning(f"Validation failed for {region_name}: {error}")
                    continue
                
                # Perform clustering analysis
                success, error, clusters = await analyze_intensity_clusters(
                    flair_path,
                    mask_path,
                    region_name,
                    eps=self.analyzer_config.clustering.eps,
                    min_samples=self.analyzer_config.clustering.min_cluster_size
                )
                
                if not success:
                    self.logger.warning(f"Clustering failed for {region_name}: {error}")
                    continue
                
                # Identify anomalies
                anomalies = identify_anomalies(
                    clusters,
                    threshold=self.analyzer_config.clustering.outlier_threshold
                )
                
                analysis_results[region_name] = {
                    "segmentation": {
                        **segmentation_results[region_name].__dict__,
                        **metrics
                    },
                    "clusters": [c.__dict__ for c in clusters],
                    "anomalies": [a.__dict__ for a in anomalies]
                }
                
                # Update progress
                progress = 0.5 + (0.5 * i / total_regions)
                await self._send_message(
                    create_status_update(
                        sender=self.config.name,
                        recipient="coordinator",
                        state="analyzing",
                        progress=progress,
                        message=f"Analyzed region {i}/{total_regions}"
                    )
                )
            
            # Send results
            await self._send_message(
                create_data_message(
                    sender=self.config.name,
                    recipient="coordinator",
                    data_type="analysis_results",
                    content={
                        "subject_id": subject_id,
                        "analysis_results": analysis_results,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            )
            
            await self._send_message(
                create_status_update(
                    sender=self.config.name,
                    recipient="coordinator",
                    state="completed",
                    progress=1.0,
                    message="Analysis complete"
                )
            )
            
        except Exception as e:
            await self._handle_error(f"Analysis failed: {str(e)}")

    async def _cleanup(self) -> None:
        """Cleanup agent resources"""
        await super()._cleanup()
        self.logger.info("Cleaning up analyzer agent")
        # Additional cleanup if needed