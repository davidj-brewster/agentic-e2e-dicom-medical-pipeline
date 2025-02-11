"""
Preprocessing Agent for handling FSL/FreeSurfer operations.
Manages image preprocessing, registration, and quality control.
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from pydantic import BaseModel, Field

from core.config import FSLConfig, FreeSurferConfig, ProcessingConfig
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
from utils.neuroimaging import (
    normalize_image,
    register_images,
    run_brain_extraction,
    run_freesurfer_command,
    run_recon_all,
    run_tissue_segmentation,
    validate_image
)
from .base import AgentConfig, BaseAgent


@dataclass
class ImageMetadata:
    """Stores metadata for neuroimaging files"""
    path: Path
    modality: str
    dimensions: Tuple[int, ...]
    voxel_size: Tuple[float, ...]
    orientation: str
    data_type: str


class PreprocessingMetrics(BaseModel):
    """Quality control metrics for preprocessing steps"""
    snr: float
    contrast_to_noise: float
    motion_parameters: Optional[Dict[str, float]] = None
    registration_cost: Optional[float] = None
    normalized_mutual_information: Optional[float] = None
    brain_volume: Optional[float] = None
    tissue_volumes: Optional[Dict[str, float]] = None
    segmentation_quality: Optional[float] = None


class PreprocessingConfig(BaseModel):
    """Preprocessor agent configuration"""
    fsl: FSLConfig
    freesurfer: FreeSurferConfig
    processing: ProcessingConfig


class PreprocessingAgent(BaseAgent):
    """
    Agent responsible for image preprocessing and registration using FSL/FreeSurfer tools.
    Handles data validation, normalization, and registration tasks.
    """

    def __init__(
        self,
        config: AgentConfig,
        preprocessing_config: PreprocessingConfig,
        message_queue: Optional[Any] = None
    ):
        super().__init__(config, message_queue)
        self.preprocessing_config = preprocessing_config
        self.current_subject: Optional[str] = None
        self.subject_dir: Optional[Path] = None
        self.env_setup_verified = False

    async def _initialize(self) -> None:
        """Initialize agent resources"""
        await super()._initialize()
        self.logger.info("Initializing preprocessing agent")
        
        # Verify environment setup
        if not await self._verify_environment():
            raise RuntimeError("Environment verification failed")

    async def _verify_environment(self) -> bool:
        """Verify FSL and FreeSurfer environment setup"""
        try:
            # Check FSL setup
            fsl_dir = self.preprocessing_config.fsl.fsl_dir
            if not fsl_dir.exists():
                await self._handle_error(f"FSL directory not found: {fsl_dir}")
                return False
            
            # Check FreeSurfer setup
            fs_home = self.preprocessing_config.freesurfer.freesurfer_home
            if not fs_home.exists():
                await self._handle_error(f"FreeSurfer directory not found: {fs_home}")
                return False
            
            # Verify FSL commands
            fsl_cmds = ["fslmaths", "fslstats", "flirt", "bet", "fast"]
            for cmd in fsl_cmds:
                cmd_path = fsl_dir / "bin" / cmd
                if not cmd_path.exists():
                    await self._handle_error(f"FSL command not found: {cmd}")
                    return False
            
            # Verify FreeSurfer commands
            fs_cmds = ["mri_convert", "recon-all"]
            for cmd in fs_cmds:
                cmd_path = fs_home / "bin" / cmd
                if not cmd_path.exists():
                    await self._handle_error(f"FreeSurfer command not found: {cmd}")
                    return False
            
            # Verify FreeSurfer license
            if not self.preprocessing_config.freesurfer.license_file.exists():
                await self._handle_error("FreeSurfer license file not found")
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
            if command == "preprocess_subject":
                await self._preprocess_subject(
                    subject_id=params["subject_id"],
                    input_files={
                        k: Path(v) for k, v in params["input_files"].items()
                    }
                )
            else:
                await self._handle_error(
                    f"Unknown command: {command}",
                    message
                )
                
        except Exception as e:
            await self._handle_error(str(e), message)

    async def _preprocess_subject(
        self,
        subject_id: str,
        input_files: Dict[str, Path]
    ) -> None:
        """Execute preprocessing pipeline for a subject"""
        try:
            self.current_subject = subject_id
            self.subject_dir = self.config.working_dir / subject_id
            
            # Create subject directories
            for subdir in ["orig", "prep", "reg", "qc", "freesurfer"]:
                (self.subject_dir / subdir).mkdir(parents=True, exist_ok=True)
            
            # Validate inputs
            metadata = await self._validate_input(input_files)
            if not metadata:
                return
            
            await self._send_message(
                create_status_update(
                    sender=self.config.name,
                    recipient="coordinator",
                    state="preprocessing",
                    progress=0.1,
                    message="Input validation complete"
                )
            )
            
            # Brain extraction
            brain_masks = {}
            for modality, meta in metadata.items():
                output_path = self.subject_dir / "prep" / f"{modality}_brain_mask.nii.gz"
                success, error = await run_brain_extraction(
                    meta.path,
                    output_path,
                    fsl_dir=str(self.preprocessing_config.fsl.fsl_dir),
                    fractional_intensity=self.preprocessing_config.processing.bet_f_value
                )
                if not success:
                    await self._handle_error(f"Brain extraction failed for {modality}: {error}")
                    return
                brain_masks[modality] = output_path
            
            await self._send_message(
                create_status_update(
                    sender=self.config.name,
                    recipient="coordinator",
                    state="preprocessing",
                    progress=0.3,
                    message="Brain extraction complete"
                )
            )
            
            # Run FreeSurfer recon-all on T1
            if "T1" in metadata:
                success, error = await run_recon_all(
                    subject_id,
                    metadata["T1"].path,
                    self.subject_dir / "freesurfer",
                    fs_home=str(self.preprocessing_config.freesurfer.freesurfer_home)
                )
                if not success:
                    await self._handle_error(f"FreeSurfer processing failed: {error}")
                    return
            
            await self._send_message(
                create_status_update(
                    sender=self.config.name,
                    recipient="coordinator",
                    state="preprocessing",
                    progress=0.5,
                    message="FreeSurfer processing complete"
                )
            )
            
            # Tissue segmentation
            segmentation_outputs = {}
            for modality, mask in brain_masks.items():
                output_prefix = self.subject_dir / "prep" / f"{modality}_seg"
                success, error, outputs = await run_tissue_segmentation(
                    mask,
                    output_prefix,
                    fsl_dir=str(self.preprocessing_config.fsl.fsl_dir),
                    num_classes=3,
                    bias_correction=self.preprocessing_config.processing.bias_correction
                )
                if not success:
                    await self._handle_error(f"Tissue segmentation failed for {modality}: {error}")
                    return
                segmentation_outputs[modality] = outputs
            
            await self._send_message(
                create_status_update(
                    sender=self.config.name,
                    recipient="coordinator",
                    state="preprocessing",
                    progress=0.7,
                    message="Tissue segmentation complete"
                )
            )
            
            # Register T2-FLAIR to T1 space
            if "T2_FLAIR" in metadata and "T1" in metadata:
                output_path = self.subject_dir / "reg" / "T2_FLAIR_reg.nii.gz"
                transform_path = output_path.with_suffix(".mat")
                
                success, error, cost = await register_images(
                    segmentation_outputs["T2_FLAIR"]["bias_corrected"],
                    segmentation_outputs["T1"]["bias_corrected"],
                    output_path,
                    transform_path,
                    cost_function=self.preprocessing_config.fsl.registration_cost,
                    dof=self.preprocessing_config.processing.registration_dof,
                    interp=self.preprocessing_config.fsl.interpolation
                )
                
                if not success:
                    await self._handle_error(f"Registration failed: {error}")
                    return
                
                # Calculate QC metrics
                valid, error, stats = await validate_image(output_path)
                if not valid:
                    await self._handle_error(f"Output validation failed: {error}")
                    return
                
                # Calculate tissue volumes
                tissue_volumes = {}
                for tissue, seg_path in segmentation_outputs["T1"].items():
                    if tissue.startswith("class_"):
                        success, stdout, stderr = await run_freesurfer_command(
                            "mri_segstats",
                            ["--seg", str(seg_path), "--sum"]
                        )
                        if success and stdout:
                            try:
                                volume = float(stdout.split("\n")[-2].split()[-1])
                                tissue_volumes[tissue] = volume
                            except (IndexError, ValueError):
                                pass
                
                metrics = PreprocessingMetrics(
                    snr=stats["snr"],
                    contrast_to_noise=stats["contrast"],
                    registration_cost=cost,
                    brain_volume=sum(tissue_volumes.values()),
                    tissue_volumes=tissue_volumes,
                    segmentation_quality=stats["entropy"]
                )
                
                # Send success message
                await self._send_message(
                    create_data_message(
                        sender=self.config.name,
                        recipient="coordinator",
                        data_type="preprocessing_results",
                        content={
                            "subject_id": subject_id,
                            "preprocessed_files": {
                                "T1": str(segmentation_outputs["T1"]["bias_corrected"]),
                                "T1_brain": str(brain_masks["T1"]),
                                "T1_seg": str(segmentation_outputs["T1"]["segmentation"]),
                                "T2_FLAIR": str(output_path),
                                "T2_FLAIR_brain": str(brain_masks["T2_FLAIR"]),
                                "T2_FLAIR_seg": str(segmentation_outputs["T2_FLAIR"]["segmentation"])
                            },
                            "metrics": metrics.dict()
                        }
                    )
                )
                
                await self._send_message(
                    create_status_update(
                        sender=self.config.name,
                        recipient="coordinator",
                        state="completed",
                        progress=1.0,
                        message="Preprocessing complete"
                    )
                )
            
        except Exception as e:
            await self._handle_error(f"Preprocessing failed: {str(e)}")

    async def _validate_input(
        self,
        input_files: Dict[str, Path]
    ) -> Dict[str, ImageMetadata]:
        """Validate input image files and extract metadata"""
        metadata = {}
        required_modalities = ["T1", "T2_FLAIR"]
        
        for modality, file_path in input_files.items():
            if not file_path.exists():
                await self._handle_error(f"Input file not found: {file_path}")
                continue
            
            try:
                img = nib.load(str(file_path))
                metadata[modality] = ImageMetadata(
                    path=file_path,
                    modality=modality,
                    dimensions=img.shape,
                    voxel_size=img.header.get_zooms(),
                    orientation=nib.aff2axcodes(img.affine),
                    data_type=img.get_data_dtype().name
                )
            except Exception as e:
                await self._handle_error(f"Error reading {modality} file: {str(e)}")
                continue
        
        missing = set(required_modalities) - set(metadata.keys())
        if missing:
            await self._handle_error(f"Missing required modalities: {missing}")
            return {}
        
        return metadata

    async def _cleanup(self) -> None:
        """Cleanup agent resources"""
        await super()._cleanup()
        self.logger.info("Cleaning up preprocessing agent")
        # Additional cleanup if needed