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
            fsl_cmds = ["fslmaths", "fslstats", "flirt"]
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
            for subdir in ["orig", "prep", "reg", "qc"]:
                (self.subject_dir / subdir).mkdir(parents=True, exist_ok=True)
            
            # Validate inputs
            metadata = await self._validate_input(input_files)
            if not metadata:
                return
            
            # Send status update
            await self._send_message(
                create_status_update(
                    sender=self.config.name,
                    recipient="coordinator",
                    state="preprocessing",
                    progress=0.2,
                    message="Input validation complete"
                )
            )
            
            # Apply bias correction if configured
            if self.preprocessing_config.processing.bias_correction:
                for modality, meta in metadata.items():
                    output_path = self.subject_dir / "prep" / f"{modality}_bias_corr.nii.gz"
                    success, error = await self._apply_bias_correction(
                        meta.path,
                        output_path
                    )
                    if not success:
                        await self._handle_error(f"Bias correction failed for {modality}: {error}")
                        return
                    metadata[modality] = meta._replace(path=output_path)
            
            # Normalize resolution
            t1_metadata = metadata["T1"]
            target_resolution = t1_metadata.voxel_size
            
            normalized_files = {}
            for modality, meta in metadata.items():
                output_path = self.subject_dir / "prep" / f"{modality}_norm.nii.gz"
                
                success, error = await normalize_image(
                    meta.path,
                    output_path,
                    target_resolution,
                    meta.voxel_size
                )
                
                if not success:
                    await self._handle_error(f"Normalization failed for {modality}: {error}")
                    return
                
                normalized_files[modality] = output_path
            
            await self._send_message(
                create_status_update(
                    sender=self.config.name,
                    recipient="coordinator",
                    state="preprocessing",
                    progress=0.5,
                    message="Resolution normalization complete"
                )
            )
            
            # Register T2-FLAIR to T1
            if "T2_FLAIR" in normalized_files:
                output_path = self.subject_dir / "reg" / "T2_FLAIR_reg.nii.gz"
                transform_path = output_path.with_suffix(".mat")
                
                success, error, cost = await register_images(
                    normalized_files["T2_FLAIR"],
                    normalized_files["T1"],
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
                
                metrics = PreprocessingMetrics(
                    snr=stats["mean"] / stats["std"] if stats["std"] > 0 else 0,
                    contrast_to_noise=(stats["max"] - stats["min"]) / stats["std"] if stats["std"] > 0 else 0,
                    registration_cost=cost
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
                                "T1": str(normalized_files["T1"]),
                                "T2_FLAIR": str(output_path)
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

    async def _apply_bias_correction(
        self,
        input_path: Path,
        output_path: Path
    ) -> Tuple[bool, Optional[str]]:
        """Apply bias field correction using FSL FAST"""
        try:
            success, stdout, stderr = await run_fsl_command(
                "fast",
                ["-B", str(input_path), "-o", str(output_path)]
            )
            
            if not success:
                return False, f"Bias correction failed: {stderr}"
            
            return True, None
            
        except Exception as e:
            return False, f"Error during bias correction: {str(e)}"

    async def _cleanup(self) -> None:
        """Cleanup agent resources"""
        await super()._cleanup()
        self.logger.info("Cleaning up preprocessing agent")
        # Additional cleanup if needed