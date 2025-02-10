"""
Preprocessing Agent for handling FSL/FreeSurfer operations.
Manages image preprocessing, registration, and quality control.
"""
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from pydantic import BaseModel

from .coordinator import AgentMessage, MessageType, Priority, TaskStatus


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


class PreprocessingAgent:
    """
    Agent responsible for image preprocessing and registration using FSL/FreeSurfer tools.
    Handles data validation, normalization, and registration tasks.
    """

    def __init__(self, coordinator_id: str):
        self.agent_id = "preprocessor"
        self.coordinator_id = coordinator_id
        self.current_subject: Optional[str] = None
        self.working_dir: Optional[Path] = None
        self.env_setup_verified = False

    async def verify_environment(self) -> bool:
        """Verify FSL and FreeSurfer environment setup"""
        required_vars = {
            "FSLDIR": os.getenv("FSLDIR"),
            "FREESURFER_HOME": os.getenv("FREESURFER_HOME")
        }
        
        if not all(required_vars.values()):
            await self._send_error("Environment variables FSLDIR and/or FREESURFER_HOME not set")
            return False
        
        # Verify critical executables
        fsl_cmds = ["fslmaths", "fslstats", "flirt"]
        fs_cmds = ["mri_convert", "recon-all"]
        
        for cmd in fsl_cmds:
            if not os.path.exists(os.path.join(required_vars["FSLDIR"], "bin", cmd)):
                await self._send_error(f"FSL command {cmd} not found")
                return False
        
        for cmd in fs_cmds:
            if not os.path.exists(os.path.join(required_vars["FREESURFER_HOME"], "bin", cmd)):
                await self._send_error(f"FreeSurfer command {cmd} not found")
                return False
        
        self.env_setup_verified = True
        return True

    async def initialize_subject(self, subject_id: str, working_dir: Path) -> None:
        """Initialize processing for a new subject"""
        self.current_subject = subject_id
        self.working_dir = working_dir
        
        # Create subject-specific directories
        subject_dir = working_dir / subject_id
        for subdir in ["orig", "prep", "reg", "qc"]:
            (subject_dir / subdir).mkdir(parents=True, exist_ok=True)

    async def validate_input(self, input_files: Dict[str, Path]) -> Dict[str, ImageMetadata]:
        """Validate input image files and extract metadata"""
        metadata = {}
        required_modalities = ["T1", "T2_FLAIR"]
        
        for modality, file_path in input_files.items():
            if not file_path.exists():
                await self._send_error(f"Input file not found: {file_path}")
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
                await self._send_error(f"Error reading {modality} file: {str(e)}")
                continue
        
        missing = set(required_modalities) - set(metadata.keys())
        if missing:
            await self._send_error(f"Missing required modalities: {missing}")
        
        return metadata

    async def normalize_resolution(self, image_path: Path, target_resolution: Tuple[float, ...]) -> Path:
        """Normalize image resolution using FSL"""
        if not self.env_setup_verified:
            await self.verify_environment()
        
        output_path = self.working_dir / self.current_subject / "prep" / f"{image_path.stem}_norm.nii.gz"
        
        try:
            # Get current resolution
            img = nib.load(str(image_path))
            current_res = img.header.get_zooms()
            
            if current_res == target_resolution:
                return image_path
            
            # Calculate scaling factors
            scale_factors = [t / c for t, c in zip(target_resolution, current_res)]
            
            # Use fslmaths for resampling
            cmd = f"fslmaths {image_path} -subsamp2 {' '.join(map(str, scale_factors))} {output_path}"
            os.system(cmd)  # In real implementation, use subprocess with proper error handling
            
            return output_path
        except Exception as e:
            await self._send_error(f"Resolution normalization failed: {str(e)}")
            return image_path

    async def register_to_t1(self, moving_path: Path, fixed_path: Path) -> Path:
        """Register T2-FLAIR to T1 space using FSL FLIRT"""
        if not self.env_setup_verified:
            await self.verify_environment()
        
        output_path = self.working_dir / self.current_subject / "reg" / f"{moving_path.stem}_reg.nii.gz"
        transform_path = output_path.with_suffix(".mat")
        
        try:
            # Run FLIRT registration
            cmd = (
                f"flirt -in {moving_path} -ref {fixed_path} "
                f"-out {output_path} -omat {transform_path} "
                "-cost corratio -dof 12 -interp trilinear"
            )
            os.system(cmd)  # In real implementation, use subprocess with proper error handling
            
            return output_path
        except Exception as e:
            await self._send_error(f"Registration failed: {str(e)}")
            return moving_path

    async def calculate_metrics(self, image_path: Path) -> PreprocessingMetrics:
        """Calculate quality control metrics for preprocessed image"""
        try:
            img = nib.load(str(image_path))
            data = img.get_fdata()
            
            # Calculate SNR
            signal = np.mean(data)
            noise = np.std(data)
            snr = signal / noise if noise > 0 else 0
            
            # Calculate CNR (simplified)
            percentile_95 = np.percentile(data, 95)
            percentile_5 = np.percentile(data, 5)
            cnr = (percentile_95 - percentile_5) / noise if noise > 0 else 0
            
            return PreprocessingMetrics(
                snr=float(snr),
                contrast_to_noise=float(cnr)
            )
        except Exception as e:
            await self._send_error(f"Metrics calculation failed: {str(e)}")
            return PreprocessingMetrics(snr=0.0, contrast_to_noise=0.0)

    async def run_preprocessing(self, subject_id: str, input_files: Dict[str, Path]) -> None:
        """Execute the complete preprocessing pipeline"""
        try:
            # Initialize subject
            working_dir = Path("work")  # In real implementation, get from config
            await self.initialize_subject(subject_id, working_dir)
            
            # Validate inputs
            metadata = await self.validate_input(input_files)
            if not metadata:
                return
            
            # Normalize resolution (to T1 resolution)
            t1_metadata = metadata["T1"]
            target_resolution = t1_metadata.voxel_size
            
            normalized_files = {}
            for modality, meta in metadata.items():
                normalized_path = await self.normalize_resolution(meta.path, target_resolution)
                normalized_files[modality] = normalized_path
            
            # Register T2-FLAIR to T1
            if "T2_FLAIR" in normalized_files:
                registered_flair = await self.register_to_t1(
                    normalized_files["T2_FLAIR"],
                    normalized_files["T1"]
                )
                
                # Calculate QC metrics
                metrics = await self.calculate_metrics(registered_flair)
                
                # Send success message with results
                await self._send_message(
                    MessageType.RESULT,
                    {
                        "subject_id": subject_id,
                        "preprocessed_files": {
                            "T1": str(normalized_files["T1"]),
                            "T2_FLAIR": str(registered_flair)
                        },
                        "metrics": metrics.dict()
                    }
                )
            
        except Exception as e:
            await self._send_error(f"Preprocessing pipeline failed: {str(e)}")

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