"""
Configuration management for neuroimaging pipeline.
Handles environment settings, pipeline parameters, and agent configurations.
"""
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, validator


class RegistrationCost(str, Enum):
    """Valid cost functions for registration"""
    CORRELATION_RATIO = "corratio"
    MUTUAL_INFO = "mutualinfo"
    NORMALIZED_CORRELATION = "normcorr"
    NORMALIZED_MUTUAL_INFO = "normmi"


class InterpolationMethod(str, Enum):
    """Valid interpolation methods"""
    TRILINEAR = "trilinear"
    NEAREST_NEIGHBOUR = "nearestneighbour"
    SPLINE = "spline"


class IntensityRange(BaseModel):
    """Intensity normalization range"""
    min: int = Field(0, ge=0)
    max: int = Field(4095, gt=0)

    @validator("max")
    def max_greater_than_min(cls, v: int, values: Dict[str, Any]) -> int:
        """Validate max is greater than min"""
        if "min" in values and v <= values["min"]:
            raise ValueError("max must be greater than min")
        return v


class BiasFieldCorrectionOptions(BaseModel):
    """Options for bias field correction"""
    convergence: float = Field(0.001, gt=0)
    shrink_factor: int = Field(4, gt=0)
    iterations: List[int] = Field(default=[50, 50, 50, 50])

    @validator("iterations")
    def validate_iterations(cls, v: List[int]) -> List[int]:
        """Validate iteration counts"""
        if not all(i > 0 for i in v):
            raise ValueError("All iteration counts must be positive")
        return v


class RegistrationOptions(BaseModel):
    """Options for image registration"""
    cost: RegistrationCost = RegistrationCost.CORRELATION_RATIO
    search_cost: RegistrationCost = RegistrationCost.CORRELATION_RATIO
    bins: int = Field(256, gt=0)
    interp: InterpolationMethod = InterpolationMethod.TRILINEAR
    dof: int = Field(12, ge=6, le=12)
    schedule: str = "default"


class FSLConfig(BaseModel):
    """FSL-specific configuration"""
    fsl_dir: Path = Field(..., description="Path to FSL installation")
    standard_brain: Path = Field(..., description="Path to standard brain template")
    registration_cost: RegistrationCost = Field(
        default=RegistrationCost.CORRELATION_RATIO,
        description="Cost function for registration"
    )
    interpolation: InterpolationMethod = Field(
        default=InterpolationMethod.TRILINEAR,
        description="Interpolation method"
    )
    
    @validator("fsl_dir")
    def validate_fsl_dir(cls, v: Path) -> Path:
        """Validate FSL directory exists"""
        if not v.exists():
            raise ValueError(f"FSL directory not found: {v}")
        return v


class FreeSurferConfig(BaseModel):
    """FreeSurfer-specific configuration"""
    freesurfer_home: Path = Field(..., description="Path to FreeSurfer installation")
    subjects_dir: Path = Field(..., description="Path to subjects directory")
    license_file: Path = Field(..., description="Path to license file")
    
    @validator("freesurfer_home")
    def validate_freesurfer_home(cls, v: Path) -> Path:
        """Validate FreeSurfer directory exists"""
        if not v.exists():
            raise ValueError(f"FreeSurfer directory not found: {v}")
        return v


class AnthropicConfig(BaseModel):
    """Anthropic API configuration"""
    api_key: str = Field(..., description="Anthropic API key")
    model: str = Field(
        default="claude-3-sonnet-20240229",
        description="Model to use"
    )
    max_tokens: int = Field(
        default=1000,
        description="Maximum tokens per request"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Sampling temperature"
    )


class ProcessingConfig(BaseModel):
    """Image processing configuration"""
    normalize_intensity: bool = Field(
        default=True,
        description="Apply intensity normalization"
    )
    bias_correction: bool = Field(
        default=True,
        description="Apply bias field correction"
    )
    skull_strip: bool = Field(
        default=True,
        description="Apply skull stripping"
    )
    registration_dof: int = Field(
        default=12,
        ge=6,
        le=12,
        description="Degrees of freedom for registration"
    )
    smoothing_fwhm: float = Field(
        default=2.0,
        gt=0.0,
        description="Smoothing kernel FWHM in mm"
    )
    intensity_range: IntensityRange = Field(
        default_factory=IntensityRange,
        description="Intensity normalization range"
    )
    bias_correction_options: BiasFieldCorrectionOptions = Field(
        default_factory=BiasFieldCorrectionOptions,
        description="Bias field correction options"
    )
    registration_options: RegistrationOptions = Field(
        default_factory=RegistrationOptions,
        description="Registration options"
    )


class SegmentationConfig(BaseModel):
    """Segmentation configuration"""
    regions_of_interest: Dict[str, int] = Field(
        default={
            "L_Thal": 10,
            "R_Thal": 49,
            "L_Caud": 11,
            "R_Caud": 50,
            "L_Puta": 12,
            "R_Puta": 51,
            "L_Pall": 13,
            "R_Pall": 52,
            "L_Hipp": 17,
            "R_Hipp": 53,
            "L_Amyg": 18,
            "R_Amyg": 54
        },
        description="Region labels and values"
    )
    probability_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability threshold for segmentation"
    )
    min_region_size: int = Field(
        default=10,
        gt=0,
        description="Minimum region size in voxels"
    )


class ClusteringConfig(BaseModel):
    """Clustering configuration"""
    outlier_threshold: float = Field(
        default=2.0,
        gt=0.0,
        description="Z-score threshold for outliers"
    )
    min_cluster_size: int = Field(
        default=5,
        gt=0,
        description="Minimum cluster size"
    )
    connectivity: int = Field(
        default=26,
        description="Voxel connectivity for clustering"
    )

    @validator("connectivity")
    def validate_connectivity(cls, v: int) -> int:
        """Validate connectivity value"""
        if v not in {6, 18, 26}:
            raise ValueError("Connectivity must be 6, 18, or 26")
        return v


class VisualizationConfig(BaseModel):
    """Visualization configuration"""
    output_dpi: int = Field(
        default=300,
        gt=0,
        description="Output DPI for images"
    )
    slice_spacing: int = Field(
        default=5,
        gt=0,
        description="Spacing between slices"
    )
    overlay_alpha: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Overlay transparency"
    )
    anomaly_cmap: str = Field(
        default="hot",
        description="Colormap for anomalies"
    )
    background_cmap: str = Field(
        default="gray",
        description="Colormap for background"
    )


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field(
        default="INFO",
        description="Log level"
    )
    file: Path = Field(
        default=Path("pipeline.log"),
        description="Log file path"
    )
    max_size_mb: int = Field(
        default=100,
        gt=0,
        description="Maximum log file size in MB"
    )
    backup_count: int = Field(
        default=5,
        ge=0,
        description="Number of backup files to keep"
    )
    timestamp: bool = Field(
        default=True,
        description="Include timestamps in logs"
    )
    process_id: bool = Field(
        default=True,
        description="Include process ID in logs"
    )

    @validator("level")
    def validate_level(cls, v: str) -> str:
        """Validate log level"""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()


class ErrorHandlingConfig(BaseModel):
    """Error handling configuration"""
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries"
    )
    retry_delay: float = Field(
        default=5.0,
        ge=0.0,
        description="Delay between retries in seconds"
    )
    continue_on_error: bool = Field(
        default=False,
        description="Continue pipeline on non-critical errors"
    )
    critical_errors: List[str] = Field(
        default=[
            "REGISTRATION_FAILED",
            "SEGMENTATION_FAILED",
            "INSUFFICIENT_RESOURCES"
        ],
        description="Error severity levels that trigger pipeline stop"
    )


class PipelineConfig(BaseModel):
    """Complete pipeline configuration"""
    # Environment
    fsl: FSLConfig
    freesurfer: FreeSurferConfig
    anthropic: AnthropicConfig
    
    # Processing
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    
    # System
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    error_handling: ErrorHandlingConfig = Field(default_factory=ErrorHandlingConfig)
    
    # Paths
    working_dir: Path = Field(
        default=Path("work"),
        description="Working directory"
    )
    output_dir: Path = Field(
        default=Path("output"),
        description="Output directory"
    )
    
    @classmethod
    def from_environment(cls) -> "PipelineConfig":
        """Create configuration from environment variables"""
        return cls(
            fsl=FSLConfig(
                fsl_dir=Path(os.getenv("FSLDIR", "/usr/local/fsl")),
                standard_brain=Path(os.getenv(
                    "FSLDIR",
                    "/usr/local/fsl"
                )) / "data" / "standard" / "MNI152_T1_2mm_brain.nii.gz"
            ),
            freesurfer=FreeSurferConfig(
                freesurfer_home=Path(os.getenv(
                    "FREESURFER_HOME",
                    "/usr/local/freesurfer"
                )),
                subjects_dir=Path(os.getenv(
                    "SUBJECTS_DIR",
                    "subjects"
                )),
                license_file=Path(os.getenv(
                    "FS_LICENSE",
                    "/usr/local/freesurfer/license.txt"
                ))
            ),
            anthropic=AnthropicConfig(
                api_key=os.getenv("ANTHROPIC_API_KEY", "")
            )
        )
    
    @classmethod
    def from_file(cls, config_path: Path) -> "PipelineConfig":
        """Load configuration from file"""
        import yaml
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)
    
    def save(self, config_path: Path) -> None:
        """Save configuration to file"""
        import yaml
        
        config_data = self.dict()
        
        # Convert paths to strings
        def convert_paths(data: Any) -> Any:
            if isinstance(data, Path):
                return str(data)
            elif isinstance(data, dict):
                return {k: convert_paths(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [convert_paths(v) for v in data]
            return data
        
        config_data = convert_paths(config_data)
        
        with open(config_path, "w") as f:
            yaml.safe_dump(config_data, f, default_flow_style=False)


def load_config(
    config_path: Optional[Path] = None,
    from_env: bool = True
) -> PipelineConfig:
    """Load pipeline configuration"""
    if config_path and config_path.exists():
        return PipelineConfig.from_file(config_path)
    elif from_env:
        return PipelineConfig.from_environment()
    else:
        raise ValueError(
            "Must provide either config_path or set from_env=True"
        )