"""
Configuration management for neuroimaging pipeline.
Handles environment settings, pipeline parameters, and agent configurations.
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, validator
import yaml

from core.workflow import ResourceRequirements


class FSLConfig(BaseModel):
    """FSL-specific configuration"""
    fsl_dir: Path = Field(..., description="Path to FSL installation")
    standard_brain: Path = Field(..., description="Path to standard brain template")
    registration_cost: str = Field(
        default="corratio",
        description="Cost function for registration"
    )
    interpolation: str = Field(
        default="trilinear",
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
        description="Degrees of freedom for registration"
    )
    smoothing_fwhm: float = Field(
        default=2.0,
        description="Smoothing kernel FWHM in mm"
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
        description="Probability threshold for segmentation"
    )
    min_region_size: int = Field(
        default=10,
        description="Minimum region size in voxels"
    )


class ClusteringConfig(BaseModel):
    """Clustering configuration"""
    outlier_threshold: float = Field(
        default=2.0,
        description="Z-score threshold for outliers"
    )
    min_cluster_size: int = Field(
        default=5,
        description="Minimum cluster size"
    )
    connectivity: int = Field(
        default=26,
        description="Voxel connectivity for clustering"
    )


class VisualizationConfig(BaseModel):
    """Visualization configuration"""
    output_dpi: int = Field(
        default=300,
        description="Output DPI for images"
    )
    slice_spacing: int = Field(
        default=5,
        description="Spacing between slices"
    )
    overlay_alpha: float = Field(
        default=0.35,
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


class AgentResourceConfig(BaseModel):
    """Resource configuration for agents"""
    preprocessor: ResourceRequirements = Field(
        default=ResourceRequirements(
            cpu_cores=2,
            memory_gb=4.0
        )
    )
    analyzer: ResourceRequirements = Field(
        default=ResourceRequirements(
            cpu_cores=2,
            memory_gb=8.0
        )
    )
    visualizer: ResourceRequirements = Field(
        default=ResourceRequirements(
            cpu_cores=1,
            memory_gb=4.0
        )
    )


class CacheConfig(BaseModel):
    """Cache configuration"""
    cache_dir: Path = Field(
        default=Path("cache"),
        description="Cache directory"
    )
    max_cache_size_gb: float = Field(
        default=10.0,
        description="Maximum cache size in GB"
    )
    cache_ttl: int = Field(
        default=7 * 24 * 60 * 60,  # 1 week in seconds
        description="Cache TTL in seconds"
    )
    similarity_threshold: float = Field(
        default=0.8,
        description="Similarity threshold for cache hits"
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
    
    # Resources
    resources: AgentResourceConfig = Field(default_factory=AgentResourceConfig)
    
    # Caching
    cache: CacheConfig = Field(default_factory=CacheConfig)
    
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