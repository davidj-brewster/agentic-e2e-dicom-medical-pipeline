"""
Image registration functionality.
Handles registration between images using SimpleITK.
"""
import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


class TransformType(Enum):
    """Available transform types"""
    RIGID = auto()       # Translation and rotation
    AFFINE = auto()      # Rigid + scaling and shearing
    BSPLINE = auto()     # Deformable B-spline
    DEMONS = auto()      # Deformable demons


class MetricType(Enum):
    """Available similarity metrics"""
    MEAN_SQUARES = auto()           # Mean squared difference
    MUTUAL_INFORMATION = auto()     # Mutual information
    CORRELATION = auto()            # Cross correlation
    MATTES = auto()                # Mattes mutual information


@dataclass
class RegistrationResult:
    """Registration result"""
    success: bool
    transform: Optional[sitk.Transform]
    metrics: Dict[str, float]
    transformed_image: Optional[nib.Nifti1Image] = None
    error: Optional[str] = None


class ImageRegistration:
    """Image registration system"""
    
    def __init__(
        self,
        fixed_image: nib.Nifti1Image,
        moving_image: nib.Nifti1Image,
        transform_type: TransformType = TransformType.AFFINE,
        metric_type: MetricType = MetricType.MUTUAL_INFORMATION
    ):
        """
        Initialize registration.
        
        Args:
            fixed_image: Target image
            moving_image: Image to register
            transform_type: Type of transform
            metric_type: Type of similarity metric
        """
        self.fixed = fixed_image
        self.moving = moving_image
        self.transform_type = transform_type
        self.metric_type = metric_type
        
        # Convert to SimpleITK images
        self.fixed_sitk = self.nifti_to_sitk(fixed_image)
        self.moving_sitk = self.nifti_to_sitk(moving_image)
        
        # Initialize registration
        self.registration = sitk.ImageRegistrationMethod()
        
        # Setup metric
        self.setup_metric()
        
        # Setup optimizer
        self.setup_optimizer()
        
        # Setup transform
        self.setup_transform()
    
    def nifti_to_sitk(self, image: nib.Nifti1Image) -> sitk.Image:
        """
        Convert NIfTI to SimpleITK image.
        
        Args:
            image: NIfTI image
            
        Returns:
            SimpleITK image
        """
        # Get data and affine
        data = image.get_fdata()
        affine = image.affine
        
        # Create SimpleITK image
        sitk_image = sitk.GetImageFromArray(data)
        
        # Set direction and spacing from affine
        direction = affine[:3, :3].flatten()
        spacing = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
        origin = affine[:3, 3]
        
        sitk_image.SetDirection(tuple(direction))
        sitk_image.SetSpacing(tuple(spacing))
        sitk_image.SetOrigin(tuple(origin))
        
        return sitk_image
    
    def sitk_to_nifti(
        self,
        image: sitk.Image,
        reference: nib.Nifti1Image
    ) -> nib.Nifti1Image:
        """
        Convert SimpleITK to NIfTI image.
        
        Args:
            image: SimpleITK image
            reference: Reference NIfTI image
            
        Returns:
            NIfTI image
        """
        # Get data
        data = sitk.GetArrayFromImage(image)
        
        # Create NIfTI image with reference header
        return nib.Nifti1Image(data, reference.affine, reference.header)
    
    def setup_metric(self) -> None:
        """Setup similarity metric"""
        if self.metric_type == MetricType.MEAN_SQUARES:
            self.registration.SetMetricAsMeanSquares()
        elif self.metric_type == MetricType.MUTUAL_INFORMATION:
            self.registration.SetMetricAsMattesMutualInformation()
        elif self.metric_type == MetricType.CORRELATION:
            self.registration.SetMetricAsCorrelation()
        else:  # MATTES
            self.registration.SetMetricAsMattesMutualInformation()
    
    def setup_optimizer(self) -> None:
        """Setup optimizer"""
        if self.transform_type in [TransformType.RIGID, TransformType.AFFINE]:
            # Regular step gradient descent
            self.registration.SetOptimizerAsRegularStepGradientDescent(
                learningRate=1.0,
                minStep=0.001,
                numberOfIterations=100,
                gradientMagnitudeTolerance=1e-8
            )
        else:
            # LBFGS optimizer for deformable registration
            self.registration.SetOptimizerAsLBFGS2(
                numberOfIterations=50,
                maximumNumberOfCorrections=5,
                maximumNumberOfFunctionEvaluations=1000,
                gradientConvergenceTolerance=1e-8
            )
    
    def setup_transform(self) -> None:
        """Setup transform"""
        if self.transform_type == TransformType.RIGID:
            transform = sitk.Euler3DTransform()
        elif self.transform_type == TransformType.AFFINE:
            transform = sitk.AffineTransform(3)
        elif self.transform_type == TransformType.BSPLINE:
            transform = sitk.BSplineTransform(3)
        else:  # DEMONS
            transform = sitk.DisplacementFieldTransform(3)
        
        # Initialize transform
        initial_transform = sitk.CenteredTransformInitializer(
            self.fixed_sitk,
            self.moving_sitk,
            transform
        )
        self.registration.SetInitialTransform(initial_transform)
    
    def register(self) -> RegistrationResult:
        """
        Perform registration.
        
        Returns:
            Registration result
        """
        try:
            # Perform registration
            transform = self.registration.Execute(
                self.fixed_sitk,
                self.moving_sitk
            )
            
            # Transform moving image
            transformed = sitk.Resample(
                self.moving_sitk,
                self.fixed_sitk,
                transform,
                sitk.sitkLinear,
                0.0,
                self.moving_sitk.GetPixelID()
            )
            
            # Convert back to NIfTI
            transformed_nifti = self.sitk_to_nifti(
                transformed,
                self.fixed
            )
            
            # Get metrics
            metrics = {
                "similarity": self.registration.GetMetricValue(),
                "iterations": self.registration.GetOptimizerIteration(),
                "stop_condition": self.registration.GetOptimizerStopConditionDescription()
            }
            
            return RegistrationResult(
                success=True,
                transform=transform,
                metrics=metrics,
                transformed_image=transformed_nifti
            )
            
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return RegistrationResult(
                success=False,
                transform=None,
                metrics={},
                error=str(e)
            )
    
    def save_transform(self, path: Path) -> bool:
        """
        Save transform parameters.
        
        Args:
            path: Output file path
            
        Returns:
            Success flag
        """
        try:
            sitk.WriteTransform(self.registration.GetTransform(), str(path))
            return True
        except Exception as e:
            logger.error(f"Failed to save transform: {e}")
            return False
    
    def load_transform(self, path: Path) -> bool:
        """
        Load transform parameters.
        
        Args:
            path: Input file path
            
        Returns:
            Success flag
        """
        try:
            transform = sitk.ReadTransform(str(path))
            self.registration.SetInitialTransform(transform)
            return True
        except Exception as e:
            logger.error(f"Failed to load transform: {e}")
            return False