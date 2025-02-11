"""
Test suite for image registration.
Tests registration functionality and transforms.
"""
import numpy as np
import pytest
import SimpleITK as sitk
from pathlib import Path

from utils.registration import (
    ImageRegistration,
    MetricType,
    RegistrationResult,
    TransformType
)
from tests.test_measurement import test_image


@pytest.fixture
def fixed_image(test_image):
    """Fixture for fixed image"""
    return test_image


@pytest.fixture
def moving_image(test_image):
    """Fixture for moving image to register"""
    # Create transformed version of test image
    data = test_image.get_fdata()
    affine = test_image.affine.copy()
    
    # Apply translation
    affine[:3, 3] += 10  # 10mm translation
    
    # Apply rotation
    theta = np.pi / 8  # 22.5 degrees
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    affine[:3, :3] = rotation @ affine[:3, :3]
    
    return type(test_image)(data, affine)


@pytest.fixture
def registration(
    fixed_image,
    moving_image
) -> ImageRegistration:
    """Fixture for image registration"""
    return ImageRegistration(
        fixed_image,
        moving_image,
        transform_type=TransformType.AFFINE,
        metric_type=MetricType.MUTUAL_INFORMATION
    )


class TestImageRegistration:
    """Tests for image registration"""
    
    def test_initialization(self, registration: ImageRegistration):
        """Test registration initialization"""
        assert registration.fixed is not None
        assert registration.moving is not None
        assert registration.fixed_sitk is not None
        assert registration.moving_sitk is not None
        assert registration.transform_type == TransformType.AFFINE
        assert registration.metric_type == MetricType.MUTUAL_INFORMATION
    
    def test_image_conversion(self, registration: ImageRegistration):
        """Test image format conversion"""
        # Test NIfTI to SimpleITK
        sitk_image = registration.nifti_to_sitk(registration.fixed)
        assert isinstance(sitk_image, sitk.Image)
        assert sitk_image.GetDimension() == 3
        assert sitk_image.GetSize() == registration.fixed.shape
        
        # Test SimpleITK to NIfTI
        nifti_image = registration.sitk_to_nifti(sitk_image, registration.fixed)
        assert type(nifti_image) == type(registration.fixed)
        assert np.allclose(nifti_image.affine, registration.fixed.affine)
        assert nifti_image.shape == registration.fixed.shape
    
    def test_metric_setup(self, registration: ImageRegistration):
        """Test metric setup"""
        # Test different metrics
        for metric_type in MetricType:
            registration.metric_type = metric_type
            registration.setup_metric()
            assert registration.registration is not None
    
    def test_optimizer_setup(self, registration: ImageRegistration):
        """Test optimizer setup"""
        # Test different transforms
        for transform_type in TransformType:
            registration.transform_type = transform_type
            registration.setup_optimizer()
            assert registration.registration is not None
    
    def test_transform_setup(self, registration: ImageRegistration):
        """Test transform setup"""
        # Test different transforms
        for transform_type in TransformType:
            registration.transform_type = transform_type
            registration.setup_transform()
            assert registration.registration is not None
    
    def test_registration(self, registration: ImageRegistration):
        """Test registration process"""
        # Perform registration
        result = registration.register()
        
        # Check result
        assert isinstance(result, RegistrationResult)
        assert result.success
        assert result.transform is not None
        assert result.metrics is not None
        assert result.transformed_image is not None
        assert result.error is None
        
        # Check metrics
        assert "similarity" in result.metrics
        assert "iterations" in result.metrics
        assert "stop_condition" in result.metrics
        
        # Check transformed image
        assert result.transformed_image.shape == registration.fixed.shape
        assert np.allclose(
            result.transformed_image.affine,
            registration.fixed.affine
        )
    
    def test_transform_io(
        self,
        registration: ImageRegistration,
        tmp_path: Path
    ):
        """Test transform I/O"""
        # Perform registration
        result = registration.register()
        assert result.success
        
        # Save transform
        transform_path = tmp_path / "transform.tfm"
        success = registration.save_transform(transform_path)
        assert success
        assert transform_path.exists()
        
        # Create new registration
        new_registration = ImageRegistration(
            registration.fixed,
            registration.moving
        )
        
        # Load transform
        success = new_registration.load_transform(transform_path)
        assert success
        
        # Compare results
        new_result = new_registration.register()
        assert new_result.success
        assert np.allclose(
            new_result.metrics["similarity"],
            result.metrics["similarity"],
            rtol=1e-5
        )
    
    def test_error_handling(self, registration: ImageRegistration):
        """Test error handling"""
        # Test invalid fixed image
        invalid_registration = ImageRegistration(
            None,
            registration.moving
        )
        result = invalid_registration.register()
        assert not result.success
        assert result.error is not None
        
        # Test invalid transform file
        success = registration.load_transform(Path("nonexistent.tfm"))
        assert not success
    
    @pytest.mark.parametrize("transform_type", list(TransformType))
    def test_transform_types(
        self,
        registration: ImageRegistration,
        transform_type: TransformType
    ):
        """Test different transform types"""
        registration.transform_type = transform_type
        registration.setup_transform()
        result = registration.register()
        assert result.success
        assert result.transform is not None
    
    @pytest.mark.parametrize("metric_type", list(MetricType))
    def test_metric_types(
        self,
        registration: ImageRegistration,
        metric_type: MetricType
    ):
        """Test different metric types"""
        registration.metric_type = metric_type
        registration.setup_metric()
        result = registration.register()
        assert result.success
        assert result.metrics["similarity"] is not None