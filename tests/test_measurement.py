"""
Test suite for measurement functionality.
Tests coordinate systems, unit conversion, and measurements.
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from utils.measurement import (
    Coordinate,
    CoordinateConverter,
    CoordinateSystem,
    MeasurementResult,
    MeasurementSystem,
    UnitConverter,
    UnitSystem
)


@pytest.fixture
def test_image() -> nib.Nifti1Image:
    """Fixture for test image"""
    # Create test image with known properties
    data = np.zeros((64, 64, 64))
    affine = np.array([
        [-1.0, 0.0, 0.0, 32.0],
        [0.0, -1.0, 0.0, 32.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    return nib.Nifti1Image(data, affine)


@pytest.fixture
def coord_converter(test_image: nib.Nifti1Image) -> CoordinateConverter:
    """Fixture for coordinate converter"""
    return CoordinateConverter(test_image.affine)


@pytest.fixture
def unit_converter(test_image: nib.Nifti1Image) -> UnitConverter:
    """Fixture for unit converter"""
    return UnitConverter(test_image.header.get_zooms()[:3])


@pytest.fixture
def measurement_system(test_image: nib.Nifti1Image) -> MeasurementSystem:
    """Fixture for measurement system"""
    return MeasurementSystem(test_image)


class TestCoordinate:
    """Tests for coordinate functionality"""
    
    def test_coordinate_conversion(self):
        """Test coordinate array conversion"""
        coord = Coordinate(1.0, 2.0, 3.0)
        arr = coord.to_array()
        assert np.array_equal(arr, np.array([1.0, 2.0, 3.0]))
        
        new_coord = Coordinate.from_array(arr)
        assert new_coord.x == 1.0
        assert new_coord.y == 2.0
        assert new_coord.z == 3.0


class TestCoordinateConverter:
    """Tests for coordinate conversion"""
    
    def test_image_to_physical(self, coord_converter: CoordinateConverter):
        """Test image to physical conversion"""
        image_coord = Coordinate(32, 32, 32)
        physical = coord_converter.image_to_physical(image_coord)
        assert np.allclose(physical.x, 0.0)
        assert np.allclose(physical.y, 0.0)
        assert np.allclose(physical.z, 32.0)
    
    def test_physical_to_image(self, coord_converter: CoordinateConverter):
        """Test physical to image conversion"""
        physical_coord = Coordinate(0.0, 0.0, 32.0)
        image = coord_converter.physical_to_image(physical_coord)
        assert np.allclose(image.x, 32.0)
        assert np.allclose(image.y, 32.0)
        assert np.allclose(image.z, 32.0)
    
    def test_image_to_world(self, coord_converter: CoordinateConverter):
        """Test image to world conversion"""
        image_coord = Coordinate(32, 32, 32)
        world = coord_converter.image_to_world(image_coord)
        assert np.allclose(world.x, 0.0)
        assert np.allclose(world.y, 0.0)
        assert np.allclose(world.z, 32.0)
    
    def test_world_to_image(self, coord_converter: CoordinateConverter):
        """Test world to image conversion"""
        world_coord = Coordinate(0.0, 0.0, 32.0)
        image = coord_converter.world_to_image(world_coord)
        assert np.allclose(image.x, 32.0)
        assert np.allclose(image.y, 32.0)
        assert np.allclose(image.z, 32.0)


class TestUnitConverter:
    """Tests for unit conversion"""
    
    def test_voxel_conversion(self, unit_converter: UnitConverter):
        """Test voxel unit conversion"""
        # Test voxels to physical
        mm = unit_converter.voxels_to_physical(1.0, axis=0)
        assert mm == 1.0  # Assuming 1mm voxel size
        
        # Test physical to voxels
        voxels = unit_converter.physical_to_voxels(1.0, axis=0)
        assert voxels == 1.0
    
    def test_unit_conversion(self, unit_converter: UnitConverter):
        """Test unit system conversion"""
        # Test mm to cm
        cm = unit_converter.convert(
            10.0,
            UnitSystem.MILLIMETERS,
            UnitSystem.CENTIMETERS
        )
        assert cm == 1.0
        
        # Test cm to inches
        inches = unit_converter.convert(
            2.54,
            UnitSystem.CENTIMETERS,
            UnitSystem.INCHES
        )
        assert np.allclose(inches, 1.0)
        
        # Test voxels to cm
        cm = unit_converter.convert(
            10.0,
            UnitSystem.VOXELS,
            UnitSystem.CENTIMETERS,
            axis=0
        )
        assert cm == 1.0


class TestMeasurementSystem:
    """Tests for measurement system"""
    
    def test_coordinate_conversion(self, measurement_system: MeasurementSystem):
        """Test coordinate system conversion"""
        # Test image to physical
        image_coord = Coordinate(32, 32, 32)
        physical = measurement_system.convert_coordinates(
            image_coord,
            CoordinateSystem.IMAGE,
            CoordinateSystem.PHYSICAL
        )
        assert np.allclose(physical.x, 0.0)
        assert np.allclose(physical.y, 0.0)
        assert np.allclose(physical.z, 32.0)
        
        # Test physical to world
        world = measurement_system.convert_coordinates(
            physical,
            CoordinateSystem.PHYSICAL,
            CoordinateSystem.WORLD
        )
        assert np.allclose(world.x, 0.0)
        assert np.allclose(world.y, 0.0)
        assert np.allclose(world.z, 32.0)
    
    def test_value_conversion(self, measurement_system: MeasurementSystem):
        """Test value conversion"""
        # Test mm to cm
        cm = measurement_system.convert_value(
            10.0,
            UnitSystem.MILLIMETERS,
            UnitSystem.CENTIMETERS
        )
        assert cm == 1.0
        
        # Test voxels to mm
        mm = measurement_system.convert_value(
            1.0,
            UnitSystem.VOXELS,
            UnitSystem.MILLIMETERS,
            axis=0
        )
        assert mm == 1.0
    
    def test_result_storage(
        self,
        measurement_system: MeasurementSystem,
        tmp_path: Path
    ):
        """Test measurement result storage"""
        # Create test result
        result = MeasurementResult(
            value=10.0,
            unit=UnitSystem.MILLIMETERS,
            points=[Coordinate(0, 0, 0), Coordinate(10, 0, 0)],
            type="distance"
        )
        
        # Add result
        measurement_system.add_result(result)
        assert len(measurement_system.results) == 1
        
        # Save results
        output_path = tmp_path / "measurements.json"
        measurement_system.save_results(output_path)
        assert output_path.exists()
        
        # Clear results
        measurement_system.clear_results()
        assert len(measurement_system.results) == 0
        
        # Load results
        measurement_system.load_results(output_path)
        assert len(measurement_system.results) == 1
        loaded = measurement_system.results[0]
        assert loaded.value == 10.0
        assert loaded.unit == UnitSystem.MILLIMETERS
        assert len(loaded.points) == 2
        assert loaded.type == "distance"