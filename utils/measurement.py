"""
Measurement system for neuroimaging visualization.
Handles coordinate systems, unit conversion, and measurements.
"""
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D

from utils.interaction import Point

logger = logging.getLogger(__name__)


class CoordinateSystem(Enum):
    """Available coordinate systems"""
    IMAGE = auto()    # Voxel coordinates (i, j, k)
    PHYSICAL = auto() # Physical coordinates (x, y, z) in mm
    WORLD = auto()    # World coordinates (RAS+)


class UnitSystem(Enum):
    """Available unit systems"""
    VOXELS = auto()
    MILLIMETERS = auto()
    CENTIMETERS = auto()
    INCHES = auto()


@dataclass
class Coordinate:
    """3D coordinate"""
    x: float
    y: float
    z: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Coordinate":
        """Create from numpy array"""
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))


@dataclass
class MeasurementResult:
    """Measurement result"""
    value: float
    unit: UnitSystem
    points: List[Coordinate]
    type: str
    metadata: Optional[Dict] = None


class CoordinateConverter:
    """Converts between coordinate systems"""
    
    def __init__(self, affine: np.ndarray):
        """
        Initialize converter.
        
        Args:
            affine: 4x4 affine transformation matrix
        """
        self.affine = affine
        self.inverse = np.linalg.inv(affine)
    
    def image_to_physical(self, coord: Coordinate) -> Coordinate:
        """Convert image to physical coordinates"""
        point = np.append(coord.to_array(), 1)
        transformed = self.affine @ point
        return Coordinate.from_array(transformed[:3])
    
    def physical_to_image(self, coord: Coordinate) -> Coordinate:
        """Convert physical to image coordinates"""
        point = np.append(coord.to_array(), 1)
        transformed = self.inverse @ point
        return Coordinate.from_array(transformed[:3])
    
    def image_to_world(self, coord: Coordinate) -> Coordinate:
        """Convert image to world coordinates"""
        # First convert to physical
        physical = self.image_to_physical(coord)
        
        # Then apply RAS+ transformation
        # Note: This assumes the physical space is LPS+
        return Coordinate(
            x=-physical.x,  # L -> R
            y=-physical.y,  # P -> A
            z=physical.z    # S -> S
        )
    
    def world_to_image(self, coord: Coordinate) -> Coordinate:
        """Convert world to image coordinates"""
        # First convert to physical
        physical = Coordinate(
            x=-coord.x,  # R -> L
            y=-coord.y,  # A -> P
            z=coord.z    # S -> S
        )
        
        # Then convert to image
        return self.physical_to_image(physical)


class UnitConverter:
    """Converts between unit systems"""
    
    # Conversion factors to millimeters
    FACTORS = {
        UnitSystem.MILLIMETERS: 1.0,
        UnitSystem.CENTIMETERS: 10.0,
        UnitSystem.INCHES: 25.4
    }
    
    def __init__(self, voxel_size: Tuple[float, float, float]):
        """
        Initialize converter.
        
        Args:
            voxel_size: Voxel dimensions in mm
        """
        self.voxel_size = voxel_size
    
    def voxels_to_physical(self, value: float, axis: int) -> float:
        """Convert voxel units to physical units (mm)"""
        return value * self.voxel_size[axis]
    
    def physical_to_voxels(self, value: float, axis: int) -> float:
        """Convert physical units (mm) to voxel units"""
        return value / self.voxel_size[axis]
    
    def convert(
        self,
        value: float,
        from_unit: UnitSystem,
        to_unit: UnitSystem,
        axis: Optional[int] = None
    ) -> float:
        """
        Convert between unit systems.
        
        Args:
            value: Value to convert
            from_unit: Source unit system
            to_unit: Target unit system
            axis: Optional axis for voxel conversion
            
        Returns:
            Converted value
        """
        if from_unit == to_unit:
            return value
        
        # Convert to millimeters first
        if from_unit == UnitSystem.VOXELS:
            if axis is None:
                raise ValueError("Axis required for voxel conversion")
            mm_value = self.voxels_to_physical(value, axis)
        else:
            mm_value = value * self.FACTORS[from_unit]
        
        # Then convert to target unit
        if to_unit == UnitSystem.VOXELS:
            if axis is None:
                raise ValueError("Axis required for voxel conversion")
            return self.physical_to_voxels(mm_value, axis)
        else:
            return mm_value / self.FACTORS[to_unit]


class MeasurementSystem:
    """Base measurement system"""
    
    def __init__(
        self,
        image: nib.Nifti1Image,
        unit_system: UnitSystem = UnitSystem.MILLIMETERS
    ):
        """
        Initialize measurement system.
        
        Args:
            image: NIfTI image for coordinate reference
            unit_system: Preferred unit system
        """
        self.image = image
        self.unit_system = unit_system
        
        # Get image properties
        self.affine = image.affine
        header = image.header
        self.voxel_size = tuple(float(x) for x in header.get_zooms()[:3])
        
        # Create converters
        self.coord_converter = CoordinateConverter(self.affine)
        self.unit_converter = UnitConverter(self.voxel_size)
        
        # Initialize results storage
        self.results: List[MeasurementResult] = []
    
    def convert_coordinates(
        self,
        coord: Coordinate,
        from_system: CoordinateSystem,
        to_system: CoordinateSystem
    ) -> Coordinate:
        """
        Convert between coordinate systems.
        
        Args:
            coord: Coordinate to convert
            from_system: Source coordinate system
            to_system: Target coordinate system
            
        Returns:
            Converted coordinate
        """
        if from_system == to_system:
            return coord
        
        # Convert to physical first
        if from_system == CoordinateSystem.IMAGE:
            physical = self.coord_converter.image_to_physical(coord)
        elif from_system == CoordinateSystem.WORLD:
            world = coord
            physical = Coordinate(
                x=-world.x,  # R -> L
                y=-world.y,  # A -> P
                z=world.z    # S -> S
            )
        else:
            physical = coord
        
        # Then convert to target system
        if to_system == CoordinateSystem.IMAGE:
            return self.coord_converter.physical_to_image(physical)
        elif to_system == CoordinateSystem.WORLD:
            return Coordinate(
                x=-physical.x,  # L -> R
                y=-physical.y,  # P -> A
                z=physical.z    # S -> S
            )
        else:
            return physical
    
    def convert_value(
        self,
        value: float,
        from_unit: UnitSystem,
        to_unit: UnitSystem,
        axis: Optional[int] = None
    ) -> float:
        """
        Convert value between unit systems.
        
        Args:
            value: Value to convert
            from_unit: Source unit system
            to_unit: Target unit system
            axis: Optional axis for voxel conversion
            
        Returns:
            Converted value
        """
        return self.unit_converter.convert(value, from_unit, to_unit, axis)
    
    def save_results(self, path: Path) -> None:
        """
        Save measurement results.
        
        Args:
            path: Output file path
        """
        try:
            # Convert results to JSON-serializable format
            data = []
            for result in self.results:
                points = [
                    {"x": p.x, "y": p.y, "z": p.z}
                    for p in result.points
                ]
                data.append({
                    "value": result.value,
                    "unit": result.unit.name,
                    "points": points,
                    "type": result.type,
                    "metadata": result.metadata
                })
            
            # Save to file
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved measurement results to: {path}")
            
        except Exception as e:
            logger.error(f"Failed to save measurement results: {e}")
    
    def load_results(self, path: Path) -> None:
        """
        Load measurement results.
        
        Args:
            path: Input file path
        """
        try:
            # Load from file
            with open(path) as f:
                data = json.load(f)
            
            # Convert to result objects
            self.results = []
            for item in data:
                points = [
                    Coordinate(
                        x=float(p["x"]),
                        y=float(p["y"]),
                        z=float(p["z"])
                    )
                    for p in item["points"]
                ]
                result = MeasurementResult(
                    value=float(item["value"]),
                    unit=UnitSystem[item["unit"]],
                    points=points,
                    type=item["type"],
                    metadata=item.get("metadata")
                )
                self.results.append(result)
            
            logger.info(f"Loaded measurement results from: {path}")
            
        except Exception as e:
            logger.error(f"Failed to load measurement results: {e}")
    
    def add_result(self, result: MeasurementResult) -> None:
        """
        Add measurement result.
        
        Args:
            result: Measurement result to add
        """
        self.results.append(result)
        logger.debug(f"Added measurement result: {result.type}")
    
    def clear_results(self) -> None:
        """Clear all measurement results"""
        self.results = []
        logger.debug("Cleared measurement results")