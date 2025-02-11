"""
Test suite for measurement tools.
Tests distance, angle, and area/volume measurements.
"""
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from utils.interaction import Point
from utils.measurement import (
    Coordinate,
    CoordinateSystem,
    MeasurementResult,
    MeasurementSystem,
    UnitSystem
)
from utils.measurement_tools import (
    AngleMode,
    AngleTool,
    AreaVolumeTool,
    DistanceTool,
    MeasurementTool,
    OrientationMode,
    PreviewState,
    ROIMode,
    ROIStatistics
)
from tests.test_measurement import test_image


@pytest.fixture
def measurement_system(test_image) -> MeasurementSystem:
    """Fixture for measurement system"""
    return MeasurementSystem(test_image)


@pytest.fixture
def distance_tool(measurement_system: MeasurementSystem) -> DistanceTool:
    """Fixture for distance tool"""
    return DistanceTool(measurement_system)


@pytest.fixture
def angle_tool(measurement_system: MeasurementSystem) -> AngleTool:
    """Fixture for angle tool"""
    return AngleTool(measurement_system)


@pytest.fixture
def area_volume_tool(measurement_system: MeasurementSystem) -> AreaVolumeTool:
    """Fixture for area/volume tool"""
    tool = AreaVolumeTool(measurement_system)
    tool.mask = np.ones((64, 64), dtype=bool)  # Mock image data
    return tool


@pytest.fixture
def axes() -> Axes:
    """Fixture for matplotlib axes"""
    fig = Figure()
    return fig.add_subplot(111)


class TestPreviewState:
    """Tests for preview state"""
    
    def test_preview_state_init(self):
        """Test preview state initialization"""
        state = PreviewState(points=[])
        assert state.points == []
        assert state.current_point is None
        assert not state.active
        assert not state.complete


class TestDistanceTool:
    """Tests for distance measurement tool"""
    
    def test_tool_initialization(self, distance_tool: DistanceTool):
        """Test tool initialization"""
        assert isinstance(distance_tool, MeasurementTool)
        assert distance_tool.preview.points == []
        assert not distance_tool.preview.complete
        assert len(distance_tool.preview_artists) == 0
        assert distance_tool.result is None
    
    def test_handle_click(self, distance_tool: DistanceTool):
        """Test click handling"""
        # First point
        p1 = Point(10, 10)
        distance_tool.handle_click(p1)
        assert len(distance_tool.preview.points) == 1
        assert not distance_tool.preview.complete
        
        # Second point
        p2 = Point(20, 20)
        distance_tool.handle_click(p2)
        assert len(distance_tool.preview.points) == 2
        assert distance_tool.preview.complete
        
        # Result should be stored
        assert distance_tool.result is not None
        assert distance_tool.result.type == "distancetool"
        assert len(distance_tool.result.points) == 2
    
    def test_handle_move(self, distance_tool: DistanceTool):
        """Test move handling"""
        # Add first point
        p1 = Point(10, 10)
        distance_tool.handle_click(p1)
        
        # Move cursor
        p2 = Point(20, 20)
        distance_tool.handle_move(p2)
        assert distance_tool.preview.current_point == p2
        
        # Complete measurement
        distance_tool.handle_click(p2)
        
        # Move should have no effect after completion
        p3 = Point(30, 30)
        distance_tool.handle_move(p3)
        assert distance_tool.preview.current_point != p3
    
    def test_calculate_distance(self, distance_tool: DistanceTool):
        """Test distance calculation"""
        # Add points
        p1 = Point(0, 0)
        p2 = Point(10, 0)
        distance_tool.handle_click(p1)
        distance_tool.handle_click(p2)
        
        # Calculate distance
        distance = distance_tool.calculate()
        assert distance == pytest.approx(10.0)  # 10mm with default 1mm voxel size
    
    def test_draw_preview(
        self,
        distance_tool: DistanceTool,
        axes: Axes
    ):
        """Test preview drawing"""
        # Add first point
        p1 = Point(10, 10)
        distance_tool.handle_click(p1)
        distance_tool.draw_preview(axes)
        assert len(distance_tool.preview_artists) == 1  # Start point
        
        # Move cursor
        p2 = Point(20, 20)
        distance_tool.handle_move(p2)
        distance_tool.draw_preview(axes)
        assert len(distance_tool.preview_artists) == 3  # Points + line
        
        # Complete measurement
        distance_tool.handle_click(p2)
        distance_tool.draw_preview(axes)
        assert len(distance_tool.preview_artists) == 4  # Points + line + label
    
    def test_reset(self, distance_tool: DistanceTool):
        """Test tool reset"""
        # Add points
        p1 = Point(10, 10)
        p2 = Point(20, 20)
        distance_tool.handle_click(p1)
        distance_tool.handle_click(p2)
        
        # Reset tool
        distance_tool.reset()
        assert len(distance_tool.preview.points) == 0
        assert not distance_tool.preview.complete
        assert len(distance_tool.preview_artists) == 0
        assert distance_tool.result is None
    
    def test_unit_conversion(self, distance_tool: DistanceTool):
        """Test distance unit conversion"""
        # Set unit system to centimeters
        distance_tool.system.unit_system = UnitSystem.CENTIMETERS
        
        # Add points 10mm apart
        p1 = Point(0, 0)
        p2 = Point(10, 0)
        distance_tool.handle_click(p1)
        distance_tool.handle_click(p2)
        
        # Calculate distance
        distance = distance_tool.calculate()
        assert distance == pytest.approx(1.0)  # 10mm = 1cm


class TestAngleTool:
    """Tests for angle measurement tool"""
    
    def test_tool_initialization(self, angle_tool: AngleTool):
        """Test tool initialization"""
        assert isinstance(angle_tool, MeasurementTool)
        assert angle_tool.preview.points == []
        assert not angle_tool.preview.complete
        assert len(angle_tool.preview_artists) == 0
        assert angle_tool.result is None
        assert angle_tool.mode == AngleMode.THREE_POINT
        assert angle_tool.orientation == OrientationMode.CLOCKWISE
    
    def test_handle_click(self, angle_tool: AngleTool):
        """Test click handling"""
        # First point
        p1 = Point(10, 10)
        angle_tool.handle_click(p1)
        assert len(angle_tool.preview.points) == 1
        assert not angle_tool.preview.complete
        
        # Second point
        p2 = Point(20, 10)
        angle_tool.handle_click(p2)
        assert len(angle_tool.preview.points) == 2
        assert not angle_tool.preview.complete
        
        # Third point
        p3 = Point(20, 20)
        angle_tool.handle_click(p3)
        assert len(angle_tool.preview.points) == 3
        assert angle_tool.preview.complete
        
        # Result should be stored
        assert angle_tool.result is not None
        assert angle_tool.result.type == "angletool"
        assert len(angle_tool.result.points) == 3
    
    def test_handle_move(self, angle_tool: AngleTool):
        """Test move handling"""
        # Add first two points
        p1 = Point(10, 10)
        p2 = Point(20, 10)
        angle_tool.handle_click(p1)
        angle_tool.handle_click(p2)
        
        # Move cursor
        p3 = Point(20, 20)
        angle_tool.handle_move(p3)
        assert angle_tool.preview.current_point == p3
        
        # Complete measurement
        angle_tool.handle_click(p3)
        
        # Move should have no effect after completion
        p4 = Point(30, 30)
        angle_tool.handle_move(p4)
        assert angle_tool.preview.current_point != p4
    
    def test_calculate_angle(self, angle_tool: AngleTool):
        """Test angle calculation"""
        # Test right angle
        angle_tool.handle_click(Point(10, 10))  # First point
        angle_tool.handle_click(Point(10, 20))  # Vertex
        angle_tool.handle_click(Point(20, 20))  # Third point
        
        angle = angle_tool.calculate()
        assert angle == pytest.approx(90.0)
        
        # Test straight angle
        angle_tool.reset()
        angle_tool.handle_click(Point(0, 10))   # First point
        angle_tool.handle_click(Point(10, 10))  # Vertex
        angle_tool.handle_click(Point(20, 10))  # Third point
        
        angle = angle_tool.calculate()
        assert angle == pytest.approx(180.0)
    
    def test_orientation_modes(self, angle_tool: AngleTool):
        """Test angle orientation modes"""
        # Points for 90-degree angle
        p1 = Point(10, 10)  # First point
        p2 = Point(10, 20)  # Vertex
        p3 = Point(20, 20)  # Third point
        
        # Test clockwise
        angle_tool.orientation = OrientationMode.CLOCKWISE
        angle_tool.handle_click(p1)
        angle_tool.handle_click(p2)
        angle_tool.handle_click(p3)
        assert angle_tool.calculate() == pytest.approx(90.0)
        
        # Test counterclockwise
        angle_tool.reset()
        angle_tool.orientation = OrientationMode.COUNTERCLOCKWISE
        angle_tool.handle_click(p1)
        angle_tool.handle_click(p2)
        angle_tool.handle_click(p3)
        assert angle_tool.calculate() == pytest.approx(270.0)
        
        # Test absolute
        angle_tool.reset()
        angle_tool.orientation = OrientationMode.ABSOLUTE
        angle_tool.handle_click(p1)
        angle_tool.handle_click(p2)
        angle_tool.handle_click(p3)
        assert angle_tool.calculate() == pytest.approx(90.0)
    
    def test_draw_preview(
        self,
        angle_tool: AngleTool,
        axes: Axes
    ):
        """Test preview drawing"""
        # Add first point
        p1 = Point(10, 10)
        angle_tool.handle_click(p1)
        angle_tool.draw_preview(axes)
        assert len(angle_tool.preview_artists) == 1  # Start point
        
        # Add second point
        p2 = Point(10, 20)
        angle_tool.handle_click(p2)
        angle_tool.draw_preview(axes)
        assert len(angle_tool.preview_artists) == 3  # Points + line
        
        # Move cursor for third point
        p3 = Point(20, 20)
        angle_tool.handle_move(p3)
        angle_tool.draw_preview(axes)
        assert len(angle_tool.preview_artists) == 5  # Points + lines
        
        # Complete measurement
        angle_tool.handle_click(p3)
        angle_tool.draw_preview(axes)
        assert len(angle_tool.preview_artists) == 7  # Points + lines + arc + label
    
    def test_reset(self, angle_tool: AngleTool):
        """Test tool reset"""
        # Add points
        p1 = Point(10, 10)
        p2 = Point(10, 20)
        p3 = Point(20, 20)
        angle_tool.handle_click(p1)
        angle_tool.handle_click(p2)
        angle_tool.handle_click(p3)
        
        # Reset tool
        angle_tool.reset()
        assert len(angle_tool.preview.points) == 0
        assert not angle_tool.preview.complete
        assert len(angle_tool.preview_artists) == 0
        assert angle_tool.result is None


class TestAreaVolumeTool:
    """Tests for area/volume measurement tool"""
    
    def test_tool_initialization(self, area_volume_tool: AreaVolumeTool):
        """Test tool initialization"""
        assert isinstance(area_volume_tool, MeasurementTool)
        assert area_volume_tool.preview.points == []
        assert not area_volume_tool.preview.complete
        assert len(area_volume_tool.preview_artists) == 0
        assert area_volume_tool.result is None
        assert area_volume_tool.mode == ROIMode.POLYGON
        assert not area_volume_tool.closed
        assert area_volume_tool.stats is None
    
    def test_handle_click(self, area_volume_tool: AreaVolumeTool):
        """Test click handling"""
        # Add points for triangle
        p1 = Point(10, 10)
        p2 = Point(20, 10)
        p3 = Point(15, 20)
        
        area_volume_tool.handle_click(p1)
        assert len(area_volume_tool.preview.points) == 1
        assert not area_volume_tool.preview.complete
        assert not area_volume_tool.closed
        
        area_volume_tool.handle_click(p2)
        assert len(area_volume_tool.preview.points) == 2
        assert not area_volume_tool.preview.complete
        assert not area_volume_tool.closed
        
        area_volume_tool.handle_click(p3)
        assert len(area_volume_tool.preview.points) == 3
        assert not area_volume_tool.preview.complete
        assert not area_volume_tool.closed
        
        # Close polygon by clicking near start point
        area_volume_tool.handle_click(Point(11, 11))  # Close to p1
        assert len(area_volume_tool.preview.points) == 3
        assert area_volume_tool.preview.complete
        assert area_volume_tool.closed
        
        # Result should be stored
        assert area_volume_tool.result is not None
        assert area_volume_tool.result.type == "areavolumetool"
        assert len(area_volume_tool.result.points) == 3
    
    def test_handle_move(self, area_volume_tool: AreaVolumeTool):
        """Test move handling"""
        # Add points
        p1 = Point(10, 10)
        p2 = Point(20, 10)
        area_volume_tool.handle_click(p1)
        area_volume_tool.handle_click(p2)
        
        # Move cursor
        p3 = Point(15, 20)
        area_volume_tool.handle_move(p3)
        assert area_volume_tool.preview.current_point == p3
        
        # Complete measurement
        area_volume_tool.handle_click(p3)
        area_volume_tool.handle_click(Point(11, 11))  # Close polygon
        
        # Move should have no effect after completion
        p4 = Point(30, 30)
        area_volume_tool.handle_move(p4)
        assert area_volume_tool.preview.current_point != p4
    
    def test_calculate_measurements(self, area_volume_tool: AreaVolumeTool):
        """Test area and volume calculations"""
        # Create triangle with area = 50 square units
        area_volume_tool.handle_click(Point(10, 10))
        area_volume_tool.handle_click(Point(20, 10))
        area_volume_tool.handle_click(Point(15, 20))
        area_volume_tool.handle_click(Point(11, 11))  # Close polygon
        
        # Calculate measurements
        measurements = area_volume_tool.calculate()
        assert measurements["area"] == pytest.approx(50.0)
        assert measurements["volume"] > 0  # Volume depends on mask
    
    def test_roi_statistics(self, area_volume_tool: AreaVolumeTool):
        """Test ROI statistics calculation"""
        # Create triangle ROI
        area_volume_tool.handle_click(Point(10, 10))
        area_volume_tool.handle_click(Point(20, 10))
        area_volume_tool.handle_click(Point(15, 20))
        area_volume_tool.handle_click(Point(11, 11))  # Close polygon
        
        # Calculate statistics
        stats = area_volume_tool.calculate_statistics()
        assert isinstance(stats, dict)
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert "volume" in stats
        assert "area" in stats
    
    def test_draw_preview(
        self,
        area_volume_tool: AreaVolumeTool,
        axes: Axes
    ):
        """Test preview drawing"""
        # Add first point
        p1 = Point(10, 10)
        area_volume_tool.handle_click(p1)
        area_volume_tool.draw_preview(axes)
        assert len(area_volume_tool.preview_artists) == 1  # Start point
        
        # Add second point
        p2 = Point(20, 10)
        area_volume_tool.handle_click(p2)
        area_volume_tool.draw_preview(axes)
        assert len(area_volume_tool.preview_artists) == 3  # Points + line
        
        # Add third point
        p3 = Point(15, 20)
        area_volume_tool.handle_click(p3)
        area_volume_tool.draw_preview(axes)
        assert len(area_volume_tool.preview_artists) == 4  # Points + lines
        
        # Close polygon
        area_volume_tool.handle_click(Point(11, 11))
        area_volume_tool.draw_preview(axes)
        assert len(area_volume_tool.preview_artists) == 6  # Points + lines + mask + label
    
    def test_reset(self, area_volume_tool: AreaVolumeTool):
        """Test tool reset"""
        # Add points
        p1 = Point(10, 10)
        p2 = Point(20, 10)
        p3 = Point(15, 20)
        area_volume_tool.handle_click(p1)
        area_volume_tool.handle_click(p2)
        area_volume_tool.handle_click(p3)
        area_volume_tool.handle_click(Point(11, 11))  # Close polygon
        
        # Reset tool
        area_volume_tool.reset()
        assert len(area_volume_tool.preview.points) == 0
        assert not area_volume_tool.preview.complete
        assert not area_volume_tool.closed
        assert len(area_volume_tool.preview_artists) == 0
        assert area_volume_tool.result is None
        assert area_volume_tool.stats is None