"""
Test suite for interactive 3D viewer.
Tests viewer functionality and tool integration.
"""
import numpy as np
import pytest
import vtk

from utils.interactive_viewer import (
    Interactive3DViewer,
    ViewerConfig,
    ViewerMode
)
from utils.measurement_tools import (
    AngleTool,
    AreaVolumeTool,
    DistanceTool,
    MeasurementSystem
)
from tests.test_measurement import test_image


@pytest.fixture
def viewer_config() -> ViewerConfig:
    """Fixture for viewer configuration"""
    return ViewerConfig(
        width=400,
        height=300,
        background_color=(0.0, 0.0, 0.0),
        tool_color=(1.0, 0.0, 0.0),
        tool_opacity=1.0,
        tool_line_width=1.0
    )


@pytest.fixture
def measurement_system(test_image) -> MeasurementSystem:
    """Fixture for measurement system"""
    return MeasurementSystem(test_image)


@pytest.fixture
def viewer(viewer_config: ViewerConfig) -> Interactive3DViewer:
    """Fixture for interactive viewer"""
    viewer = Interactive3DViewer(viewer_config)
    
    # Create test sphere
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(10)
    sphere.SetPhiResolution(30)
    sphere.SetThetaResolution(30)
    
    # Create mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())
    
    # Create actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # Add to renderer
    viewer.renderer.AddActor(actor)
    
    return viewer


class TestInteractiveViewer:
    """Tests for interactive viewer"""
    
    def test_viewer_initialization(self, viewer: Interactive3DViewer):
        """Test viewer initialization"""
        assert viewer.mode == ViewerMode.NAVIGATE
        assert not viewer.tools
        assert viewer.active_tool is None
        assert not viewer.results
        
        # Check VTK objects
        assert isinstance(viewer.renderer, vtk.vtkRenderer)
        assert isinstance(viewer.window, vtk.vtkRenderWindow)
        assert isinstance(viewer.interactor, vtk.vtkRenderWindowInteractor)
        
        # Check configuration
        bg_color = viewer.renderer.GetBackground()
        assert bg_color == viewer.config.background_color
        assert viewer.window.GetSize() == (
            viewer.config.width,
            viewer.config.height
        )
    
    def test_add_tool(
        self,
        viewer: Interactive3DViewer,
        measurement_system: MeasurementSystem
    ):
        """Test tool addition"""
        # Add distance tool
        tool = DistanceTool(measurement_system)
        viewer.add_tool(tool)
        
        assert "distancetool" in viewer.tools
        assert len(tool.actors) == 3  # points, lines, labels
        
        # Check actor properties
        for actor in tool.actors.values():
            color = actor.GetProperty().GetColor()
            opacity = actor.GetProperty().GetOpacity()
            line_width = actor.GetProperty().GetLineWidth()
            
            assert color == viewer.config.tool_color
            assert opacity == viewer.config.tool_opacity
            assert line_width == viewer.config.tool_line_width
    
    def test_switch_tool(
        self,
        viewer: Interactive3DViewer,
        measurement_system: MeasurementSystem
    ):
        """Test tool switching"""
        # Add tools
        distance_tool = DistanceTool(measurement_system)
        angle_tool = AngleTool(measurement_system)
        viewer.add_tool(distance_tool)
        viewer.add_tool(angle_tool)
        
        # Switch to distance tool
        viewer.switch_tool("distancetool")
        assert viewer.mode == ViewerMode.MEASURE
        assert viewer.active_tool == distance_tool
        for actor in distance_tool.actors.values():
            assert actor.GetVisibility()
        
        # Switch to angle tool
        viewer.switch_tool("angletool")
        assert viewer.mode == ViewerMode.MEASURE
        assert viewer.active_tool == angle_tool
        for actor in angle_tool.actors.values():
            assert actor.GetVisibility()
        for actor in distance_tool.actors.values():
            assert not actor.GetVisibility()
        
        # Switch to invalid tool
        viewer.switch_tool("invalidtool")
        assert viewer.mode == ViewerMode.NAVIGATE
        assert viewer.active_tool is None
    
    def test_mouse_interaction(
        self,
        viewer: Interactive3DViewer,
        measurement_system: MeasurementSystem
    ):
        """Test mouse interaction"""
        # Add distance tool
        tool = DistanceTool(measurement_system)
        viewer.add_tool(tool)
        viewer.switch_tool("distancetool")
        
        # Simulate left click
        viewer.interactor.SetEventPosition(200, 150)
        viewer.on_left_click(None, "LeftButtonPressEvent")
        assert len(tool.preview.points) == 1
        
        # Simulate mouse move
        viewer.interactor.SetEventPosition(250, 200)
        viewer.on_mouse_move(None, "MouseMoveEvent")
        assert tool.preview.current_point is not None
        
        # Simulate second click
        viewer.on_left_click(None, "LeftButtonPressEvent")
        assert len(tool.preview.points) == 2
        assert tool.result is not None
        assert len(viewer.results) == 1
    
    def test_key_interaction(
        self,
        viewer: Interactive3DViewer,
        measurement_system: MeasurementSystem
    ):
        """Test key interaction"""
        # Add tool
        tool = DistanceTool(measurement_system)
        viewer.add_tool(tool)
        viewer.switch_tool("distancetool")
        
        # Test navigation mode
        viewer.interactor.SetKeySym("n")
        viewer.on_key_press(None, "KeyPressEvent")
        assert viewer.mode == ViewerMode.NAVIGATE
        for actor in tool.actors.values():
            assert not actor.GetVisibility()
        
        # Test measurement mode
        viewer.interactor.SetKeySym("m")
        viewer.on_key_press(None, "KeyPressEvent")
        assert viewer.mode == ViewerMode.MEASURE
        for actor in tool.actors.values():
            assert actor.GetVisibility()
        
        # Test escape key
        viewer.interactor.SetEventPosition(200, 150)
        viewer.on_left_click(None, "LeftButtonPressEvent")
        viewer.interactor.SetKeySym("Escape")
        viewer.on_key_press(None, "KeyPressEvent")
        assert len(tool.preview.points) == 0
    
    def test_screenshot(
        self,
        viewer: Interactive3DViewer,
        tmp_path: Path
    ):
        """Test screenshot capture"""
        # Save screenshot
        output_path = tmp_path / "test.png"
        viewer.save_screenshot(output_path)
        assert output_path.exists()
        
        # Test default path
        viewer.save_screenshot()
        assert Path("screenshot.png").exists()
    
    def test_cleanup(self, viewer: Interactive3DViewer):
        """Test viewer cleanup"""
        viewer.stop()
        assert not viewer.window