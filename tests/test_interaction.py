"""
Test suite for interaction functionality.
Tests interaction handlers and manager.
"""
import pytest
from matplotlib.backend_bases import MouseEvent, KeyEvent
from matplotlib.figure import Figure

from utils.interaction import (
    CrosshairHandler,
    InteractionManager,
    InteractionMode,
    PanHandler,
    Point,
    ViewState,
    WindowLevelHandler,
    ZoomHandler,
    connect_interaction_manager
)


@pytest.fixture
def view_state() -> ViewState:
    """Fixture for view state"""
    return ViewState(
        center=Point(0, 0),
        zoom=1.0,
        window=1.0,
        level=0.5,
        orientation="axial"
    )


@pytest.fixture
def interaction_manager() -> InteractionManager:
    """Fixture for interaction manager"""
    return InteractionManager()


@pytest.fixture
def figure() -> Figure:
    """Fixture for matplotlib figure"""
    return Figure()


def create_mouse_event(
    figure: Figure,
    button: int,
    x: float,
    y: float,
    dblclick: bool = False
) -> MouseEvent:
    """Create mock mouse event"""
    return MouseEvent(
        name="mock",
        canvas=figure.canvas,
        x=x,
        y=y,
        button=button,
        dblclick=dblclick,
        xdata=x,
        ydata=y
    )


def create_key_event(figure: Figure, key: str) -> KeyEvent:
    """Create mock key event"""
    return KeyEvent(
        name="mock",
        canvas=figure.canvas,
        key=key
    )


class TestPanHandler:
    """Tests for pan interaction handler"""
    
    def test_pan_interaction(self, view_state: ViewState, figure: Figure):
        """Test pan interaction"""
        handler = PanHandler()
        handler.view_state = view_state
        
        # Test mouse press
        event = create_mouse_event(figure, button=1, x=0, y=0)
        handler.on_mouse_press(event)
        assert handler.active
        assert handler.start_point == Point(0, 0)
        
        # Test mouse move
        event = create_mouse_event(figure, button=1, x=10, y=5)
        handler.on_mouse_move(event)
        assert handler.view_state.center.x == -10
        assert handler.view_state.center.y == -5
        
        # Test mouse release
        event = create_mouse_event(figure, button=1, x=10, y=5)
        handler.on_mouse_release(event)
        assert not handler.active
        assert handler.start_point is None


class TestZoomHandler:
    """Tests for zoom interaction handler"""
    
    def test_zoom_interaction(self, view_state: ViewState, figure: Figure):
        """Test zoom interaction"""
        handler = ZoomHandler()
        handler.view_state = view_state
        
        # Test mouse press
        event = create_mouse_event(figure, button=3, x=0, y=0)
        handler.on_mouse_press(event)
        assert handler.active
        assert handler.start_point == Point(0, 0)
        
        # Test mouse move
        event = create_mouse_event(figure, button=3, x=0, y=50)
        handler.on_mouse_move(event)
        assert handler.view_state.zoom > 1.0
        
        # Test mouse release
        event = create_mouse_event(figure, button=3, x=0, y=50)
        handler.on_mouse_release(event)
        assert not handler.active
        assert handler.start_point is None
        
        # Test key press
        event = create_key_event(figure, key="+")
        handler.on_key_press(event)
        assert handler.view_state.zoom > 1.0


class TestWindowLevelHandler:
    """Tests for window/level interaction handler"""
    
    def test_window_level_interaction(self, view_state: ViewState, figure: Figure):
        """Test window/level interaction"""
        handler = WindowLevelHandler()
        handler.view_state = view_state
        
        # Test mouse press
        event = create_mouse_event(figure, button=2, x=0, y=0)
        handler.on_mouse_press(event)
        assert handler.active
        assert handler.start_point == Point(0, 0)
        
        # Test mouse move
        event = create_mouse_event(figure, button=2, x=10, y=5)
        handler.on_mouse_move(event)
        assert handler.view_state.window > 1.0
        assert handler.view_state.level > 0.5
        
        # Test mouse release
        event = create_mouse_event(figure, button=2, x=10, y=5)
        handler.on_mouse_release(event)
        assert not handler.active
        assert handler.start_point is None


class TestCrosshairHandler:
    """Tests for crosshair interaction handler"""
    
    def test_crosshair_interaction(self, view_state: ViewState, figure: Figure):
        """Test crosshair interaction"""
        handler = CrosshairHandler()
        handler.view_state = view_state
        
        # Test double click
        event = create_mouse_event(figure, button=1, x=0, y=0, dblclick=True)
        handler.on_mouse_press(event)
        assert handler.active
        assert handler.current_point == Point(0, 0)
        
        # Test mouse move
        event = create_mouse_event(figure, button=1, x=10, y=5)
        handler.on_mouse_move(event)
        assert handler.current_point == Point(10, 5)
        
        # Test key press
        event = create_key_event(figure, key="up")
        handler.on_key_press(event)
        assert handler.current_point.y == 6
        
        # Test deactivate
        event = create_mouse_event(figure, button=1, x=10, y=6, dblclick=True)
        handler.on_mouse_press(event)
        assert not handler.active


class TestInteractionManager:
    """Tests for interaction manager"""
    
    def test_mode_switching(self, interaction_manager: InteractionManager):
        """Test mode switching"""
        # Test initial mode
        assert interaction_manager.mode == InteractionMode.NONE
        
        # Test mode switching
        interaction_manager.set_mode(InteractionMode.PAN)
        assert interaction_manager.mode == InteractionMode.PAN
        
        # Test key press mode switching
        event = create_key_event(Figure(), key="z")
        interaction_manager.handle_key_press(event)
        assert interaction_manager.mode == InteractionMode.ZOOM
        
        # Test escape
        event = create_key_event(Figure(), key="escape")
        interaction_manager.handle_key_press(event)
        assert interaction_manager.mode == InteractionMode.NONE
    
    def test_event_handling(
        self,
        interaction_manager: InteractionManager,
        figure: Figure
    ):
        """Test event handling"""
        # Set pan mode
        interaction_manager.set_mode(InteractionMode.PAN)
        
        # Test mouse press
        event = create_mouse_event(figure, button=1, x=0, y=0)
        interaction_manager.handle_mouse_press(event)
        assert interaction_manager.handlers[InteractionMode.PAN].active
        
        # Test mouse move
        event = create_mouse_event(figure, button=1, x=10, y=5)
        interaction_manager.handle_mouse_move(event)
        assert interaction_manager.view_state.center.x == -10
        assert interaction_manager.view_state.center.y == -5
        
        # Test mouse release
        event = create_mouse_event(figure, button=1, x=10, y=5)
        interaction_manager.handle_mouse_release(event)
        assert not interaction_manager.handlers[InteractionMode.PAN].active


def test_connect_interaction_manager(
    interaction_manager: InteractionManager,
    figure: Figure
):
    """Test connecting interaction manager to figure"""
    connect_interaction_manager(figure, interaction_manager)
    
    # Verify event connections
    assert len(figure.canvas.callbacks.callbacks) == 4  # press, move, release, key
    
    # Test event handling
    event = create_key_event(figure, key="p")
    figure.canvas.callbacks.process("key_press_event", event)
    assert interaction_manager.mode == InteractionMode.PAN