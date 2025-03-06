"""
Utilities for neuroimaging visualization.
Provides async visualization operations with multiple backends.
"""
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import vtk
from matplotlib.figure import Figure
from vtk.util import numpy_support

from utils.interaction import InteractionManager

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from nilearn import plotting
import vtk
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkPoints, vtkUnsignedCharArray
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkContourFilter
from vtkmodules.vtkFiltersGeneral import vtkImageMarchingCubes
from vtkmodules.vtkFiltersSources import vtkPlaneSource
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCamera,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
logger = logging.getLogger(__name__)


class ViewType(str, Enum):
    """Types of visualization views"""
    AXIAL = "axial"
    SAGITTAL = "sagittal"
    CORONAL = "coronal"
    THREE_D = "3d"
    MULTIPLANAR = "multiplanar"


@dataclass
class SliceSelection:
    """Selected slices for multi-planar visualization"""
    sagittal: List[int]
    coronal: List[int]
    axial: List[int]


@dataclass
class ColorMap:
    """Color map configuration"""
    name: str
    min_value: float = 0.0
    max_value: float = 1.0
    alpha: float = 1.0


@dataclass
class ViewportConfig:
    """Viewport configuration"""
    width: int
    height: int
    background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_position: Optional[Tuple[float, float, float]] = None
    camera_focus: Optional[Tuple[float, float, float]] = None
    up_vector: Optional[Tuple[float, float, float]] = None


async def select_informative_slices(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    n_slices: int = 5
) -> SliceSelection:
    """
    Select informative slices from volume data.
    
    Args:
        data: Volume data
        mask: Optional mask to focus on specific regions
        n_slices: Number of slices to select
        
    Returns:
        Selected slice indices for each orientation
    """
    try:
        if mask is None:
            # Use intensity variation to find informative slices
            variation = np.var(data, axis=(1, 2)), np.var(data, axis=(0, 2)), np.var(data, axis=(0, 1))
        else:
            # Use masked region
            x_indices, y_indices, z_indices = np.where(mask > 0)
            variation = x_indices, y_indices, z_indices
        
        def select_slices(indices: np.ndarray) -> List[int]:
            if len(indices) == 0:
                return []
            
            if isinstance(indices, tuple):
                indices = np.arange(len(indices))
            
            min_idx = np.min(indices)
            max_idx = np.max(indices)
            
            if max_idx - min_idx < n_slices:
                return list(range(min_idx, max_idx + 1))
            
            # Select evenly spaced slices
            return list(np.linspace(min_idx, max_idx, n_slices, dtype=int))
        
        return SliceSelection(
            sagittal=select_slices(variation[0]),
            coronal=select_slices(variation[1]),
            axial=select_slices(variation[2])
        )
        
    except Exception as e:
        logger.error(f"Error selecting slices: {str(e)}")
        return SliceSelection(sagittal=[], coronal=[], axial=[])


async def create_multiplanar_figure(
    background: np.ndarray,
    overlay: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    slices: Optional[SliceSelection] = None,
    background_cmap: ColorMap = ColorMap("gray"),
    overlay_cmap: ColorMap = ColorMap("hot", alpha=0.35),
    dpi: int = 300,
    title: Optional[str] = None,
    interactive: bool = True
) -> Tuple[bool, Optional[str], Optional[plt.Figure], Optional[InteractionManager]]:
    """
    Create multi-planar view figure.
    
    Args:
        background: Background volume data
        overlay: Optional overlay volume data
        mask: Optional mask data
        slices: Optional slice selection
        background_cmap: Background colormap
        overlay_cmap: Overlay colormap
        dpi: Output DPI
        title: Optional figure title
        
    Returns:
        Tuple of (success, error_message, figure)
    """
    try:
        # Select slices if not provided
        if slices is None:
            slices = await select_informative_slices(background, mask)
        
        # Create figure
        n_rows = 3  # One row each for sagittal, coronal, axial
        n_cols = max(len(slices.sagittal), len(slices.coronal), len(slices.axial))
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4*n_cols, 4*n_rows),
            dpi=dpi
        )
        
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot slices
        views = [
            ("Sagittal", slices.sagittal, (1, 2)),
            ("Coronal", slices.coronal, (0, 2)),
            ("Axial", slices.axial, (0, 1))
        ]
        
        for row, (view_name, slice_indices, display_axes) in enumerate(views):
            for col, slice_idx in enumerate(slice_indices):
                ax = axes[row, col]
                
                # Extract slice data
                if row == 0:  # Sagittal
                    bg_slice = background[slice_idx, :, :]
                    overlay_slice = overlay[slice_idx, :, :] if overlay is not None else None
                    mask_slice = mask[slice_idx, :, :] if mask is not None else None
                elif row == 1:  # Coronal
                    bg_slice = background[:, slice_idx, :]
                    overlay_slice = overlay[:, slice_idx, :] if overlay is not None else None
                    mask_slice = mask[:, slice_idx, :] if mask is not None else None
                else:  # Axial
                    bg_slice = background[:, :, slice_idx]
                    overlay_slice = overlay[:, :, slice_idx] if overlay is not None else None
                    mask_slice = mask[:, :, slice_idx] if mask is not None else None
                
                # Normalize slice data
                bg_slice = np.clip(
                    (bg_slice - bg_slice.min()) / (bg_slice.max() - bg_slice.min()),
                    0, 1
                )
                
                # Plot background
                ax.imshow(
                    bg_slice,
                    cmap=background_cmap.name,
                    vmin=background_cmap.min_value,
                    vmax=background_cmap.max_value
                )
                
                # Plot overlay
                if overlay_slice is not None and mask_slice is not None:
                    masked_overlay = overlay_slice * mask_slice
                    if np.any(masked_overlay > 0):
                        ax.imshow(
                            masked_overlay,
                            cmap=overlay_cmap.name,
                            alpha=overlay_cmap.alpha,
                            vmin=overlay_cmap.min_value,
                            vmax=overlay_cmap.max_value
                        )
                
                # Add slice information
                ax.set_title(f"{view_name} - Slice {slice_idx}")
                ax.axis('off')
        
        if title:
            fig.suptitle(title)
        
        # Add interaction manager if requested
        interaction_manager = None
        if interactive:
            from utils.interaction import InteractionManager, connect_interaction_manager
            interaction_manager = InteractionManager()
            connect_interaction_manager(fig, interaction_manager)
            
            # Add help text
            fig.text(
                0.01, 0.99,
                "Controls:\n" +
                "p: Pan mode (left click + drag)\n" +
                "z: Zoom mode (right click + drag, or +/-)\n" +
                "w: Window/level mode (middle click + drag)\n" +
                "c: Crosshair mode (double click to toggle)\n" +
                "esc: Exit current mode",
                fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        return True, None, fig, interaction_manager
        
    except Exception as e:
        return False, f"Error creating multi-planar figure: {str(e)}", None


async def create_3d_rendering(
    volume: np.ndarray,
    affine: np.ndarray,
    config: ViewportConfig,
    overlay: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    volume_cmap: ColorMap = ColorMap("gray"),
    overlay_cmap: ColorMap = ColorMap("hot", alpha=0.35)
) -> Tuple[bool, Optional[str], Optional[vtk.vtkRenderer]]:
    """
    Create 3D volume rendering.
    
    Args:
        volume: Volume data
        affine: Affine transformation matrix
        config: Viewport configuration
        overlay: Optional overlay volume
        mask: Optional mask volume
        volume_cmap: Volume colormap
        overlay_cmap: Overlay colormap
        
    Returns:
        Tuple of (success, error_message, renderer)
    """
    try:
        # Create renderer
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(*config.background_color)
        
        # Convert volume to VTK
        vtk_data = numpy_support.numpy_to_vtk(
            volume.ravel(),
            deep=True,
            array_type=vtk.VTK_FLOAT
        )
        
        # Create image data
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(*volume.shape)
        image_data.GetPointData().SetScalars(vtk_data)
        
        # Create volume mapper
        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputData(image_data)
        
        # Create volume property
        prop = vtk.vtkVolumeProperty()
        prop.ShadeOn()
        prop.SetInterpolationTypeToLinear()
        
        # Create color transfer function
        color = vtk.vtkColorTransferFunction()
        color.AddRGBPoint(volume_cmap.min_value, 0.0, 0.0, 0.0)
        color.AddRGBPoint(volume_cmap.max_value, 1.0, 1.0, 1.0)
        prop.SetColor(color)
        
        # Create opacity transfer function
        opacity = vtk.vtkPiecewiseFunction()
        opacity.AddPoint(volume_cmap.min_value, 0.0)
        opacity.AddPoint(volume_cmap.max_value, volume_cmap.alpha)
        prop.SetScalarOpacity(opacity)
        
        # Create volume
        volume_actor = vtk.vtkVolume()
        volume_actor.SetMapper(mapper)
        volume_actor.SetProperty(prop)
        
        # Add volume to renderer
        renderer.AddVolume(volume_actor)
        
        # Add overlay if provided
        if overlay is not None and mask is not None:
            masked_overlay = overlay * mask
            if np.any(masked_overlay > 0):
                # Create overlay volume
                overlay_data = numpy_support.numpy_to_vtk(
                    masked_overlay.ravel(),
                    deep=True,
                    array_type=vtk.VTK_FLOAT
                )
                
                overlay_image = vtk.vtkImageData()
                overlay_image.SetDimensions(*masked_overlay.shape)
                overlay_image.GetPointData().SetScalars(overlay_data)
                
                # Create overlay mapper and property
                overlay_mapper = vtk.vtkSmartVolumeMapper()
                overlay_mapper.SetInputData(overlay_image)
                
                overlay_prop = vtk.vtkVolumeProperty()
                overlay_prop.ShadeOn()
                overlay_prop.SetInterpolationTypeToLinear()
                
                # Create overlay color transfer function
                overlay_color = vtk.vtkColorTransferFunction()
                overlay_color.AddRGBPoint(overlay_cmap.min_value, 1.0, 0.0, 0.0)
                overlay_color.AddRGBPoint(overlay_cmap.max_value, 1.0, 1.0, 0.0)
                overlay_prop.SetColor(overlay_color)
                
                # Create overlay opacity transfer function
                overlay_opacity = vtk.vtkPiecewiseFunction()
                overlay_opacity.AddPoint(overlay_cmap.min_value, 0.0)
                overlay_opacity.AddPoint(overlay_cmap.max_value, overlay_cmap.alpha)
                overlay_prop.SetScalarOpacity(overlay_opacity)
                
                # Create overlay volume
                overlay_actor = vtk.vtkVolume()
                overlay_actor.SetMapper(overlay_mapper)
                overlay_actor.SetProperty(overlay_prop)
                
                # Add overlay to renderer
                renderer.AddVolume(overlay_actor)
        
        # Set up camera
        camera = renderer.GetActiveCamera()
        if config.camera_position:
            camera.SetPosition(*config.camera_position)
        if config.camera_focus:
            camera.SetFocalPoint(*config.camera_focus)
        if config.up_vector:
            camera.SetViewUp(*config.up_vector)
        
        # Reset camera to show everything
        renderer.ResetCamera()
        
        return True, None, renderer
        
    except Exception as e:
        return False, f"Error creating 3D rendering: {str(e)}", None


async def save_figure(
    fig: plt.Figure,
    output_path: Path,
    dpi: int = 300
) -> Tuple[bool, Optional[str]]:
    """
    Save matplotlib figure to file.
    
    Args:
        fig: Figure to save
        output_path: Output file path
        dpi: Output DPI
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save figure
        fig.savefig(output_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        
        return True, None
        
    except Exception as e:
        return False, f"Error saving figure: {str(e)}"


async def save_vtk_screenshot(
    renderer: vtk.vtkRenderer,
    output_path: Path,
    width: int,
    height: int
) -> Tuple[bool, Optional[str]]:
    """
    Save VTK renderer screenshot to file.
    
    Args:
        renderer: VTK renderer
        output_path: Output file path
        width: Image width
        height: Image height
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Create render window
        window = vtk.vtkRenderWindow()
        window.SetOffScreenRendering(1)
        window.AddRenderer(renderer)
        window.SetSize(width, height)
        
        # Render scene
        window.Render()
        
        # Create window to image filter
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(window)
        w2i.Update()
        
        # Create PNG writer
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(output_path))
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        
        return True, None
        
    except Exception as e:
        return False, f"Error saving VTK screenshot: {str(e)}"