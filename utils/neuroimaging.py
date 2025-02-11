"""
Utilities for handling FSL and FreeSurfer commands asynchronously.
Provides safe command execution and result validation.
"""
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


async def run_command(
    command: str,
    args: List[str],
    env: Optional[Dict[str, str]] = None
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Run a command asynchronously and capture output.
    
    Args:
        command: Command to execute
        args: List of command arguments
        env: Optional environment variables
        
    Returns:
        Tuple of (success, stdout, stderr)
    """
    try:
        process = await asyncio.create_subprocess_exec(
            command,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        stdout, stderr = await process.communicate()
        success = process.returncode == 0
        
        if stdout:
            stdout = stdout.decode().strip()
        if stderr:
            stderr = stderr.decode().strip()
            
        if not success:
            logger.error(f"Command failed: {command} {' '.join(args)}")
            logger.error(f"Error output: {stderr}")
            
        return success, stdout, stderr
        
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        return False, None, str(e)


async def run_fsl_command(
    command: str,
    args: List[str],
    fsl_dir: Optional[str] = None
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Run an FSL command with proper environment setup.
    
    Args:
        command: FSL command name
        args: Command arguments
        fsl_dir: Optional FSL directory override
        
    Returns:
        Tuple of (success, stdout, stderr)
    """
    env = {}
    if fsl_dir:
        env["FSLDIR"] = fsl_dir
        env["PATH"] = f"{fsl_dir}/bin:{env.get('PATH', '')}"
    
    return await run_command(command, args, env)


async def run_freesurfer_command(
    command: str,
    args: List[str],
    fs_home: Optional[str] = None
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Run a FreeSurfer command with proper environment setup.
    
    Args:
        command: FreeSurfer command name
        args: Command arguments
        fs_home: Optional FreeSurfer home directory override
        
    Returns:
        Tuple of (success, stdout, stderr)
    """
    env = {}
    if fs_home:
        env["FREESURFER_HOME"] = fs_home
        env["PATH"] = f"{fs_home}/bin:{env.get('PATH', '')}"
    
    return await run_command(command, args, env)


async def normalize_image(
    input_path: Path,
    output_path: Path,
    target_resolution: Tuple[float, ...],
    current_resolution: Optional[Tuple[float, ...]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Normalize image resolution using FSL.
    
    Args:
        input_path: Input image path
        output_path: Output image path
        target_resolution: Target voxel dimensions
        current_resolution: Current voxel dimensions (if known)
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        if current_resolution and current_resolution == target_resolution:
            # No resampling needed
            return True, None
            
        scale_factors = [
            t / c for t, c in zip(target_resolution, current_resolution)
        ] if current_resolution else [1.0, 1.0, 1.0]
        
        success, stdout, stderr = await run_fsl_command(
            "fslmaths",
            [
                str(input_path),
                "-subsamp2",
                *map(str, scale_factors),
                str(output_path)
            ]
        )
        
        if not success:
            return False, f"Normalization failed: {stderr}"
            
        return True, None
        
    except Exception as e:
        return False, f"Error during normalization: {str(e)}"


async def register_images(
    moving_path: Path,
    fixed_path: Path,
    output_path: Path,
    transform_path: Optional[Path] = None,
    cost_function: str = "corratio",
    dof: int = 12,
    interp: str = "trilinear"
) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Register images using FSL FLIRT.
    
    Args:
        moving_path: Moving image path
        fixed_path: Fixed image path
        output_path: Output image path
        transform_path: Optional transform matrix path
        cost_function: Cost function for registration
        dof: Degrees of freedom
        interp: Interpolation method
        
    Returns:
        Tuple of (success, error_message, cost_value)
    """
    try:
        args = [
            "-in", str(moving_path),
            "-ref", str(fixed_path),
            "-out", str(output_path),
            "-cost", cost_function,
            "-dof", str(dof),
            "-interp", interp
        ]
        
        if transform_path:
            args.extend(["-omat", str(transform_path)])
        
        success, stdout, stderr = await run_fsl_command("flirt", args)
        
        if not success:
            return False, f"Registration failed: {stderr}", None
            
        # Extract cost value from output if available
        cost_value = None
        if stdout and "Final cost" in stdout:
            try:
                cost_line = next(
                    line for line in stdout.split("\n")
                    if "Final cost" in line
                )
                cost_value = float(cost_line.split("=")[1].strip())
            except (StopIteration, IndexError, ValueError):
                pass
        
        return True, None, cost_value
        
    except Exception as e:
        return False, f"Error during registration: {str(e)}", None


async def validate_image(
    image_path: Path,
    expected_dims: Optional[Tuple[int, ...]] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> Tuple[bool, Optional[str], Dict[str, float]]:
    """
    Validate image data and calculate basic statistics.
    
    Args:
        image_path: Image path to validate
        expected_dims: Expected dimensions (if any)
        min_value: Minimum expected value
        max_value: Maximum expected value
        
    Returns:
        Tuple of (valid, error_message, statistics)
    """
    try:
        # Get image statistics
        success, stdout, stderr = await run_fsl_command(
            "fslstats",
            [str(image_path), "-R", "-m", "-s"]
        )
        
        if not success:
            return False, f"Failed to get image statistics: {stderr}", {}
            
        # Parse statistics
        stats = stdout.split()
        range_min, range_max = map(float, stats[:2])
        mean, std = map(float, stats[2:])
        
        statistics = {
            "min": range_min,
            "max": range_max,
            "mean": mean,
            "std": std
        }
        
        # Validate dimensions if specified
        if expected_dims:
            success, stdout, stderr = await run_fsl_command(
                "fslinfo",
                [str(image_path)]
            )
            
            if success:
                dims = []
                for line in stdout.split("\n"):
                    if line.startswith("dim"):
                        dims.append(int(line.split()[1]))
                        
                if tuple(dims[:3]) != expected_dims:
                    return False, f"Unexpected dimensions: got {dims[:3]}, expected {expected_dims}", statistics
        
        # Validate value range
        if min_value is not None and range_min < min_value:
            return False, f"Values below minimum: {range_min} < {min_value}", statistics
            
        if max_value is not None and range_max > max_value:
            return False, f"Values above maximum: {range_max} > {max_value}", statistics
        
        return True, None, statistics
        
    except Exception as e:
        return False, f"Error during validation: {str(e)}", {}