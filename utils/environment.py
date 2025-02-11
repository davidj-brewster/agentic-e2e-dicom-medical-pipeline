"""
Environment validation utilities for neuroimaging pipeline.
Checks for required FreeSurfer and FSL installations and configurations.
"""
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from rich.console import Console
from rich.table import Table

from core.config import PipelineConfig

console = Console()


@dataclass
class EnvironmentCheck:
    """Results of an environment check"""
    name: str
    status: bool
    message: str
    details: Optional[Dict[str, str]] = None


class EnvironmentValidator:
    """Validates system environment for neuroimaging pipeline"""

    def __init__(self):
        self.checks: List[EnvironmentCheck] = []

    def check_freesurfer(self) -> EnvironmentCheck:
        """Check FreeSurfer installation and configuration"""
        try:
            freesurfer_home = os.getenv("FREESURFER_HOME")
            if not freesurfer_home:
                return EnvironmentCheck(
                    name="FreeSurfer",
                    status=False,
                    message="FREESURFER_HOME environment variable not set"
                )

            freesurfer_path = Path(freesurfer_home)
            if not freesurfer_path.exists():
                return EnvironmentCheck(
                    name="FreeSurfer",
                    status=False,
                    message=f"FreeSurfer directory not found: {freesurfer_path}"
                )

            # Check for critical executables
            required_bins = [
                "recon-all",
                "mri_convert",
                "mri_segment",
                "freeview"
            ]
            
            missing_bins = []
            for bin_name in required_bins:
                bin_path = freesurfer_path / "bin" / bin_name
                if not bin_path.exists():
                    missing_bins.append(bin_name)

            if missing_bins:
                return EnvironmentCheck(
                    name="FreeSurfer",
                    status=False,
                    message=f"Missing required FreeSurfer executables: {', '.join(missing_bins)}"
                )

            # Check license file
            license_file = freesurfer_path / "license.txt"
            if not license_file.exists():
                return EnvironmentCheck(
                    name="FreeSurfer",
                    status=False,
                    message="FreeSurfer license file not found"
                )

            # Try running a simple FreeSurfer command
            try:
                result = subprocess.run(
                    ["mri_info", "--version"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                version = result.stdout.strip()
            except subprocess.CalledProcessError:
                return EnvironmentCheck(
                    name="FreeSurfer",
                    status=False,
                    message="Failed to execute FreeSurfer command"
                )

            return EnvironmentCheck(
                name="FreeSurfer",
                status=True,
                message="FreeSurfer installation validated",
                details={
                    "version": version,
                    "path": str(freesurfer_path),
                    "license": "Present"
                }
            )

        except Exception as e:
            return EnvironmentCheck(
                name="FreeSurfer",
                status=False,
                message=f"FreeSurfer validation failed: {str(e)}"
            )

    def check_fsl(self) -> EnvironmentCheck:
        """Check FSL installation and configuration"""
        try:
            fsl_dir = os.getenv("FSLDIR")
            if not fsl_dir:
                return EnvironmentCheck(
                    name="FSL",
                    status=False,
                    message="FSLDIR environment variable not set"
                )

            fsl_path = Path(fsl_dir)
            if not fsl_path.exists():
                return EnvironmentCheck(
                    name="FSL",
                    status=False,
                    message=f"FSL directory not found: {fsl_path}"
                )

            # Check for critical executables
            required_bins = [
                "fslmaths",
                "fslstats",
                "flirt",
                "first",
                "run_first_all"
            ]
            
            missing_bins = []
            for bin_name in required_bins:
                bin_path = fsl_path / "bin" / bin_name
                if not bin_path.exists():
                    missing_bins.append(bin_name)

            if missing_bins:
                return EnvironmentCheck(
                    name="FSL",
                    status=False,
                    message=f"Missing required FSL executables: {', '.join(missing_bins)}"
                )

            # Try running a simple FSL command
            try:
                result = subprocess.run(
                    ["fslversion"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                version = result.stdout.strip()
            except subprocess.CalledProcessError:
                return EnvironmentCheck(
                    name="FSL",
                    status=False,
                    message="Failed to execute FSL command"
                )

            return EnvironmentCheck(
                name="FSL",
                status=True,
                message="FSL installation validated",
                details={
                    "version": version,
                    "path": str(fsl_path)
                }
            )

        except Exception as e:
            return EnvironmentCheck(
                name="FSL",
                status=False,
                message=f"FSL validation failed: {str(e)}"
            )

    def check_python_dependencies(self) -> EnvironmentCheck:
        """Check required Python packages"""
        try:
            import nibabel
            import numpy
            import scipy
            import nilearn
            import sklearn
            import matplotlib
            import anthropic
            
            return EnvironmentCheck(
                name="Python Dependencies",
                status=True,
                message="All required Python packages found",
                details={
                    "nibabel": nibabel.__version__,
                    "numpy": numpy.__version__,
                    "scipy": scipy.__version__,
                    "nilearn": nilearn.__version__,
                    "scikit-learn": sklearn.__version__,
                    "matplotlib": matplotlib.__version__,
                    "anthropic": anthropic.__version__
                }
            )

        except ImportError as e:
            return EnvironmentCheck(
                name="Python Dependencies",
                status=False,
                message=f"Missing Python dependency: {str(e)}"
            )
        except Exception as e:
            return EnvironmentCheck(
                name="Python Dependencies",
                status=False,
                message=f"Python dependency check failed: {str(e)}"
            )

    def check_system_requirements(self) -> EnvironmentCheck:
        """Check system requirements (disk space, memory, etc.)"""
        try:
            # Check available disk space (require at least 10GB)
            required_space = 10 * 1024 * 1024 * 1024  # 10GB in bytes
            _, _, free = shutil.disk_usage("/")
            
            if free < required_space:
                return EnvironmentCheck(
                    name="System Requirements",
                    status=False,
                    message=f"Insufficient disk space. Required: 10GB, Available: {free/(1024**3):.1f}GB"
                )

            # Check available memory (require at least 8GB)
            import psutil
            memory = psutil.virtual_memory()
            required_memory = 8 * 1024 * 1024 * 1024  # 8GB in bytes
            
            if memory.available < required_memory:
                return EnvironmentCheck(
                    name="System Requirements",
                    status=False,
                    message=f"Insufficient memory. Required: 8GB, Available: {memory.available/(1024**3):.1f}GB"
                )

            return EnvironmentCheck(
                name="System Requirements",
                status=True,
                message="System requirements met",
                details={
                    "disk_space_gb": f"{free/(1024**3):.1f}",
                    "memory_gb": f"{memory.available/(1024**3):.1f}"
                }
            )

        except Exception as e:
            return EnvironmentCheck(
                name="System Requirements",
                status=False,
                message=f"System requirements check failed: {str(e)}"
            )

    def validate_environment(self) -> Tuple[bool, List[EnvironmentCheck]]:
        """Run all environment checks"""
        self.checks = [
            self.check_freesurfer(),
            self.check_fsl(),
            self.check_python_dependencies(),
            self.check_system_requirements()
        ]
        
        all_passed = all(check.status for check in self.checks)
        return all_passed, self.checks

    def print_report(self) -> None:
        """Print environment validation report"""
        table = Table(title="Environment Validation Report")
        
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Message")
        table.add_column("Details", style="dim")
        
        for check in self.checks:
            status_style = "green" if check.status else "red"
            status_text = "✓ Pass" if check.status else "✗ Fail"
            
            details_text = ""
            if check.details:
                details_text = "\n".join(
                    f"{k}: {v}" for k, v in check.details.items()
                )
            
            table.add_row(
                check.name,
                f"[{status_style}]{status_text}[/{status_style}]",
                check.message,
                details_text
            )
        
        console.print(table)


def setup_environment(config: PipelineConfig) -> Optional[str]:
    """
    Set up environment variables and validate environment.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Error message if setup failed, None if successful
    """
    try:
        # Set environment variables
        os.environ["FREESURFER_HOME"] = str(config.freesurfer.freesurfer_home)
        os.environ["SUBJECTS_DIR"] = str(config.freesurfer.subjects_dir)
        os.environ["FSLDIR"] = str(config.fsl.fsl_dir)
        
        # Source FreeSurfer setup script
        setup_script = config.freesurfer.freesurfer_home / "SetUpFreeSurfer.sh"
        if setup_script.exists():
            try:
                subprocess.run(
                    ["bash", "-c", f"source {setup_script} && env"],
                    capture_output=True,
                    text=True,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                return f"Failed to source FreeSurfer setup script: {e}"
        
        # Source FSL setup script
        setup_script = config.fsl.fsl_dir / "etc/fslconf/fsl.sh"
        if setup_script.exists():
            try:
                subprocess.run(
                    ["bash", "-c", f"source {setup_script} && env"],
                    capture_output=True,
                    text=True,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                return f"Failed to source FSL setup script: {e}"
        
        # Validate environment
        validator = EnvironmentValidator()
        success, checks = validator.validate_environment()
        
        if not success:
            failed_checks = [check for check in checks if not check.status]
            return "\n".join(check.message for check in failed_checks)
        
        return None
        
    except Exception as e:
        return f"Environment setup failed: {str(e)}"


def verify_environment(config: PipelineConfig) -> Optional[str]:
    """
    Verify environment setup.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Error message if verification failed, None if successful
    """
    validator = EnvironmentValidator()
    success, checks = validator.validate_environment()
    
    if not success:
        failed_checks = [check for check in checks if not check.status]
        return "\n".join(check.message for check in failed_checks)
    
    return None


if __name__ == "__main__":
    validator = EnvironmentValidator()
    passed, checks = validator.validate_environment()
    
    validator.print_report()
    
    if not passed:
        console.print("\n[red]Environment validation failed![/red]")
        console.print("Please address the issues above before running the pipeline.")
        exit(1)
    else:
        console.print("\n[green]Environment validation successful![/green]")
        console.print("The system is ready to run the pipeline.")