"""
Main entry point for the neuroimaging analysis pipeline.
Handles argument parsing, configuration, and pipeline execution.
"""
import argparse
import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import rich
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from core.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("neuroimaging")
console = Console()

def setup_environment() -> Optional[str]:
    """Verify and setup required environment variables"""
    required_vars = {
        "FREESURFER_HOME": os.getenv("FREESURFER_HOME"),
        "FSLDIR": os.getenv("FSLDIR"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY")
    }
    
    missing = [var for var, value in required_vars.items() if not value]
    
    if missing:
        return f"Missing required environment variables: {', '.join(missing)}"
    
    # Source FreeSurfer setup if not already done
    if not os.getenv("SUBJECTS_DIR"):
        fs_setup = os.path.join(required_vars["FREESURFER_HOME"], "SetUpFreeSurfer.sh")
        if os.path.exists(fs_setup):
            os.system(f"source {fs_setup}")
    
    # Add FSL binaries to PATH if not already there
    fsl_bin = os.path.join(required_vars["FSLDIR"], "bin")
    if fsl_bin not in os.environ["PATH"]:
        os.environ["PATH"] = f"{fsl_bin}:{os.environ['PATH']}"
    
    return None


def create_working_directory(base_dir: Path) -> Path:
    """Create timestamped working directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    working_dir = base_dir / f"processing_{timestamp}"
    try:
        working_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating working directory: {str(e)}")
        raise e
    return working_dir


async def process_subject(
    pipeline: Pipeline,
    subject_dir: Path,
    working_dir: Path,
    progress: Progress
) -> Dict[str, any]:
    """Process a single subject with progress tracking"""
    subject_id = subject_dir.name
    task_id = progress.add_task(f"Processing subject {subject_id}...", total=None)
    
    try:
        # Find input files
        input_files = {
            "T1": next(subject_dir.glob("*(T1|MPRAGE)*.nii.gz"), None),
            "T2_FLAIR": next(subject_dir.glob("*(T2_)?FLAIR*.nii.gz"), None)
        }
        
        if not all(input_files.values()):
            missing = [k for k, v in input_files.items() if not v]
            raise ValueError(f"Missing required input files: {', '.join(missing)}")
        
        # Process subject
        result = await pipeline.process_subject(
            subject_id,
            input_files,
            working_dir
        )
        
        progress.update(task_id, completed=True)
        return result
        
    except Exception as e:
        logger.error(f"Error processing subject {subject_id}: {str(e)}")
        progress.update(task_id, completed=True)
        return {
            "status": "failed",
            "error": str(e)
        }


async def main(args: argparse.Namespace) -> None:
    """Main execution function"""
    try:
        # Verify environment
        if error := setup_environment():
            logger.error(error)
            return
        
        # Create working directory
        working_dir = create_working_directory(args.output_dir)
        logger.info(f"Created working directory: {working_dir}")
        
        # Initialize pipeline
        pipeline = Pipeline(os.getenv("ANTHROPIC_API_KEY"))
        
        # Process dataset
        dataset_dir = Path(args.input_dir)
        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return
        
        # Find subject directories
        subject_dirs = [
            d for d in dataset_dir.iterdir()
            if d.is_dir() and any(d.glob("*.nii.gz"))
        ]
        
        if not subject_dirs:
            logger.error(f"No subject directories found in {dataset_dir}")
            return
        
        logger.info(f"Found {len(subject_dirs)} subjects to process")
        
        # Process subjects with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            tasks = [
                process_subject(pipeline, subject_dir, working_dir, progress)
                for subject_dir in subject_dirs
            ]
            results = await asyncio.gather(*tasks)
        
        # Summarize results
        success_count = sum(1 for r in results if r["status"] != "failed")
        logger.info(f"\nProcessing complete:")
        logger.info(f"- Successfully processed: {success_count}/{len(results)} subjects")
        logger.info(f"- Results saved to: {working_dir}")
        
        # Save summary report
        summary_path = working_dir / "processing_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Processing Summary\n")
            f.write(f"=================\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input Directory: {dataset_dir}\n")
            f.write(f"Output Directory: {working_dir}\n\n")
            
            for subject_dir, result in zip(subject_dirs, results):
                f.write(f"\nSubject: {subject_dir.name}\n")
                f.write(f"Status: {result['status']}\n")
                if result["status"] == "failed":
                    f.write(f"Error: {result.get('error', 'Unknown error')}\n")
                else:
                    f.write(f"Workflow ID: {result['workflow_id']}\n")
                    if "visualization_paths" in result:
                        f.write("Visualization Outputs:\n")
                        for vis_type, path in result["visualization_paths"].items():
                            f.write(f"- {vis_type}: {path}\n")
        
        logger.info(f"Summary report saved to: {summary_path}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Neuroimaging Analysis Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to input dataset directory containing subject subdirectories"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Base directory for output files"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level"
    )
    
    args = parser.parse_args()
    
    # Update log level if specified
    logger.setLevel(args.log_level)
    
    # Run pipeline
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise