"""
Command-line interface commands.
Implements command structure and execution logic.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import vtk
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from core.config import load_config
from core.pipeline import Pipeline
from utils.environment import setup_environment, verify_environment
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)

from utils.visualization import (
    ColorMap,
    ViewportConfig,
    create_3d_rendering,
    create_multiplanar_figure,
    save_figure,
    save_vtk_screenshot
)
from utils.interactive_viewer import (
    Interactive3DViewer,
    ViewerConfig
)
from utils.measurement_tools import (
    AngleTool,
    AreaVolumeTool,
    DistanceTool,
    MeasurementSystem
)
from utils.registration import (
    ImageRegistration,
    TransformType,
    MetricType
)
from utils.overlay import (
    OverlayVisualizer,
    OverlayMode,
    VisualizationConfig
)
from utils.statistics import (
    StatisticalComparison,
    StatisticalResult,
    StatisticalTest
)
from utils.registration import (
    ImageRegistration,
    TransformType,
    MetricType
)
from utils.overlay import (
    OverlayVisualizer,
    OverlayMode,
    VisualizationConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("neuroimaging")
console = Console()


@dataclass
class Argument:
    """Command line argument definition"""
    name: str
    help: str
    type: type = str
    default: Any = None
    required: bool = False
    choices: Optional[List[str]] = None


class Command(ABC):
    """Base command class"""
    name: str
    help: str
    arguments: List[Argument]

    def __init__(self):
        self.config = load_config()

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add command arguments to parser"""
        for arg in self.arguments:
            kwargs = {
                "help": arg.help,
                "type": arg.type,
                "default": arg.default,
                "required": arg.required
            }
            if arg.choices:
                kwargs["choices"] = arg.choices
            parser.add_argument(arg.name, **kwargs)

    @abstractmethod
    async def execute(self, args: Namespace) -> None:
        """Execute command"""
        pass


class SetupEnvironmentCommand(Command):
    """Setup environment command"""
    name = "setup-environment"
    help = "Setup system environment"
    arguments = [
        Argument(
            "--freesurfer-home",
            help="FreeSurfer installation directory",
            type=Path
        ),
        Argument(
            "--fsl-dir",
            help="FSL installation directory",
            type=Path
        ),
        Argument(
            "--anthropic-key",
            help="Anthropic API key"
        )
    ]

    async def execute(self, args: Namespace) -> None:
        """Execute setup environment command"""
        try:
            # Update environment variables
            if args.freesurfer_home:
                self.config.freesurfer.freesurfer_home = args.freesurfer_home
            if args.fsl_dir:
                self.config.fsl.fsl_dir = args.fsl_dir
            if args.anthropic_key:
                self.config.anthropic.api_key = args.anthropic_key
            
            # Setup environment
            if error := setup_environment(self.config):
                logger.error(f"Environment setup failed: {error}")
                return
            
            logger.info("Environment setup complete")
            
        except Exception as e:
            logger.error(f"Environment setup failed: {e}")


class ValidateCommand(Command):
    """Validate system command"""
    name = "validate"
    help = "Validate system setup"
    arguments = []

    async def execute(self, args: Namespace) -> None:
        """Execute validation command"""
        try:
            # Verify environment
            if error := verify_environment(self.config):
                logger.error(f"Validation failed: {error}")
                return
            
            logger.info("System validation complete")
            logger.info("✓ FreeSurfer installation")
            logger.info("✓ FSL installation")
            logger.info("✓ Environment variables")
            logger.info("✓ Required binaries")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")


class ProcessSubjectCommand(Command):
    """Process single subject command"""
    name = "process-subject"
    help = "Process single subject"
    arguments = [
        Argument(
            "subject_dir",
            help="Subject directory path",
            type=Path,
            required=True
        ),
        Argument(
            "--output-dir",
            help="Output directory path",
            type=Path,
            default=Path("output")
        )
    ]

    async def execute(self, args: Namespace) -> None:
        """Execute subject processing command"""
        try:
            # Verify subject directory
            if not args.subject_dir.exists():
                logger.error(f"Subject directory not found: {args.subject_dir}")
                return
            
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            working_dir = args.output_dir / f"processing_{timestamp}"
            working_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize pipeline
            pipeline = Pipeline(self.config)
            
            # Find input files
            input_files = {
                "T1": next(args.subject_dir.glob("*T1*.nii.gz"), None),
                "T2_FLAIR": next(args.subject_dir.glob("*T2_FLAIR*.nii.gz"), None)
            }
            
            if not all(input_files.values()):
                missing = [k for k, v in input_files.items() if not v]
                logger.error(f"Missing required input files: {', '.join(missing)}")
                return
            
            # Process subject with progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task_id = progress.add_task(
                    f"Processing subject {args.subject_dir.name}...",
                    total=None
                )
                
                result = await pipeline.process_subject(
                    args.subject_dir.name,
                    input_files
                )
                
                progress.update(task_id, completed=True)
            
            # Display results
            if result["status"] == "completed":
                logger.info("\nProcessing complete:")
                logger.info(f"- Workflow ID: {result['workflow_id']}")
                logger.info(f"- Results saved to: {working_dir}")
                
                if "visualization_paths" in result:
                    logger.info("\nVisualization outputs:")
                    for vis_type, path in result["visualization_paths"].items():
                        logger.info(f"- {vis_type}: {path}")
            else:
                logger.error(f"\nProcessing failed: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")


class ProcessDatasetCommand(Command):
    """Process dataset command"""
    name = "process-dataset"
    help = "Process multiple subjects"
    arguments = [
        Argument(
            "dataset_dir",
            help="Dataset directory path",
            type=Path,
            required=True
        ),
        Argument(
            "--output-dir",
            help="Output directory path",
            type=Path,
            default=Path("output")
        )
    ]

    async def execute(self, args: Namespace) -> None:
        """Execute dataset processing command"""
        try:
            # Verify dataset directory
            if not args.dataset_dir.exists():
                logger.error(f"Dataset directory not found: {args.dataset_dir}")
                return
            
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            working_dir = args.output_dir / f"processing_{timestamp}"
            working_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize pipeline
            pipeline = Pipeline(self.config)
            
            # Process dataset
            results = await pipeline.process_dataset(args.dataset_dir)
            
            # Display results
            success_count = sum(1 for r in results.values() if r["status"] == "completed")
            logger.info(f"\nProcessing complete:")
            logger.info(f"- Successfully processed: {success_count}/{len(results)} subjects")
            logger.info(f"- Results saved to: {working_dir}")
            
            # Save summary report
            summary_path = working_dir / "processing_summary.txt"
            with open(summary_path, "w") as f:
                f.write(f"Processing Summary\n")
                f.write(f"=================\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input Directory: {args.dataset_dir}\n")
                f.write(f"Output Directory: {working_dir}\n\n")
                
                for subject_id, result in results.items():
                    f.write(f"\nSubject: {subject_id}\n")
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
            logger.error(f"Processing failed: {e}")


class ListWorkflowsCommand(Command):
    """List workflows command"""
    name = "list-workflows"
    help = "List active and completed workflows"
    arguments = [
        Argument(
            "--status",
            help="Filter by workflow status",
            choices=["active", "completed", "failed"],
            default=None
        )
    ]

    async def execute(self, args: Namespace) -> None:
        """Execute list workflows command"""
        try:
            # Initialize pipeline
            pipeline = Pipeline(self.config)
            
            # Get workflows from coordinator
            workflows = await pipeline.agents["coordinator"].list_workflows(
                status=args.status
            )
            
            if not workflows:
                logger.info("No workflows found")
                return
            
            # Display workflows
            logger.info("\nWorkflows:")
            for workflow in workflows:
                logger.info(f"\nWorkflow ID: {workflow['workflow_id']}")
                logger.info(f"Status: {workflow['status']}")
                logger.info(f"Subject: {workflow['subject_id']}")
                logger.info(f"Started: {workflow['start_time']}")
                if workflow.get("end_time"):
                    logger.info(f"Completed: {workflow['end_time']}")
                if workflow.get("error"):
                    logger.info(f"Error: {workflow['error']}")
            
        except Exception as e:
            logger.error(f"Failed to list workflows: {e}")


class ShowWorkflowCommand(Command):
    """Show workflow details command"""
    name = "show-workflow"
    help = "Display workflow details"
    arguments = [
        Argument(
            "workflow_id",
            help="Workflow ID",
            required=True
        )
    ]

    async def execute(self, args: Namespace) -> None:
        """Execute show workflow command"""
        try:
            # Initialize pipeline
            pipeline = Pipeline(self.config)
            
            # Get workflow details
            workflow = await pipeline.agents["coordinator"].get_workflow_details(
                args.workflow_id
            )
            
            if not workflow:
                logger.error(f"Workflow not found: {args.workflow_id}")
                return
            
            # Display workflow details
            logger.info(f"\nWorkflow Details:")
            logger.info(f"ID: {workflow['workflow_id']}")
            logger.info(f"Status: {workflow['status']}")
            logger.info(f"Subject: {workflow['subject_id']}")
            logger.info(f"Started: {workflow['start_time']}")
            if workflow.get("end_time"):
                logger.info(f"Completed: {workflow['end_time']}")
            
            logger.info("\nSteps:")
            for step in workflow["steps"]:
                logger.info(f"\n- {step['step_id']}:")
                logger.info(f"  Status: {step['status']}")
                logger.info(f"  Agent: {step['agent']}")
                logger.info(f"  Started: {step['start_time']}")
                if step.get("end_time"):
                    logger.info(f"  Completed: {step['end_time']}")
                if step.get("error"):
                    logger.info(f"  Error: {step['error']}")
            
            if workflow.get("metrics"):
                logger.info("\nMetrics:")
                metrics = workflow["metrics"]
                logger.info(f"- Total Time: {metrics['total_time']:.2f}s")
                logger.info(f"- CPU Usage: {metrics['resource_usage']['cpu']['mean']:.1f}%")
                logger.info(f"- Memory Usage: {metrics['resource_usage']['memory']['mean']:.1f}%")
            
        except Exception as e:
            logger.error(f"Failed to show workflow: {e}")


class CancelWorkflowCommand(Command):
    """Cancel workflow command"""
    name = "cancel-workflow"
    help = "Cancel running workflow"
    arguments = [
        Argument(
            "workflow_id",
            help="Workflow ID",
            required=True
        )
    ]

    async def execute(self, args: Namespace) -> None:
        """Execute cancel workflow command"""
        try:
            # Initialize pipeline
            pipeline = Pipeline(self.config)
            
            # Cancel workflow
            success = await pipeline.agents["coordinator"].cancel_workflow(
                args.workflow_id
            )
            
            if success:
                logger.info(f"Workflow cancelled: {args.workflow_id}")
            else:
                logger.error(f"Failed to cancel workflow: {args.workflow_id}")
            
        except Exception as e:
            logger.error(f"Failed to cancel workflow: {e}")


class ListCacheCommand(Command):
    """List cached workflows command"""
    name = "list-cache"
    help = "Show cached workflows"
    arguments = [
        Argument(
            "--pattern",
            help="Filter by workflow pattern",
            default=None
        ),
        Argument(
            "--days",
            help="Filter by age in days",
            type=int,
            default=None
        ),
        Argument(
            "--min-score",
            help="Filter by minimum similarity score",
            type=float,
            default=None
        )
    ]

    async def execute(self, args: Namespace) -> None:
        """Execute list cache command"""
        try:
            # Initialize pipeline
            pipeline = Pipeline(self.config)
            
            # Get cache entries
            entries = await pipeline.workflow_cache.list_entries(
                pattern=args.pattern,
                max_age_days=args.days,
                min_score=args.min_score
            )
            
            if not entries:
                logger.info("No cached workflows found")
                return
            
            # Display cache entries
            logger.info("\nCached Workflows:")
            for entry in entries:
                logger.info(f"\nSubject: {entry.subject_id}")
                logger.info(f"Similarity Score: {entry.similarity_score:.2f}")
                logger.info(f"Timestamp: {entry.timestamp}")
                logger.info(f"Steps: {len(entry.pattern.steps)}")
                logger.info("Metrics:")
                for name, value in entry.pattern.metrics.items():
                    logger.info(f"- {name}: {value}")
            
        except Exception as e:
            logger.error(f"Failed to list cache: {e}")


class ShowCacheCommand(Command):
    """Show cache details command"""
    name = "show-cache"
    help = "Display cache details"
    arguments = [
        Argument(
            "subject_id",
            help="Subject ID",
            required=True
        )
    ]

    async def execute(self, args: Namespace) -> None:
        """Execute show cache command"""
        try:
            # Initialize pipeline
            pipeline = Pipeline(self.config)
            
            # Get cache entry
            entry = await pipeline.workflow_cache.get_entry(args.subject_id)
            
            if not entry:
                logger.error(f"Cache entry not found: {args.subject_id}")
                return
            
            # Display cache details
            logger.info(f"\nCache Details for {args.subject_id}:")
            logger.info(f"Timestamp: {entry.timestamp}")
            logger.info(f"Similarity Score: {entry.similarity_score:.2f}")
            
            logger.info("\nWorkflow Steps:")
            for step in entry.pattern.steps:
                logger.info(f"\n- {step['step_id']}:")
                logger.info(f"  Tool: {step['tool']}")
                logger.info(f"  Parameters:")
                for name, value in step['parameters'].items():
                    logger.info(f"    {name}: {value}")
            
            logger.info("\nPerformance Metrics:")
            for name, value in entry.pattern.metrics.items():
                logger.info(f"- {name}: {value}")
            
            logger.info("\nPrompt Templates:")
            for step_id, template in entry.prompt_templates.items():
                logger.info(f"\n{step_id}:")
                logger.info(template)
            
        except Exception as e:
            logger.error(f"Failed to show cache: {e}")


class ClearCacheCommand(Command):
    """Clear cache command"""
    name = "clear-cache"
    help = "Remove cached items"
    arguments = [
        Argument(
            "--pattern",
            help="Filter by workflow pattern",
            default=None
        ),
        Argument(
            "--days",
            help="Filter by age in days",
            type=int,
            default=None
        ),
        Argument(
            "--min-score",
            help="Filter by minimum similarity score",
            type=float,
            default=None
        ),
        Argument(
            "--force",
            help="Force removal without confirmation",
            action="store_true"
        )
    ]

    async def execute(self, args: Namespace) -> None:
        """Execute clear cache command"""
        try:
            # Initialize pipeline
            pipeline = Pipeline(self.config)
            
            # Get entries to remove
            entries = await pipeline.workflow_cache.list_entries(
                pattern=args.pattern,
                max_age_days=args.days,
                min_score=args.min_score
            )
            
            if not entries:
                logger.info("No cached workflows found")
                return
            
            # Confirm removal
            if not args.force:
                logger.info(f"\nWill remove {len(entries)} cache entries:")
                for entry in entries:
                    logger.info(f"- {entry.subject_id} ({entry.timestamp})")
                
                response = input("\nProceed with removal? [y/N] ")
                if response.lower() != 'y':
                    logger.info("Operation cancelled")
                    return
            
            # Remove entries
            removed = await pipeline.workflow_cache.clear_entries(
                pattern=args.pattern,
                max_age_days=args.days,
                min_score=args.min_score
            )
            
            logger.info(f"Removed {removed} cache entries")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")


class OptimizeCacheCommand(Command):
    """Optimize cache command"""
    name = "optimize-cache"
    help = "Optimize cache storage"
    arguments = [
        Argument(
            "--target-size",
            help="Target cache size in GB",
            type=float,
            default=None
        ),
        Argument(
            "--min-score",
            help="Minimum similarity score to keep",
            type=float,
            default=None
        ),
        Argument(
            "--dry-run",
            help="Show what would be done without making changes",
            action="store_true"
        )
    ]

    async def execute(self, args: Namespace) -> None:
        """Execute optimize cache command"""
        try:
            # Initialize pipeline
            pipeline = Pipeline(self.config)
            
            # Get current cache stats
            stats = await pipeline.workflow_cache.get_stats()
            logger.info("\nCurrent Cache Stats:")
            logger.info(f"Total Size: {stats['size_gb']:.2f} GB")
            logger.info(f"Entry Count: {stats['entry_count']}")
            logger.info(f"Average Score: {stats['avg_score']:.2f}")
            
            # Optimize cache
            if args.dry_run:
                logger.info("\nDry run - no changes will be made")
            
            result = await pipeline.workflow_cache.optimize(
                target_size_gb=args.target_size,
                min_score=args.min_score,
                dry_run=args.dry_run
            )
            
            if args.dry_run:
                logger.info("\nWould remove:")
                logger.info(f"- {result['entries_to_remove']} entries")
                logger.info(f"- {result['size_to_free']:.2f} GB")
                logger.info("\nWould result in:")
                logger.info(f"- {result['final_entry_count']} entries")
                logger.info(f"- {result['final_size_gb']:.2f} GB")
                logger.info(f"- {result['final_avg_score']:.2f} average score")
            else:
                logger.info("\nOptimization complete:")
                logger.info(f"- Removed {result['entries_removed']} entries")
                logger.info(f"- Freed {result['size_freed']:.2f} GB")
                logger.info(f"- New size: {result['final_size_gb']:.2f} GB")
                logger.info(f"- New average score: {result['final_avg_score']:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to optimize cache: {e}")


class ViewResultsCommand(Command):
    """Display analysis results command"""
    name = "view-results"
    help = "Display analysis results with interactive viewer"
    arguments = [
        Argument(
            "subject_id",
            help="Subject ID",
            required=True
        ),
        Argument(
            "--view-type",
            help="View type",
            choices=["axial", "sagittal", "coronal", "all"],
            default="all"
        ),
        Argument(
            "--overlay",
            help="Overlay type",
            choices=["mask", "clusters", "anomalies"],
            default="anomalies"
        ),
        Argument(
            "--output",
            help="Save screenshot to file",
            type=Path,
            default=None
        )
    ]

    async def execute(self, args: Namespace) -> None:
        """Execute view results command"""
        try:
            # Initialize pipeline
            pipeline = Pipeline(self.config)
            
            # Get subject results
            results = await pipeline.get_subject_results(args.subject_id)
            if not results:
                logger.error(f"No results found for subject: {args.subject_id}")
                return
            
            # Load images
            t1_path = results["preprocessed_files"]["T1"]
            flair_path = results["preprocessed_files"]["T2_FLAIR"]
            
            t1_img = nib.load(str(t1_path))
            flair_img = nib.load(str(flair_path))
            
            t1_data = t1_img.get_fdata()
            flair_data = flair_img.get_fdata()
            
            # Create overlay based on type
            if args.overlay == "mask":
                overlay_data = np.zeros_like(t1_data)
                for region, result in results["analysis_results"].items():
                    mask_path = result["segmentation"]["mask_path"]
                    mask_data = nib.load(str(mask_path)).get_fdata()
                    overlay_data += mask_data
            else:  # clusters or anomalies
                overlay_data = np.zeros_like(t1_data)
                for region, result in results["analysis_results"].items():
                    for cluster in (
                        result["anomalies"] if args.overlay == "anomalies"
                        else result["clusters"]
                    ):
                        mins, maxs = cluster["bounding_box"]
                        slice_indices = tuple(
                            slice(min_, max_+1)
                            for min_, max_ in zip(mins, maxs)
                        )
                        overlay_data[slice_indices] = cluster["outlier_score"]
            
            # Create figure based on view type
            if args.view_type == "all":
                success, error, fig = await create_multiplanar_figure(
                    background=t1_data,
                    overlay=overlay_data,
                    mask=overlay_data > 0,
                    background_cmap=ColorMap(
                        name=self.config.visualization.background_cmap
                    ),
                    overlay_cmap=ColorMap(
                        name=self.config.visualization.anomaly_cmap,
                        alpha=self.config.visualization.overlay_alpha
                    ),
                    dpi=self.config.visualization.output_dpi,
                    title=f"Analysis Results - {args.subject_id}"
                )
                
                if not success:
                    logger.error(f"Failed to create figure: {error}")
                    return
                
                # Save or display figure
                if args.output:
                    success, error = await save_figure(
                        fig,
                        args.output,
                        dpi=self.config.visualization.output_dpi
                    )
                    if success:
                        logger.info(f"Saved figure to: {args.output}")
                    else:
                        logger.error(f"Failed to save figure: {error}")
                else:
                    plt.show()
            else:
                # Create single view
                view_map = {
                    "axial": (0, 1),
                    "sagittal": (1, 2),
                    "coronal": (0, 2)
                }
                display_axes = view_map[args.view_type]
                
                # Create figure
                fig, ax = plt.subplots(
                    figsize=(10, 10),
                    dpi=self.config.visualization.output_dpi
                )
                
                # Get middle slice
                slice_idx = t1_data.shape[display_axes[0]] // 2
                
                # Extract slice data
                if args.view_type == "sagittal":
                    bg_slice = t1_data[slice_idx, :, :]
                    overlay_slice = overlay_data[slice_idx, :, :]
                elif args.view_type == "coronal":
                    bg_slice = t1_data[:, slice_idx, :]
                    overlay_slice = overlay_data[:, slice_idx, :]
                else:  # axial
                    bg_slice = t1_data[:, :, slice_idx]
                    overlay_slice = overlay_data[:, :, slice_idx]
                
                # Normalize background
                bg_slice = np.clip(
                    (bg_slice - bg_slice.min()) / (bg_slice.max() - bg_slice.min()),
                    0, 1
                )
                
                # Plot background
                ax.imshow(
                    bg_slice,
                    cmap=self.config.visualization.background_cmap
                )
                
                # Plot overlay
                if np.any(overlay_slice > 0):
                    ax.imshow(
                        overlay_slice,
                        cmap=self.config.visualization.anomaly_cmap,
                        alpha=self.config.visualization.overlay_alpha
                    )
                
                ax.set_title(f"{args.view_type.title()} View - Slice {slice_idx}")
                ax.axis('off')
                
                # Save or display figure
                if args.output:
                    success, error = await save_figure(
                        fig,
                        args.output,
                        dpi=self.config.visualization.output_dpi
                    )
                    if success:
                        logger.info(f"Saved figure to: {args.output}")
                    else:
                        logger.error(f"Failed to save figure: {error}")
                else:
                    plt.show()
            
        except Exception as e:
            logger.error(f"Failed to view results: {e}")


class ExportReportCommand(Command):
    """Generate analysis report command"""
    name = "export-report"
    help = "Generate analysis report in various formats"
    arguments = [
        Argument(
            "subject_id",
            help="Subject ID",
            required=True
        ),
        Argument(
            "--format",
            help="Report format",
            choices=["html", "pdf", "dicom"],
            default="html"
        ),
        Argument(
            "--template",
            help="Report template file",
            type=Path,
            default=None
        ),
        Argument(
            "--output",
            help="Output file path",
            type=Path,
            default=None
        )
    ]

    async def execute(self, args: Namespace) -> None:
        """Execute export report command"""
        try:
            # Initialize pipeline
            pipeline = Pipeline(self.config)
            
            # Get subject results
            results = await pipeline.get_subject_results(args.subject_id)
            if not results:
                logger.error(f"No results found for subject: {args.subject_id}")
                return
            
            # Create report directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = (
                args.output.parent if args.output
                else self.config.output_dir / "reports" / f"report_{timestamp}"
            )
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate visualizations
            vis_dir = report_dir / "images"
            vis_dir.mkdir(exist_ok=True)
            
            # Create multi-planar view
            mpr_path = vis_dir / "multiplanar.png"
            await ViewResultsCommand().execute(Namespace(
                subject_id=args.subject_id,
                view_type="all",
                overlay="anomalies",
                output=mpr_path
            ))
            
            # Create report
            if args.format == "html":
                template = args.template or Path("templates/report.html")
                if not template.exists():
                    logger.error(f"Template not found: {template}")
                    return
                
                # Load template
                with open(template) as f:
                    template_content = f.read()
                
                # Replace placeholders
                report_content = template_content.format(
                    subject_id=args.subject_id,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    multiplanar_image=mpr_path.relative_to(report_dir),
                    results=results
                )
                
                # Save report
                report_path = args.output or (report_dir / "report.html")
                with open(report_path, "w") as f:
                    f.write(report_content)
                
                logger.info(f"Saved HTML report to: {report_path}")
                
            elif args.format == "pdf":
                # TODO: Implement PDF export
                logger.error("PDF export not yet implemented")
                return
                
            else:  # dicom
                # TODO: Implement DICOM export
                logger.error("DICOM export not yet implemented")
                return
            
        except Exception as e:
            logger.error(f"Failed to export report: {e}")


class View3DCommand(Command):
    """Launch 3D viewer command"""
    name = "view-3d"
    help = "Launch interactive 3D viewer"
    arguments = [
        Argument(
            "subject_id",
            help="Subject ID",
            required=True
        ),
        Argument(
            "--mode",
            help="Rendering mode",
            choices=["surface", "volume"],
            default="volume"
        ),
        Argument(
            "--overlay",
            help="Overlay type",
            choices=["mask", "clusters", "anomalies"],
            default="anomalies"
        ),
        Argument(
            "--output",
            help="Save screenshot to file",
            type=Path,
            default=None
        )
    ]

    async def execute(self, args: Namespace) -> None:
        """Execute view 3D command"""
        try:
            # Initialize pipeline
            pipeline = Pipeline(self.config)
            
            # Get subject results
            results = await pipeline.get_subject_results(args.subject_id)
            if not results:
                logger.error(f"No results found for subject: {args.subject_id}")
                return
            
            # Load images
            t1_path = results["preprocessed_files"]["T1"]
            t1_img = nib.load(str(t1_path))
            t1_data = t1_img.get_fdata()
            
            # Create overlay based on type
            if args.overlay == "mask":
                overlay_data = np.zeros_like(t1_data)
                for region, result in results["analysis_results"].items():
                    mask_path = result["segmentation"]["mask_path"]
                    mask_data = nib.load(str(mask_path)).get_fdata()
                    overlay_data += mask_data
            else:  # clusters or anomalies
                overlay_data = np.zeros_like(t1_data)
                for region, result in results["analysis_results"].items():
                    for cluster in (
                        result["anomalies"] if args.overlay == "anomalies"
                        else result["clusters"]
                    ):
                        mins, maxs = cluster["bounding_box"]
                        slice_indices = tuple(
                            slice(min_, max_+1)
                            for min_, max_ in zip(mins, maxs)
                        )
                        overlay_data[slice_indices] = cluster["outlier_score"]
            
            # Create interactive viewer
            viewer = Interactive3DViewer(
                ViewerConfig(
                    width=self.config.visualization.window_width,
                    height=self.config.visualization.window_height,
                    background_color=self.config.visualization.background_color,
                    tool_color=self.config.visualization.tool_color,
                    tool_opacity=self.config.visualization.tool_opacity,
                    tool_line_width=self.config.visualization.tool_line_width
                )
            )
            
            # Create measurement system
            measurement_system = MeasurementSystem(t1_img)
            
            # Add measurement tools
            viewer.add_tool(DistanceTool(measurement_system))
            viewer.add_tool(AngleTool(measurement_system))
            viewer.add_tool(AreaVolumeTool(measurement_system))
            
            # Create volume actor
            volume_actor = vtk.vtkVolume()
            if args.mode == "surface":
                # Create surface
                surface = vtk.vtkMarchingCubes()
                surface.SetInputData(self.create_vtk_image(t1_data))
                surface.SetValue(0, 128)  # Threshold
                
                # Create mapper
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(surface.GetOutputPort())
                
                # Create actor
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                viewer.renderer.AddActor(actor)
                
            else:  # volume
                # Create transfer functions
                color = vtk.vtkColorTransferFunction()
                opacity = vtk.vtkPiecewiseFunction()
                
                # Add control points
                for x in range(0, 256, 32):
                    color.AddRGBPoint(
                        x,
                        *self.config.visualization.volume_color_points[x // 32]
                    )
                    opacity.AddPoint(
                        x,
                        self.config.visualization.volume_opacity_points[x // 32]
                    )
                
                # Create volume property
                property = vtk.vtkVolumeProperty()
                property.SetColor(color)
                property.SetScalarOpacity(opacity)
                property.ShadeOn()
                
                # Create mapper
                mapper = vtk.vtkSmartVolumeMapper()
                mapper.SetInputData(self.create_vtk_image(t1_data))
                
                # Create actor
                volume_actor.SetMapper(mapper)
                volume_actor.SetProperty(property)
                viewer.renderer.AddVolume(volume_actor)
            
            # Create overlay if needed
            if args.overlay:
                if args.mode == "surface":
                    # Create surface
                    surface = vtk.vtkMarchingCubes()
                    surface.SetInputData(self.create_vtk_image(overlay_data))
                    surface.SetValue(0, 0.5)  # Threshold
                    
                    # Create mapper
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(surface.GetOutputPort())
                    
                    # Create actor
                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetOpacity(
                        self.config.visualization.overlay_alpha
                    )
                    viewer.renderer.AddActor(actor)
                    
                else:  # volume
                    # Create transfer functions
                    color = vtk.vtkColorTransferFunction()
                    opacity = vtk.vtkPiecewiseFunction()
                    
                    # Add control points
                    color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
                    color.AddRGBPoint(1.0, 1.0, 0.0, 0.0)  # Red for anomalies
                    opacity.AddPoint(0.0, 0.0)
                    opacity.AddPoint(1.0, self.config.visualization.overlay_alpha)
                    
                    # Create volume property
                    property = vtk.vtkVolumeProperty()
                    property.SetColor(color)
                    property.SetScalarOpacity(opacity)
                    property.ShadeOn()
                    
                    # Create mapper
                    mapper = vtk.vtkSmartVolumeMapper()
                    mapper.SetInputData(self.create_vtk_image(overlay_data))
                    
                    # Create actor
                    overlay_actor = vtk.vtkVolume()
                    overlay_actor.SetMapper(mapper)
                    overlay_actor.SetProperty(property)
                    viewer.renderer.AddVolume(overlay_actor)
            
            # Switch to initial tool if specified
            if args.tool:
                viewer.switch_tool(args.tool)
            
            # Start viewer
            viewer.start()
            
            # Save results if any
            if viewer.results:
                results_dir = results["output_dir"] / "measurements"
                results_dir.mkdir(exist_ok=True)
                
                # Save individual results
                for i, result in enumerate(viewer.results, 1):
                    result_path = results_dir / f"measurement_{i}.json"
                    with open(result_path, "w") as f:
                        json.dump({
                            "tool": result["tool"],
                            "value": result["result"].value,
                            "unit": result["result"].unit.name,
                            "points": [
                                {
                                    "x": p.x,
                                    "y": p.y,
                                    "z": p.z
                                }
                                for p in result["result"].points
                            ],
                            "type": result["result"].type,
                            "metadata": result["result"].metadata
                        }, f, indent=2)
                
                # Create summary file
                summary_path = results_dir / "summary.json"
                with open(summary_path, "w") as f:
                    json.dump({
                        "count": len(viewer.results),
                        "tools": [r["tool"] for r in viewer.results],
                        "timestamp": datetime.now().isoformat()
                    }, f, indent=2)
                
                logger.info(f"Saved {len(viewer.results)} measurements to: {results_dir}")
            
            # Save screenshot if requested
            if args.output:
                viewer.save_screenshot(args.output)
                logger.info(f"Saved screenshot to: {args.output}")
            
            # Cleanup
            viewer.stop()
            
        except Exception as e:
            logger.error(f"Failed to view 3D: {e}")


class CompareSubjectsCommand(Command):
    """Compare multiple subjects command"""
    name = "compare-subjects"
    help = "Compare multiple subjects"
    arguments = [
        Argument(
            "subject_ids",
            help="Subject IDs (space-separated)",
            nargs="+",
            required=True
        ),
        Argument(
            "--mode",
            help="Comparison mode",
            choices=[m.name.lower() for m in OverlayMode],
            default="blend"
        ),
        Argument(
            "--metric",
            help="Comparison metric",
            choices=["mse", "correlation"],
            default="correlation"
        ),
        Argument(
            "--registration",
            help="Registration type",
            choices=[t.name.lower() for t in TransformType],
            default="affine"
        ),
        Argument(
            "--interactive",
            help="Enable interactive mode",
            action="store_true"
        ),
        Argument(
            "--output",
            help="Output directory",
            type=Path,
            default=None
        )
    ]

    async def execute(self, args: Namespace) -> None:
        """Execute compare subjects command"""
        try:
            # Initialize pipeline
            pipeline = Pipeline(self.config)
            
            # Get results for all subjects
            results = {}
            for subject_id in args.subject_ids:
                subject_results = await pipeline.get_subject_results(subject_id)
                if not subject_results:
                    logger.error(f"No results found for subject: {subject_id}")
                    return
                results[subject_id] = subject_results
            
            # Create output directory
            if args.output:
                output_dir = args.output
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = self.config.output_dir / "comparisons" / f"comparison_{timestamp}"
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load base subject
            base_subject = args.subject_ids[0]
            base_results = results[base_subject]
            base_img = nib.load(str(base_results["preprocessed_files"]["T1"]))
            base_data = base_img.get_fdata()
            
            # Register and compare each subject
            registered_images = []
            registered_masks = []
            for subject_id in args.subject_ids[1:]:
                # Load moving image
                subject_results = results[subject_id]
                moving_img = nib.load(str(subject_results["preprocessed_files"]["T1"]))
                
                # Load masks if available
                base_mask = None
                moving_mask = None
                if "segmentation" in base_results["analysis_results"]:
                    base_mask = nib.load(
                        str(base_results["analysis_results"]["segmentation"]["mask_path"])
                    ).get_fdata()
                if "segmentation" in subject_results["analysis_results"]:
                    moving_mask = nib.load(
                        str(subject_results["analysis_results"]["segmentation"]["mask_path"])
                    ).get_fdata()
                
                # Register image
                registration = ImageRegistration(
                    base_img,
                    moving_img,
                    transform_type=TransformType[args.registration.upper()],
                    metric_type=MetricType.MUTUAL_INFORMATION
                )
                
                result = registration.register()
                if not result.success:
                    logger.error(f"Registration failed for subject {subject_id}: {result.error}")
                    return
                
                registered_images.append(result.transformed_image)
                
                # Transform mask if available
                if moving_mask is not None:
                    registered_masks.append(
                        registration.transform_image(moving_mask)
                    )
                
                # Save transform if output directory provided
                if args.output:
                    transform_path = output_dir / f"{subject_id}_transform.tfm"
                    registration.save_transform(transform_path)
            
            # Create statistical comparison
            comparison = StatisticalComparison(
                [base_data] + [img.get_fdata() for img in registered_images],
                [base_mask] + registered_masks if base_mask else None,
                args.subject_ids
            )
            
            # Perform statistical test
            result = comparison.perform_statistical_test(
                StatisticalTest[args.test.upper()],
                alpha=args.alpha
            )
            
            # Create visualizer
            visualizer = OverlayVisualizer(
                base_data,
                registered_images[0].get_fdata(),
                VisualizationConfig(
                    window=self.config.visualization.window_width,
                    level=self.config.visualization.window_level,
                    blend_factor=0.5,
                    colormap=self.config.visualization.background_cmap,
                    overlay_colormap=self.config.visualization.anomaly_cmap,
                    overlay_alpha=self.config.visualization.overlay_alpha
                )
            )
            
            # Set visualization mode
            visualizer.mode = OverlayMode[args.mode.upper()]
            
            if args.interactive:
                # Create control window
                window = visualizer.create_controls()
                
                # Create notebook for tabs
                notebook = ttk.Notebook(window)
                notebook.pack(fill="both", expand=True)
                
                # Visualization tab
                vis_frame = ttk.Frame(notebook)
                notebook.add(vis_frame, text="Visualization")
                
                fig = plt.figure(figsize=(12, 8))
                canvas = FigureCanvasTkAgg(fig, vis_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)
                
                toolbar = NavigationToolbar2Tk(canvas, vis_frame)
                toolbar.update()
                
                ax = fig.add_subplot(111)
                
                def update_view():
                    """Update visualization"""
                    # Clear axis
                    ax.clear()
                    
                    # Get current visualization
                    result = visualizer.render()
                    
                    if isinstance(result, Figure):
                        # Copy figure contents
                        for child in result.get_children():
                            if isinstance(child, plt.Axes):
                                # Copy axis contents
                                for artist in child.get_children():
                                    artist.remove()
                                    ax.add_artist(copy.copy(artist))
                    else:
                        # Display array
                        ax.imshow(result)
                    
                    # Add metrics
                    metrics = visualizer.calculate_metrics()
                    ax.set_title(
                        f"MSE: {metrics['mse']:.3f}, "
                        f"Correlation: {metrics['correlation']:.3f}"
                    )
                    
                    # Update display
                    fig.canvas.draw_idle()
                
                # Add update callback
                window.after(100, update_view)
                
                # Statistics tab
                stats_frame = ttk.Frame(notebook)
                notebook.add(stats_frame, text="Statistics")
                
                # Add ROI statistics
                roi_frame = ttk.LabelFrame(stats_frame, text="ROI Statistics")
                roi_frame.pack(fill="x", padx=5, pady=5)
                
                stats = comparison.calculate_roi_statistics()
                for label, stat in stats.items():
                    group_frame = ttk.LabelFrame(roi_frame, text=label)
                    group_frame.pack(fill="x", padx=5, pady=5)
                    
                    for name, value in stat.items():
                        ttk.Label(
                            group_frame,
                            text=f"{name}: {value:.3f}"
                        ).pack(anchor="w")
                
                # Add statistical test results
                test_frame = ttk.LabelFrame(stats_frame, text="Statistical Test")
                test_frame.pack(fill="x", padx=5, pady=5)
                
                ttk.Label(
                    test_frame,
                    text=f"Test: {result.test_type.name}"
                ).pack(anchor="w")
                ttk.Label(
                    test_frame,
                    text=f"Statistic: {result.statistic:.3f}"
                ).pack(anchor="w")
                ttk.Label(
                    test_frame,
                    text=f"P-value: {result.p_value:.3e}"
                ).pack(anchor="w")
                ttk.Label(
                    test_frame,
                    text=f"Effect size: {result.effect_size:.3f}"
                ).pack(anchor="w")
                
                if result.confidence_interval:
                    ci_low, ci_high = result.confidence_interval
                    ttk.Label(
                        test_frame,
                        text=f"95% CI: ({ci_low:.3f}, {ci_high:.3f})"
                    ).pack(anchor="w")
                
                # Save button
                def save_results():
                    comparison.save_results(output_dir)
                    messagebox.showinfo(
                        "Success",
                        f"Results saved to: {output_dir}"
                    )
                
                ttk.Button(
                    window,
                    text="Save Results",
                    command=save_results
                ).pack(pady=10)
                
                # Start event loop
                window.mainloop()
                
            else:
                # Generate static visualization
                result = visualizer.render()
                
                if isinstance(result, Figure):
                    fig = result
                else:
                    fig = plt.figure(figsize=(12, 8))
                    plt.imshow(result)
                    
                    # Add metrics
                    metrics = visualizer.calculate_metrics()
                    plt.title(
                        f"MSE: {metrics['mse']:.3f}, "
                        f"Correlation: {metrics['correlation']:.3f}"
                    )
                
                # Save results
                comparison.save_results(output_dir)
                
                # Save or display figure
                if args.output:
                    output_path = output_dir / "comparison.png"
                    success, error = await save_figure(
                        fig,
                        output_path,
                        dpi=self.config.visualization.output_dpi
                    )
                    if success:
                        logger.info(f"Saved comparison to: {output_path}")
                    else:
                        logger.error(f"Failed to save comparison: {error}")
                else:
                    plt.show()
                
                # Display results
                logger.info("\nStatistical Results:")
                logger.info(f"Test: {result.test_type.name}")
                logger.info(f"Statistic: {result.statistic:.3f}")
                logger.info(f"P-value: {result.p_value:.3e}")
                logger.info(f"Effect size: {result.effect_size:.3f}")
                
                if result.confidence_interval:
                    ci_low, ci_high = result.confidence_interval
                    logger.info(
                        f"95% CI: ({ci_low:.3f}, {ci_high:.3f})"
                    )
            
        except Exception as e:
            logger.error(f"Failed to compare subjects: {e}")


# Register all commands
COMMANDS = {
    cmd.name: cmd for cmd in [
        SetupEnvironmentCommand(),
        ValidateCommand(),
        ProcessSubjectCommand(),
        ProcessDatasetCommand(),
        ListWorkflowsCommand(),
        ShowWorkflowCommand(),
        CancelWorkflowCommand(),
        ListCacheCommand(),
        ShowCacheCommand(),
        ClearCacheCommand(),
        OptimizeCacheCommand(),
        ViewResultsCommand(),
        ExportReportCommand(),
        View3DCommand(),
        CompareSubjectsCommand()
    ]
}