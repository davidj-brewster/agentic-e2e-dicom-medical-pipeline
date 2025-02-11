"""
Main pipeline orchestration module.
Handles agent initialization, workflow caching, and pipeline execution.
"""
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import anthropic
from pydantic import BaseModel

from agents.analyzer import AnalyzerAgent
from agents.base import AgentConfig
from agents.coordinator import CoordinatorAgent
from agents.preprocessor import PreprocessingAgent
from agents.visualizer import VisualizerAgent
from core.config import (
    AnthropicConfig,
    CacheConfig,
    PipelineConfig,
    load_config
)
from utils.pipeline import (
    PipelineMonitor,
    ResourceMonitor,
    WorkflowCache,
    WorkflowPattern
)

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main pipeline orchestrator.
    Manages agent initialization, workflow execution, and caching.
    """

    def __init__(self, config_path: Optional[Path] = None):
        # Load configuration
        self.config = load_config(config_path)
        
        # Set up logging
        logging.basicConfig(
            level=self.config.logging.level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=self.config.logging.file
        )
        
        # Initialize Anthropic client
        self.client = anthropic.Client(api_key=self.config.anthropic.api_key)
        
        # Initialize utilities
        self.resource_monitor = ResourceMonitor(
            gpu_enabled=bool(self.config.resources.gpu_memory_gb)
        )
        self.workflow_cache = WorkflowCache(
            cache_dir=self.config.cache.cache_dir,
            anthropic_client=self.client,
            max_cache_size=self.config.cache.max_cache_size_gb,
            similarity_threshold=self.config.cache.similarity_threshold
        )
        self.pipeline_monitor = PipelineMonitor(self.resource_monitor)
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Create working directories
        self.config.working_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all pipeline agents"""
        try:
            # Create coordinator
            coordinator = CoordinatorAgent(
                config=AgentConfig(
                    name="coordinator",
                    capabilities={"workflow_management", "agent_coordination"},
                    resource_limits=self.config.resources.coordinator,
                    working_dir=self.config.working_dir
                )
            )
            
            # Create preprocessor
            preprocessor = PreprocessingAgent(
                config=AgentConfig(
                    name="preprocessor",
                    capabilities={"preprocessing", "registration"},
                    resource_limits=self.config.resources.preprocessor,
                    working_dir=self.config.working_dir
                ),
                preprocessing_config=self.config.processing
            )
            
            # Create analyzer
            analyzer = AnalyzerAgent(
                config=AgentConfig(
                    name="analyzer",
                    capabilities={"segmentation", "analysis"},
                    resource_limits=self.config.resources.analyzer,
                    working_dir=self.config.working_dir
                ),
                analyzer_config=self.config.analysis
            )
            
            # Create visualizer
            visualizer = VisualizerAgent(
                config=AgentConfig(
                    name="visualizer",
                    capabilities={"visualization", "reporting"},
                    resource_limits=self.config.resources.visualizer,
                    working_dir=self.config.working_dir
                ),
                visualization_config=self.config.visualization
            )
            
            return {
                "coordinator": coordinator,
                "preprocessor": preprocessor,
                "analyzer": analyzer,
                "visualizer": visualizer
            }
            
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            raise

    async def _check_resources(
        self,
        workflow_pattern: WorkflowPattern
    ) -> Tuple[bool, Optional[str]]:
        """Check if required resources are available"""
        try:
            total_cpu = 0
            total_memory = 0
            total_disk = 0
            total_gpu = 0
            
            # Sum resource requirements
            for step in workflow_pattern.steps:
                resources = step.get("resources", {})
                total_cpu += resources.get("cpu_cores", 0)
                total_memory += resources.get("memory_gb", 0)
                total_disk += resources.get("disk_space_gb", 0)
                total_gpu += resources.get("gpu_memory_gb", 0)
            
            # Check resources
            return await self.resource_monitor.check_resources(
                required_cpu=total_cpu,
                required_memory=total_memory,
                required_disk=total_disk,
                required_gpu=total_gpu
            )
            
        except Exception as e:
            return False, f"Error checking resources: {e}"

    async def process_subject(
        self,
        subject_id: str,
        input_files: Dict[str, Path]
    ) -> Dict[str, Any]:
        """Process a single subject through the pipeline"""
        try:
            logger.info(f"Processing subject: {subject_id}")
            
            # Start monitoring
            await self.pipeline_monitor.start_monitoring()
            
            # Find similar workflow pattern
            cached_pattern = await self.workflow_cache.find_similar_pattern(
                subject_id,
                {"input_files": input_files}
            )
            
            if cached_pattern:
                logger.info(f"Found similar workflow pattern (score: {cached_pattern.similarity_score:.2f})")
                # Optimize pattern for current subject
                pattern = await self.workflow_cache.optimize_workflow(
                    cached_pattern.pattern,
                    {"input_files": input_files},
                    cached_pattern.prompt_templates
                )
            else:
                logger.info("No similar workflow pattern found, using default")
                pattern = WorkflowPattern(
                    steps=[],  # Will be populated by coordinator
                    metrics={},
                    parameters={"input_files": input_files}
                )
            
            # Check resources
            resources_ok, error = await self._check_resources(pattern)
            if not resources_ok:
                raise RuntimeError(f"Insufficient resources: {error}")
            
            # Initialize workflow
            workflow_id = await self.agents["coordinator"].initialize_workflow(
                subject_id,
                pattern
            )
            
            try:
                # Run preprocessing
                start_time = datetime.utcnow()
                await self.agents["preprocessor"].run_preprocessing(
                    subject_id,
                    input_files
                )
                self.pipeline_monitor.record_step_time(
                    "preprocessing",
                    (datetime.utcnow() - start_time).total_seconds()
                )
                
                # Get preprocessed files
                preprocessed_files = {
                    "T1": self.config.working_dir / subject_id / "prep" / "T1_norm.nii.gz",
                    "T2_FLAIR": self.config.working_dir / subject_id / "reg" / "T2_FLAIR_reg.nii.gz"
                }
                
                # Run analysis
                start_time = datetime.utcnow()
                await self.agents["analyzer"].run_analysis(
                    subject_id,
                    preprocessed_files["T1"],
                    preprocessed_files["T2_FLAIR"]
                )
                self.pipeline_monitor.record_step_time(
                    "analysis",
                    (datetime.utcnow() - start_time).total_seconds()
                )
                
                # Get analysis results
                analysis_results = await self.agents["coordinator"].get_analysis_results(
                    workflow_id
                )
                
                # Run visualization
                start_time = datetime.utcnow()
                await self.agents["visualizer"].run_visualization(
                    subject_id,
                    preprocessed_files["T1"],
                    preprocessed_files["T2_FLAIR"],
                    analysis_results
                )
                self.pipeline_monitor.record_step_time(
                    "visualization",
                    (datetime.utcnow() - start_time).total_seconds()
                )
                
                # Get final status
                status = await self.agents["coordinator"].get_workflow_status(
                    workflow_id
                )
                
                if status["status"] == "completed":
                    # Cache successful workflow
                    await self.workflow_cache.save_pattern(
                        subject_id,
                        WorkflowPattern(
                            steps=pattern.steps,
                            metrics=self.pipeline_monitor.get_metrics(),
                            parameters={"input_files": input_files}
                        )
                    )
                
                # Stop monitoring
                await self.pipeline_monitor.stop_monitoring()
                
                return {
                    "workflow_id": workflow_id,
                    "status": status["status"],
                    "metrics": self.pipeline_monitor.get_metrics(),
                    "analysis_results": analysis_results,
                    "visualization_paths": await self.agents["coordinator"].get_visualization_paths(
                        workflow_id
                    )
                }
                
            except Exception as e:
                logger.error(f"Error processing subject {subject_id}: {e}")
                return {
                    "workflow_id": workflow_id,
                    "status": "failed",
                    "error": str(e)
                }
                
        except Exception as e:
            logger.error(f"Error initializing pipeline for subject {subject_id}: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }

    async def process_dataset(
        self,
        dataset_dir: Path
    ) -> Dict[str, Any]:
        """Process all subjects in a dataset"""
        try:
            logger.info(f"Processing dataset: {dataset_dir}")
            results = {}
            
            # Find all subject directories
            subject_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
            
            for subject_dir in subject_dirs:
                subject_id = subject_dir.name
                
                # Find input files
                input_files = {
                    "T1": next(subject_dir.glob("*T1*.nii.gz"), None),
                    "T2_FLAIR": next(subject_dir.glob("*T2_FLAIR*.nii.gz"), None)
                }
                
                if all(input_files.values()):
                    # Process subject
                    results[subject_id] = await self.process_subject(
                        subject_id,
                        input_files
                    )
                else:
                    logger.warning(f"Missing input files for subject: {subject_id}")
                    results[subject_id] = {
                        "status": "failed",
                        "error": "Missing required input files"
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }