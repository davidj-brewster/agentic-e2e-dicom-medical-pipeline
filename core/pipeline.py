"""
Main pipeline orchestration module.
Handles agent initialization, workflow caching, and pipeline execution.
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
from pydantic import BaseModel

from agents.analyzer import AnalyzerAgent
from agents.coordinator import CoordinatorAgent
from agents.preprocessor import PreprocessingAgent
from agents.visualizer import VisualizerAgent


class WorkflowCache(BaseModel):
    """Cache structure for successful workflow patterns"""
    subject_id: str
    workflow_steps: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    prompt_templates: Dict[str, str]
    timestamp: datetime = datetime.utcnow()


class Pipeline:
    """
    Main pipeline orchestrator.
    Manages agent initialization, workflow execution, and caching.
    """

    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Client(api_key=anthropic_api_key)
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize agents
        self.coordinator = CoordinatorAgent()
        self.preprocessor = PreprocessingAgent(self.coordinator.agent_id)
        self.analyzer = AnalyzerAgent(self.coordinator.agent_id)
        self.visualizer = VisualizerAgent(self.coordinator.agent_id)
        
        # Register agents with coordinator
        self._register_agents()

    def _register_agents(self) -> None:
        """Register all agents with the coordinator"""
        asyncio.run(self.coordinator.register_agent(
            self.preprocessor.agent_id,
            ["preprocessing", "registration"]
        ))
        self.logger.info(f"Registered preprocessor agent: {self.preprocessor.agent_id}")
        asyncio.run(self.coordinator.register_agent(
            self.analyzer.agent_id,
            ["segmentation", "clustering"]
        ))
        self.logger.info(f"Registered analyzer agent: {self.analyzer.agent_id}")
        asyncio.run(self.coordinator.register_agent(
            self.visualizer.agent_id,
            ["visualization", "reporting"]
        ))
        self.logger.info(f"Registered visualizer agent: {self.visualizer.agent_id}")

    async def _cache_workflow(
        self,
        subject_id: str,
        workflow_steps: List[Dict[str, Any]],
        performance_metrics: Dict[str, Any]
    ) -> None:
        """Cache successful workflow pattern"""
        try:
            # Generate prompt templates using Anthropic API
            prompt_templates = {}
            
            for step in workflow_steps:
                # Create a prompt template for the step
                prompt = f"""
                Task: Process neuroimaging data for subject analysis
                Step: {step['step_id']}
                Tool: {step['tool']}
                Parameters: {json.dumps(step['parameters'], indent=2)}
                Success Metrics: {json.dumps(step.get('success_metrics', {}), indent=2)}
                
                Generate a template for processing similar subjects.
                """
                
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    model=self.client.default_model,
                    max_tokens=1000,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                
                prompt_templates[step['step_id']] = response.content
                self.logger.debug(f"Prompt template for {step['step_id']}: {response.content}")
            
            # Create cache entry
            cache_entry = WorkflowCache(
                subject_id=subject_id,
                workflow_steps=workflow_steps,
                performance_metrics=performance_metrics,
                prompt_templates=prompt_templates
            )
            
            # Save cache to file
            cache_path = f"{self.cache_dir}/workflow_cache_{subject_id}.json"
            cache_path.write_text(cache_entry.json(indent=2))
            
        except Exception as e:
            self.logger.warn(f"Error caching workflow: {str(e)}. Continuing without cached step.")

    async def _load_cached_workflow(self, subject_id: str) -> Optional[WorkflowCache]:
        """Load cached workflow pattern for similar subject"""
        try:
            # Find most relevant cache entry
            cache_files = list(self.cache_dir.glob("workflow_cache_*.json"))
            if not cache_files:
                self.logger.debug("No workflow cache files found")
                return None
            
            # In a real implementation, this would use similarity matching
            # For now, just load the most recent cache
            latest_cache = max(cache_files, key=lambda p: p.stat().st_mtime)
            
            cache_data = json.loads(latest_cache.read_text())
            self.logger.info(f"Loaded cached workflow for {subject_id}")
            self.logger.debug(f"Cache data: {"\n".join(cache_data)}")
            return WorkflowCache(**cache_data)
            
        except Exception as e:
            print(f"Error loading workflow cache: {str(e)} (subject {subject_id})")
            return None

    async def _optimize_workflow(
        self,
        cached_workflow: WorkflowCache,
        subject_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Optimize workflow based on cached pattern and current subject"""
        try:
            optimized_steps = []
            
            for step in cached_workflow.workflow_steps:
                # Use Anthropic API to adapt the step for the current subject
                prompt = f"""
                Previous successful workflow step:
                {json.dumps(step, indent=2)}
                
                Template:
                {cached_workflow.prompt_templates[step['step_id']]}
                
                Current subject data:
                {json.dumps(subject_data, indent=2)}
                
                Optimize this workflow step for the current subject.
                Return only the optimized step configuration as valid JSON.
                """
                
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    model="claude-3-sonnet-20240229",
                    max_tokens=2000,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                
                try:
                    optimized_step = json.loads(response.content)
                    self.logger.debug(f"Optimized step: {optimized_step} from {step}")
                    optimized_steps.append(optimized_step)
                except json.JSONDecodeError as e:
                    # If optimization fails, use original step
                    self.logger.warn(f"Failed to optimize step: {step}: {e}")
                    optimized_steps.append(step)
            
            return optimized_steps
            
        except Exception as e:
            print(f"Error optimizing workflow: {str(e)}")
            return cached_workflow.workflow_steps

    async def process_subject(
        self,
        subject_id: str,
        input_files: Dict[str, Path],
        working_dir: Path
    ) -> Dict[str, Any]:
        """Process a single subject through the pipeline"""
        try:
            # Initialize workflow
            workflow_id = await self.coordinator.initialize_workflow(
                subject_id,
                {"input_files": {k: str(v) for k, v in input_files.items()}}
            )
            
            # Load cached workflow if available
            cached_workflow = await self._load_cached_workflow(subject_id)
            if cached_workflow:
                # Optimize workflow based on cache
                workflow_steps = await self._optimize_workflow(
                    cached_workflow,
                    {"subject_id": subject_id, "input_files": input_files}
                )
                # Update coordinator's workflow steps
                self.coordinator.workflow_steps = workflow_steps
                self.logger.debug(f"Optimized workflow steps: {'\n'.join(workflow_steps)}")
            
            # Run preprocessing
            await self.preprocessor.run_preprocessing(
                subject_id,
                input_files
            )
            
            # Wait for preprocessing to complete
            while True:
                status = await self.coordinator.monitor_workflow(workflow_id)
                if status["status"] != "in_progress":
                    break
                await asyncio.sleep(2)
            
            if status["status"] == "failed":
                self.logger.warn("async def process_subject with subject_id: {subject_id}: Preprocessing failed")
                raise Exception("Preprocessing failed")
            
            # Get preprocessed file paths
            preprocessed_files = {
                "T1": working_dir / subject_id / "prep" / "T1_norm.nii.gz",
                "T2_FLAIR": working_dir / subject_id / "reg" / "T2_FLAIR_reg.nii.gz"
            }
            
            # Run analysis
            await self.analyzer.run_analysis(
                subject_id,
                preprocessed_files["T1"],
                preprocessed_files["T2_FLAIR"],
                working_dir
            )
            
            # Wait for analysis to complete
            while True:
                status = await self.coordinator.monitor_workflow(workflow_id)
                if status["status"] != "in_progress":
                    break
                await asyncio.sleep(2)
            
            if status["status"] == "failed":
                self.logger.warn("async def process_subject with subject_id: {subject_id}: Analysis failed (self.analyzer.run_analysis)")
                raise Exception("Analysis failed (self.analyzer.run_analysis)")
            
            # Get analysis results
            analysis_results = None
            for message in self.coordinator.message_queue:
                if (message.sender == self.analyzer.agent_id and
                    "analysis_results" in message.payload):
                    analysis_results = message.payload["analysis_results"]
                    self.logger.debug(f"Analysis results: {analysis_results}")
                    break
            
            if not analysis_results:
                self.logger.warn("async def process_subject with subject_id: {subject_id}: No analysis results found")
                raise Exception("No analysis results found")
            
            # Run visualization
            await self.visualizer.run_visualization(
                subject_id,
                preprocessed_files["T1"],
                preprocessed_files["T2_FLAIR"],
                analysis_results,
                working_dir
            )
            
            # Wait for visualization to complete
            while True:
                status = await self.coordinator.monitor_workflow(workflow_id)
                if status["status"] != "in_progress":
                    break
                await asyncio.sleep(1)
            
            if status["status"] == "failed":
                raise Exception("Visualization failed")
            
            # Cache successful workflow
            await self._cache_workflow(
                subject_id,
                self.coordinator.workflow_steps,
                status
            )
            
            # Return results
            return {
                "workflow_id": workflow_id,
                "status": status,
                "analysis_results": analysis_results,
                "visualization_paths": {
                    message.payload["visualization_paths"]
                    for message in self.coordinator.message_queue
                    if message.sender == self.visualizer.agent_id
                    and "visualization_paths" in message.payload
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing subject {subject_id}: {str(e)}")
            return {
                "workflow_id": workflow_id if 'workflow_id' in locals() else None,
                "status": "failed",
                "error": str(e)
            }

    async def process_dataset(
        self,
        dataset_dir: Path,
        working_dir: Path
    ) -> Dict[str, Any]:
        """Process all subjects in a dataset"""
        try:
            self.logger.debug(f"Processing dataset: {dataset_dir} from {working_dir}")
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
                    self.logger.info("Processing subject: {subject_id}")
                    self.logger.debug("Input files: {''.join(input_files)}")
                    results[subject_id] = await self.process_subject(
                        subject_id,
                        input_files,
                        working_dir
                    )
                else:
                    self.logger.warn("Subject: {subject_id} failed due to missing input files.")
                    results[subject_id] = {
                        "status": "failed",
                        "error": "Missing required input files"
                    }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing dataset: {str(e)}")
            return {"status": "failed", "error": str(e)}