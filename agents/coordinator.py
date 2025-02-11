"""
Coordinator Agent for managing the neuroimaging analysis workflow.
Handles task orchestration, inter-agent communication, and workflow state management.
"""
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from core.messages import (
    Message,
    MessageType,
    Priority,
    create_command,
    create_error,
    create_status_update
)
from core.workflow import (
    ResourceRequirements,
    WorkflowCache,
    WorkflowDefinition,
    WorkflowManager,
    WorkflowState,
    WorkflowStep
)
from .base import BaseAgent, AgentConfig


class CoordinatorAgent(BaseAgent):
    """
    Coordinator Agent responsible for managing the neuroimaging analysis workflow.
    Handles task distribution, monitoring, and workflow optimization.
    """

    def __init__(
        self,
        config: AgentConfig,
        workflow_manager: Optional[WorkflowManager] = None
    ):
        super().__init__(config)
        self.workflow_manager = workflow_manager or WorkflowManager()
        self.active_workflows: Dict[UUID, WorkflowDefinition] = {}
        self.agent_capabilities: Dict[str, Set[str]] = {}

    async def _initialize(self) -> None:
        """Initialize coordinator resources"""
        await super()._initialize()
        self.logger.info("Initializing coordinator agent")
        # Additional initialization if needed

    async def _cleanup(self) -> None:
        """Cleanup coordinator resources"""
        await super()._cleanup()
        self.logger.info("Cleaning up coordinator agent")
        # Additional cleanup if needed

    async def _handle_command(self, message: Message) -> None:
        """Handle command messages"""
        command = message.payload.command
        params = message.payload.parameters

        try:
            if command == "initialize_workflow":
                workflow_id = await self._initialize_workflow(
                    params["subject_id"],
                    params.get("input_data", {})
                )
                await self._send_message(
                    create_status_update(
                        sender=self.config.name,
                        recipient=message.sender,
                        state="workflow_initialized",
                        details={"workflow_id": str(workflow_id)}
                    )
                )

            elif command == "register_agent":
                await self._register_agent(
                    params["agent_id"],
                    params["capabilities"]
                )
                await self._send_message(
                    create_status_update(
                        sender=self.config.name,
                        recipient=message.sender,
                        state="agent_registered"
                    )
                )

            else:
                await self._handle_error(
                    f"Unknown command: {command}",
                    message,
                )

        except Exception as e:
            await self._handle_error(str(e), message)

    async def _handle_data(self, message: Message) -> None:
        """Handle data messages"""
        data_type = message.payload.data_type
        content = message.payload.content

        try:
            if data_type == "workflow_result":
                workflow_id = UUID(content["workflow_id"])
                if workflow_id in self.active_workflows:
                    await self._process_workflow_result(workflow_id, content)
            else:
                self.logger.warning(f"Unhandled data type: {data_type}")

        except Exception as e:
            await self._handle_error(str(e), message)

    async def _handle_query(self, message: Message) -> None:
        """Handle query messages"""
        query_type = message.payload.query_type
        params = message.payload.parameters

        try:
            if query_type == "workflow_status":
                workflow_id = UUID(params["workflow_id"])
                status = await self._get_workflow_status(workflow_id)
                await self._send_message(
                    create_status_update(
                        sender=self.config.name,
                        recipient=message.sender,
                        state="query_response",
                        details=status
                    )
                )
            else:
                self.logger.warning(f"Unhandled query type: {query_type}")

        except Exception as e:
            await self._handle_error(str(e), message)

    async def _initialize_workflow(
        self,
        subject_id: str,
        input_data: Dict[str, Any]
    ) -> UUID:
        """Initialize a new workflow for a subject"""
        # Check cache for similar workflow
        cached = await self.workflow_manager.get_cached_workflow(subject_id)
        
        # Create new workflow
        workflow = self.workflow_manager.create_workflow(
            name=f"Workflow-{subject_id}",
            description=f"Neuroimaging analysis for subject {subject_id}"
        )
        
        # Define workflow stages
        preprocessing = self.workflow_manager.add_stage(
            workflow.workflow_id,
            "Preprocessing",
            "Image preprocessing and quality control"
        )
        
        analysis = self.workflow_manager.add_stage(
            workflow.workflow_id,
            "Analysis",
            "Segmentation and clustering analysis"
        )
        
        visualization = self.workflow_manager.add_stage(
            workflow.workflow_id,
            "Visualization",
            "Result visualization and report generation"
        )
        
        # Add steps with optimized parameters from cache if available
        if cached:
            self.logger.info(f"Using cached workflow parameters for {subject_id}")
            params = cached.optimized_parameters
        else:
            params = {}
        
        # Add workflow steps
        steps = [
            WorkflowStep(
                name="Input Validation",
                step_type="validation",
                command="validate_input",
                parameters={
                    "subject_id": subject_id,
                    "input_data": input_data,
                    **params.get("validation", {})
                },
                resources=ResourceRequirements(
                    cpu_cores=1,
                    memory_gb=4
                )
            ),
            WorkflowStep(
                name="FSL Preprocessing",
                step_type="preprocessing",
                command="preprocess_images",
                parameters={
                    "subject_id": subject_id,
                    **params.get("preprocessing", {})
                },
                resources=ResourceRequirements(
                    cpu_cores=4,
                    memory_gb=8
                )
            ),
            WorkflowStep(
                name="FreeSurfer Segmentation",
                step_type="segmentation",
                command="segment_brain",
                parameters={
                    "subject_id": subject_id,
                    **params.get("segmentation", {})
                },
                resources=ResourceRequirements(
                    cpu_cores=4,
                    memory_gb=16
                )
            ),
            WorkflowStep(
                name="Clustering Analysis",
                step_type="analysis",
                command="cluster_regions",
                parameters={
                    "subject_id": subject_id,
                    **params.get("clustering", {})
                },
                resources=ResourceRequirements(
                    cpu_cores=2,
                    memory_gb=8
                )
            ),
            WorkflowStep(
                name="Visualization",
                step_type="visualization",
                command="generate_visualizations",
                parameters={
                    "subject_id": subject_id,
                    **params.get("visualization", {})
                },
                resources=ResourceRequirements(
                    cpu_cores=2,
                    memory_gb=8,
                    gpu_memory_gb=2
                )
            )
        ]
        
        # Add steps to appropriate stages
        for step in steps[:2]:
            self.workflow_manager.add_step(
                workflow.workflow_id,
                preprocessing.stage_id,
                step
            )
        
        for step in steps[2:4]:
            self.workflow_manager.add_step(
                workflow.workflow_id,
                analysis.stage_id,
                step
            )
        
        self.workflow_manager.add_step(
            workflow.workflow_id,
            visualization.stage_id,
            steps[-1]
        )
        
        self.active_workflows[workflow.workflow_id] = workflow
        return workflow.workflow_id

    async def _register_agent(
        self,
        agent_id: str,
        capabilities: List[str]
    ) -> None:
        """Register a new agent with the coordinator"""
        self.agent_capabilities[agent_id] = set(capabilities)
        self.logger.info(f"Registered agent {agent_id} with capabilities: {capabilities}")

    async def _process_workflow_result(
        self,
        workflow_id: UUID,
        result: Dict[str, Any]
    ) -> None:
        """Process workflow step results"""
        try:
            # Update workflow state
            step_id = UUID(result["step_id"])
            new_state = WorkflowState(result["state"])
            metrics = result.get("metrics")
            
            self.workflow_manager.update_step_state(
                workflow_id,
                step_id,
                new_state,
                metrics
            )
            
            # Check if workflow is complete
            status = await self._get_workflow_status(workflow_id)
            if status["status"] == "completed":
                # Cache successful workflow
                await self.workflow_manager.cache_workflow(
                    workflow_id,
                    result["subject_id"],
                    status["performance_metrics"],
                    result["optimized_parameters"]
                )
                
            elif status["status"] == "failed":
                self.logger.error(f"Workflow {workflow_id} failed")
                # Implement failure recovery logic
                
        except Exception as e:
            self.logger.error(f"Error processing workflow result: {e}")
            raise

    async def _get_workflow_status(
        self,
        workflow_id: UUID
    ) -> Dict[str, Any]:
        """Get current status of workflow execution"""
        return self.workflow_manager.get_workflow_status(workflow_id)

    async def _send_heartbeat(self) -> None:
        """Send coordinator heartbeat"""
        await super()._send_heartbeat()
        # Add coordinator-specific status information
        active_workflows = len(self.active_workflows)
        registered_agents = len(self.agent_capabilities)
        
        await self._send_message(
            create_status_update(
                sender=self.config.name,
                recipient="system",
                state="active",
                details={
                    "active_workflows": active_workflows,
                    "registered_agents": registered_agents
                }
            )
        )