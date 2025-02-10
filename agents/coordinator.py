"""
Coordinator Agent for managing the neuroimaging analysis workflow.
Handles task orchestration, inter-agent communication, and workflow state management.
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel


class TaskStatus(Enum):
    """Workflow task status states"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageType(Enum):
    """Types of inter-agent messages"""
    COMMAND = "command"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    DATA = "data"
    RESULT = "result"


class Priority(Enum):
    """Message priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class AgentMessage(BaseModel):
    """Standard message format for inter-agent communication"""
    message_id: UUID = uuid4()
    sender: str
    recipient: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: datetime = datetime.utcnow()
    priority: Priority = Priority.NORMAL


@dataclass
class WorkflowStep:
    """Represents a single step in the processing workflow"""
    step_id: str
    tool: str
    parameters: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None


class WorkflowCache(BaseModel):
    """Cache structure for storing successful workflow patterns"""
    workflow_id: UUID
    subject_id: str
    pipeline_steps: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    timestamp: datetime = datetime.utcnow()


class CoordinatorAgent:
    """
    Coordinator Agent responsible for managing the neuroimaging analysis workflow.
    Handles task distribution, monitoring, and workflow optimization.
    """

    def __init__(self):
        self.workflow_steps: List[WorkflowStep] = []
        self.workflow_cache: Dict[UUID, WorkflowCache] = {}
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        self.message_queue: List[AgentMessage] = []

    async def initialize_workflow(self, subject_id: str, input_data: Dict[str, Any]) -> UUID:
        """Initialize a new workflow for a subject"""
        workflow_id = uuid4()
        
        # Define the standard workflow steps
        self.workflow_steps = [
            WorkflowStep(
                step_id="input_validation",
                tool="data_validator",
                parameters={"subject_id": subject_id, "input_data": input_data}
            ),
            WorkflowStep(
                step_id="preprocessing",
                tool="fsl_preprocessor",
                parameters={"subject_id": subject_id}
            ),
            WorkflowStep(
                step_id="registration",
                tool="fsl_registration",
                parameters={"subject_id": subject_id}
            ),
            WorkflowStep(
                step_id="segmentation",
                tool="freesurfer_segmentation",
                parameters={"subject_id": subject_id}
            ),
            WorkflowStep(
                step_id="clustering",
                tool="intensity_clustering",
                parameters={"subject_id": subject_id}
            ),
            WorkflowStep(
                step_id="visualization",
                tool="freeview_renderer",
                parameters={"subject_id": subject_id}
            )
        ]
        
        return workflow_id

    async def register_agent(self, agent_id: str, capabilities: List[str]) -> None:
        """Register a new agent with the coordinator"""
        self.active_agents[agent_id] = {
            "capabilities": capabilities,
            "status": "idle",
            "last_heartbeat": datetime.utcnow()
        }

    async def send_message(self, message: AgentMessage) -> None:
        """Send a message to another agent"""
        self.message_queue.append(message)
        # In a real implementation, this would use proper message queuing
        await self._process_message_queue()

    async def _process_message_queue(self) -> None:
        """Process pending messages in the queue"""
        while self.message_queue:
            message = self.message_queue.pop(0)
            # Handle message based on type
            if message.message_type == MessageType.COMMAND:
                await self._handle_command(message)
            elif message.message_type == MessageType.STATUS_UPDATE:
                await self._handle_status_update(message)
            elif message.message_type == MessageType.ERROR:
                await self._handle_error(message)

    async def _handle_command(self, message: AgentMessage) -> None:
        """Handle incoming command messages"""
        if message.recipient not in self.active_agents:
            await self.send_message(
                AgentMessage(
                    sender="coordinator",
                    recipient=message.sender,
                    message_type=MessageType.ERROR,
                    payload={"error": f"Agent {message.recipient} not found"}
                )
            )
            return
        
        # Update agent status and forward command
        self.active_agents[message.recipient]["status"] = "busy"
        # Implementation would forward command to actual agent

    async def _handle_status_update(self, message: AgentMessage) -> None:
        """Handle status update messages from agents"""
        agent_id = message.sender
        if agent_id in self.active_agents:
            self.active_agents[agent_id]["last_heartbeat"] = datetime.utcnow()
            self.active_agents[agent_id]["status"] = message.payload.get("status", "idle")

    async def _handle_error(self, message: AgentMessage) -> None:
        """Handle error messages from agents"""
        # Log error and update workflow step status
        step_id = message.payload.get("step_id")
        error_message = message.payload.get("error")
        
        for step in self.workflow_steps:
            if step.step_id == step_id:
                step.status = TaskStatus.FAILED
                step.error = error_message
                break

    async def cache_workflow(self, workflow_id: UUID, metrics: Dict[str, Any]) -> None:
        """Cache successful workflow patterns for optimization"""
        if not self.workflow_steps:
            return

        cache_entry = WorkflowCache(
            workflow_id=workflow_id,
            subject_id=self.workflow_steps[0].parameters["subject_id"],
            pipeline_steps=[{
                "step_id": step.step_id,
                "tool": step.tool,
                "parameters": step.parameters,
                "success_metrics": metrics.get(step.step_id, {})
            } for step in self.workflow_steps if step.status == TaskStatus.COMPLETED],
            performance_metrics=metrics
        )
        
        self.workflow_cache[workflow_id] = cache_entry

    async def get_cached_workflow(self, subject_id: str) -> Optional[WorkflowCache]:
        """Retrieve cached workflow pattern for similar subjects"""
        # In a real implementation, this would include similarity matching logic
        for cache in self.workflow_cache.values():
            if cache.subject_id == subject_id:
                return cache
        return None

    async def monitor_workflow(self, workflow_id: UUID) -> Dict[str, Any]:
        """Monitor and report workflow progress"""
        total_steps = len(self.workflow_steps)
        completed_steps = sum(1 for step in self.workflow_steps 
                            if step.status == TaskStatus.COMPLETED)
        failed_steps = sum(1 for step in self.workflow_steps 
                          if step.status == TaskStatus.FAILED)
        
        return {
            "workflow_id": workflow_id,
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "progress": (completed_steps / total_steps) * 100 if total_steps > 0 else 0,
            "status": "failed" if failed_steps > 0 else 
                     "completed" if completed_steps == total_steps else 
                     "in_progress"
        }