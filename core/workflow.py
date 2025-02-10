"""
Workflow definitions and state management.
Defines workflow steps, states, and transitions for the neuroimaging pipeline.
"""
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

from .messages import ErrorSeverity, Priority


class WorkflowState(Enum):
    """Possible states for workflow steps"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    RETRYING = "retrying"


class StepType(Enum):
    """Types of workflow steps"""
    PREPROCESSING = "preprocessing"
    REGISTRATION = "registration"
    SEGMENTATION = "segmentation"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    VALIDATION = "validation"
    CLEANUP = "cleanup"


class ResourceType(Enum):
    """Types of resources used in workflow steps"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"


class ResourceRequirements(BaseModel):
    """Resource requirements for a workflow step"""
    cpu_cores: Optional[int] = None
    memory_gb: Optional[float] = None
    gpu_memory_gb: Optional[float] = None
    disk_space_gb: Optional[float] = None
    network_bandwidth_mbps: Optional[float] = None


class StepDependency(BaseModel):
    """Dependency relationship between workflow steps"""
    step_id: UUID
    dependency_type: str = "completion"  # completion, data, resource, etc.
    conditions: Optional[Dict[str, Any]] = None


class StepValidation(BaseModel):
    """Validation criteria for workflow step outputs"""
    validation_type: str
    parameters: Dict[str, Any]
    severity: ErrorSeverity = ErrorSeverity.ERROR
    retry_count: int = 0
    timeout: Optional[float] = None


class StepMetrics(BaseModel):
    """Metrics collected during step execution"""
    duration: float
    start_time: datetime
    end_time: datetime
    resource_usage: Dict[ResourceType, float]
    error_count: int = 0
    retry_count: int = 0
    output_size_bytes: Optional[int] = None
    custom_metrics: Optional[Dict[str, Any]] = None


class WorkflowStep(BaseModel):
    """Definition of a single workflow step"""
    step_id: UUID = Field(default_factory=uuid4)
    name: str
    step_type: StepType
    state: WorkflowState = WorkflowState.PENDING
    priority: Priority = Priority.NORMAL
    
    # Dependencies and requirements
    dependencies: List[StepDependency] = []
    resources: ResourceRequirements
    
    # Execution details
    command: str
    parameters: Dict[str, Any]
    environment: Optional[Dict[str, str]] = None
    working_dir: Optional[Path] = None
    timeout: Optional[float] = None
    
    # Validation and metrics
    validations: List[StepValidation] = []
    metrics: Optional[StepMetrics] = None
    
    # Input/Output
    inputs: Dict[str, Path] = {}
    outputs: Dict[str, Path] = {}
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 60.0  # seconds
    
    @validator("working_dir", pre=True)
    def validate_working_dir(cls, v):
        """Ensure working directory is a Path object"""
        return Path(v) if v else None


class WorkflowStage(BaseModel):
    """Group of related workflow steps"""
    stage_id: UUID = Field(default_factory=uuid4)
    name: str
    description: Optional[str] = None
    steps: List[WorkflowStep]
    parallel_execution: bool = False
    timeout: Optional[float] = None


class WorkflowDefinition(BaseModel):
    """Complete workflow definition"""
    workflow_id: UUID = Field(default_factory=uuid4)
    name: str
    description: Optional[str] = None
    version: str
    stages: List[WorkflowStage]
    
    # Global settings
    global_timeout: Optional[float] = None
    max_parallel_steps: Optional[int] = None
    error_handling_policy: Dict[str, Any] = {}


class WorkflowCache(BaseModel):
    """Cache entry for workflow optimization"""
    cache_id: UUID = Field(default_factory=uuid4)
    workflow_id: UUID
    subject_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Cached data
    successful_steps: List[WorkflowStep]
    performance_metrics: Dict[str, Any]
    optimized_parameters: Dict[str, Any]
    
    # Cache metadata
    validity_period: Optional[float] = None  # seconds
    tags: List[str] = []


class WorkflowManager:
    """Manages workflow execution and state transitions"""
    
    def __init__(self):
        self.workflows: Dict[UUID, WorkflowDefinition] = {}
        self.step_states: Dict[UUID, Dict[UUID, WorkflowState]] = {}
        self.cache: Dict[UUID, WorkflowCache] = {}

    def create_workflow(
        self,
        name: str,
        description: Optional[str] = None,
        version: str = "1.0.0"
    ) -> WorkflowDefinition:
        """Create a new workflow definition"""
        workflow = WorkflowDefinition(
            name=name,
            description=description,
            version=version,
            stages=[]
        )
        self.workflows[workflow.workflow_id] = workflow
        self.step_states[workflow.workflow_id] = {}
        return workflow

    def add_stage(
        self,
        workflow_id: UUID,
        name: str,
        description: Optional[str] = None,
        parallel_execution: bool = False
    ) -> WorkflowStage:
        """Add a new stage to a workflow"""
        workflow = self.workflows[workflow_id]
        stage = WorkflowStage(
            name=name,
            description=description,
            parallel_execution=parallel_execution,
            steps=[]
        )
        workflow.stages.append(stage)
        return stage

    def add_step(
        self,
        workflow_id: UUID,
        stage_id: UUID,
        step: WorkflowStep
    ) -> None:
        """Add a step to a workflow stage"""
        workflow = self.workflows[workflow_id]
        stage = next(s for s in workflow.stages if s.stage_id == stage_id)
        stage.steps.append(step)
        self.step_states[workflow_id][step.step_id] = step.state

    def update_step_state(
        self,
        workflow_id: UUID,
        step_id: UUID,
        new_state: WorkflowState,
        metrics: Optional[StepMetrics] = None
    ) -> None:
        """Update the state of a workflow step"""
        workflow = self.workflows[workflow_id]
        for stage in workflow.stages:
            for step in stage.steps:
                if step.step_id == step_id:
                    step.state = new_state
                    if metrics:
                        step.metrics = metrics
                    self.step_states[workflow_id][step_id] = new_state
                    break

    def get_ready_steps(self, workflow_id: UUID) -> List[WorkflowStep]:
        """Get steps that are ready to execute"""
        workflow = self.workflows[workflow_id]
        ready_steps = []
        
        for stage in workflow.stages:
            for step in stage.steps:
                if self._is_step_ready(workflow_id, step):
                    ready_steps.append(step)
                if not stage.parallel_execution and ready_steps:
                    break
            if ready_steps:
                break
        
        return ready_steps

    def _is_step_ready(self, workflow_id: UUID, step: WorkflowStep) -> bool:
        """Check if a step is ready to execute"""
        if step.state != WorkflowState.PENDING:
            return False
            
        # Check dependencies
        for dep in step.dependencies:
            dep_state = self.step_states[workflow_id].get(dep.step_id)
            if dep_state != WorkflowState.COMPLETED:
                return False
        
        return True

    def cache_workflow(
        self,
        workflow_id: UUID,
        subject_id: str,
        performance_metrics: Dict[str, Any],
        optimized_parameters: Dict[str, Any]
    ) -> WorkflowCache:
        """Cache successful workflow for optimization"""
        workflow = self.workflows[workflow_id]
        successful_steps = [
            step for stage in workflow.stages
            for step in stage.steps
            if step.state == WorkflowState.COMPLETED
        ]
        
        cache_entry = WorkflowCache(
            workflow_id=workflow_id,
            subject_id=subject_id,
            successful_steps=successful_steps,
            performance_metrics=performance_metrics,
            optimized_parameters=optimized_parameters
        )
        
        self.cache[cache_entry.cache_id] = cache_entry
        return cache_entry

    def get_cached_workflow(
        self,
        subject_id: str,
        max_age: Optional[float] = None
    ) -> Optional[WorkflowCache]:
        """Retrieve cached workflow for similar subject"""
        if not self.cache:
            return None
            
        # Find most recent cache entry for subject
        relevant_caches = [
            cache for cache in self.cache.values()
            if cache.subject_id == subject_id
        ]
        
        if not relevant_caches:
            return None
            
        newest_cache = max(
            relevant_caches,
            key=lambda c: c.timestamp
        )
        
        # Check age if specified
        if max_age is not None:
            age = (datetime.utcnow() - newest_cache.timestamp).total_seconds()
            if age > max_age:
                return None
        
        return newest_cache

    def get_workflow_status(self, workflow_id: UUID) -> Dict[str, Any]:
        """Get current status of workflow execution"""
        workflow = self.workflows[workflow_id]
        total_steps = sum(len(stage.steps) for stage in workflow.stages)
        completed_steps = sum(
            1 for state in self.step_states[workflow_id].values()
            if state == WorkflowState.COMPLETED
        )
        failed_steps = sum(
            1 for state in self.step_states[workflow_id].values()
            if state == WorkflowState.FAILED
        )
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "version": workflow.version,
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "progress": (completed_steps / total_steps) * 100 if total_steps > 0 else 0,
            "status": "failed" if failed_steps > 0 else
                      "completed" if completed_steps == total_steps else
                      "in_progress"
        }