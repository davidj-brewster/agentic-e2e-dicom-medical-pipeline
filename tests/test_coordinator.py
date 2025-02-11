"""
Test suite for coordinator agent functionality.
Tests workflow initialization, agent registration, and workflow management.
"""
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from uuid import UUID

import pytest

from agents.base import AgentConfig
from agents.coordinator import CoordinatorAgent
from core.messages import (
    Message,
    MessageType,
    Priority,
    create_command,
    create_data_message,
    create_status_update
)
from core.workflow import (
    ResourceRequirements,
    WorkflowManager,
    WorkflowState
)
from tests.mocks import (
    MockMessageQueue,
    MockWorkflowValidator,
    setup_mock_command_runner
)


@pytest.fixture
def workflow_manager() -> WorkflowManager:
    """Fixture for workflow manager"""
    return WorkflowManager()


@pytest.fixture
def coordinator_config() -> AgentConfig:
    """Fixture for coordinator agent configuration"""
    return AgentConfig(
        name="coordinator",
        capabilities={"workflow_management", "agent_coordination"},
        resource_limits=ResourceRequirements(
            cpu_cores=2,
            memory_gb=4.0
        ),
        working_dir=Path("coordinator_work")
    )


@pytest.fixture
def coordinator_agent(
    coordinator_config: AgentConfig,
    workflow_manager: WorkflowManager
) -> CoordinatorAgent:
    """Fixture for coordinator agent"""
    return CoordinatorAgent(
        config=coordinator_config,
        workflow_manager=workflow_manager
    )


class TestCoordinatorAgent:
    """Tests for coordinator agent functionality"""
    
    @pytest.mark.asyncio
    async def test_workflow_initialization(self, coordinator_agent: CoordinatorAgent):
        """Test workflow initialization"""
        # Initialize workflow
        workflow_id = await coordinator_agent._initialize_workflow(
            subject_id="test_subject",
            input_data={"t1": "test.nii.gz"}
        )
        
        # Verify workflow creation
        assert workflow_id in coordinator_agent.active_workflows
        workflow = coordinator_agent.active_workflows[workflow_id]
        
        # Check stages
        assert len(workflow.stages) == 3  # Preprocessing, Analysis, Visualization
        
        # Check steps in each stage
        preprocessing_steps = workflow.stages[0].steps
        analysis_steps = workflow.stages[1].steps
        visualization_steps = workflow.stages[2].steps
        
        assert len(preprocessing_steps) == 2  # Validation and Preprocessing
        assert len(analysis_steps) == 2      # Segmentation and Clustering
        assert len(visualization_steps) == 1  # Visualization
        
        # Verify step configurations
        validation_step = preprocessing_steps[0]
        assert validation_step.name == "Input Validation"
        assert validation_step.parameters["subject_id"] == "test_subject"
        
        preprocessing_step = preprocessing_steps[1]
        assert preprocessing_step.name == "FSL Preprocessing"
        assert preprocessing_step.resources.cpu_cores == 4
        
        segmentation_step = analysis_steps[0]
        assert segmentation_step.name == "FreeSurfer Segmentation"
        assert segmentation_step.resources.memory_gb == 16
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, coordinator_agent: CoordinatorAgent):
        """Test agent registration"""
        # Register test agent
        await coordinator_agent._register_agent(
            agent_id="test_agent",
            capabilities=["preprocessing", "analysis"]
        )
        
        # Verify registration
        assert "test_agent" in coordinator_agent.agent_capabilities
        capabilities = coordinator_agent.agent_capabilities["test_agent"]
        assert "preprocessing" in capabilities
        assert "analysis" in capabilities
    
    @pytest.mark.asyncio
    async def test_workflow_result_processing(self, coordinator_agent: CoordinatorAgent):
        """Test workflow result processing"""
        # Initialize workflow
        workflow_id = await coordinator_agent._initialize_workflow(
            subject_id="test_subject",
            input_data={}
        )
        
        # Create mock result
        result = {
            "step_id": str(coordinator_agent.active_workflows[workflow_id].stages[0].steps[0].step_id),
            "state": "completed",
            "subject_id": "test_subject",
            "metrics": {
                "duration": 10.5,
                "memory_usage": 2.1
            },
            "optimized_parameters": {
                "preprocessing": {"method": "fast"}
            }
        }
        
        # Process result
        await coordinator_agent._process_workflow_result(workflow_id, result)
        
        # Verify state update
        status = await coordinator_agent._get_workflow_status(workflow_id)
        assert status["completed_steps"] == 1
    
    @pytest.mark.asyncio
    async def test_command_handling(self, coordinator_agent: CoordinatorAgent):
        """Test command message handling"""
        # Create initialize workflow command
        command = create_command(
            sender="test",
            recipient="coordinator",
            command="initialize_workflow",
            parameters={
                "subject_id": "test_subject",
                "input_data": {"t1": "test.nii.gz"}
            }
        )
        
        # Handle command
        await coordinator_agent._handle_command(command)
        
        # Verify workflow creation
        assert len(coordinator_agent.active_workflows) == 1
        
        # Get messages sent by coordinator
        messages = coordinator_agent.message_queue.get_messages("test")
        assert len(messages) == 1
        assert messages[0].message_type == MessageType.STATUS_UPDATE
        assert messages[0].payload.state == "workflow_initialized"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, coordinator_agent: CoordinatorAgent):
        """Test error handling in coordinator"""
        # Create invalid command
        command = create_command(
            sender="test",
            recipient="coordinator",
            command="invalid_command",
            parameters={}
        )
        
        # Handle command
        await coordinator_agent._handle_command(command)
        
        # Verify error message
        messages = coordinator_agent.message_queue.get_messages("test")
        assert len(messages) == 1
        assert messages[0].message_type == MessageType.ERROR
        assert "Unknown command" in messages[0].payload.message
    
    @pytest.mark.asyncio
    async def test_workflow_status_query(self, coordinator_agent: CoordinatorAgent):
        """Test workflow status querying"""
        # Initialize workflow
        workflow_id = await coordinator_agent._initialize_workflow(
            subject_id="test_subject",
            input_data={}
        )
        
        # Get status
        status = await coordinator_agent._get_workflow_status(workflow_id)
        
        # Verify status information
        assert status["workflow_id"] == workflow_id
        assert status["total_steps"] == 5  # Total number of steps across all stages
        assert status["completed_steps"] == 0
        assert status["status"] == "in_progress"
    
    @pytest.mark.asyncio
    async def test_workflow_caching(self, coordinator_agent: CoordinatorAgent):
        """Test workflow caching functionality"""
        # Initialize and complete workflow
        workflow_id = await coordinator_agent._initialize_workflow(
            subject_id="test_subject",
            input_data={}
        )
        
        # Complete all steps
        workflow = coordinator_agent.active_workflows[workflow_id]
        for stage in workflow.stages:
            for step in stage.steps:
                coordinator_agent.workflow_manager.update_step_state(
                    workflow_id,
                    step.step_id,
                    WorkflowState.COMPLETED,
                    {
                        "duration": 10.0,
                        "memory_usage": 2.0
                    }
                )
        
        # Process completion result
        await coordinator_agent._process_workflow_result(
            workflow_id,
            {
                "step_id": str(workflow.stages[-1].steps[-1].step_id),
                "state": "completed",
                "subject_id": "test_subject",
                "metrics": {"total_duration": 50.0},
                "optimized_parameters": {
                    "preprocessing": {"method": "fast"},
                    "segmentation": {"algorithm": "quick"}
                }
            }
        )
        
        # Initialize new workflow for same subject
        new_workflow_id = await coordinator_agent._initialize_workflow(
            subject_id="test_subject",
            input_data={}
        )
        
        # Verify cached parameters are used
        new_workflow = coordinator_agent.active_workflows[new_workflow_id]
        preprocessing_step = new_workflow.stages[0].steps[1]  # FSL Preprocessing step
        assert preprocessing_step.parameters.get("method") == "fast"