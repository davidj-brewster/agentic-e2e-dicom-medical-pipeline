"""
Test suite for core components of the neuroimaging pipeline.
Tests message protocol, workflow management, and base agent functionality.
"""
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import pytest
from pydantic import ValidationError

from agents.base import AgentConfig, BaseAgent
from core.messages import (
    Message,
    MessageType,
    Priority,
    create_command,
    create_data_message,
    create_error,
    create_status_update
)
from core.workflow import (
    ResourceRequirements,
    StepType,
    WorkflowManager,
    WorkflowState,
    WorkflowStep
)
from tests.mocks import (
    MockAgent,
    MockCommandRunner,
    MockMessageQueue,
    MockWorkflowValidator,
    setup_mock_command_runner
)


@pytest.fixture
def message_queue() -> MockMessageQueue:
    """Fixture for mock message queue"""
    return MockMessageQueue()


@pytest.fixture
def command_runner() -> MockCommandRunner:
    """Fixture for mock command runner"""
    return setup_mock_command_runner()


@pytest.fixture
def workflow_validator() -> MockWorkflowValidator:
    """Fixture for workflow validator"""
    return MockWorkflowValidator()


@pytest.fixture
def agent_config() -> AgentConfig:
    """Fixture for agent configuration"""
    return AgentConfig(
        name="test_agent",
        capabilities={"test"},
        resource_limits=ResourceRequirements(
            cpu_cores=2,
            memory_gb=4.0
        ),
        working_dir=Path("test_work")
    )


@pytest.fixture
def mock_agent(
    agent_config: AgentConfig,
    message_queue: MockMessageQueue,
    command_runner: MockCommandRunner
) -> MockAgent:
    """Fixture for mock agent"""
    return MockAgent(
        config=agent_config,
        message_queue=message_queue,
        command_runner=command_runner
    )


class TestMessageProtocol:
    """Tests for message protocol implementation"""
    
    def test_create_command(self):
        """Test command message creation"""
        message = create_command(
            sender="test",
            recipient="agent",
            command="process",
            parameters={"input": "test.nii.gz"}
        )
        
        assert message.message_type == MessageType.COMMAND
        assert message.sender == "test"
        assert message.recipient == "agent"
        assert message.payload.command == "process"
        assert message.payload.parameters == {"input": "test.nii.gz"}
    
    def test_create_status_update(self):
        """Test status update message creation"""
        message = create_status_update(
            sender="agent",
            recipient="coordinator",
            state="running",
            progress=0.5,
            message="Processing data"
        )
        
        assert message.message_type == MessageType.STATUS_UPDATE
        assert message.payload.state == "running"
        assert message.payload.progress == 0.5
        assert message.payload.message == "Processing data"
    
    def test_create_error(self):
        """Test error message creation"""
        message = create_error(
            sender="agent",
            recipient="coordinator",
            error_message="Process failed"
        )
        
        assert message.message_type == MessageType.ERROR
        assert message.payload.message == "Process failed"
    
    def test_message_validation(self):
        """Test message validation"""
        with pytest.raises(ValidationError):
            Message(
                sender="test",
                recipient="agent",
                message_type="invalid",
                payload={}
            )


class TestWorkflowManagement:
    """Tests for workflow management"""
    
    def test_workflow_creation(self):
        """Test workflow creation and configuration"""
        manager = WorkflowManager()
        workflow = manager.create_workflow(
            name="test_workflow",
            description="Test workflow"
        )
        
        assert workflow.name == "test_workflow"
        assert workflow.description == "Test workflow"
        assert len(workflow.stages) == 0
    
    def test_stage_addition(self):
        """Test adding stages to workflow"""
        manager = WorkflowManager()
        workflow = manager.create_workflow("test_workflow")
        
        stage = manager.add_stage(
            workflow.workflow_id,
            name="preprocessing",
            parallel_execution=True
        )
        
        assert len(workflow.stages) == 1
        assert stage.name == "preprocessing"
        assert stage.parallel_execution is True
    
    def test_step_addition(self):
        """Test adding steps to workflow stage"""
        manager = WorkflowManager()
        workflow = manager.create_workflow("test_workflow")
        stage = manager.add_stage(workflow.workflow_id, "preprocessing")
        
        step = WorkflowStep(
            name="normalize",
            step_type=StepType.PREPROCESSING,
            command="fslmaths",
            parameters={},
            resources=ResourceRequirements(
                cpu_cores=1,
                memory_gb=2.0
            )
        )
        
        manager.add_step(workflow.workflow_id, stage.stage_id, step)
        assert len(stage.steps) == 1
    
    def test_step_state_updates(self):
        """Test workflow step state updates"""
        manager = WorkflowManager()
        workflow = manager.create_workflow("test_workflow")
        stage = manager.add_stage(workflow.workflow_id, "preprocessing")
        
        step = WorkflowStep(
            name="normalize",
            step_type=StepType.PREPROCESSING,
            command="fslmaths",
            parameters={},
            resources=ResourceRequirements(
                cpu_cores=1,
                memory_gb=2.0
            )
        )
        
        manager.add_step(workflow.workflow_id, stage.stage_id, step)
        manager.update_step_state(
            workflow.workflow_id,
            step.step_id,
            WorkflowState.RUNNING
        )
        
        assert manager.step_states[workflow.workflow_id][step.step_id] == WorkflowState.RUNNING


class TestBaseAgent:
    """Tests for base agent functionality"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, mock_agent: MockAgent):
        """Test agent initialization"""
        await mock_agent._initialize()
        assert mock_agent.state.status == "ready"
    
    @pytest.mark.asyncio
    async def test_message_handling(
        self,
        mock_agent: MockAgent,
        message_queue: MockMessageQueue
    ):
        """Test message handling"""
        # Create test command
        message = create_command(
            sender="test",
            recipient=mock_agent.config.name,
            command="test_command",
            parameters={}
        )
        
        # Add message to queue
        await message_queue.add_message(message)
        
        # Process message
        messages = message_queue.get_messages(mock_agent.config.name)
        for msg in messages:
            await mock_agent._process_message(msg)
        
        # Verify command was processed
        assert "test_command" in mock_agent.processed_commands
    
    @pytest.mark.asyncio
    async def test_resource_management(self, mock_agent: MockAgent):
        """Test resource management"""
        requirements = ResourceRequirements(
            cpu_cores=1,
            memory_gb=2.0
        )
        
        # Check and allocate resources
        assert await mock_agent._check_resources(requirements)
        await mock_agent._allocate_resources(requirements)
        
        # Verify resource allocation
        assert mock_agent.state.resource_usage.get("cpu") == 1
        assert mock_agent.state.resource_usage.get("memory") == 2.0
        
        # Release resources
        await mock_agent._release_resources(requirements)
        
        # Verify resource release
        assert mock_agent.state.resource_usage.get("cpu") == 0
        assert mock_agent.state.resource_usage.get("memory") == 0


class TestMockSystem:
    """Tests for mock testing system"""
    
    @pytest.mark.asyncio
    async def test_command_execution(self, command_runner: MockCommandRunner):
        """Test mock command execution"""
        success, output, error = await command_runner.execute(
            "test_command",
            "arg1",
            kwarg1="value1"
        )
        
        assert success
        assert "Mock output" in output
        assert error is None
        
        executions = command_runner.get_executions("test_command")
        assert len(executions) == 1
        assert executions[0].args == ["arg1"]
        assert executions[0].kwargs == {"kwarg1": "value1"}
    
    @pytest.mark.asyncio
    async def test_message_queue_delays(self, message_queue: MockMessageQueue):
        """Test message queue delivery delays"""
        message_queue.set_delivery_delay("test_recipient", 0.1)
        
        start_time = datetime.utcnow()
        
        await message_queue.add_message(
            create_command(
                sender="test",
                recipient="test_recipient",
                command="test",
                parameters={}
            )
        )
        
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        assert elapsed >= 0.1
    
    def test_workflow_validation(self, workflow_validator: MockWorkflowValidator):
        """Test workflow validation"""
        step_id = uuid4()
        
        # Record state transitions
        workflow_validator.record_transition(step_id, WorkflowState.PENDING)
        workflow_validator.record_transition(step_id, WorkflowState.RUNNING)
        workflow_validator.record_transition(step_id, WorkflowState.COMPLETED)
        
        # Validate sequence
        valid, error = workflow_validator.validate_sequence(
            step_id,
            [
                WorkflowState.PENDING,
                WorkflowState.RUNNING,
                WorkflowState.COMPLETED
            ]
        )
        
        assert valid
        assert error is None