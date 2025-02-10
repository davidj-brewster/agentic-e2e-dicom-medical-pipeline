"""
Mock testing framework for neuroimaging pipeline.
Provides mocks for FSL/FreeSurfer commands, agent communication, and workflow validation.
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import UUID

import nibabel as nib
import numpy as np
from pydantic import BaseModel

from agents.base import AgentConfig, BaseAgent
from core.messages import Message, MessageQueue, MessageType
from core.workflow import WorkflowState


@dataclass
class CommandExecution:
    """Record of a mock command execution"""
    command: str
    args: List[str]
    kwargs: Dict[str, Any]
    timestamp: datetime
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None


class MockCommandRunner:
    """Mocks command execution for testing"""
    
    def __init__(self):
        self.executions: List[CommandExecution] = []
        self.command_handlers: Dict[str, Callable] = {}
        self.should_fail: Set[str] = set()
    
    def register_handler(
        self,
        command: str,
        handler: Callable
    ) -> None:
        """Register a handler for a specific command"""
        self.command_handlers[command] = handler
    
    def set_failure(self, command: str) -> None:
        """Set a command to fail when executed"""
        self.should_fail.add(command)
    
    def clear_failure(self, command: str) -> None:
        """Clear failure setting for a command"""
        self.should_fail.discard(command)
    
    async def execute(
        self,
        command: str,
        *args: str,
        **kwargs: Any
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Execute a mock command"""
        execution = CommandExecution(
            command=command,
            args=list(args),
            kwargs=kwargs,
            timestamp=datetime.utcnow(),
            success=True
        )
        
        try:
            if command in self.should_fail:
                raise RuntimeError(f"Mock failure for command: {command}")
            
            if command in self.command_handlers:
                output = await self.command_handlers[command](*args, **kwargs)
                execution.output = output
            else:
                execution.output = f"Mock output for {command}"
            
        except Exception as e:
            execution.success = False
            execution.error = str(e)
        
        self.executions.append(execution)
        return (
            execution.success,
            execution.output,
            execution.error
        )
    
    def get_executions(
        self,
        command: Optional[str] = None
    ) -> List[CommandExecution]:
        """Get record of command executions"""
        if command:
            return [e for e in self.executions if e.command == command]
        return self.executions


class MockMessageQueue(MessageQueue):
    """Mock message queue for testing agent communication"""
    
    def __init__(self):
        super().__init__()
        self.message_history: List[Message] = []
        self.delivery_delays: Dict[str, float] = {}
    
    async def add_message(self, message: Message) -> None:
        """Add a message with optional delay"""
        self.message_history.append(message)
        
        if message.recipient in self.delivery_delays:
            await asyncio.sleep(self.delivery_delays[message.recipient])
        
        await super().add_message(message)
    
    def set_delivery_delay(
        self,
        recipient: str,
        delay: float
    ) -> None:
        """Set message delivery delay for a recipient"""
        self.delivery_delays[recipient] = delay
    
    def get_message_history(
        self,
        message_type: Optional[MessageType] = None
    ) -> List[Message]:
        """Get history of messages"""
        if message_type:
            return [m for m in self.message_history
                   if m.message_type == message_type]
        return self.message_history


class MockAgent(BaseAgent):
    """Mock agent for testing"""
    
    def __init__(
        self,
        config: AgentConfig,
        message_queue: Optional[MockMessageQueue] = None,
        command_runner: Optional[MockCommandRunner] = None
    ):
        super().__init__(config, message_queue or MockMessageQueue())
        self.command_runner = command_runner or MockCommandRunner()
        self.processed_commands: List[str] = []
    
    async def _handle_command(self, message: Message) -> None:
        """Handle command messages"""
        command = message.payload.command
        self.processed_commands.append(command)
        
        success, output, error = await self.command_runner.execute(
            command,
            **message.payload.parameters
        )
        
        if not success:
            await self._handle_error(
                f"Command failed: {error}",
                message
            )


def create_mock_nifti(
    shape: Tuple[int, ...] = (64, 64, 64),
    affine: Optional[np.ndarray] = None,
    data_type: np.dtype = np.float32
) -> Tuple[Path, np.ndarray]:
    """Create a mock NIfTI file for testing"""
    if affine is None:
        affine = np.eye(4)
    
    # Create random data
    data = np.random.rand(*shape).astype(data_type)
    
    # Create temporary file
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    file_path = temp_dir / "mock.nii.gz"
    
    # Save NIfTI file
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(file_path))
    
    return file_path, data


class MockWorkflowValidator:
    """Validates workflow execution in tests"""
    
    def __init__(self):
        self.state_transitions: Dict[UUID, List[Tuple[datetime, WorkflowState]]] = {}
    
    def record_transition(
        self,
        step_id: UUID,
        new_state: WorkflowState
    ) -> None:
        """Record a workflow state transition"""
        if step_id not in self.state_transitions:
            self.state_transitions[step_id] = []
        
        self.state_transitions[step_id].append(
            (datetime.utcnow(), new_state)
        )
    
    def validate_sequence(
        self,
        step_id: UUID,
        expected_states: List[WorkflowState]
    ) -> Tuple[bool, Optional[str]]:
        """Validate state transition sequence"""
        if step_id not in self.state_transitions:
            return False, f"No transitions recorded for step {step_id}"
        
        actual_states = [
            state for _, state in self.state_transitions[step_id]
        ]
        
        if actual_states != expected_states:
            return False, (
                f"Invalid state sequence for step {step_id}\n"
                f"Expected: {expected_states}\n"
                f"Actual: {actual_states}"
            )
        
        return True, None
    
    def get_execution_time(
        self,
        step_id: UUID,
        start_state: WorkflowState,
        end_state: WorkflowState
    ) -> Optional[float]:
        """Get execution time between states"""
        if step_id not in self.state_transitions:
            return None
        
        transitions = self.state_transitions[step_id]
        
        start_time = None
        end_time = None
        
        for timestamp, state in transitions:
            if state == start_state:
                start_time = timestamp
            elif state == end_state:
                end_time = timestamp
                break
        
        if start_time and end_time:
            return (end_time - start_time).total_seconds()
        return None


# FSL command mocks
async def mock_fslmaths(*args: str, **kwargs: Any) -> str:
    """Mock fslmaths command"""
    input_path = args[0]
    output_path = args[-1]
    
    # Create mock output
    if Path(input_path).exists():
        img = nib.load(input_path)
        data = img.get_fdata()
        
        # Apply mock operation
        if "-mul" in args:
            data *= float(args[args.index("-mul") + 1])
        elif "-add" in args:
            data += float(args[args.index("-add") + 1])
        
        # Save output
        nib.save(nib.Nifti1Image(data, img.affine), output_path)
        return f"Mock fslmaths output saved to {output_path}"
    
    raise FileNotFoundError(f"Input file not found: {input_path}")


async def mock_flirt(*args: str, **kwargs: Any) -> str:
    """Mock flirt registration command"""
    input_path = kwargs.get("in")
    reference_path = kwargs.get("ref")
    output_path = kwargs.get("out")
    
    if not all([input_path, reference_path, output_path]):
        raise ValueError("Missing required parameters")
    
    if not all(Path(p).exists() for p in [input_path, reference_path]):
        raise FileNotFoundError("Input or reference file not found")
    
    # Create mock registered output
    ref_img = nib.load(reference_path)
    mock_data = np.random.rand(*ref_img.shape)
    nib.save(nib.Nifti1Image(mock_data, ref_img.affine), output_path)
    
    return f"Mock registration output saved to {output_path}"


async def mock_first(*args: str, **kwargs: Any) -> str:
    """Mock FIRST segmentation command"""
    input_path = kwargs.get("in")
    output_path = kwargs.get("out")
    
    if not all([input_path, output_path]):
        raise ValueError("Missing required parameters")
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create mock segmentation
    img = nib.load(input_path)
    mock_seg = np.zeros(img.shape, dtype=np.int16)
    
    # Add some "segments"
    for i, size in enumerate([10, 15, 20], start=1):
        center = np.array(img.shape) // 2
        mock_seg[
            center[0]-size:center[0]+size,
            center[1]-size:center[1]+size,
            center[2]-size:center[2]+size
        ] = i
    
    nib.save(nib.Nifti1Image(mock_seg, img.affine), output_path)
    return f"Mock segmentation output saved to {output_path}"


# Register default mock handlers
def setup_mock_command_runner() -> MockCommandRunner:
    """Set up mock command runner with default handlers"""
    runner = MockCommandRunner()
    runner.register_handler("fslmaths", mock_fslmaths)
    runner.register_handler("flirt", mock_flirt)
    runner.register_handler("run_first_all", mock_first)
    return runner