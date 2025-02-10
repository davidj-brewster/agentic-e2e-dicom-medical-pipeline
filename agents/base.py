"""
Base agent class providing common functionality for all agents.
Implements messaging, state management, and error handling.
"""
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union, Tuple
from uuid import UUID

from pydantic import BaseModel

from core.messages import (
    ErrorInfo,
    ErrorSeverity,
    Message,
    MessageQueue,
    MessageType,
    Priority,
    create_command,
    create_data_message,
    create_error,
    create_result,
    create_status_update
)
from core.workflow import ResourceRequirements, WorkflowState


class AgentConfig(BaseModel):
    """Configuration for agent behavior"""
    name: str
    capabilities: Set[str]
    max_parallel_tasks: int = 1
    resource_limits: ResourceRequirements
    working_dir: Path
    message_timeout: float = 30.0
    retry_limit: int = 3
    retry_delay: float = 5.0


class AgentState(BaseModel):
    """Current state of an agent"""
    status: str = "idle"
    current_tasks: Dict[UUID, Dict[str, Any]] = {}
    resource_usage: Dict[str, float] = {}
    last_heartbeat: datetime = datetime.utcnow()
    error_count: int = 0
    processed_messages: int = 0


class BaseAgent:
    """
    Base agent implementation providing common functionality.
    All specialized agents should inherit from this class.
    """

    def __init__(
        self,
        config: AgentConfig,
        message_queue: Optional[MessageQueue] = None
    ):
        self.config = config
        self.state = AgentState()
        self.message_queue = message_queue or MessageQueue()
        self.logger = logging.getLogger(f"agent.{config.name}")
        
        # Set up logging
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Create working directory
        self.config.working_dir.mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        """Start the agent"""
        self.logger.info(f"Starting agent: {self.config.name}")
        try:
            await self._initialize()
            await self._run_message_loop()
        except Exception as e:
            self.logger.error(f"Agent failed: {str(e)}")
            raise

    async def stop(self) -> None:
        """Stop the agent"""
        self.logger.info(f"Stopping agent: {self.config.name}")
        try:
            await self._cleanup()
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
            raise

    async def _initialize(self) -> None:
        """Initialize agent resources"""
        self.state.status = "initializing"
        # Override in subclasses for specific initialization
        self.state.status = "ready"

    async def _cleanup(self) -> None:
        """Clean up agent resources"""
        self.state.status = "cleaning_up"
        # Override in subclasses for specific cleanup
        self.state.status = "stopped"

    async def _run_message_loop(self) -> None:
        """Main message processing loop"""
        self.state.status = "running"
        
        while True:
            try:
                # Process incoming messages
                messages = self.message_queue.get_messages(self.config.name)
                for message in messages:
                    await self._process_message(message)
                self.message_queue.clear_messages(self.config.name)
                
                # Update state
                self.state.last_heartbeat = datetime.utcnow()
                
                # Send heartbeat
                await self._send_heartbeat()
                
                # Sleep briefly
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Message loop error: {str(e)}")
                self.state.error_count += 1
                await self._handle_error(str(e))

    async def _process_message(self, message: Message) -> None:
        """Process an incoming message"""
        self.state.processed_messages += 1
        
        try:
            if message.message_type == MessageType.COMMAND:
                await self._handle_command(message)
            elif message.message_type == MessageType.DATA:
                await self._handle_data(message)
            elif message.message_type == MessageType.QUERY:
                await self._handle_query(message)
            elif message.message_type == MessageType.ERROR:
                await self._handle_error_message(message)
            else:
                self.logger.warning(f"Unhandled message type: {message.message_type}")
        
        except Exception as e:
            self.logger.error(f"Message processing error: {str(e)}")
            await self._handle_error(str(e), message)

    async def _handle_command(self, message: Message) -> None:
        """Handle command messages"""
        # Override in subclasses
        raise NotImplementedError

    async def _handle_data(self, message: Message) -> None:
        """Handle data messages"""
        # Override in subclasses
        raise NotImplementedError

    async def _handle_query(self, message: Message) -> None:
        """Handle query messages"""
        # Override in subclasses
        raise NotImplementedError

    async def _handle_error_message(self, message: Message) -> None:
        """Handle error messages"""
        error_info: ErrorInfo = message.payload
        self.logger.error(
            f"Received error from {message.sender}: {error_info.message}"
        )
        self.state.error_count += 1

    async def _handle_error(
        self,
        error_message: str,
        original_message: Optional[Message] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR
    ) -> None:
        """Handle errors during message processing"""
        self.logger.error(f"Error: {error_message}")
        
        # Create error message
        error = create_error(
            sender=self.config.name,
            recipient=original_message.sender if original_message else "coordinator",
            error_message=error_message,
            severity=severity
        )
        
        # Send error message
        await self._send_message(error)

    async def _send_message(self, message: Message) -> None:
        """Send a message to another agent"""
        self.message_queue.add_message(message)

    async def _send_heartbeat(self) -> None:
        """Send heartbeat status update"""
        message = create_status_update(
            sender=self.config.name,
            recipient="coordinator",
            state=self.state.status,
            details={
                "current_tasks": len(self.state.current_tasks),
                "error_count": self.state.error_count,
                "processed_messages": self.state.processed_messages,
                "resource_usage": self.state.resource_usage
            }
        )
        await self._send_message(message)

    async def _check_resources(
        self,
        requirements: ResourceRequirements
    ) -> bool:
        """Check if required resources are available"""
        # Check CPU usage
        if (requirements.cpu_cores and
            self.state.resource_usage.get("cpu", 0) +
            requirements.cpu_cores > self.config.resource_limits.cpu_cores):
            return False
        
        # Check memory usage
        if (requirements.memory_gb and
            self.state.resource_usage.get("memory", 0) +
            requirements.memory_gb > self.config.resource_limits.memory_gb):
            return False
        
        # Check GPU memory
        if (requirements.gpu_memory_gb and
            self.state.resource_usage.get("gpu_memory", 0) +
            requirements.gpu_memory_gb > self.config.resource_limits.gpu_memory_gb):
            return False
        
        return True

    async def _allocate_resources(
        self,
        requirements: ResourceRequirements
    ) -> None:
        """Allocate resources for a task"""
        if requirements.cpu_cores:
            self.state.resource_usage["cpu"] = (
                self.state.resource_usage.get("cpu", 0) +
                requirements.cpu_cores
            )
        
        if requirements.memory_gb:
            self.state.resource_usage["memory"] = (
                self.state.resource_usage.get("memory", 0) +
                requirements.memory_gb
            )
        
        if requirements.gpu_memory_gb:
            self.state.resource_usage["gpu_memory"] = (
                self.state.resource_usage.get("gpu_memory", 0) +
                requirements.gpu_memory_gb
            )

    async def _release_resources(
        self,
        requirements: ResourceRequirements
    ) -> None:
        """Release allocated resources"""
        if requirements.cpu_cores:
            self.state.resource_usage["cpu"] = max(
                0,
                self.state.resource_usage.get("cpu", 0) -
                requirements.cpu_cores
            )
        
        if requirements.memory_gb:
            self.state.resource_usage["memory"] = max(
                0,
                self.state.resource_usage.get("memory", 0) -
                requirements.memory_gb
            )
        
        if requirements.gpu_memory_gb:
            self.state.resource_usage["gpu_memory"] = max(
                0,
                self.state.resource_usage.get("gpu_memory", 0) -
                requirements.gpu_memory_gb
            )

    def _can_handle_task(self, task_type: str) -> bool:
        """Check if agent can handle a specific task type"""
        return task_type in self.config.capabilities

    async def _validate_output(
        self,
        output_path: Path,
        validation_func: Any
    ) -> Tuple[bool, Optional[str]]:
        """Validate task output"""
        try:
            if not output_path.exists():
                return False, "Output file not found"
            
            # Run validation
            is_valid = await validation_func(output_path)
            if not is_valid:
                return False, "Output validation failed"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    async def _retry_task(
        self,
        task_id: UUID,
        max_retries: int = 3,
        delay: float = 5.0
    ) -> bool:
        """Retry a failed task"""
        if task_id not in self.state.current_tasks:
            return False
        
        task = self.state.current_tasks[task_id]
        if task.get("retry_count", 0) >= max_retries:
            return False
        
        # Increment retry count
        task["retry_count"] = task.get("retry_count", 0) + 1
        
        # Wait before retrying
        await asyncio.sleep(delay)
        
        # Retry task
        try:
            await self._handle_command(task["original_message"])
            return True
        except Exception:
            return False