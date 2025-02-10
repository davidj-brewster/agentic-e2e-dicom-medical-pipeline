"""
Core message protocol for inter-agent communication.
Defines message types, formats, and validation using pydantic models.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of messages that can be exchanged between agents"""
    COMMAND = "command"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    DATA = "data"
    RESULT = "result"
    QUERY = "query"
    RESPONSE = "response"


class Priority(Enum):
    """Message priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class ErrorSeverity(Enum):
    """Error severity levels for error messages"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorInfo(BaseModel):
    """Detailed error information"""
    message: str
    severity: ErrorSeverity
    details: Optional[Dict[str, Any]] = None
    traceback: Optional[str] = None
    recovery_suggestion: Optional[str] = None


class StatusInfo(BaseModel):
    """Status update information"""
    state: str
    progress: Optional[float] = None
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DataPayload(BaseModel):
    """Data message payload"""
    data_type: str
    content: Any
    metadata: Optional[Dict[str, Any]] = None


class CommandPayload(BaseModel):
    """Command message payload"""
    command: str
    parameters: Dict[str, Any]
    timeout: Optional[float] = None
    retry_count: Optional[int] = None


class QueryPayload(BaseModel):
    """Query message payload"""
    query_type: str
    parameters: Dict[str, Any]
    response_format: Optional[str] = None


class ResponsePayload(BaseModel):
    """Response message payload"""
    query_id: UUID
    content: Any
    metadata: Optional[Dict[str, Any]] = None


class ResultPayload(BaseModel):
    """Result message payload"""
    result_type: str
    content: Any
    metrics: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class Message(BaseModel):
    """Base message model for all inter-agent communication"""
    message_id: UUID = Field(default_factory=uuid4)
    correlation_id: Optional[UUID] = None
    sender: str
    recipient: str
    message_type: MessageType
    priority: Priority = Priority.NORMAL
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: Union[
        CommandPayload,
        DataPayload,
        ErrorInfo,
        QueryPayload,
        ResponsePayload,
        ResultPayload,
        StatusInfo
    ]
    metadata: Optional[Dict[str, Any]] = None


class MessageBatch(BaseModel):
    """Batch of messages for efficient transmission"""
    batch_id: UUID = Field(default_factory=uuid4)
    messages: List[Message]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MessageQueue:
    """Simple message queue implementation"""
    
    def __init__(self):
        self.queues: Dict[str, List[Message]] = {}
    
    def add_message(self, message: Message) -> None:
        """Add a message to the recipient's queue"""
        if message.recipient not in self.queues:
            self.queues[message.recipient] = []
        self.queues[message.recipient].append(message)
    
    def get_messages(self, recipient: str) -> List[Message]:
        """Get all messages for a recipient"""
        return self.queues.get(recipient, [])
    
    def clear_messages(self, recipient: str) -> None:
        """Clear all messages for a recipient"""
        self.queues[recipient] = []


def create_command(
    sender: str,
    recipient: str,
    command: str,
    parameters: Dict[str, Any],
    priority: Priority = Priority.NORMAL
) -> Message:
    """Create a command message"""
    return Message(
        sender=sender,
        recipient=recipient,
        message_type=MessageType.COMMAND,
        priority=priority,
        payload=CommandPayload(
            command=command,
            parameters=parameters
        )
    )


def create_status_update(
    sender: str,
    recipient: str,
    state: str,
    progress: Optional[float] = None,
    message: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> Message:
    """Create a status update message"""
    return Message(
        sender=sender,
        recipient=recipient,
        message_type=MessageType.STATUS_UPDATE,
        payload=StatusInfo(
            state=state,
            progress=progress,
            message=message,
            details=details
        )
    )


def create_error(
    sender: str,
    recipient: str,
    error_message: str,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    details: Optional[Dict[str, Any]] = None,
    traceback: Optional[str] = None,
    recovery_suggestion: Optional[str] = None
) -> Message:
    """Create an error message"""
    logger.info(f"Creating error message: {error_message} to {recipient} ({severity}) with details {''.join(details)}.\nRecovery suggestion is {recovery_suggestion}")
    return Message(
        sender=sender,
        recipient=recipient,
        message_type=MessageType.ERROR,
        priority=Priority.HIGH,
        payload=ErrorInfo(
            message=error_message,
            severity=severity,
            details=details,
            traceback=traceback,
            recovery_suggestion=recovery_suggestion
        )
    )


def create_data_message(
    sender: str,
    recipient: str,
    data_type: str,
    content: Any,
    metadata: Optional[Dict[str, Any]] = None
) -> Message:
    """Create a data message"""
    return Message(
        sender=sender,
        recipient=recipient,
        message_type=MessageType.DATA,
        payload=DataPayload(
            data_type=data_type,
            content=content,
            metadata=metadata
        )
    )


def create_result(
    sender: str,
    recipient: str,
    result_type: str,
    content: Any,
    metrics: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Message:
    """Create a result message"""
    return Message(
        sender=sender,
        recipient=recipient,
        message_type=MessageType.RESULT,
        payload=ResultPayload(
            result_type=result_type,
            content=content,
            metrics=metrics,
            metadata=metadata
        )
    )


def create_query(
    sender: str,
    recipient: str,
    query_type: str,
    parameters: Dict[str, Any],
    response_format: Optional[str] = None
) -> Message:
    """Create a query message"""
    return Message(
        sender=sender,
        recipient=recipient,
        message_type=MessageType.QUERY,
        payload=QueryPayload(
            query_type=query_type,
            parameters=parameters,
            response_format=response_format
        )
    )


def create_response(
    sender: str,
    recipient: str,
    query_id: UUID,
    content: Any,
    metadata: Optional[Dict[str, Any]] = None
) -> Message:
    """Create a response message"""
    return Message(
        sender=sender,
        recipient=recipient,
        message_type=MessageType.RESPONSE,
        correlation_id=query_id,
        payload=ResponsePayload(
            query_id=query_id,
            content=content,
            metadata=metadata
        )
    )