"""
WebSocket manager for real-time workflow progress updates
Implements WebSocket connections for TTD-DR Framework API
"""

import json
import asyncio
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from .models import (
    WebSocketMessage, WebSocketMessageType, StatusUpdateMessage,
    NodeCompletedMessage, QualityUpdateMessage, WorkflowStatus
)
from .auth import verify_token, TokenData

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for workflow updates"""
    
    def __init__(self):
        # Active connections: execution_id -> set of websockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # User connections: user_id -> set of websockets
        self.user_connections: Dict[str, Set[WebSocket]] = {}
        # Connection metadata: websocket -> connection info
        self.connection_metadata: Dict[WebSocket, Dict] = {}
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, execution_id: str, 
                     user_id: Optional[str] = None) -> bool:
        """
        Accept WebSocket connection and register it
        
        Args:
            websocket: WebSocket connection
            execution_id: Workflow execution ID to monitor
            user_id: Optional authenticated user ID
            
        Returns:
            True if connection was accepted, False otherwise
        """
        try:
            await websocket.accept()
            
            async with self._lock:
                # Register connection for execution
                if execution_id not in self.active_connections:
                    self.active_connections[execution_id] = set()
                self.active_connections[execution_id].add(websocket)
                
                # Register connection for user if authenticated
                if user_id:
                    if user_id not in self.user_connections:
                        self.user_connections[user_id] = set()
                    self.user_connections[user_id].add(websocket)
                
                # Store connection metadata
                self.connection_metadata[websocket] = {
                    "execution_id": execution_id,
                    "user_id": user_id,
                    "connected_at": datetime.now(),
                    "last_ping": datetime.now()
                }
            
            logger.info(f"WebSocket connected for execution {execution_id}, user {user_id}")
            
            # Send initial connection confirmation
            await self.send_to_connection(websocket, {
                "type": "connection_established",
                "execution_id": execution_id,
                "timestamp": datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection: {e}")
            return False
    
    async def disconnect(self, websocket: WebSocket):
        """
        Disconnect and cleanup WebSocket connection
        
        Args:
            websocket: WebSocket connection to disconnect
        """
        async with self._lock:
            # Get connection metadata
            metadata = self.connection_metadata.get(websocket, {})
            execution_id = metadata.get("execution_id")
            user_id = metadata.get("user_id")
            
            # Remove from execution connections
            if execution_id and execution_id in self.active_connections:
                self.active_connections[execution_id].discard(websocket)
                if not self.active_connections[execution_id]:
                    del self.active_connections[execution_id]
            
            # Remove from user connections
            if user_id and user_id in self.user_connections:
                self.user_connections[user_id].discard(websocket)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            
            # Remove metadata
            if websocket in self.connection_metadata:
                del self.connection_metadata[websocket]
        
        logger.info(f"WebSocket disconnected for execution {execution_id}, user {user_id}")
    
    async def send_to_connection(self, websocket: WebSocket, message: Dict):
        """
        Send message to specific WebSocket connection
        
        Args:
            websocket: Target WebSocket connection
            message: Message to send
        """
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"Failed to send message to WebSocket: {e}")
            await self.disconnect(websocket)
    
    async def broadcast_to_execution(self, execution_id: str, message: WebSocketMessage):
        """
        Broadcast message to all connections monitoring an execution
        
        Args:
            execution_id: Execution ID to broadcast to
            message: Message to broadcast
        """
        if execution_id not in self.active_connections:
            logger.debug(f"No active connections for execution {execution_id}")
            return
        
        # Convert message to dict
        message_dict = message.dict()
        
        # Get connections to broadcast to
        connections = self.active_connections[execution_id].copy()
        
        # Send to all connections
        disconnected_connections = []
        for websocket in connections:
            try:
                await websocket.send_text(json.dumps(message_dict, default=str))
            except Exception as e:
                logger.error(f"Failed to broadcast to WebSocket: {e}")
                disconnected_connections.append(websocket)
        
        # Clean up disconnected connections
        for websocket in disconnected_connections:
            await self.disconnect(websocket)
        
        logger.debug(f"Broadcasted message to {len(connections) - len(disconnected_connections)} connections")
    
    async def broadcast_to_user(self, user_id: str, message: WebSocketMessage):
        """
        Broadcast message to all connections for a specific user
        
        Args:
            user_id: User ID to broadcast to
            message: Message to broadcast
        """
        if user_id not in self.user_connections:
            logger.debug(f"No active connections for user {user_id}")
            return
        
        # Convert message to dict
        message_dict = message.dict()
        
        # Get connections to broadcast to
        connections = self.user_connections[user_id].copy()
        
        # Send to all connections
        disconnected_connections = []
        for websocket in connections:
            try:
                await websocket.send_text(json.dumps(message_dict, default=str))
            except Exception as e:
                logger.error(f"Failed to broadcast to WebSocket: {e}")
                disconnected_connections.append(websocket)
        
        # Clean up disconnected connections
        for websocket in disconnected_connections:
            await self.disconnect(websocket)
    
    async def send_status_update(self, execution_id: str, status: WorkflowStatus,
                               current_node: Optional[str] = None, progress: float = 0.0,
                               **kwargs):
        """
        Send status update to all connections monitoring an execution
        
        Args:
            execution_id: Execution ID
            status: Current workflow status
            current_node: Currently executing node
            progress: Progress percentage (0-100)
            **kwargs: Additional data to include
        """
        message = StatusUpdateMessage(
            execution_id=execution_id,
            status=status,
            current_node=current_node,
            progress=progress,
            **kwargs
        )
        await self.broadcast_to_execution(execution_id, message)
    
    async def send_node_completed(self, execution_id: str, node_name: str,
                                duration: float, **kwargs):
        """
        Send node completion notification
        
        Args:
            execution_id: Execution ID
            node_name: Name of completed node
            duration: Node execution duration in seconds
            **kwargs: Additional data to include
        """
        message = NodeCompletedMessage(
            execution_id=execution_id,
            node_name=node_name,
            duration=duration,
            **kwargs
        )
        await self.broadcast_to_execution(execution_id, message)
    
    async def send_quality_update(self, execution_id: str, quality_metrics, **kwargs):
        """
        Send quality metrics update
        
        Args:
            execution_id: Execution ID
            quality_metrics: QualityMetrics object
            **kwargs: Additional data to include
        """
        message = QualityUpdateMessage(
            execution_id=execution_id,
            quality_metrics=quality_metrics,
            **kwargs
        )
        await self.broadcast_to_execution(execution_id, message)
    
    async def send_error(self, execution_id: str, error_message: str, **kwargs):
        """
        Send error notification
        
        Args:
            execution_id: Execution ID
            error_message: Error message
            **kwargs: Additional data to include
        """
        message = WebSocketMessage(
            type=WebSocketMessageType.ERROR,
            execution_id=execution_id,
            data={
                "error": error_message,
                **kwargs
            }
        )
        await self.broadcast_to_execution(execution_id, message)
    
    async def send_workflow_completed(self, execution_id: str, final_report: Optional[str] = None,
                                    quality_score: Optional[float] = None, **kwargs):
        """
        Send workflow completion notification
        
        Args:
            execution_id: Execution ID
            final_report: Final research report
            quality_score: Final quality score
            **kwargs: Additional data to include
        """
        message = WebSocketMessage(
            type=WebSocketMessageType.WORKFLOW_COMPLETED,
            execution_id=execution_id,
            data={
                "final_report": final_report,
                "quality_score": quality_score,
                **kwargs
            }
        )
        await self.broadcast_to_execution(execution_id, message)
    
    async def ping_connections(self):
        """Send ping to all connections to keep them alive"""
        current_time = datetime.now()
        
        for websocket, metadata in self.connection_metadata.items():
            try:
                await websocket.ping()
                metadata["last_ping"] = current_time
            except Exception as e:
                logger.error(f"Failed to ping WebSocket: {e}")
                await self.disconnect(websocket)
    
    def get_connection_stats(self) -> Dict:
        """Get statistics about active connections"""
        return {
            "total_connections": len(self.connection_metadata),
            "executions_monitored": len(self.active_connections),
            "authenticated_users": len(self.user_connections),
            "connections_per_execution": {
                exec_id: len(connections) 
                for exec_id, connections in self.active_connections.items()
            }
        }

# Global connection manager instance
connection_manager = ConnectionManager()

async def authenticate_websocket(websocket: WebSocket, token: Optional[str]) -> Optional[TokenData]:
    """
    Authenticate WebSocket connection using JWT token
    
    Args:
        websocket: WebSocket connection
        token: JWT token from query parameters
        
    Returns:
        TokenData if authentication successful, None otherwise
    """
    if not token:
        return None
    
    try:
        token_data = verify_token(token)
        if token_data is None:
            await websocket.close(code=4001, reason="Invalid token")
            return None
        
        # Check if token is expired
        if datetime.utcnow() > token_data.exp:
            await websocket.close(code=4001, reason="Token expired")
            return None
        
        return token_data
        
    except Exception as e:
        logger.error(f"WebSocket authentication error: {e}")
        await websocket.close(code=4001, reason="Authentication failed")
        return None

async def handle_websocket_messages(websocket: WebSocket, execution_id: str):
    """
    Handle incoming WebSocket messages
    
    Args:
        websocket: WebSocket connection
        execution_id: Execution ID being monitored
    """
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                # Handle different message types
                if message_type == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
                
                elif message_type == "subscribe":
                    # Handle subscription to additional executions
                    additional_execution_id = message.get("execution_id")
                    if additional_execution_id:
                        async with connection_manager._lock:
                            if additional_execution_id not in connection_manager.active_connections:
                                connection_manager.active_connections[additional_execution_id] = set()
                            connection_manager.active_connections[additional_execution_id].add(websocket)
                
                elif message_type == "unsubscribe":
                    # Handle unsubscription from executions
                    unsubscribe_execution_id = message.get("execution_id")
                    if unsubscribe_execution_id:
                        async with connection_manager._lock:
                            if unsubscribe_execution_id in connection_manager.active_connections:
                                connection_manager.active_connections[unsubscribe_execution_id].discard(websocket)
                
                else:
                    logger.warning(f"Unknown WebSocket message type: {message_type}")
                    
            except json.JSONDecodeError:
                logger.error("Invalid JSON received from WebSocket")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Message processing error"
                }))
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for execution {execution_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await connection_manager.disconnect(websocket)

# Background task to ping connections periodically
async def ping_task():
    """Background task to ping WebSocket connections"""
    while True:
        try:
            await connection_manager.ping_connections()
            await asyncio.sleep(30)  # Ping every 30 seconds
        except Exception as e:
            logger.error(f"Error in ping task: {e}")
            await asyncio.sleep(60)  # Wait longer on error