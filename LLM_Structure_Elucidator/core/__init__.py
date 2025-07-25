"""
Core functionality for the LLM Structure Elucidator.
"""

from .app import app
from .socket import socketio, init_socketio
from .agents import agent_coordinator

# Initialize socketio with the app to avoid circular imports
init_socketio(app)

__all__ = ['app', 'socketio', 'agent_coordinator']
