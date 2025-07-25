"""
Socket.IO setup and configuration.
"""
from flask_socketio import SocketIO
from .app import app

# Initialize Socket.IO
socketio = SocketIO(app, cors_allowed_origins="*")
