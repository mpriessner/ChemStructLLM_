"""
Socket.IO setup and configuration.
"""
from flask_socketio import SocketIO

# Initialize Socket.IO - app will be set later to avoid circular imports
socketio = SocketIO(cors_allowed_origins="*")

def init_socketio(app):
    """Initialize socketio with the Flask app."""
    socketio.init_app(app)
    return socketio
