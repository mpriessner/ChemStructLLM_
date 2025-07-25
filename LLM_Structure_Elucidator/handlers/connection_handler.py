"""
Connection event handlers for Socket.IO.
"""
from flask_socketio import emit
from core.socket import socketio

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')
    emit('connection_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')
    emit('connection_status', {'status': 'disconnected'})
