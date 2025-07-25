"""
Chat-related event handlers for Socket.IO.
"""
from flask import session
from flask_socketio import emit
from core.socket import socketio

@socketio.on('clear_chat')
def clear_chat():
    """Clear chat history."""
    try:
        session.clear()
        emit('chat_cleared', {'status': 'success'})
    except Exception as e:
        print(f"Error in clear_chat: {str(e)}")
        emit('error', {'message': 'Failed to clear chat history'})
