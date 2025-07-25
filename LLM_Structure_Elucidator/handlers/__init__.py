"""
Socket.IO event handlers for the LLM Structure Elucidator.
"""

from .message_handler import handle_message
from .plot_handler import handle_plot_request
from .molecule_handler import get_molecule_image
from .audio_handler import handle_audio_request, handle_tts_request
from .connection_handler import handle_connect, handle_disconnect
from .chat_handler import clear_chat

__all__ = [
    'handle_message',
    'handle_plot_request',
    'handle_audio_request',
    'handle_tts_request',
    'handle_connect',
    'handle_disconnect',
    'get_molecule_image',
    'clear_chat'
]
