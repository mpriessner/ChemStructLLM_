"""
Main entry point for the LLM Structure Elucidator application.
"""
from core import app, socketio

if __name__ == '__main__':
    print("Starting LLM Structure Elucidator...")
    print("Access the application at: https://localhost:5001")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True, ssl_context='adhoc')
