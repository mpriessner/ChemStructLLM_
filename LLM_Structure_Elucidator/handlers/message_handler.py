"""
Message handling functionality for Socket.IO events.
"""
from flask import jsonify
from flask_socketio import emit
from core.socket import socketio
import traceback

# Import agent_coordinator lazily to avoid circular import
def get_agent_coordinator():
    from core.agents import agent_coordinator
    return agent_coordinator

@socketio.on('message')
def handle_message(data):
    """Handle incoming messages."""
    print("\n[Message Handler] ====== Starting Message Handling ======")
    print(f"[Message Handler] Received data: {data}")
    
    try:
        # Extract message and model choice
        message = data.get('message', '').lower().strip()
        model_choice = data.get('model', 'claude-3-haiku')
        
        # Get current molecule context
        from handlers.molecule_handler import get_current_molecule
        current_molecule = get_current_molecule()
        context = {
            'current_molecule': current_molecule
        } if current_molecule else {}
        
        print(f"[Message Handler] Processing message: '{message}' with model: {model_choice}")
        print(f"[Message Handler] Context: {context}")
        
        # Process message with AI agent
        coordinator = get_agent_coordinator()
        response = coordinator.process_message(message, model_choice, context=context)
        
        # Handle different response types
        if response.get('type') == 'plot':
            handle_plot_response(response, model_choice)
        else:
            handle_text_response(response, model_choice)
            
        print("[Message Handler] ====== Message Handling Complete ======\n")
            
    except Exception as e:
        print(f"[Message Handler] ERROR: {str(e)}")
        error_msg = f"Error processing message: {str(e)}"
        emit('message', {
            'content': error_msg,
            'type': 'error',
            'model': model_choice
        })

def handle_plot_response(response, model_choice):
    """Handle plot-type responses."""
    print("\n[Message Handler] ====== Processing Plot Response ======")
    
    try:
        # Extract plot details
        plot_type = response.get('plot_type')
        plot_params = response.get('parameters', {})
        print(f"[Message Handler] Plot type: {plot_type}")
        
        # Emit the agent's response first
        print("[Message Handler] Emitting agent response")
        emit('message', {
            'content': f"Generating {plot_type.upper()} plot...",
            'type': 'info',
            'model': model_choice
        })
        
        # Then trigger the plot with parameters
        print("[Message Handler] Triggering plot handler")
        plot_request = {
            'plot_type': plot_type,
            'parameters': plot_params
        }
        emit('plot_request', plot_request)
        print("[Message Handler] ====== Plot Response Processing Complete ======\n")
        
    except Exception as e:
        print(f"[Message Handler] ERROR in handle_plot_response: {str(e)}")
        emit('message', {
            'content': f"Error processing plot request: {str(e)}",
            'type': 'error',
            'model': model_choice
        })

def handle_text_response(response, model_choice):
    """Handle text-type responses."""
    print("[Message Handler] Non-plot response, emitting message")
    emit('message', {
        'content': response.get('content', ''),
        'type': response.get('type', 'text'),
        'model': model_choice
    })


@socketio.on('request_random_molecule')
def get_random_molecule():
    """Generate and return a random molecule for testing."""
    try:
        from utils.visualization import generate_random_molecule, create_molecule_response
        
        # Generate random molecule
        mol, smiles = generate_random_molecule()
        if mol is None or smiles is None:
            return jsonify({
                'error': 'Failed to generate random molecule'
            }), 500
            
        # Create visualization response
        response_2d = create_molecule_response(smiles, is_3d=False)
        if not response_2d:
            return jsonify({
                'error': 'Failed to create molecule visualization'
            }), 500
            
        return jsonify(response_2d)
        
    except Exception as e:
        print(f"Error generating random molecule: {str(e)}")
        return jsonify({
            'error': f'Error: {str(e)}'
        }), 500
