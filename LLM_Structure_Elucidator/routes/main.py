"""
Main routes for the LLM Structure Elucidator application.
"""
from flask import Blueprint, request, jsonify, session, render_template, send_from_directory
from services.ai_handler import ai_handler
from core.agents import agent_coordinator
from handlers.molecule_handler import MoleculeHandler
import json
import traceback
import os
import base64

main = Blueprint('main', __name__)
molecule_handler = MoleculeHandler()

@main.route('/', methods=['GET'])
def home():
    """Home page route."""
    if 'conversation' not in session:
        session['conversation'] = []
    return render_template('index.html', conversation=session['conversation'])

@main.route('/chat', methods=['POST'])
async def chat():
    """Chat endpoint for handling user messages."""
    print("\n[Chat Route] ====== Starting Chat Request ======")
    try:
        # Get user input and model choice - handle both form and JSON data
        if request.is_json:
            data = request.get_json()
            user_input = data.get('user_input', '').strip()
            model_choice = data.get('model_choice', 'claude-3-5-haiku')
        else:
            user_input = request.form.get('user_input', '').strip()
            model_choice = request.form.get('model_choice', 'claude-3-5-haiku')
            
        print(f"[Chat Route] Received input: '{user_input}'")
        print(f"[Chat Route] Model choice: {model_choice}")
        
        # Initialize session if needed
        if 'conversation' not in session:
            print("[Chat Route] Initializing conversation session")
            session['conversation'] = []
            
        # Process message with agent coordinator
        response = await agent_coordinator.process_message(user_input, model_choice)
        # print(f"[Chat Route] print out the response of this agent from just before. {response}")

        # Update conversation history based on response type
        session['conversation'].append(('user', user_input))
        
        if response['type'] == 'clarification':
            # Handle nested content structure
            clarification_message = response['content']['content'] if isinstance(response['content'], dict) else response['content']
            if 'metadata' in response and 'reasoning' in response['metadata']:
                clarification_message += f"\n\nReasoning: {response['metadata']['reasoning']}"
            session['conversation'].append(('bot', clarification_message))
            
        elif response['type'] == 'error':
            # For errors, show both error message and reasoning
            error_message = f"Error: {response['content']}"
            if 'metadata' in response and 'reasoning' in response['metadata']:
                error_message += f"\n\nReasoning: {response['metadata']['reasoning']}"
            print(f"[main.py] Error encountered: {error_message}")
        
            session['conversation'].append(('bot', error_message))
            
        elif response['type'] == 'text':
            # For text responses, show content and reasoning if confidence is low
            text_message = response['content']
            if 'metadata' in response and response['metadata'].get('confidence', 1.0) < 0.7:
                text_message += f"\n\nReasoning: {response['metadata'].get('reasoning', 'No reasoning provided')}"
            session['conversation'].append(('bot', text_message))

        elif response['type'] == 'molecule_plot':
            # For molecule plot responses, show a message and pass through the structured data
            message = "Generating molecule visualization..."
            if 'metadata' in response and response['metadata'].get('confidence', 1.0) < 0.7:
                message += f"\n\nNote: {response['metadata'].get('reasoning', 'No reasoning provided')}"
            session['conversation'].append(('bot', message))
            
            # The frontend expects the molecule data in a specific structure
            # The coordinator has already formatted this correctly, so we just pass it through
            # print(f"[Chat Route] Sending molecule plot data to frontend: {response}")
            
        else:
            # For other types (like plot), show a generic message with reasoning if confidence is low
            message = f"Generating {response['type']} visualization..."
            if 'metadata' in response and response['metadata'].get('confidence', 1.0) < 0.7:
                message += f"\n\nNote: {response['metadata'].get('reasoning', 'No reasoning provided')}"
            session['conversation'].append(('bot', message))
            
        session.modified = True
        
        print("[Chat Route] ====== Chat Request Complete ======\n")
        #print(f"[Chat Route] Response from agent: {response}")

        return jsonify(response)
        
    except Exception as e:
        print(f"[Chat Route] ERROR: {str(e)}")
        import traceback
        print(f"[Chat Route] Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500



@main.route('/clear_chat', methods=['POST'])
def clear_chat():
    """Clear the chat history."""
    if 'conversation' in session:
        session['conversation'] = []
        session.modified = True
    return jsonify({'status': 'success'})

@main.route('/generate_molecule_image', methods=['POST'])
def generate_molecule_image():
    """Generate and return a molecule visualization response."""
    print("[Route] Received molecule image generation request")
    try:
        # Get SMILES from request
        data = request.get_json()
        print(f"[Route] Request data: {data}")
        
        smiles = data.get('smiles', 'CC(=O)O')  # Default to acetic acid if no SMILES provided
        print(f"[Route] Using SMILES: {smiles}")
        
        # Generate the response with image and metadata
        response = molecule_handler.generate_molecule_response(smiles)
        print(f"[Route] Generated response: {response is not None}")
        
        if response is None:
            error_msg = f"Failed to generate molecule response for SMILES: {smiles}"
            print(f"[Route] Error: {error_msg}")
            return jsonify({'error': error_msg}), 500
            
        print("[Route] Successfully generated molecule response")
        return jsonify(response)
    
    except Exception as e:
        error_msg = f"Error generating molecule response: {str(e)}"
        print(f"[Route] Exception: {error_msg}")
        print(f"[Route] Exception type: {type(e)}")
        return jsonify({'error': error_msg}), 500

@main.route('/test_data/<path:filename>')
def serve_test_data(filename):
    """Serve files from the test_data directory."""
    test_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_data')
    return send_from_directory(test_data_dir, filename)

@main.route('/nmr_images/<filename>')
def serve_nmr_image(filename):
    """Serve NMR image files as base64-encoded strings."""
    try:
        # Define the NMR images directory path
        nmr_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '_temp_folder')
        
        # Construct the full file path
        file_path = os.path.join(nmr_dir, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({'error': 'Image not found'}), 404
            
        # Read and encode the image
        with open(file_path, 'rb') as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
        return jsonify({
            'image': img_base64,
            'filename': filename,
            'type': 'base64'
        })
        
    except Exception as e:
        print(f"Error serving NMR image: {str(e)}")
        return jsonify({'error': str(e)}), 500
