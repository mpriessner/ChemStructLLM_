"""
Audio handling routes for the LLM Structure Elucidator.
"""
from flask import Blueprint, request, jsonify
import requests
import openai
import base64
import os
from config.settings import ELEVENLABS_KEY, OPENAI_API_KEY

audio = Blueprint('audio', __name__)

# Configure OpenAI
openai.api_key = OPENAI_API_KEY  # In v0.28.0, we set the API key directly on the openai module

@audio.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Handle audio transcription."""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        model_choice = request.form.get('model_choice', 'openai')

        # Save the file temporarily
        temp_path = 'temp_audio.webm'
        audio_file.save(temp_path)

        try:
            # Call Whisper API for transcription using new syntax
            with open(temp_path, 'rb') as audio:
                transcript = openai.Audio.transcriptions.create(
                    model="whisper-1",
                    file=audio
                )

            if transcript and hasattr(transcript, 'text'):
                return jsonify({'text': transcript.text})
            else:
                return jsonify({'error': 'Failed to transcribe audio'}), 500

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"Removed temporary file: {temp_path}")

    except Exception as e:
        print(f"Error in transcribe_audio: {str(e)}")
        return jsonify({'error': str(e)}), 500

@audio.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    """Convert text to speech using ElevenLabs API."""
    try:
        data = request.get_json()
        text = data.get('text')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        CHUNK_SIZE = 1024
        url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_KEY
        }

        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            audio_content = response.content
            audio_base64 = base64.b64encode(audio_content).decode('utf-8')
            return jsonify({'audio': audio_base64})
        else:
            return jsonify({'error': 'Failed to generate speech'}), response.status_code

    except Exception as e:
        print(f"Error in text_to_speech: {str(e)}")
        return jsonify({'error': str(e)}), 500
