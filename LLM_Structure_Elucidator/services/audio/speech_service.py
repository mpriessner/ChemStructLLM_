"""
Speech service for text-to-speech and speech-to-text conversions.
"""
from typing import Optional, Dict, Any, BinaryIO
import requests
import json
import base64
from pathlib import Path
import tempfile
import os

class SpeechService:
    """Service for handling speech-related operations."""
    
    def __init__(self, elevenlabs_key: str, openai_key: str):
        """Initialize the speech service."""
        self.elevenlabs_key = elevenlabs_key
        self.openai_key = openai_key
        self.elevenlabs_voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default voice
        self.openai_model = "whisper-1"  # Default model
    
    async def text_to_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """Convert text to speech using ElevenLabs API."""
        try:
            voice_id = voice_id or self.elevenlabs_voice_id
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            
            headers = {
                "xi-api-key": self.elevenlabs_key,
                "Content-Type": "application/json"
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
            response.raise_for_status()
            
            # If output path is not provided, create a temporary file
            if not output_path:
                temp_dir = Path(tempfile.gettempdir())
                output_path = str(temp_dir / f"speech_{hash(text)}.mp3")
            
            # Save the audio file
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            return output_path
            
        except Exception as e:
            print(f"Error in text_to_speech: {str(e)}")
            return None
    
    async def speech_to_text(
        self,
        audio_file: BinaryIO,
        language: str = "en",
        prompt: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Convert speech to text using OpenAI's Whisper API."""
        try:
            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {
                "Authorization": f"Bearer {self.openai_key}"
            }
            
            data = {
                "model": self.openai_model,
                "language": language
            }
            
            if prompt:
                data["prompt"] = prompt
            
            files = {
                "file": audio_file
            }
            
            response = requests.post(url, headers=headers, data=data, files=files)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            print(f"Error in speech_to_text: {str(e)}")
            return None
    
    def set_voice(self, voice_id: str) -> None:
        """Set the ElevenLabs voice ID."""
        self.elevenlabs_voice_id = voice_id
    
    def set_model(self, model: str) -> None:
        """Set the OpenAI Whisper model."""
        self.openai_model = model
