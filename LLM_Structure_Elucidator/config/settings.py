"""
Configuration settings for the LLM Structure Elucidator application.
"""
import os
from datetime import datetime
from .config import anthropic_api_key, openai_api_key, elevenlabs_key, gemini_api_key, deepseek_api_key, kimi_api_key

# API Keys
ANTHROPIC_API_KEY = anthropic_api_key
OPENAI_API_KEY = openai_api_key
ELEVENLABS_KEY = elevenlabs_key
GEMINI_API_KEY = gemini_api_key
DEEPSEEK_API_KEY = deepseek_api_key
KIMI_API_KEY = kimi_api_key

# Azure DeepSeek Configuration
DEEPSEEK_AZURE_ENDPOINT = "https://deepseek7114915948.services.ai.azure.com/models"
DEEPSEEK_AZURE_API_KEY = "A3KWGWtHtx6wkhQr9pqGohdcqN4N2nEXE6H7agwliDgIsDkEarZbJQQJ99BBACHYHv6XJ3w3AAAAACOGXUoW"

# Flask Settings
SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')  # Change this in production

# File Upload Settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Sample SMILES strings for random molecule generation
SAMPLE_SMILES = [
    'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
    'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',  # Testosterone
    'CC(C)(C)NCC(O)C1=CC(O)=CC(O)=C1',  # Salbutamol
    'CC1=C(C=C(C=C1)O)C(=O)CC2=CC=C(C=C2)O',  # Benzestrol
]

# # AI Model Settings
# AI_MODELS = {
#     'claude-3-5-haiku': {
#         'provider': 'anthropic',
#         'model': 'claude-3-haiku-20240307',
#         'max_tokens': 1024,
#         'temperature': 0.7,
#     },
#     'claude-3-5-sonnet': {
#         'provider': 'anthropic',
#         'model': 'claude-3-sonnet-20240229',
#         'max_tokens': 1024,
#         'temperature': 0.7,
#     },
#     'gpt-4o': {
#         'provider': 'openai',
#         'model': 'gpt-4o',
#         'max_tokens': 1024,
#         'temperature': 0.7,
#     },
#     'gemini-pro': {
#         'provider': 'google',
#         'model': 'gemini-pro',
#         'max_tokens': 2048,
#         'temperature': 0.7,
#     },
#     'gemini-exp': {
#         'provider': 'google',
#         'model': 'gemini-exp-1114',
#         'max_tokens': 2048,
#         'temperature': 0.7,
#     },
#     'gemini-flash': {
#         'provider': 'google',
#         'model': 'gemini-1.5-flash',
#         'max_tokens': 2048,
#         'temperature': 0.7,
#     }
# }
