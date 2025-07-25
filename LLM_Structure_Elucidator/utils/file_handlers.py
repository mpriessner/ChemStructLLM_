"""
File handling utilities for the LLM Structure Elucidator.
"""
import os
from datetime import datetime
from config.settings import UPLOAD_FOLDER

def save_uploaded_file(file):
    """Save uploaded file to the uploads directory with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return filepath
