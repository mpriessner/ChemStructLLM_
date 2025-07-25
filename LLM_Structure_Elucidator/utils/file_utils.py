"""
Utility functions for file handling operations.
"""
import os
from datetime import datetime
from typing import Optional

# Store uploaded SMILES data (global state)
uploaded_smiles = {}

# Track most recently uploaded file
_latest_upload = {
    'filepath': None,
    'timestamp': None,
    'filetype': None
}

def save_uploaded_file(file) -> str:
    """Save uploaded file to the uploads directory with timestamp"""
    from core.app import UPLOAD_FOLDER  # Import here to avoid circular import
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # Update latest upload info
    _latest_upload.update({
        'filepath': filepath,
        'timestamp': datetime.now(),
        'filetype': file.filename.split('.')[-1].lower()
    })
    
    return filepath

def get_latest_upload() -> Optional[dict]:
    """Get information about the most recently uploaded file."""
    return _latest_upload if _latest_upload['filepath'] else None
