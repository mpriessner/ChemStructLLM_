"""
Routes for handling molecular structure functionality.
"""
import os
from flask import Blueprint, send_from_directory, current_app

structure = Blueprint('structure', __name__)

@structure.route('/static/images/<path:filename>')
def serve_image(filename):
    """Serve static images from the static/images directory."""
    return send_from_directory(
        os.path.join(current_app.static_folder, 'images'),
        filename
    )
