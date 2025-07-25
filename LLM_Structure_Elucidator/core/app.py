"""
Flask application setup and configuration.
"""
import os
from flask import Flask, request, jsonify, session, render_template
from dotenv import load_dotenv
from config.settings import SECRET_KEY

# Models and utilities
#from models.molecule import MoleculeHandler
# from models.ai_models import AIModelHandler
from utils.visualization import create_molecule_response, create_plot_response
from utils.file_utils import save_uploaded_file
from utils.nmr_utils import generate_random_2d_correlation_points, generate_nmr_peaks

# Load environment variables
load_dotenv()

# Initialize Flask app with correct template and static paths
template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates')
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')
app = Flask(__name__, 
           template_folder=template_dir,
           static_folder=static_dir)
app.config['SECRET_KEY'] = SECRET_KEY

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize API clients
#ai_handler = AIModelHandler()

# Register blueprints
from routes.main import main
from routes.file_upload import file_upload
from routes.audio import audio

app.register_blueprint(main)
app.register_blueprint(file_upload)
app.register_blueprint(audio)
