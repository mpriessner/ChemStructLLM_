"""
Backward compatibility entry point for the LLM Structure Elucidator.
Please use run.py instead.
"""
import warnings

# Import all modules
from core import app, socketio
from services.llm_service import LLMService
from services import services
from agents.coordinator import CoordinatorAgent, AgentType
from handlers import (
    handle_message,
    handle_plot_request,
    handle_molecule_request,
    handle_audio_request,
    handle_tts_request,
    handle_connect,
    handle_disconnect,
    get_molecule_image,
    clear_chat
)
from utils.file_utils import uploaded_smiles, save_uploaded_file
from utils.visualization import create_molecule_response
from agents.specialized.molecule_plot_agent import MoleculePlotAgent
from agents.specialized.nmr_plot_agent import NMRPlotAgent
from agents.specialized.text_response_agent import TextResponseAgent
from routes.main import main
from routes.file_upload import file_upload
from routes.audio import audio
from agents.coordinator import CoordinatorAgent, AgentType

warnings.warn("app.py is deprecated. Please use run.py instead.", DeprecationWarning)

# Register blueprints
app.register_blueprint(main)
app.register_blueprint(file_upload)
app.register_blueprint(audio)

# Initialize services
llm_service = LLMService()

# Initialize the coordinator with agents
coordinator = CoordinatorAgent(llm_service)

# Register specialized agents
molecule_agent = MoleculePlotAgent(llm_service)
nmr_agent = NMRPlotAgent(llm_service)
text_agent = TextResponseAgent(llm_service)

coordinator.add_agent(AgentType.MOLECULE_PLOT, molecule_agent)
coordinator.add_agent(AgentType.NMR_PLOT, nmr_agent)
coordinator.add_agent(AgentType.TEXT_RESPONSE, text_agent)

if __name__ == '__main__':
    print("Warning: app.py is deprecated. Please use run.py instead.")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc')