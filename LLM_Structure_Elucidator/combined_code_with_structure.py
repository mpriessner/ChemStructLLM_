Repository Structure:
LLM_Structure_Elucidator/
    LICENSE
    get_library_docs.py
    requirements.txt
    run.py
    .gitignore
    README.md
    app.py
    sample_8285_13C.npz
    combined_code_with_structure.py
    tests/
        NEEDS_IMPLEMENTATION
        test_peak_matching_v2.py
        test_molecular_visual_comparison.py
        test_peak_matching_v3.py
        conftest.py
        test_api_integration.py
        test_retrosynthesis.py
        __pycache__/
            test_molecular_visual_comparison.cpython-312.pyc
            test_molecular_visual_comparison.cpython-37.pyc
        integration/
            test_api.py
        unit/
            test_analysis.py
            test_molecule.py
        fixtures/
            molecule_data.py
    results/
    handlers/
        molecule_handler.py
        audio_handler.py
        message_handler.py
        plot_handler.py
        connection_handler.py
        chat_handler.py
        __init__.py
        __pycache__/
            molecule_handler.cpython-312.pyc
            message_handler.cpython-312.pyc
            chat_handler.cpython-312.pyc
            connection_handler.cpython-312.pyc
            plot_handler.cpython-312.pyc
            audio_handler.cpython-312.pyc
            __init__.cpython-312.pyc
    monitoring/
        __init__.py
        logging/
            logger.py
        metrics/
            performance_metrics.py
        alerts/
            alert_manager.py
    templates/
        index.html
    static/
        test_data/
            test_smiles.csv
        js/
            main.js
            modules/
                visualization.js
                chat.js
                tabs.js
                audio.js
                resizer.js
                analysis/
                    moleculeVisualization.js
                    analysisState.js
                    plotVisualization.js
                common/
                    visualization.js
                    state.js
                structure/
                    structureTabInit.js
                    structureViewZoom.js
                    index.js
                    molecule2D.js
                    structureView2D.js
        css/
            styles.css
            structureView.css
            base.css
            layout.css
            visualization.css
            tabs.css
            components.css
            chat.css
        images/
            molecule_comparison_Dummy.png
            molecule_comparison.png
    models/
        molecule.py
        __init__.py
        __pycache__/
            __init__.cpython-312.pyc
            __init__.cpython-37.pyc
            molecule.cpython-312.pyc
    data/
        molecular_data/
            molecular_data_.json
            molecular_data.json
    agents/
        __init__.py
        tools/
            analysis_tools__.py
            stout_tool.py
            mol2mol_tool.py
            nmr_simulation_tool.py
            structure_visualization_tool.py
            mmst_tool.py
            candidate_ranking_tool.py
            ___molecular_visual_comparison_tool.py
            threshold_calculation_tool.py
            spectral_comparison_tool.py
            candidate_analyzer_tool.py
            stout_operations.py
            retro_synthesis_tool.py
            analysis_enums.py
            peak_matching_tool.py
            __init__.py
            forward_synthesis_tool.py
            data_extraction_tool.py
            final_analysis_tool.py
            __pycache__/
                structure_visualization_tool.cpython-312.pyc
                retro_synthesis_tool.cpython-312.pyc
                __init__.cpython-312.pyc
                nmr_simulation_tool.cpython-312.pyc
                stout_operations.cpython-312.pyc
                spectral_comparison_tool.cpython-312.pyc
                enhanced_peak_matching_tool_v3.cpython-312.pyc
                candidate_analyzer_tool.cpython-312.pyc
                final_analysis_tool.cpython-312.pyc
                mmst_tool.cpython-312.pyc
                molecular_visual_comparison_tool.cpython-312.pyc
                candidate_ranking_tool.cpython-312.pyc
                mol2mol_tool.cpython-312.pyc
                forward_synthesis_tool.cpython-312.pyc
                peak_matching_tool.cpython-312.pyc
                enhanced_peak_matching_tool_v2.cpython-312.pyc
                enhanced_peak_matching_tool.cpython-37.pyc
                retrosynthesis_tool.cpython-312.pyc
                threshold_calculation_tool.cpython-312.pyc
                data_extraction_tool.cpython-312.pyc
                stout_tool.cpython-312.pyc
                enhanced_peak_matching_tool_v3.cpython-37.pyc
                enhanced_peak_matching_tool_v2.cpython-37.pyc
                analysis_enums.cpython-312.pyc
        __pycache__/
            __init__.cpython-312.pyc
            __init__.cpython-37.pyc
            coordinator.cpython-312.pyc
        memory/
            conversation.py
            knowledge_base.py
            __init__.py
        _agents_descriptions_/
            Instructions_for_analysis.txt
            MMST_imports.py
            orchestration_agent.txt
            AI_conversation_context.txt
            IC_MMST.py
        orchestrator/
            __init__.py
            workflow_definitions.py
            orchestrator_backup.py
            orchestrator.py
            __pycache__/
                orchestrator.cpython-312.pyc
                __init__.cpython-312.pyc
                workflow_definitions.cpython-312.pyc
                command_generator.cpython-312.pyc
            .ipynb_checkpoints/
                workflow_definitions-checkpoint.py
        coordinator/
            coordinator.py
            __init__.py
            __pycache__/
                __init__.cpython-312.pyc
                coordinator.cpython-312.pyc
        scripts/
            sgnn_sbatch.sh
            chemformer_forward_script.py
            mmst_script.py
            sgnn_local.sh
            peak_matching_script.py
            mmst_local.sh
            peak_matching_test.sh
            chemformer_retro_sbatch.sh
            sgnn_script.py
            stout_local.sh
            mol2mol_sbatch.sh
            peak_matching_local.sh
            mol2mol_script.py
            mmst_sbatch.sh
            chemformer_forward_sbatch.sh
            stout_script.py
            mol2mol_local.sh
            imports_MMST.py
            chemformer_retro_script.py
            chemformer_retro_local.sh
            chemformer_forward_local.sh
            logs/
                .gitkeep
                retro_sbatch.txt
            _working_scripts_backup/
                chemformer_retro_local.sh
                chemformer_forward_script.py
                chemformer_retro_script.py
                chemformer_retro_sbatch.sh
                Mol2Mol_script.py
                sgnn_sbatch.sh
                chemformer_forward_sbatch.sh
                mol2mol_local.sh
                sgnn_local.sh
                SGNN_script.py
                mol2mol_sbatch.sh
                chemformer_forward_local.sh
            __pycache__/
                imports_MMST.cpython-37.pyc
        specialized/
            nmr_plot_agent.py
            text_response_agent.py
            analysis_agent.py
            tool_agent.py
            __init__.py
            script_modifier_agent.py
            molecule_plot_agent.py
            __pycache__/
                __init__.cpython-37.pyc
                analysis_agent.cpython-312.pyc
                nmr_plot_agent.cpython-312.pyc
                molecule_plot_agent.cpython-312.pyc
                __init__.cpython-312.pyc
                tool_agent.cpython-312.pyc
                molecule_plot_agent.cpython-37.pyc
                text_response_agent.cpython-312.pyc
            config/
                tool_descriptions.py
                __init__.py
                __pycache__/
                    tool_descriptions.cpython-312.pyc
                    __init__.cpython-312.pyc
        base/
            base_agent.py
            __init__.py
            __pycache__/
                base_agent.cpython-37.pyc
                __init__.cpython-37.pyc
                base_agent.cpython-312.pyc
                __init__.cpython-312.pyc
    core/
        agents.py
        app.py
        __init__.py
        socket.py
        __pycache__/
            __init__.cpython-37.pyc
            agents.cpython-312.pyc
            __init__.cpython-312.pyc
            socket.cpython-312.pyc
            app.cpython-37.pyc
            app.cpython-312.pyc
    services/
        __init__.py
        llm_service.py
        ai_handler.py
        storage/
            vector_store.py
        __pycache__/
            llm_service.cpython-37.pyc
            __init__.cpython-312.pyc
            event_service.cpython-312.pyc
            ai_handler.cpython-312.pyc
            llm_service.cpython-312.pyc
            __init__.cpython-37.pyc
        audio/
            speech_service.py
    experiments/
    utils/
        sbatch_utils.py
        nmr_utils.py
        visualization.py
        __init__.py
        results_manager.py
        file_utils.py
        file_handlers.py
        __pycache__/
            sbatch_utils.cpython-312.pyc
            nmr_utils.cpython-312.pyc
            file_utils.cpython-312.pyc
            __init__.cpython-312.pyc
            visualization.cpython-312.pyc
    __pycache__/
        app.cpython-312.pyc
    test_data/
        test_smiles_with_nmr copy 2.csv
        original.txt
        test_smiles_with_nmr_.csv
        test_file_pubchem_id.txt
        test_smiles_with_nmr--big.csv
        test_smiles_with_nmr copy.csv
        test_smiles.csv
        test_smiles_with_nmr.csv
    routes/
        main.py
        audio.py
        file_upload.py
        __init__.py
        structure.py
        data_routes.py
        __pycache__/
            file_upload.cpython-312.pyc
            audio.cpython-312.pyc
            __init__.cpython-312.pyc
            main.cpython-312.pyc
    config/
        __init__.py
        settings.py
        config.py
        config.template.py
        __pycache__/
            settings.cpython-312.pyc
            config.cpython-312.pyc
            __init__.cpython-312.pyc

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/get_library_docs.py ---
import importlib
import sys
from io import StringIO

def generate_library_docs(library_name):
    """Generate documentation for a library and save it to a file."""
    try:
        # Import the library
        library = importlib.import_module(library_name)
        
        # Redirect stdout to capture help output
        old_stdout = sys.stdout
        result = StringIO()
        sys.stdout = result
        
        # Get help information
        help(library)
        
        # Restore stdout
        sys.stdout = old_stdout
        
        # Save to file
        with open(f'{library_name}_documentation.txt', 'w', encoding='utf-8') as f:
            f.write(result.getvalue())
            
        print(f"Documentation for {library_name} has been saved to {library_name}_documentation.txt")
        
    except ImportError:
        print(f"Could not import {library_name}. Make sure it's installed.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        library_name = sys.argv[1]
        generate_library_docs(library_name)
    else:
        print("Please provide a library name as argument")


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/run.py ---
"""
Main entry point for the LLM Structure Elucidator application.
"""
from core import app, socketio

if __name__ == '__main__':
    print("Starting LLM Structure Elucidator...")
    print("Access the application at: https://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc')


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/app.py ---
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

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/combined_code_with_structure.py ---
Repository Structure:
LLM_Structure_Elucidator/
    LICENSE
    get_library_docs.py
    requirements.txt
    run.py
    .gitignore
    README.md
    app.py
    sample_8285_13C.npz
    combined_code_with_structure.py
    tests/
        NEEDS_IMPLEMENTATION
        test_peak_matching_v2.py
        test_molecular_visual_comparison.py
        test_peak_matching_v3.py
        conftest.py
        test_api_integration.py
        test_retrosynthesis.py
        __pycache__/
            test_molecular_visual_comparison.cpython-312.pyc
            test_molecular_visual_comparison.cpython-37.pyc
        integration/
            test_api.py
        unit/
            test_analysis.py
            test_molecule.py
        fixtures/
            molecule_data.py
    results/
    handlers/
        molecule_handler.py
        audio_handler.py
        message_handler.py
        plot_handler.py
        connection_handler.py
        chat_handler.py
        __init__.py
        __pycache__/
            molecule_handler.cpython-312.pyc
            message_handler.cpython-312.pyc
            chat_handler.cpython-312.pyc
            connection_handler.cpython-312.pyc
            plot_handler.cpython-312.pyc
            audio_handler.cpython-312.pyc
            __init__.cpython-312.pyc
    monitoring/
        __init__.py
        logging/
            logger.py
        metrics/
            performance_metrics.py
        alerts/
            alert_manager.py
    templates/
        index.html
    static/
        test_data/
            test_smiles.csv
        js/
            main.js
            modules/
                visualization.js
                chat.js
                tabs.js
                audio.js
                resizer.js
                analysis/
                    moleculeVisualization.js
                    analysisState.js
                    plotVisualization.js
                common/
                    visualization.js
                    state.js
                structure/
                    structureTabInit.js
                    structureViewZoom.js
                    index.js
                    molecule2D.js
                    structureView2D.js
        css/
            styles.css
            structureView.css
            base.css
            layout.css
            visualization.css
            tabs.css
            components.css
            chat.css
        images/
            molecule_comparison_Dummy.png
            molecule_comparison.png
    models/
        molecule.py
        __init__.py
        __pycache__/
            __init__.cpython-312.pyc
            __init__.cpython-37.pyc
            molecule.cpython-312.pyc
    data/
        molecular_data/
            molecular_data_.json
            molecular_data.json
    agents/
        __init__.py
        tools/
            analysis_tools__.py
            stout_tool.py
            mol2mol_tool.py
            nmr_simulation_tool.py
            structure_visualization_tool.py
            mmst_tool.py
            candidate_ranking_tool.py
            ___molecular_visual_comparison_tool.py
            threshold_calculation_tool.py
            spectral_comparison_tool.py
            candidate_analyzer_tool.py
            stout_operations.py
            retro_synthesis_tool.py
            analysis_enums.py
            peak_matching_tool.py
            __init__.py
            forward_synthesis_tool.py
            data_extraction_tool.py
            final_analysis_tool.py
            __pycache__/
                structure_visualization_tool.cpython-312.pyc
                retro_synthesis_tool.cpython-312.pyc
                __init__.cpython-312.pyc
                nmr_simulation_tool.cpython-312.pyc
                stout_operations.cpython-312.pyc
                spectral_comparison_tool.cpython-312.pyc
                enhanced_peak_matching_tool_v3.cpython-312.pyc
                candidate_analyzer_tool.cpython-312.pyc
                final_analysis_tool.cpython-312.pyc
                mmst_tool.cpython-312.pyc
                molecular_visual_comparison_tool.cpython-312.pyc
                candidate_ranking_tool.cpython-312.pyc
                mol2mol_tool.cpython-312.pyc
                forward_synthesis_tool.cpython-312.pyc
                peak_matching_tool.cpython-312.pyc
                enhanced_peak_matching_tool_v2.cpython-312.pyc
                enhanced_peak_matching_tool.cpython-37.pyc
                retrosynthesis_tool.cpython-312.pyc
                threshold_calculation_tool.cpython-312.pyc
                data_extraction_tool.cpython-312.pyc
                stout_tool.cpython-312.pyc
                enhanced_peak_matching_tool_v3.cpython-37.pyc
                enhanced_peak_matching_tool_v2.cpython-37.pyc
                analysis_enums.cpython-312.pyc
        __pycache__/
            __init__.cpython-312.pyc
            __init__.cpython-37.pyc
            coordinator.cpython-312.pyc
        memory/
            conversation.py
            knowledge_base.py
            __init__.py
        _agents_descriptions_/
            Instructions_for_analysis.txt
            MMST_imports.py
            orchestration_agent.txt
            AI_conversation_context.txt
            IC_MMST.py
        orchestrator/
            __init__.py
            workflow_definitions.py
            orchestrator_backup.py
            orchestrator.py
            __pycache__/
                orchestrator.cpython-312.pyc
                __init__.cpython-312.pyc
                workflow_definitions.cpython-312.pyc
                command_generator.cpython-312.pyc
            .ipynb_checkpoints/
                workflow_definitions-checkpoint.py
        coordinator/
            coordinator.py
            __init__.py
            __pycache__/
                __init__.cpython-312.pyc
                coordinator.cpython-312.pyc
        scripts/
            sgnn_sbatch.sh
            chemformer_forward_script.py
            mmst_script.py
            sgnn_local.sh
            peak_matching_script.py
            mmst_local.sh
            peak_matching_test.sh
            chemformer_retro_sbatch.sh
            sgnn_script.py
            stout_local.sh
            mol2mol_sbatch.sh
            peak_matching_local.sh
            mol2mol_script.py
            mmst_sbatch.sh
            chemformer_forward_sbatch.sh
            stout_script.py
            mol2mol_local.sh
            imports_MMST.py
            chemformer_retro_script.py
            chemformer_retro_local.sh
            chemformer_forward_local.sh
            logs/
                .gitkeep
                retro_sbatch.txt
            _working_scripts_backup/
                chemformer_retro_local.sh
                chemformer_forward_script.py
                chemformer_retro_script.py
                chemformer_retro_sbatch.sh
                Mol2Mol_script.py
                sgnn_sbatch.sh
                chemformer_forward_sbatch.sh
                mol2mol_local.sh
                sgnn_local.sh
                SGNN_script.py
                mol2mol_sbatch.sh
                chemformer_forward_local.sh
            __pycache__/
                imports_MMST.cpython-37.pyc
        specialized/
            nmr_plot_agent.py
            text_response_agent.py
            analysis_agent.py
            tool_agent.py
            __init__.py
            script_modifier_agent.py
            molecule_plot_agent.py
            __pycache__/
                __init__.cpython-37.pyc
                analysis_agent.cpython-312.pyc
                nmr_plot_agent.cpython-312.pyc
                molecule_plot_agent.cpython-312.pyc
                __init__.cpython-312.pyc
                tool_agent.cpython-312.pyc
                molecule_plot_agent.cpython-37.pyc
                text_response_agent.cpython-312.pyc
            config/
                tool_descriptions.py
                __init__.py
                __pycache__/
                    tool_descriptions.cpython-312.pyc
                    __init__.cpython-312.pyc
        base/
            base_agent.py
            __init__.py
            __pycache__/
                base_agent.cpython-37.pyc
                __init__.cpython-37.pyc
                base_agent.cpython-312.pyc
                __init__.cpython-312.pyc
    core/
        agents.py
        app.py
        __init__.py
        socket.py
        __pycache__/
            __init__.cpython-37.pyc
            agents.cpython-312.pyc
            __init__.cpython-312.pyc
            socket.cpython-312.pyc
            app.cpython-37.pyc
            app.cpython-312.pyc
    services/
        __init__.py
        llm_service.py
        ai_handler.py
        storage/
            vector_store.py
        __pycache__/
            llm_service.cpython-37.pyc
            __init__.cpython-312.pyc
            event_service.cpython-312.pyc
            ai_handler.cpython-312.pyc
            llm_service.cpython-312.pyc
            __init__.cpython-37.pyc
        audio/
            speech_service.py
    experiments/
    utils/
        sbatch_utils.py
        nmr_utils.py
        visualization.py
        __init__.py
        results_manager.py
        file_utils.py
        file_handlers.py
        __pycache__/
            sbatch_utils.cpython-312.pyc
            nmr_utils.cpython-312.pyc
            file_utils.cpython-312.pyc
            __init__.cpython-312.pyc
            visualization.cpython-312.pyc
    __pycache__/
        app.cpython-312.pyc
    test_data/
        test_smiles_with_nmr copy 2.csv
        original.txt
        test_smiles_with_nmr_.csv
        test_file_pubchem_id.txt
        test_smiles_with_nmr--big.csv
        test_smiles_with_nmr copy.csv
        test_smiles.csv
        test_smiles_with_nmr.csv
    routes/
        main.py
        audio.py
        file_upload.py
        __init__.py
        structure.py
        data_routes.py
        __pycache__/
            file_upload.cpython-312.pyc
            audio.cpython-312.pyc
            __init__.cpython-312.pyc
            main.cpython-312.pyc
    config/
        __init__.py
        settings.py
        config.py
        config.template.py
        __pycache__/
            settings.cpython-312.pyc
            config.cpython-312.pyc
            __init__.cpython-312.pyc



--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/tests/test_peak_matching_v2.py ---
## is definitely broken becuase changed the folder location of this file but can fix it later if needed
import os
import sys
import asyncio
import json
import tempfile
from types import SimpleNamespace

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
tools_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, tools_dir)

# Import the enhanced peak matching tool
from enhanced_peak_matching_tool_v2 import EnhancedPeakMatchingTool

# Sample SMILES strings for testing
SMILES1 = "CC(=O)O"  # Acetic acid
SMILES2 = "CCC(=O)O"  # Propionic acid

# Sample HSQC peak data with correct format
SAMPLE_HSQC_PEAKS = {
    "HSQC": {
        "F2 (ppm)": [2.1, 11.5],  # H shifts
        "F1 (ppm)": [20.8, 173.1],  # C shifts
        "Intensity": [1.0, 0.8],
        "atom_index": [1, 2]
    }
}

async def run_tests():
    # Initialize the tool
    tool = EnhancedPeakMatchingTool()
    
    # Create a temporary log file for the config
    temp_log = tempfile.NamedTemporaryFile(delete=False, suffix='.log')
    temp_log.close()
    
    # Create config object with required fields
    config = SimpleNamespace(
        # Basic config
        log_file=temp_log.name,

        # SGNN required parameters
        SGNN_gen_folder_path=os.path.join(tempfile.gettempdir(), "sgnn_gen"),
        SGNN_size_filter=550,  # Maximum molecular weight filter
        SGNN_csv_save_folder=os.path.join(tempfile.gettempdir(), "sgnn_save"),
        ML_dump_folder=os.path.join(tempfile.gettempdir(), "ml_dump"),
        data_type="sgnn",

    )
    
    # Create necessary directories
    for directory in [config.SGNN_gen_folder_path, config.SGNN_csv_save_folder, config.ML_dump_folder]:
        os.makedirs(directory, exist_ok=True)
    
    # Common context for all tests
    context = {
        'mode': 'single',
        'spectra': ['HSQC'],
        'matching_mode': 'hung_dist_nn',
        'error_type': 'sum',
        'config': config
    }

    try:
        # Test 1: SMILES vs SMILES
        print("\n=== Test 1: SMILES vs SMILES ===")
        smiles_vs_smiles_input = {
            'smiles1': SMILES1,
            'smiles2': SMILES2
        }
        result1 = await tool.match_peaks(smiles_vs_smiles_input, context)
        print(json.dumps(result1, indent=2))

        # Test 2: SMILES vs Peaks
        print("\n=== Test 2: SMILES vs Peaks ===")
        smiles_vs_peaks_input = {
            'smiles': SMILES1,
            'peaks': SAMPLE_HSQC_PEAKS
        }
        result2 = await tool.match_peaks(smiles_vs_peaks_input, context)
        print(json.dumps(result2, indent=2))

        # Test 3: Peaks vs Peaks
        print("\n=== Test 3: Peaks vs Peaks ===")
        modified_peaks = {
            "HSQC": {
                "F2 (ppm)": [2.2, 11.6],  # Slightly shifted H values
                "F1 (ppm)": [21.0, 173.5],  # Slightly shifted C values
                "Intensity": [1.0, 0.8],
                "atom_index": [1, 2]
            }
        }
        peaks_vs_peaks_input = {
            'peaks1': SAMPLE_HSQC_PEAKS,
            'peaks2': modified_peaks
        }
        result3 = await tool.match_peaks(peaks_vs_peaks_input, context)
        print(json.dumps(result3, indent=2))

    finally:
        # Clean up the temporary log file
        os.unlink(temp_log.name)

if __name__ == "__main__":
    asyncio.run(run_tests())


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/tests/test_molecular_visual_comparison.py ---
"""
Test script for the Molecular Visual Comparison Tool.
Tests both single molecule comparison and batch processing functionality.
"""
import os
import sys
import asyncio
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add the project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the tool directly from its file to avoid __init__ imports
tool_path = project_root / "agents" / "tools" / "molecular_visual_comparison_tool.py"
if not tool_path.exists():
    raise FileNotFoundError(f"Tool not found at: {tool_path}")
    
import importlib.util
spec = importlib.util.spec_from_file_location("molecular_visual_comparison_tool", tool_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MolecularVisualComparisonTool = module.MolecularVisualComparisonTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('molecular_comparison_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Test SMILES strings
TEST_SMILES = {
    'aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O',
    'paracetamol': 'CC(=O)NC1=CC=C(O)C=C1',
    'ibuprofen': 'CC(C)CC1=CC=C(CC(C)C(=O)O)C=C1',
    'starting_materials': 'O=C(O)C1=CC=CC=C1O.CC(=O)OC1=CC=CC=C1C(=O)O'  # Salicylic acid + Acetic anhydride
}

def get_api_key():
    """Get Anthropic API key from environment variable or config file."""
    # First try environment variable
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        return api_key
        
    # Then try config file
    try:
        from LLM_Structure_Elucidator.config.config import anthropic_api_key
        return anthropic_api_key
    except ImportError:
        logger.error("Please set ANTHROPIC_API_KEY environment variable or create config.py from config.template.py")
        raise ValueError("Anthropic API key not found")

# Create test CSV
def create_test_csv():
    """Create a CSV file with test SMILES strings."""
    test_dir = Path('test_data')
    test_dir.mkdir(exist_ok=True)
    
    smiles_data = {
        'SMILES': [
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CC(=O)NC1=CC=C(O)C=C1',      # Paracetamol
            'CC(C)CC1=CC=C(CC(C)C(=O)O)C=C1',  # Ibuprofen
            'CC1=CC=C(O)C=C1',            # p-Cresol
            'CC(=O)C1=CC=CC=C1'           # Acetophenone
        ],
        'Name': [
            'Aspirin',
            'Paracetamol',
            'Ibuprofen',
            'p-Cresol',
            'Acetophenone'
        ]
    }
    
    df = pd.DataFrame(smiles_data)
    csv_path = test_dir / 'test_molecules.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Created test CSV file at {csv_path}")
    return str(csv_path)

async def test_single_comparison():
    """Test single molecule comparison functionality."""
    logger.info("Running single molecule comparison test")
    
    # Create test directory
    test_dir = Path('test_data')
    test_dir.mkdir(exist_ok=True)
    
    # Initialize tool
    tool = MolecularVisualComparisonTool()
    
    # Test guess vs target comparison
    input_data = {
        'comparison_type': 'guess_vs_target',
        'guess_smiles': TEST_SMILES['aspirin'],
        'target_smiles': TEST_SMILES['paracetamol']
    }
    
    context = {
        'run_dir': str(test_dir),
        'run_id': f"single_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'anthropic_api_key': get_api_key()
    }
    
    try:
        result = await tool.compare_structures(input_data, context)
        logger.info("Guess vs Target comparison result:")
        logger.info(result)
        assert result['status'] == 'success', "Comparison failed"
        
    except Exception as e:
        logger.error(f"Error in guess vs target comparison: {str(e)}")
        raise
    
    # Test guess vs starting materials comparison
    input_data = {
        'comparison_type': 'guess_vs_starting',
        'guess_smiles': TEST_SMILES['aspirin'],
        'starting_materials_smiles': TEST_SMILES['starting_materials']
    }
    
    try:
        result = await tool.compare_structures(input_data, context)
        logger.info("Guess vs Starting Materials comparison result:")
        logger.info(result)
        assert result['status'] == 'success', "Comparison failed"
        
    except Exception as e:
        logger.error(f"Error in guess vs starting materials comparison: {str(e)}")
        raise

async def test_batch_comparison():
    """Test batch processing functionality."""
    logger.info("Running batch processing test")
    
    # Create test directory and CSV
    test_dir = Path('test_data')
    test_dir.mkdir(exist_ok=True)
    csv_path = create_test_csv()
    
    # Initialize tool
    tool = MolecularVisualComparisonTool()
    
    # Test batch vs target comparison
    input_data = {
        'comparison_type': 'batch_vs_target',
        'guess_smiles_csv': str(csv_path),
        'target_smiles': TEST_SMILES['aspirin']
    }
    
    context = {
        'run_dir': str(test_dir),
        'run_id': f"batch_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'anthropic_api_key': get_api_key()
    }
    
    try:
        result = await tool.compare_structures(input_data, context)
        logger.info("Batch vs Target comparison result:")
        logger.info(result)
        assert result['status'] == 'success', "Batch comparison failed"
        
    except Exception as e:
        logger.error(f"Error in batch vs target comparison: {str(e)}")
        raise
    
    # Test batch vs starting materials comparison
    input_data = {
        'comparison_type': 'batch_vs_starting',
        'guess_smiles_csv': str(csv_path),
        'starting_materials_smiles': TEST_SMILES['starting_materials']
    }
    
    try:
        result = await tool.compare_structures(input_data, context)
        logger.info("Batch vs Starting Materials comparison result:")
        logger.info(result)
        assert result['status'] == 'success', "Batch comparison failed"
        
    except Exception as e:
        logger.error(f"Error in batch vs starting materials comparison: {str(e)}")
        raise

async def main():
    """Run all tests."""
    logger.info("Starting Molecular Visual Comparison Tool Tests")
    
    # Check for API key
    try:
        get_api_key()
    except ValueError as e:
        logger.error(str(e))
        return
    
    try:
        # Test single comparison
        await test_single_comparison()
        
        # Test batch comparison
        await test_batch_comparison()
        
        logger.info("\nAll tests completed")
        
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/tests/test_peak_matching_v3.py ---
## is definitely broken becuase changed the folder location of this file but can fix it later if needed
import os
import sys
import asyncio
import json
import tempfile
import logging
from types import SimpleNamespace
import numpy as np

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
tools_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, tools_dir)

# Import the enhanced peak matching tool
from enhanced_peak_matching_tool_v2 import EnhancedPeakMatchingTool
from utils_MMT.agents_code_v15_4_3 import generate_shifts_batch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample SMILES strings for testing
SMILES1 = "CC(=O)O"  # Acetic acid
SMILES2 = "CCC(=O)O"  # Propionic acid

# Sample peak data for each spectrum type
SAMPLE_HSQC_PEAKS = {
    "HSQC": {
        "F2 (ppm)": [2.1, 11.5],  # H shifts
        "F1 (ppm)": [20.8, 173.1],  # C shifts
        "Intensity": [1.0, 0.8],
        "atom_index": [1, 2]
    }
}

SAMPLE_COSY_PEAKS = {
    "COSY": {
        "F2 (ppm)": [2.1, 3.5],  # H shifts
        "F1 (ppm)": [2.3, 3.7],  # H shifts
        "Intensity": [1.0, 0.8],
        "atom_index": [1, 2]
    }
}

SAMPLE_1H_PEAKS = {
    "1H": {
        "shifts": [1.2, 2.3, 3.4, 4.5],
        "Intensity": [1.0, 1.0, 1.0, 1.0],  # All intensities set to 1.0
        "atom_index": [0, 1, 2, 3]
    }
}

SAMPLE_13C_PEAKS = {
    "13C": {
        "shifts": [20.8, 45.2, 173.1],
        "Intensity": [1.0, 1.0, 1.0],  # All intensities set to 1.0
        "atom_index": [0, 1, 2]
    }
}

def normalize_1d_intensities(peaks_dict, spectrum_type):
    """Normalize intensities to 1.0 for 1D spectra."""
    if spectrum_type in ["1H", "13C"]:
        if "shifts" in peaks_dict:
            peaks_dict["Intensity"] = [1.0] * len(peaks_dict["shifts"])
    return peaks_dict

async def test_spectrum(tool, spectrum_type, context, generated_peaks=None):
    """Test a specific spectrum type with all scenarios."""
    logger.info(f"\n{'='*20} Testing {spectrum_type} Spectrum {'='*20}")
    
    # Get sample peaks based on spectrum type
    sample_peaks = {
        "HSQC": SAMPLE_HSQC_PEAKS,
        "COSY": SAMPLE_COSY_PEAKS,
        "1H": SAMPLE_1H_PEAKS,
        "13C": SAMPLE_13C_PEAKS
    }[spectrum_type]
    
    # If we have generated peaks, normalize intensities for 1D spectra
    if generated_peaks and spectrum_type in generated_peaks:
        peaks_dict = generated_peaks[spectrum_type].copy()
        peaks_dict = normalize_1d_intensities(peaks_dict, spectrum_type)
        generated_peaks = {spectrum_type: peaks_dict}
        logger.info(f"Generated peaks for {spectrum_type}:")
        logger.info(json.dumps(generated_peaks, indent=2))
    
    # Test scenarios
    scenarios = [
        {
            "name": "SMILES vs SMILES",
            "input": {"smiles1": SMILES1, "smiles2": SMILES2},
        },
        {
            "name": "SMILES vs Sample Peaks",
            "input": {"smiles": SMILES1, "peaks": sample_peaks},
        },
        {
            "name": "SMILES vs Generated Peaks",
            "input": {"smiles": SMILES1, "peaks": generated_peaks} if generated_peaks else None,
        },
        {
            "name": "Sample Peaks vs Sample Peaks",
            "input": {"peaks1": sample_peaks, "peaks2": sample_peaks},
        }
    ]
    
    # Update context for current spectrum
    test_context = context.copy()
    test_context["spectra"] = [spectrum_type]
    
    # Run each scenario
    for scenario in scenarios:
        if scenario["input"] is None:
            logger.info(f"Skipping {scenario['name']} - No generated peaks available")
            continue
            
        logger.info(f"\n{'-'*20} {scenario['name']} {'-'*20}")
        logger.info(f"Input data:")
        logger.info(json.dumps(scenario["input"], indent=2))
        
        try:
            result = await tool.match_peaks(scenario["input"], test_context)
            logger.info("Result:")
            logger.info(json.dumps(result, indent=2))
            
            # Detailed validation
            assert result["status"] == "success", f"Failed {scenario['name']}"
            assert spectrum_type in result["results"], f"No results for {spectrum_type}"
            
            # Validate spectrum-specific results
            spectrum_result = result["results"][spectrum_type]
            if spectrum_result["status"] == "success":
                logger.info(f" {scenario['name']} - Success")
                logger.info(f"  Score: {spectrum_result.get('score', 'N/A')}")
                logger.info(f"  Matches: {len(spectrum_result.get('matches', []))}")
            else:
                logger.warning(f" {scenario['name']} - {spectrum_result.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"Error in {scenario['name']}: {str(e)}", exc_info=True)

async def run_tests():
    """Run all spectrum tests."""
    logger.info("Starting comprehensive NMR peak matching tests")
    
    # Initialize the tool
    tool = EnhancedPeakMatchingTool()
    
    # Create a temporary log file for the config
    temp_log = tempfile.NamedTemporaryFile(delete=False, suffix='.log')
    temp_log.close()
    
    # Create config object with required fields
    config = SimpleNamespace(
        # Basic config
        log_file=temp_log.name,
        
        # SGNN required parameters
        SGNN_gen_folder_path=os.path.join(tempfile.gettempdir(), "sgnn_gen"),
        SGNN_size_filter=550,  # Maximum molecular weight filter
        SGNN_csv_save_folder=os.path.join(tempfile.gettempdir(), "sgnn_save"),
        ML_dump_folder=os.path.join(tempfile.gettempdir(), "ml_dump"),
        data_type="sgnn",
    )
    
    # Create necessary directories
    for directory in [config.SGNN_gen_folder_path, config.SGNN_csv_save_folder, config.ML_dump_folder]:
        os.makedirs(directory, exist_ok=True)
    
    # Common context for all tests
    context = {
        'mode': 'single',
        'matching_mode': 'hung_dist_nn',
        'error_type': 'sum',
        'config': config
    }
    
    try:
        # Generate peaks from SMILES1 for testing
        logger.info("\nGenerating peaks from SMILES")
        nmr_data, _, _, _ = generate_shifts_batch(config, [SMILES1])
        generated_peaks = nmr_data[0] if nmr_data else None
        
        if generated_peaks:
            logger.info("Successfully generated NMR data:")
            for spectrum_type in generated_peaks:
                logger.info(f"{spectrum_type}: {len(generated_peaks[spectrum_type].get('shifts', []))} peaks")
        
        # Test each spectrum type
        for spectrum_type in ["HSQC", "COSY", "1H", "13C"]:
            await test_spectrum(tool, spectrum_type, context, generated_peaks)
            
    except Exception as e:
        logger.error("Test execution failed", exc_info=True)
        
    finally:
        # Clean up the temporary log file
        os.unlink(temp_log.name)
        logger.info("\nTest execution completed")

if __name__ == "__main__":
    asyncio.run(run_tests())

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/tests/conftest.py ---
"""
Global pytest configuration and fixtures.
"""
import pytest
import os
import tempfile
from pathlib import Path

@pytest.fixture(scope="session")
def test_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture(scope="session")
def test_file(test_dir):
    """Create a test file with sample content."""
    file_path = Path(test_dir) / "test_file.txt"
    with open(file_path, "w") as f:
        f.write("Test content for file upload")
    return file_path

@pytest.fixture(autouse=True)
def setup_environment():
    """Set up environment variables for testing."""
    # Store original environment
    original_env = dict(os.environ)
    
    # Set test environment variables
    os.environ.update({
        'OPENAI_API_KEY': 'test_api_key',
        'FLASK_ENV': 'testing',
        'TESTING': 'true'
    })
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture
def mock_openai_response():
    """Mock response from OpenAI API."""
    return {
        'choices': [{
            'message': {
                'content': 'Test response from OpenAI',
                'role': 'assistant'
            }
        }]
    }

@pytest.fixture
def mock_anthropic_response():
    """Mock response from Anthropic API."""
    return {
        'content': [{
            'text': 'Test response from Claude',
            'type': 'text'
        }],
        'role': 'assistant'
    }


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/tests/test_api_integration.py ---


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/tests/test_retrosynthesis.py ---
"""Test script for retrosynthesis predictions through the tool agent."""
import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.specialized.tool_agent import ToolAgent
from agents.tools.retrosynthesis_tool import RetrosynthesisTool

async def test_retrosynthesis():
    """Test retrosynthesis prediction with a simple molecule."""
    # Test molecule (aspirin)
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    
    # Initialize tool
    retro_tool = RetrosynthesisTool()
    
    # Test local prediction
    result = await retro_tool.predict_retrosynthesis(
        test_smiles,
        context={'use_slurm': False}  # Run locally
    )
    
    print("Retrosynthesis prediction result:")
    print(result)

if __name__ == "__main__":
    asyncio.run(test_retrosynthesis())


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/tests/integration/test_api.py ---
"""
Integration tests for API endpoints and workflows.
"""
import pytest
from unittest.mock import patch
from flask import url_for
import json
from ...app import app, socketio
from ..fixtures.molecule_data import SAMPLE_SMILES, SAMPLE_NMR_DATA

@pytest.fixture
def client():
    """Create a test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def socket_client():
    """Create a test client for WebSocket connections."""
    app.config['TESTING'] = True
    return socketio.test_client(app)

class TestAPIEndpoints:
    def test_home_endpoint(self, client):
        """Test the home page endpoint."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'LLM Structure Elucidator' in response.data

    def test_upload_endpoint(self, client):
        """Test file upload endpoint."""
        data = {
            'file': (open('tests/fixtures/test_file.txt', 'rb'), 'test_file.txt')
        }
        response = client.post('/upload', data=data)
        assert response.status_code == 200
        assert 'file_id' in json.loads(response.data)

    @patch('services.llm_service.LLMService.get_completion')
    def test_molecule_analysis_workflow(self, mock_llm, socket_client):
        """Test complete molecule analysis workflow."""
        # Mock LLM response
        mock_llm.return_value = {
            'content': 'Analysis of ethanol structure...',
            'confidence': 0.95
        }

        # Connect to WebSocket
        assert socket_client.is_connected()

        # Send molecule query
        socket_client.emit('message', {
            'message': 'Analyze ethanol structure',
            'model': 'claude-3-haiku'
        })

        # Receive response
        received = socket_client.get_received()
        assert len(received) > 0
        assert 'content' in received[0]['args'][0]

    def test_error_handling(self, client, socket_client):
        """Test error handling in API endpoints."""
        # Test invalid file upload
        response = client.post('/upload', data={})
        assert response.status_code == 400

        # Test invalid WebSocket message
        socket_client.emit('message', {})
        received = socket_client.get_received()
        assert 'error' in received[0]['args'][0]

    @patch('services.llm_service.LLMService.get_completion')
    def test_concurrent_requests(self, mock_llm, socket_client):
        """Test handling of concurrent requests."""
        # Mock LLM responses
        mock_llm.side_effect = [
            {'content': f'Response {i}', 'confidence': 0.9}
            for i in range(3)
        ]

        # Send multiple concurrent requests
        for i in range(3):
            socket_client.emit('message', {
                'message': f'Query {i}',
                'model': 'claude-3-haiku'
            })

        # Check all responses
        received = socket_client.get_received()
        assert len(received) == 3
        for i, response in enumerate(received):
            assert f'Response {i}' in response['args'][0]['content']

    def test_session_management(self, socket_client):
        """Test session management and context preservation."""
        # Connect and get session
        assert socket_client.is_connected()
        
        # Send multiple related queries
        queries = [
            'Analyze ethanol structure',
            'Show me its NMR spectrum',
            'Explain the peaks'
        ]
        
        for query in queries:
            socket_client.emit('message', {
                'message': query,
                'model': 'claude-3-haiku'
            })
            received = socket_client.get_received()
            assert len(received) > 0

        # Verify context is maintained
        socket_client.emit('message', {
            'message': 'What molecule were we discussing?',
            'model': 'claude-3-haiku'
        })
        received = socket_client.get_received()
        assert 'ethanol' in received[0]['args'][0]['content'].lower()


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/tests/unit/test_analysis.py ---
"""
Unit tests for analysis-related functionality.
"""
import pytest
from unittest.mock import Mock, patch
from ...agents.analysis.analysis_agent import AnalysisAgent
from ...services.llm_service import LLMService
from ..fixtures.molecule_data import SAMPLE_NMR_DATA

@pytest.mark.asyncio
class TestAnalysisAgent:
    @pytest.fixture
    async def analysis_agent(self):
        """Create an AnalysisAgent instance for testing."""
        llm_service = Mock(spec=LLMService)
        agent = AnalysisAgent(llm_service)
        return agent

    async def test_nmr_data_processing(self, analysis_agent, sample_nmr_data):
        """Test NMR data processing functionality."""
        # Test peak processing
        peaks = sample_nmr_data['peaks']
        processed_peaks = await analysis_agent.process_peaks(peaks)
        assert processed_peaks is not None
        assert len(processed_peaks) == len(peaks)

        # Test correlation processing
        correlations = sample_nmr_data['correlations']
        processed_correlations = await analysis_agent.process_correlations(correlations)
        assert processed_correlations is not None
        assert len(processed_correlations) == len(correlations)

    async def test_process_analysis_query(self, analysis_agent, mock_llm_response):
        """Test processing of analysis-related queries."""
        # Mock LLM response
        analysis_agent.llm_service.get_completion.return_value = \
            mock_llm_response['analysis_query']

        # Test analysis query processing
        query = "Analyze the NMR spectrum of ethanol"
        response = await analysis_agent.process(query, {'nmr_data': SAMPLE_NMR_DATA['ethanol']})
        
        assert response['content'] == mock_llm_response['analysis_query']['content']
        assert response['confidence'] == mock_llm_response['analysis_query']['confidence']

    async def test_plot_generation(self, analysis_agent, sample_nmr_data):
        """Test plot generation functionality."""
        # Test 1D NMR plot
        plot_1d = await analysis_agent.generate_1d_plot(sample_nmr_data['peaks'])
        assert plot_1d is not None
        assert isinstance(plot_1d, dict)
        assert 'image' in plot_1d

        # Test 2D correlation plot
        plot_2d = await analysis_agent.generate_2d_plot(sample_nmr_data['correlations'])
        assert plot_2d is not None
        assert isinstance(plot_2d, dict)
        assert 'image' in plot_2d

    async def test_error_handling(self, analysis_agent):
        """Test error handling in analysis processing."""
        # Test with invalid data
        with pytest.raises(ValueError):
            await analysis_agent.process_peaks([])

        # Test with empty query
        response = await analysis_agent.process("", {})
        assert 'error' in response

    @patch('matplotlib.pyplot')
    async def test_plot_generation_failure(self, mock_plt, analysis_agent):
        """Test handling of plot generation failures."""
        mock_plt.savefig.side_effect = Exception('Plot generation failed')
        
        with pytest.raises(Exception):
            await analysis_agent.generate_1d_plot(SAMPLE_NMR_DATA['ethanol']['peaks'])


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/tests/unit/test_molecule.py ---
"""
Unit tests for molecule-related functionality.
"""
import pytest
from rdkit import Chem
from unittest.mock import Mock, patch
from ...agents.molecule.molecule_agent import MoleculeAgent
from ...services.llm_service import LLMService
from ..fixtures.molecule_data import SAMPLE_SMILES

@pytest.mark.asyncio
class TestMoleculeAgent:
    @pytest.fixture
    async def molecule_agent(self):
        """Create a MoleculeAgent instance for testing."""
        llm_service = Mock(spec=LLMService)
        agent = MoleculeAgent(llm_service)
        return agent

    async def test_molecule_generation(self, molecule_agent, sample_molecule):
        """Test molecule generation from SMILES."""
        # Test valid SMILES
        mol = Chem.MolFromSmiles(SAMPLE_SMILES['ethanol'])
        assert mol is not None
        assert Chem.MolToSmiles(mol) == SAMPLE_SMILES['ethanol']

        # Test invalid SMILES
        mol = Chem.MolFromSmiles(SAMPLE_SMILES['invalid'])
        assert mol is None

    async def test_process_molecule_query(self, molecule_agent, mock_llm_response):
        """Test processing of molecule-related queries."""
        # Mock LLM response
        molecule_agent.llm_service.get_completion.return_value = \
            mock_llm_response['molecule_query']

        # Test molecule query processing
        query = "What is the structure of ethanol?"
        response = await molecule_agent.process(query, {})
        
        assert response['content'] == mock_llm_response['molecule_query']['content']
        assert response['confidence'] == mock_llm_response['molecule_query']['confidence']
        
        # Verify LLM was called with correct parameters
        molecule_agent.llm_service.get_completion.assert_called_once()

    async def test_molecule_visualization(self, molecule_agent, sample_molecule):
        """Test molecule visualization functions."""
        # Test 2D rendering
        img_data = molecule_agent.generate_2d_image(sample_molecule)
        assert img_data is not None
        assert isinstance(img_data, str)  # Should return base64 encoded string

        # Test 3D rendering (if implemented)
        if hasattr(molecule_agent, 'generate_3d_image'):
            img_data_3d = molecule_agent.generate_3d_image(sample_molecule)
            assert img_data_3d is not None
            assert isinstance(img_data_3d, str)

    async def test_error_handling(self, molecule_agent):
        """Test error handling in molecule processing."""
        # Test with invalid SMILES
        with pytest.raises(ValueError):
            await molecule_agent.process_molecule(SAMPLE_SMILES['invalid'])

        # Test with empty query
        response = await molecule_agent.process("", {})
        assert 'error' in response

    @patch('rdkit.Chem.MolFromSmiles')
    async def test_molecule_processing_failure(self, mock_mol_from_smiles, molecule_agent):
        """Test handling of RDKit failures."""
        mock_mol_from_smiles.return_value = None
        
        response = await molecule_agent.process_molecule(SAMPLE_SMILES['ethanol'])
        assert response['error'] == 'Failed to generate molecule'


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/tests/fixtures/molecule_data.py ---
"""
Test fixtures for molecule-related tests.
"""
import pytest
from rdkit import Chem

# Sample SMILES strings for testing
SAMPLE_SMILES = {
    'ethanol': 'CCO',
    'aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O',
    'caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    'invalid': 'NOT_A_VALID_SMILES'
}

# Sample NMR data
SAMPLE_NMR_DATA = {
    'ethanol': {
        'peaks': [(1.2, 3, 'CH3'), (3.7, 2, 'CH2'), (2.5, 1, 'OH')],
        'correlations': [(1.2, 3.7, 'strong'), (3.7, 2.5, 'weak')]
    }
}

@pytest.fixture
def sample_molecule():
    """Fixture providing a sample molecule (ethanol)."""
    return Chem.MolFromSmiles(SAMPLE_SMILES['ethanol'])

@pytest.fixture
def sample_complex_molecule():
    """Fixture providing a more complex molecule (aspirin)."""
    return Chem.MolFromSmiles(SAMPLE_SMILES['aspirin'])

@pytest.fixture
def sample_nmr_data():
    """Fixture providing sample NMR data."""
    return SAMPLE_NMR_DATA['ethanol']

@pytest.fixture
def mock_llm_response():
    """Fixture providing mock LLM responses."""
    return {
        'molecule_query': {
            'content': 'The molecule appears to be ethanol (CH3CH2OH), a simple alcohol.',
            'confidence': 0.95
        },
        'analysis_query': {
            'content': 'The NMR spectrum shows characteristic peaks for ethanol: '
                      'a triplet at 1.2 ppm (CH3), a quartet at 3.7 ppm (CH2), '
                      'and a broad singlet at 2.5 ppm (OH).',
            'confidence': 0.90
        }
    }


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/handlers/molecule_handler.py ---
"""
Molecule request handlers for Socket.IO events and HTTP requests.
"""
import os
import pandas as pd
import random
import json
from pathlib import Path
from flask import request, jsonify
from flask_socketio import emit
from core.socket import socketio
from utils.visualization import create_molecule_response
from rdkit import Chem
from core.app import app
from typing import Dict

# Path to molecular data JSON file
MOLECULAR_DATA_PATH = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "data" / "molecular_data" / "molecular_data.json"

# Global variable to track current molecule
_current_molecule = None

def get_current_molecule():
    """Get the currently selected molecule."""
    global _current_molecule
    print(f"[Molecule Handler] Getting current molecule:")
    #print(f"  - Current molecule state: {json.dumps(_current_molecule, indent=2) if _current_molecule else None}")
    return _current_molecule

def set_current_molecule(smiles: str, name: str = None, nmr_data: Dict = None, sample_id: str = None):
    """Set the current molecule."""
    global _current_molecule
    print(f"\n[Molecule Handler] Setting current molecule:")
    print(f"  - SMILES: {smiles}")
    print(f"  - Name: {name}")
    print(f"  - Sample ID: {sample_id}")
    print(f"  - NMR Data Present: {bool(nmr_data)}")
    if nmr_data:
        print(f"  - NMR Data Keys: {list(nmr_data.keys())}")
        
        # Normalize NMR data keys to use _exp suffix
        normalized_nmr_data = {}
        key_mapping = {
            '1h': '1H_exp',
            '13c': '13C_exp',
            'hsqc': 'HSQC_exp',
            'cosy': 'COSY_exp',
            '1h_exp': '1H_exp',
            '13c_exp': '13C_exp',
            'hsqc_exp': 'HSQC_exp',
            'cosy_exp': 'COSY_exp',
            '1H': '1H_exp',
            '13C': '13C_exp',
            'HSQC': 'HSQC_exp',
            'COSY': 'COSY_exp'
        }
        
        for key, value in nmr_data.items():
            normalized_key = key_mapping.get(key.lower(), key)
            if value is not None:  # Only include non-null values
                normalized_nmr_data[normalized_key] = value
    else:
        normalized_nmr_data = {}
    
    _current_molecule = {
        'smiles': smiles,
        'name': name or 'Unknown',
        'sample_id': sample_id or 'unknown',
        'nmr_data': normalized_nmr_data
    }
    print(f"[Molecule Handler] Current molecule successfully set")
    #print(f"[Molecule Handler] Full molecule state: {json.dumps(_current_molecule, indent=2)}")

def get_molecular_data():
    """Get all molecular data from JSON storage."""
    print(f"\n[Molecule Handler] Attempting to read molecular data from: {MOLECULAR_DATA_PATH}")
    try:
        if not MOLECULAR_DATA_PATH.exists():
            print(f"[Molecule Handler] ERROR: Molecular data file not found at {MOLECULAR_DATA_PATH}")
            return None
            
        with open(MOLECULAR_DATA_PATH, 'r') as f:
            data = json.load(f)
            print(f"[Molecule Handler] Successfully loaded molecular data:")
            print(f"  - Number of molecules: {len(data)}")
            print(f"  - Sample IDs: {list(data.keys())}")
        return data
    except Exception as e:
        print(f"[Molecule Handler] ERROR reading molecular data:")
        print(f"  - Error type: {type(e)}")
        print(f"  - Error message: {str(e)}")
        import traceback
        print(f"  - Traceback:\n{traceback.format_exc()}")
        return None

def get_first_molecule_json():
    """Get the first molecule's data from JSON storage."""
    try:
        data = get_molecular_data()
        if not data:
            print("[Molecule Handler] Error: No molecular data found")
            return None
            
        # Get first molecule's data
        first_molecule_id = next(iter(data))
        first_molecule = data[first_molecule_id]
        
        # Set as current molecule
        set_current_molecule(
            smiles=first_molecule.get('smiles'),
            name=first_molecule.get('name', 'Unknown'),
            sample_id=first_molecule_id,
            nmr_data=first_molecule.get('nmr_data', {})
        )
        
        return {
            'status': 'success',
            'sample_id': first_molecule_id,
            'smiles': first_molecule.get('smiles'),
            # 'inchi': first_molecule.get('inchi'),
            # 'inchi_key': first_molecule.get('inchi_key'),
            'nmr_data': first_molecule.get('nmr_data', {})
        }
    except Exception as e:
        print(f"[Molecule Handler] Error getting first molecule: {str(e)}")
        return None

# Add Flask routes for JSON data access
@app.route('/get_molecular_data', methods=['GET'])
def handle_get_molecular_data():
    """Handle request to get all molecular data from JSON storage."""
    data = get_molecular_data()
    if data is None:
        return jsonify({"error": "No molecular data found"}), 404
    return jsonify({"status": "success", "data": data})

@app.route('/get_first_molecule_json', methods=['GET'])
def handle_get_first_molecule_json():
    """Handle request to get first molecule with all associated data from JSON."""
    result = get_first_molecule_json()
    if result is None:
        return jsonify({"error": "No molecules found in database"}), 404
    return jsonify(result)


def get_nmr_data_from_json(smiles: str):
    """Get NMR data for a SMILES string from JSON storage."""
    print(f"\n[Molecule Handler] Searching for NMR data for SMILES: {smiles}")
    try:
        data = get_molecular_data()
        if not data:
            print("[Molecule Handler] No molecular data available")
            return None
            
        # Find molecule with matching SMILES
        for sample_id, molecule in data.items():
            if molecule.get('smiles') == smiles:
                print(f"[Molecule Handler] Found NMR data for SMILES: {smiles} (sample-id: {sample_id})")
                
                # Get NMR data and keep original keys
                nmr_data = molecule.get('nmr_data', {})
                result = {
                    'sample_id': sample_id,
                    '1H_exp': nmr_data.get('1H_exp'),
                    '13C_exp': nmr_data.get('13C_exp'),
                    'HSQC_exp': nmr_data.get('HSQC_exp'),
                    'COSY_exp': nmr_data.get('COSY_exp')
                }
                
                print(f"[Molecule Handler] Available spectra: {[k for k,v in result.items() if v and k != 'sample_id']}")
                return result
                
        print(f"[Molecule Handler] No NMR data found for SMILES: {smiles}")
        return {}
        
    except Exception as e:
        print(f"[Molecule Handler] ERROR retrieving NMR data:")
        print(f"  - Error type: {type(e)}")
        print(f"  - Error message: {str(e)}")
        import traceback
        print(f"  - Traceback:\n{traceback.format_exc()}")
        return None

class MoleculeHandler:
    def generate_molecule_response(self, smiles):
        """Generate a complete molecule visualization response."""
        print(f"[Molecule Handler] Generating response for SMILES: {smiles}")
        try:
            response = create_molecule_response(smiles)
            print(f"[Molecule Handler] Response generated: {response is not None}")
            return response
        except Exception as e:
            print(f"[Molecule Handler] Error generating response: {str(e)}")
            print(f"[Molecule Handler] Error type: {type(e)}")
            return None

@socketio.on('get_molecule_image')
def get_molecule_image(data=None):
    """Generate and return a molecule image."""
    print("[Molecule Handler] Handling get_molecule_image request")
    try:
        # Get first molecule from JSON instead of random SMILES
        first_molecule = get_first_molecule_json()
        if not first_molecule or not first_molecule.get('smiles'):
            raise ValueError("Failed to get valid molecule from data")
            
        smiles = first_molecule['smiles']
        print(f"[Molecule Handler] Using SMILES: {smiles}")
        
        # Generate response
        response = create_molecule_response(smiles, is_3d=False)
        if response is None:
            raise ValueError("Failed to create molecule response")
        
        # Format molecular weight if present
        if 'molecular_weight' in response:
            response['molecular_weight'] = f"{response['molecular_weight']:.2f} g/mol"
        
        print("[Molecule Handler] Emitting molecule_image")
        emit('molecule_image', response)
        
    except Exception as e:
        error_msg = f"Failed to generate molecule image: {str(e)}"
        print(f"[Molecule Handler] Exception: {error_msg}")
        emit('message', {
            'content': error_msg,
            'type': 'error'
        })


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/handlers/audio_handler.py ---
"""
Audio request handlers for Socket.IO events.
"""
import os
from flask_socketio import emit
from core.socket import socketio
from utils.file_utils import save_uploaded_file

@socketio.on('transcribe_audio')
def handle_audio_request():
    """Handle audio transcription request."""
    try:
        if 'file' not in request.files:
            return {'error': 'No file provided'}, 400
        
        file = request.files['file']
        if file.filename == '':
            return {'error': 'No file selected'}, 400
            
        # Save uploaded file
        save_uploaded_file(file)
        
        # Process audio file (implementation details in audio route)
        # This is a placeholder for the actual audio processing logic
        result = {'message': 'Audio processing not implemented'}
        emit('transcription_result', result)
            
    except Exception as e:
        print(f"Error in handle_audio_request: {str(e)}")
        emit('error', {'message': 'Failed to process audio file'})

@socketio.on('text_to_speech')
def handle_tts_request(data):
    """Handle text-to-speech request."""
    try:
        text = data.get('text', '')
        if not text:
            emit('error', {'message': 'No text provided for TTS'})
            return
            
        # Process TTS request (implementation details in audio route)
        # This is a placeholder for the actual TTS logic
        result = {'message': 'TTS not implemented'}
        emit('tts_result', result)
            
    except Exception as e:
        print(f"Error in handle_tts_request: {str(e)}")
        emit('error', {'message': 'Failed to generate speech'})


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/handlers/message_handler.py ---
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


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/handlers/plot_handler.py ---
"""
Plot request handler for Socket.IO events.
"""
import base64
import pandas as pd
import numpy as np
from flask_socketio import emit
from core.socket import socketio
import traceback
import json

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print("\n[Plot Handler] ====== Client Connected to Plot Handler ======")
    print("[Plot Handler] Socket.IO connection established")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print("\n[Plot Handler] ====== Client Disconnected from Plot Handler ======")

@socketio.on('plot_request')
def handle_plot_request(request_data):
    """Handle plot generation requests."""
    print("\n[Plot Handler] ====== Starting Plot Generation ======")

    try:
        # Validate request data
        if not request_data or not isinstance(request_data, dict):
            error = "Invalid plot request data format"
            print(f"[Plot Handler] ERROR: {error}")
            emit('message', {'content': error, 'type': 'error'})
            return
        
        # Extract and validate plot type
        plot_type = request_data.get('plot_type', '').lower()
        parameters = request_data.get('parameters', {})
        print(f"[Plot Handler] Plot type: {plot_type}")
        
        if not plot_type:
            error = "No plot type specified"
            print(f"[Plot Handler] ERROR: {error}")
            emit('message', {'content': error, 'type': 'error'})
            return
            
        print("[Plot Handler] Creating plot data...")
        
        # Extract NMR data if available
        nmr_data = parameters.get('nmr_data', {})
        x_data = nmr_data.get('x')
        y_data = nmr_data.get('y')
        z_data = nmr_data.get('z') if plot_type in ['hsqc', 'cosy'] else None
        
        # Create plot data based on type
        if plot_type in ['hsqc', 'cosy']:
            print(f"[Plot Handler] Generating {plot_type.upper()} plot data")
            plot_data, layout = generate_2d_plot(plot_type, parameters, x_data, y_data, z_data)
        else:  # 1D spectra (proton or carbon)
            print(f"[Plot Handler] Generating 1D data for {plot_type}")
            plot_data, layout = generate_1d_plot(plot_type, parameters, x_data, y_data)
        
        print("[Plot Handler] Plot data created successfully")

        # Apply plot style
        style = parameters.get('style', 'default')
        apply_plot_style(plot_data, layout, style)
        
        # Create response
        response = {
            'data': [plot_data],
            'layout': layout,
            'type': plot_type,
            'config': {
                'responsive': True,
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['sendDataToCloud', 'lasso2d', 'select2d'],
                'useResizeHandler': True
            }
        }
        
        # First emit a status message
        print("[Plot Handler] Emitting status message")
        emit('message', {
            'content': f"Generating {plot_type.upper()} plot...",
            'type': 'info'
        })
        
        # Then emit the plot data
        print("[Plot Handler] Emitting plot data to client")
        emit('plot', response)
        
        # Finally emit a success message
        emit('message', {
            'content': f"Generated {plot_type.upper()} plot successfully",
            'type': 'success'
        })
        print("[Plot Handler] ====== Plot Generation Complete ======\n")
        
    except Exception as e:
        print(f"[Plot Handler] ERROR: Failed to generate plot: {str(e)}")
        print(f"[Plot Handler] Traceback: {traceback.format_exc()}")
        emit('message', {
            'content': f"Error generating plot: {str(e)}",
            'type': 'error'
        })

def generate_2d_plot(plot_type, parameters, x_data=None, y_data=None, z_data=None):
    """Generate 2D plot data (HSQC or COSY).
    
    Args:
        plot_type (str): Type of plot ('hsqc' or 'cosy')
        parameters (dict): Plot parameters
        x_data (list, optional): X-axis data points
        y_data (list, optional): Y-axis data points
        z_data (list, optional): Z-axis data points for coloring
    
    Returns:
        tuple: (plot_data, layout) for plotly
    """
    if all(data is not None for data in [x_data, y_data, z_data]):
        print("[Plot Handler] Using provided 2D NMR data")
    else:
        print("[Plot Handler] Generating default 2D data")
        # Generate default example data
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)
        
        x_data = X.flatten().tolist()
        y_data = Y.flatten().tolist()
        z_data = Z.flatten().tolist()
    
    plot_data = {
        'type': 'scatter',
        'mode': 'markers',
        'x': x_data,
        'y': y_data,
        'marker': {
            'size': 8,
            'color': z_data,
            'colorscale': 'Viridis',
            'showscale': True
        }
    }
    
    layout = {
        'title': parameters.get('title', f'{plot_type.upper()} NMR Spectrum'),
        'xaxis': {
            'title': parameters.get('x_label', 'F2 (ppm)'),
            'autorange': 'reversed'
        },
        'yaxis': {
            'title': parameters.get('y_label', 'F1 (ppm)'),
            'autorange': 'reversed'
        },
        'showlegend': False,
        'autosize': True,
        'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
    }
    
    return plot_data, layout

def generate_1d_plot(plot_type, parameters, x_data=None, y_data=None):
    """Generate 1D plot data (proton or carbon NMR).
    
    Args:
        plot_type (str): Type of plot ('proton' or 'carbon')
        parameters (dict): Plot parameters
        x_data (list, optional): X-axis data points (chemical shifts)
        y_data (list, optional): Y-axis data points (intensities)
    
    Returns:
        tuple: (plot_data, layout) for plotly
    """
    if x_data is not None and y_data is not None:
        print("[Plot Handler] Using provided 1D NMR data")
    else:
        print("[Plot Handler] Generating default 1D data")
        # Generate default example data
        x = np.linspace(0, 10, 1000)
        y = np.exp(-(x - 5)**2) + 0.5 * np.exp(-(x - 7)**2)
        
        x_data = x.tolist()
        y_data = y.tolist()
    
    plot_data = {
        'type': 'scatter',
        'mode': 'lines',
        'x': x_data,
        'y': y_data,
        'line': {
            'color': 'blue',
            'width': 1
        }
    }
    
    layout = {
        'title': parameters.get('title', f'{plot_type.upper()} NMR Spectrum'),
        'xaxis': {
            'title': parameters.get('x_label', 'Chemical Shift (ppm)'),
            'autorange': 'reversed'
        },
        'yaxis': {
            'title': parameters.get('y_label', 'Intensity'),
        },
        'showlegend': False,
        'autosize': True,
        'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
    }
    
    return plot_data, layout

def apply_plot_style(plot_data, layout, style):
    """Apply visual style to plot."""
    if style == 'publication':
        layout['font'] = {'family': 'Arial', 'size': 14}
        layout['margin'] = {'l': 60, 'r': 20, 't': 40, 'b': 60}
        if 'line' in plot_data:
            plot_data['line']['color'] = 'black'
    elif style == 'presentation':
        layout['font'] = {'family': 'Arial', 'size': 16}
        layout['margin'] = {'l': 80, 'r': 40, 't': 60, 'b': 80}
        if 'line' in plot_data:
            plot_data['line']['color'] = '#1f77b4'  # Professional blue


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/handlers/connection_handler.py ---
"""
Connection event handlers for Socket.IO.
"""
from flask_socketio import emit
from core.socket import socketio

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')
    emit('connection_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')
    emit('connection_status', {'status': 'disconnected'})


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/handlers/chat_handler.py ---
"""
Chat-related event handlers for Socket.IO.
"""
from flask import session
from flask_socketio import emit
from core.socket import socketio

@socketio.on('clear_chat')
def clear_chat():
    """Clear chat history."""
    try:
        session.clear()
        emit('chat_cleared', {'status': 'success'})
    except Exception as e:
        print(f"Error in clear_chat: {str(e)}")
        emit('error', {'message': 'Failed to clear chat history'})


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/handlers/__init__.py ---
"""
Socket.IO event handlers for the LLM Structure Elucidator.
"""

from .message_handler import handle_message
from .plot_handler import handle_plot_request
from .molecule_handler import get_molecule_image
from .audio_handler import handle_audio_request, handle_tts_request
from .connection_handler import handle_connect, handle_disconnect
from .chat_handler import clear_chat

__all__ = [
    'handle_message',
    'handle_plot_request',
    'handle_audio_request',
    'handle_tts_request',
    'handle_connect',
    'handle_disconnect',
    'get_molecule_image',
    'clear_chat'
]


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/monitoring/__init__.py ---
"""
Monitoring package for LLM Structure Elucidator.
"""
from .metrics.performance_metrics import PerformanceMetrics
from .logging.logger import LLMLogger
from .alerts.alert_manager import AlertManager

__all__ = ['PerformanceMetrics', 'LLMLogger', 'AlertManager']


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/monitoring/logging/logger.py ---
"""
Centralized logging configuration and setup.
"""
import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional
from pathlib import Path

class LLMLogger:
    """Centralized logging for the LLM Structure Elucidator."""
    
    def __init__(self, log_dir: Optional[str] = None):
        """Initialize logger with optional custom log directory."""
        self.log_dir = log_dir or os.path.join(os.path.dirname(__file__), '../../logs')
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Create loggers for different components
        self.app_logger = self._setup_logger('app', 'app.log')
        self.api_logger = self._setup_logger('api', 'api.log')
        self.llm_logger = self._setup_logger('llm', 'llm.log')
        self.error_logger = self._setup_logger('error', 'error.log', level=logging.ERROR)
    
    def _setup_logger(self, name: str, filename: str, level: int = logging.INFO) -> logging.Logger:
        """Set up a logger with file and console handlers."""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        if not logger.handlers:
            # File handler
            file_handler = logging.handlers.RotatingFileHandler(
                os.path.join(self.log_dir, filename),
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def log_request(self, method: str, path: str, params: dict, response_time: float) -> None:
        """Log API request details."""
        self.api_logger.info(
            f"Request: {method} {path} - Params: {params} - Response Time: {response_time:.2f}s"
        )
    
    def log_llm_interaction(self, model: str, prompt_tokens: int, completion_tokens: int, 
                          duration: float, success: bool) -> None:
        """Log LLM interaction details."""
        self.llm_logger.info(
            f"Model: {model} - Prompt Tokens: {prompt_tokens} - "
            f"Completion Tokens: {completion_tokens} - Duration: {duration:.2f}s - "
            f"Success: {success}"
        )
    
    def log_error(self, error: Exception, context: dict = None) -> None:
        """Log error with context."""
        context = context or {}
        self.error_logger.error(
            f"Error: {str(error)} - Type: {type(error).__name__} - Context: {context}",
            exc_info=True
        )
    
    def log_app_event(self, event_type: str, details: dict) -> None:
        """Log application events."""
        self.app_logger.info(f"Event: {event_type} - Details: {details}")
    
    def get_recent_errors(self, hours: int = 24) -> list:
        """Get list of errors from the last N hours."""
        errors = []
        error_log_path = os.path.join(self.log_dir, 'error.log')
        
        if not os.path.exists(error_log_path):
            return errors
        
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        with open(error_log_path, 'r') as f:
            for line in f:
                try:
                    # Parse timestamp from log line
                    timestamp_str = line.split(' - ')[0]
                    timestamp = datetime.strptime(
                        timestamp_str, '%Y-%m-%d %H:%M:%S,%f'
                    ).timestamp()
                    
                    if timestamp >= cutoff_time:
                        errors.append(line.strip())
                except (ValueError, IndexError):
                    continue
        
        return errors


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/monitoring/metrics/performance_metrics.py ---
"""
Performance metrics collection and tracking.
"""
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import statistics

@dataclass
class MetricPoint:
    """Single metric measurement point."""
    value: float
    timestamp: datetime
    labels: Dict[str, str]

class PerformanceMetrics:
    """Collect and track performance metrics."""
    
    def __init__(self):
        self.metrics = {
            'response_time': [],
            'token_count': [],
            'api_latency': [],
            'error_rate': [],
            'request_count': 0
        }
        self._start_time: Optional[float] = None
    
    def start_request(self) -> None:
        """Start timing a request."""
        self._start_time = time.time()
    
    def end_request(self, success: bool = True) -> float:
        """End timing a request and record metrics."""
        if not self._start_time:
            raise ValueError("start_request() must be called before end_request()")
        
        duration = time.time() - self._start_time
        self.metrics['response_time'].append(
            MetricPoint(
                value=duration,
                timestamp=datetime.now(),
                labels={'success': str(success)}
            )
        )
        self.metrics['request_count'] += 1
        self._start_time = None
        return duration
    
    def record_token_count(self, count: int, model: str) -> None:
        """Record token usage for a request."""
        self.metrics['token_count'].append(
            MetricPoint(
                value=count,
                timestamp=datetime.now(),
                labels={'model': model}
            )
        )
    
    def record_api_latency(self, latency: float, provider: str) -> None:
        """Record API latency."""
        self.metrics['api_latency'].append(
            MetricPoint(
                value=latency,
                timestamp=datetime.now(),
                labels={'provider': provider}
            )
        )
    
    def record_error(self, error_type: str) -> None:
        """Record an error occurrence."""
        self.metrics['error_rate'].append(
            MetricPoint(
                value=1.0,
                timestamp=datetime.now(),
                labels={'error_type': error_type}
            )
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of collected metrics."""
        summary = {}
        
        # Response time stats
        response_times = [m.value for m in self.metrics['response_time']]
        if response_times:
            summary['response_time'] = {
                'mean': statistics.mean(response_times),
                'median': statistics.median(response_times),
                'p95': statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else None
            }
        
        # Token usage by model
        token_counts = {}
        for metric in self.metrics['token_count']:
            model = metric.labels['model']
            if model not in token_counts:
                token_counts[model] = []
            token_counts[model].append(metric.value)
        
        summary['token_usage'] = {
            model: {
                'total': sum(counts),
                'mean': statistics.mean(counts)
            }
            for model, counts in token_counts.items()
        }
        
        # API latency by provider
        latencies = {}
        for metric in self.metrics['api_latency']:
            provider = metric.labels['provider']
            if provider not in latencies:
                latencies[provider] = []
            latencies[provider].append(metric.value)
        
        summary['api_latency'] = {
            provider: {
                'mean': statistics.mean(lats),
                'p95': statistics.quantiles(lats, n=20)[18] if len(lats) >= 20 else None
            }
            for provider, lats in latencies.items()
        }
        
        # Error rate
        total_requests = self.metrics['request_count']
        total_errors = len(self.metrics['error_rate'])
        summary['error_rate'] = total_errors / total_requests if total_requests > 0 else 0
        
        return summary


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/monitoring/alerts/alert_manager.py ---
"""
Alert management and notification system.
"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
import json
import os
from pathlib import Path

@dataclass
class Alert:
    """Alert configuration and state."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    message_template: str
    severity: str
    cooldown: timedelta
    last_triggered: Optional[datetime] = None

@dataclass
class AlertNotification:
    """Alert notification details."""
    alert_name: str
    message: str
    severity: str
    timestamp: datetime
    context: Dict[str, Any]

class AlertManager:
    """Manage system alerts and notifications."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize alert manager with optional config path."""
        self.alerts: Dict[str, Alert] = {}
        self.notifications: List[AlertNotification] = []
        self.config = self._load_config(config_path)
        self._setup_default_alerts()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load alert configuration from file."""
        if not config_path:
            config_path = os.path.join(os.path.dirname(__file__), 'alert_config.json')
        
        if not os.path.exists(config_path):
            # Create default config if it doesn't exist
            default_config = {
                'email': {
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': '',
                    'password': '',
                    'from_address': '',
                    'to_addresses': []
                },
                'thresholds': {
                    'error_rate': 0.1,
                    'response_time': 5.0,
                    'api_latency': 2.0
                }
            }
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _setup_default_alerts(self) -> None:
        """Set up default system alerts."""
        self.add_alert(
            name='high_error_rate',
            condition=lambda metrics: metrics.get('error_rate', 0) > self.config['thresholds']['error_rate'],
            message_template='High error rate detected: {error_rate:.2%}',
            severity='critical',
            cooldown=timedelta(minutes=30)
        )
        
        self.add_alert(
            name='slow_response_time',
            condition=lambda metrics: metrics.get('response_time', {}).get('p95', 0) > self.config['thresholds']['response_time'],
            message_template='Slow response time detected. P95: {response_time[p95]:.2f}s',
            severity='warning',
            cooldown=timedelta(minutes=15)
        )
        
        self.add_alert(
            name='api_latency',
            condition=lambda metrics: any(
                provider['mean'] > self.config['thresholds']['api_latency']
                for provider in metrics.get('api_latency', {}).values()
            ),
            message_template='High API latency detected for providers: {affected_providers}',
            severity='warning',
            cooldown=timedelta(minutes=15)
        )
    
    def add_alert(self, name: str, condition: Callable[[Dict[str, Any]], bool],
                 message_template: str, severity: str, cooldown: timedelta) -> None:
        """Add a new alert configuration."""
        self.alerts[name] = Alert(
            name=name,
            condition=condition,
            message_template=message_template,
            severity=severity,
            cooldown=cooldown
        )
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[AlertNotification]:
        """Check all alerts against current metrics."""
        triggered_alerts = []
        current_time = datetime.now()
        
        for alert in self.alerts.values():
            # Skip if alert is in cooldown
            if (alert.last_triggered and 
                current_time - alert.last_triggered < alert.cooldown):
                continue
            
            try:
                if alert.condition(metrics):
                    # Format message with metrics
                    message = alert.message_template.format(**metrics)
                    
                    # Create notification
                    notification = AlertNotification(
                        alert_name=alert.name,
                        message=message,
                        severity=alert.severity,
                        timestamp=current_time,
                        context=metrics
                    )
                    
                    triggered_alerts.append(notification)
                    self.notifications.append(notification)
                    
                    # Update last triggered time
                    alert.last_triggered = current_time
                    
                    # Send notification
                    self._send_notification(notification)
            except Exception as e:
                print(f"Error checking alert {alert.name}: {str(e)}")
        
        return triggered_alerts
    
    def _send_notification(self, notification: AlertNotification) -> None:
        """Send alert notification via configured channels."""
        # Email notification
        if self.config['email']['to_addresses']:
            try:
                msg = MIMEText(
                    f"Alert: {notification.alert_name}\n"
                    f"Severity: {notification.severity}\n"
                    f"Time: {notification.timestamp}\n"
                    f"Message: {notification.message}\n"
                    f"Context: {json.dumps(notification.context, indent=2)}"
                )
                
                msg['Subject'] = f"[{notification.severity.upper()}] LLM Structure Elucidator Alert"
                msg['From'] = self.config['email']['from_address']
                msg['To'] = ', '.join(self.config['email']['to_addresses'])
                
                with smtplib.SMTP(
                    self.config['email']['smtp_server'],
                    self.config['email']['smtp_port']
                ) as server:
                    server.starttls()
                    if self.config['email']['username'] and self.config['email']['password']:
                        server.login(
                            self.config['email']['username'],
                            self.config['email']['password']
                        )
                    server.send_message(msg)
            except Exception as e:
                print(f"Error sending email notification: {str(e)}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[AlertNotification]:
        """Get list of alerts from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.notifications
            if alert.timestamp >= cutoff_time
        ]


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/models/molecule.py ---
"""
Molecule-related functionality for the LLM Structure Elucidator.
"""
import random
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem
from config.settings import SAMPLE_SMILES

class MoleculeHandler:
    @staticmethod
    def generate_random_molecule():
        """Generate a random molecule from the sample SMILES strings."""
        try:
            # Select a random SMILES string
            smiles = random.choice(SAMPLE_SMILES)
            
            # Create molecule from SMILES
            mol = Chem.MolFromSmiles(smiles)
            
            return mol
        except Exception as e:
            print(f"Error generating molecule: {str(e)}")
            return None

    @staticmethod
    def validate_smiles(smiles):
        """Validate a SMILES string by attempting to create a molecule."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    @staticmethod
    def calculate_molecular_weight(mol):
        """Calculate the molecular weight of a molecule."""
        try:
            return Descriptors.ExactMolWt(mol)
        except:
            return None

    @staticmethod
    def generate_2d_image(mol, size=(300, 300)):
        """Generate a 2D image of a molecule."""
        try:
            return Draw.MolToImage(mol, size=size)
        except Exception as e:
            print(f"Error generating 2D image: {str(e)}")
            return None


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/models/__init__.py ---



--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/__init__.py ---
"""
Agent module exports.
"""

from .base.base_agent import BaseAgent
from .coordinator.coordinator import CoordinatorAgent, AgentType
from .orchestrator.orchestrator import OrchestrationAgent
from .specialized.molecule_plot_agent import MoleculePlotAgent
from .specialized.nmr_plot_agent import NMRPlotAgent
from .specialized.text_response_agent import TextResponseAgent
from .specialized.tool_agent import ToolAgent

__all__ = [
    'BaseAgent',
    'CoordinatorAgent',
    'AgentType',
    'OrchestrationAgent',
    'MoleculePlotAgent',
    'NMRPlotAgent',
    'TextResponseAgent',
    'ToolAgent'
]

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/tools/analysis_tools__.py ---
"""
Tools for analysis-related operations.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
from utils.nmr_utils import generate_nmr_peaks, generate_random_2d_correlation_points

class AnalysisTools:
    """Collection of tools for data analysis and interpretation."""
    
    @staticmethod
    def analyze_nmr_spectrum(peaks: List[Dict[str, float]]) -> Dict[str, Any]:
        """Analyze NMR spectrum peaks."""
        try:
            # Convert peaks to numpy array for analysis
            peak_values = np.array([peak["ppm"] for peak in peaks])
            intensities = np.array([peak["intensity"] for peak in peaks])
            
            return {
                "num_peaks": len(peaks),
                "max_intensity": float(np.max(intensities)),
                "min_intensity": float(np.min(intensities)),
                "mean_intensity": float(np.mean(intensities)),
                "chemical_shift_range": (float(np.min(peak_values)), float(np.max(peak_values)))
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def find_correlations(points: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Analyze 2D correlation points."""
        try:
            # Convert points to numpy arrays
            x_values = np.array([p[0] for p in points])
            y_values = np.array([p[1] for p in points])
            
            # Calculate correlation coefficient
            correlation = float(np.corrcoef(x_values, y_values)[0, 1])
            
            return {
                "num_points": len(points),
                "correlation": correlation,
                "x_range": (float(np.min(x_values)), float(np.max(x_values))),
                "y_range": (float(np.min(y_values)), float(np.max(y_values)))
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def generate_test_data() -> Dict[str, Any]:
        """Generate test data for analysis."""
        try:
            peaks = generate_nmr_peaks()
            correlations = generate_random_2d_correlation_points()
            
            return {
                "peaks": peaks,
                "correlations": correlations,
                "peak_analysis": AnalysisTools.analyze_nmr_spectrum(peaks),
                "correlation_analysis": AnalysisTools.find_correlations(correlations)
            }
        except Exception as e:
            return {"error": str(e)}


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/tools/stout_tool.py ---
"""
STOUT (Structure to IUPAC Name) tool for SMILES/IUPAC name conversion.
"""

import json
import uuid
from pathlib import Path
import subprocess
from typing import Dict, Any, Optional, List, Union
from .data_extraction_tool import DataExtractionTool

class STOUTTool:
    """Tool for converting between SMILES and IUPAC names using STOUT."""
    
    def __init__(self):
        """Initialize the STOUT tool."""
        # Get path relative to this file's location
        self.base_path = Path(__file__).parent.parent.parent
        self.stout_dir = self.base_path / "_temp_folder/stout"
        self.stout_dir.mkdir(parents=True, exist_ok=True)
        self.script_path = Path(__file__).parent.parent / "scripts" / "stout_local.sh"
        self.data_tool = DataExtractionTool()
        self.intermediate_dir = self.base_path / "_temp_folder" / "intermediate_results"

    
    async def convert_smiles_to_iupac(self, smiles: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Convert SMILES to IUPAC name.
        
        Args:
            smiles: The SMILES string to convert
            context: Optional context dictionary
            
        Returns:
            Dictionary containing conversion result or error
        """
        result = await self._run_conversion(smiles, "forward")
        return result
    
    async def convert_iupac_to_smiles(self, iupac: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Convert IUPAC name to SMILES."""
        return await self._run_conversion(iupac, "reverse")
    
    async def process_molecule_batch(self, molecules: List[Dict], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a batch of molecules to get their IUPAC names.
        
        Args:
            molecules: List of dictionaries containing molecule information with SMILES
            context: Optional context dictionary
            
        Returns:
            Dictionary containing batch conversion results
        """
        # Extract sample_id from context
        sample_id = None
        if context and 'sample_id' in context:
            sample_id = context['sample_id']
        elif molecules and isinstance(molecules[0], dict) and 'sample_id' in molecules[0]:
            sample_id = molecules[0]['sample_id']
            
        if not sample_id:
            raise ValueError("No sample_id provided in context or molecules")

        # Load or create intermediate data
        intermediate_data = self._load_or_create_intermediate(sample_id, context)
        
        # Generate unique filenames for batch processing
        job_id = uuid.uuid4().hex
        input_file = self.stout_dir / f"batch_input_{job_id}.json"
        output_file = self.stout_dir / f"batch_output_{job_id}.json"
        
        try:
            # Prepare batch input in required format
            batch_input = []
            for molecule in molecules:
                smiles = molecule.get('SMILES') or molecule.get('smiles')
                if smiles:
                    batch_input.append({
                        'smiles': smiles,
                        'molecule_id': molecule.get('sample_id'),
                        'original_data': molecule
                    })
            
            if not batch_input:
                return {
                    'status': 'error',
                    'error': 'No valid SMILES found in input molecules'
                }
            
            # Write batch input to file
            input_file.write_text(json.dumps(batch_input))
            
            # Run conversion script with batch mode
            subprocess.run(
                [str(self.script_path), str(input_file), str(output_file), 'forward', '--batch'],
                check=True,
                timeout=30  # Align timeout with shell script
            )
            
            # Read and parse results
            results = json.loads(output_file.read_text())
            
            # Update original molecules with IUPAC names and save to intermediate file
            for result, orig_molecule in zip(results, molecules):
                if result['status'] == 'success':
                    orig_molecule['iupac_name'] = result['result']
                    # intermediate_data['step_outputs']['stout'][orig_molecule.get('sample_id')] = {
                    #     'smiles': orig_molecule.get('smiles'),
                    #     'iupac_name': result['result'],
                    #     'status': 'success'
                    # }
                else:
                    orig_molecule['iupac_error'] = result.get('error', 'Unknown conversion error')
                    # intermediate_data['step_outputs']['stout'][orig_molecule.get('sample_id')] = {
                    #     'smiles': orig_molecule.get('smiles'),
                    #     'iupac_name': None,
                    #     'status': 'error'
                    # }
            
            # Save updated intermediate data
            self._save_intermediate(sample_id, intermediate_data)
            
            return {
                'status': 'success',
                'results': results,
                'updated_molecules': molecules  # Return updated molecules for caller's use
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'error',
                'error': 'Batch conversion timed out after 30 seconds'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': f'Batch processing failed: {str(e)}'
            }
        finally:
            # Cleanup
            input_file.unlink(missing_ok=True)
            output_file.unlink(missing_ok=True)

    async def _run_conversion(self, input_str: str, mode: str) -> Dict[str, Any]:
        """Run the conversion process."""
        # Generate unique filenames
        job_id = uuid.uuid4().hex
        input_file = self.stout_dir / f"input_{job_id}.txt"
        output_file = self.stout_dir / f"output_{job_id}.json"
        
        try:
            # Write input to file
            input_file.write_text(input_str)
            
            # Run conversion script
            subprocess.run(
                [str(self.script_path), str(input_file), str(output_file), mode],
                check=True,
                timeout=35
            )
            
            # Read and parse result
            result = json.loads(output_file.read_text())
            return result
            
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": "Conversion timed out",
                "mode": mode
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "mode": mode
            }
        finally:
            # Cleanup
            input_file.unlink(missing_ok=True)
            output_file.unlink(missing_ok=True)

    async def _load_or_create_intermediate(self, sample_id: str, context: Dict[str, Any] = None) -> Dict:
        """Load existing intermediate file or create new one from master data."""
        intermediate_path = self._get_intermediate_path(sample_id)
        
        # Try loading existing intermediate
        if intermediate_path.exists():
            with open(intermediate_path, 'r') as f:
                data = json.load(f)
                if 'molecule_data' not in data:
                    data['molecule_data'] = {}
                # if 'step_outputs' not in data:
                #     data['step_outputs'] = {}
                # if 'stout' not in data['step_outputs']:
                #     data['step_outputs']['stout'] = {}
                return data
                
        # If no intermediate exists, try getting data from master file
        master_path = self.base_path / "data" / "molecular_data" / "molecular_data.json"
        if master_path.exists():
            with open(master_path, 'r') as f:
                master_data = json.load(f)
                if sample_id in master_data:
                    # Create new intermediate with this sample's data
                    intermediate_data = {
                        'molecule_data': master_data[sample_id],
                        # 'step_outputs': {'stout': {}}
                    }
                    self._save_intermediate(sample_id, intermediate_data)
                    return intermediate_data
        
        # Create empty intermediate if no data found
        intermediate_data = {
            'molecule_data': {},
            # 'step_outputs': {'stout': {}}
        }
        self._save_intermediate(sample_id, intermediate_data)
        return intermediate_data

    def _save_intermediate(self, sample_id: str, data: Dict) -> None:
        """Save data to intermediate file."""
        intermediate_path = self._get_intermediate_path(sample_id)
        with open(intermediate_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _get_intermediate_path(self, sample_id: str) -> Path:
        """Get path to intermediate file for a sample."""
        return self.intermediate_dir / f"{sample_id}_intermediate.json"


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/tools/mol2mol_tool.py ---
"""Tool for generating molecular analogues using Mol2Mol network."""
import os
import logging
import shutil
import pandas as pd
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess
import json

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.sbatch_utils import execute_sbatch, wait_for_job_completion
from models.molecule import MoleculeHandler

# Constants for paths and directories
BASE_DIR = Path(__file__).parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "agents" / "scripts"
TEMP_DIR = BASE_DIR / "_temp_folder"
SBATCH_SCRIPT = SCRIPTS_DIR / "mol2mol_sbatch.sh"
LOCAL_SCRIPT = SCRIPTS_DIR / "mol2mol_local.sh"

# Constants for Mol2Mol execution
MOL2MOL_OUTPUT_CHECK_INTERVAL = 5  # seconds
MOL2MOL_OUTPUT_TIMEOUT = 600  # 10 minutes
MOL2MOL_OUTPUT_FILE = "generated_molecules.csv"
MOL2MOL_RUNNING_LOCK = "mol2mol_running.lock"
MOL2MOL_COMPLETE_LOCK = "mol2mol_complete.lock"
MOL2MOL_INPUT_FILENAME = "mol2mol_selection.csv"


class Mol2MolTool:
    """Tool for generating molecular analogues using Mol2Mol network."""
    
    def __init__(self):
        """Initialize the Mol2Mol tool with required directories."""
        # Ensure required directories exist
        self.scripts_dir = SCRIPTS_DIR
        
        # Create directories if they don't exist
        self.temp_dir = TEMP_DIR
        self.temp_dir.mkdir(exist_ok=True)
        self.intermediate_dir = self.temp_dir / "intermediate_results"
        self.intermediate_dir.mkdir(exist_ok=True)
        
        # Validate script existence
        if not SBATCH_SCRIPT.exists():
            raise FileNotFoundError(f"Required SBATCH script not found at {SBATCH_SCRIPT}")
        
        # Validate local script existence
        if not LOCAL_SCRIPT.exists():
            raise FileNotFoundError(f"Required LOCAL script not found at {LOCAL_SCRIPT}")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def _prepare_input_file(self, smiles: str, sample_id: str = None) -> Path:
        """Prepare input file for Mol2Mol.
        
        Args:
            smiles: SMILES string of input molecule
            sample_id: Optional sample ID
            
        Returns:
            Path to created input file
        """
        # Validate SMILES
        if not MoleculeHandler.validate_smiles(smiles):
            raise ValueError(f"Invalid SMILES string: {smiles}")
            
        # Create input DataFrame
        input_data = {
            'SMILES': [smiles],
            'sample-id': [sample_id if sample_id else 'MOL_1']
        }
        df = pd.DataFrame(input_data)
        
        # Save to CSV
        input_file = self.temp_dir / MOL2MOL_INPUT_FILENAME
        await asyncio.to_thread(df.to_csv, input_file, index=False)
        self.logger.info(f"Created input file at: {input_file}")
        
        return input_file

    async def _wait_for_output(self, run_id: str) -> Path:
        """Wait for Mol2Mol generation to complete and return output file path.
        
        Args:
            run_id: Unique identifier for this run (not used anymore, kept for compatibility)
            
        Returns:
            Path to the output file
            
        Raises:
            TimeoutError: If generation doesn't complete within timeout period
        """
        start_time = datetime.now()
        output_file = self.temp_dir / MOL2MOL_OUTPUT_FILE
        running_lock = self.temp_dir / MOL2MOL_RUNNING_LOCK
        complete_lock = self.temp_dir / MOL2MOL_COMPLETE_LOCK
        
        # Create running lock file
        running_lock.touch()
        self.logger.info(f"Created running lock file at: {running_lock}")
        
        while True:
            # Check if complete lock exists and running lock is gone
            if complete_lock.exists() and not running_lock.exists():
                if output_file.exists():
                    try:
                        # Validate output file
                        df = pd.read_csv(output_file)
                        if not df.empty:
                            self.logger.info(f"Found valid output file at: {output_file}")
                            # Clean up lock files
                            if complete_lock.exists():
                                complete_lock.unlink()
                            return output_file
                    except Exception as e:
                        self.logger.warning(f"Found incomplete or invalid output file: {e}")
            
            # Check timeout
            if (datetime.now() - start_time).total_seconds() > MOL2MOL_OUTPUT_TIMEOUT:
                # Clean up lock files
                if running_lock.exists():
                    running_lock.unlink()
                if complete_lock.exists():
                    complete_lock.unlink()
                raise TimeoutError("Timeout waiting for Mol2Mol generation to complete")
            
            # Wait before next check
            await asyncio.sleep(MOL2MOL_OUTPUT_CHECK_INTERVAL)

    # async def _check_existing_predictions(self, molecule_id: str) -> Optional[List[str]]:
    #     """Check if mol2mol predictions already exist for a given molecule ID.
        
    #     Args:
    #         molecule_id: ID of the molecule to check
            
    #     Returns:
    #         List of predicted SMILES if they exist, None otherwise
    #     """
    #     try:
    #         master_data_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
    #         if not master_data_path.exists():
    #             return None
            
    #         # Read master data
    #         with open(master_data_path, 'r') as f:
    #             master_data = json.load(f)
            
    #         # Check if molecule exists and has predictions
    #         if molecule_id in master_data and 'mol2mol_predictions' in master_data[molecule_id]:
    #             predictions = master_data[molecule_id]['mol2mol_predictions']
    #             if predictions:  # Only return if there are actual predictions
    #                 return predictions
    #         return None
            
    #     except Exception as e:
    #         self.logger.error(f"Error checking existing predictions: {str(e)}")
    #         return None

    async def generate_analogues(self, smiles: str, sample_id: str = None) -> Dict[str, Any]:
        """Generate analogues for a given SMILES string.
        
        Args:
            smiles: Input SMILES string
            sample_id: Optional sample ID
            
        Returns:
            Dict containing status and results/error message
        """
        self.logger.warning("mol2mol_tool.generate_analogues execute")

        try:
            if not sample_id:
                raise ValueError("No sample_id provided")

            # Load or create intermediate data
            context = {'smiles': smiles, 'sample_id': sample_id}
            intermediate_data = self._load_or_create_intermediate(sample_id, context)
            
            # Check if predictions already exist
            if ('molecule_data' in intermediate_data and 
                'mol2mol_results' in intermediate_data['molecule_data'] and
                intermediate_data['molecule_data']['mol2mol_results']['status'] == 'success'):
                self.logger.info(f"Found existing mol2mol predictions for sample {sample_id}")
                return {
                    'status': 'success',
                    'message': 'Mol2mol predictions already exist',
                    'predictions': next(iter(intermediate_data['molecule_data']['mol2mol_results']['generated_analogues_target'].values()))
                }

            # Create unique output filename
            output_filename = f"generated_molecules_{sample_id}.csv"
            output_file = self.temp_dir / output_filename
            
            # Create input file
            input_file = await self._prepare_input_file(smiles, sample_id)
            
            # Check if using SLURM
            use_slurm = False  # Default to local execution
            
            # Check CUDA availability for local execution
            try:
                import torch
                if not torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID') is None:
                    self.logger.warning("CUDA not available. Switching to SLURM execution.")
                    use_slurm = True
            except ImportError:
                self.logger.warning("PyTorch not found. Switching to SLURM execution.")
                use_slurm = True
            
            if use_slurm:
                # Submit SLURM job
                self.logger.info("Submitting Mol2Mol SLURM job")
                try:
                    process = await asyncio.create_subprocess_exec(
                        'sbatch',
                        str(SBATCH_SCRIPT),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode != 0:
                        return {
                            'status': 'error',
                            'message': f'SLURM submission failed: {stderr.decode()}'
                        }
                        
                    # Extract job ID from sbatch output
                    job_id = stdout.decode().strip().split()[-1]
                    self.logger.info(f"SLURM job submitted with ID: {job_id}")
                    
                    # Wait for job completion
                    output_file = await self._wait_for_output(job_id)
                    
                except Exception as e:
                    return {
                        'status': 'error',
                        'message': f'SLURM execution failed: {str(e)}'
                    }
            else:
                # Execute locally
                self.logger.info("Running Mol2Mol locally")
                LOCAL_SCRIPT.chmod(0o755)  # Make script executable
                try:
                    process = await asyncio.create_subprocess_exec(
                        str(LOCAL_SCRIPT),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()  # Wait for process to finish
                    
                    if process.returncode != 0:
                        return {
                            'status': 'error',
                            'message': f'Local execution failed: {stderr.decode()}'
                        }
                    
                    # Process finished, check output file
                    temp_output = self.temp_dir / MOL2MOL_OUTPUT_FILE
                    if not temp_output.exists():
                        return {
                            'status': 'error',
                            'message': f'Output file not found at {temp_output}'
                        }
                    
                    # If using unique filename, rename the output file
                    if output_filename != MOL2MOL_OUTPUT_FILE:
                        temp_output.rename(output_file)
                    
                    # Validate output file
                    try:
                        df = pd.read_csv(output_file)
                        if df.empty:
                            return {
                                'status': 'error',
                                'message': 'Generated file is empty'
                            }
                    except Exception as e:
                        return {
                            'status': 'error',
                            'message': f'Failed to read output file: {str(e)}'
                        }
                        
                except Exception as e:
                    return {
                        'status': 'error',
                        'message': f'Local execution failed: {str(e)}'
                    }
            
            # Read predictions and store in intermediate file
            predictions_df = pd.read_csv(output_file)
            # Extract predictions - first row contains input SMILES, subsequent rows are predictions
            sample_id_col = next(iter(predictions_df.columns))  # Get first column name (sample ID)
            predictions = predictions_df[sample_id_col][1:].tolist()  # Skip first row (input SMILES)
            self.logger.info(f"Generated predictions for sample {sample_id}: {predictions}")
            # self.logger.info(f"Intermediate data for sample {sample_id}: {json.dumps(intermediate_data, indent=2)}")
            
            # Store predictions in intermediate data under molecule_data
            if 'molecule_data' not in intermediate_data:
                intermediate_data['molecule_data'] = {}
                
            intermediate_data['molecule_data']['mol2mol_results'] = {
                'generated_analogues_target': {
                    smiles: predictions  # Target SMILES -> list of generated analogues
                },
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            self._save_intermediate(sample_id, intermediate_data)
            
            # Clean up temporary files
            if input_file.exists():
                input_file.unlink()
            if output_file.exists():
                output_file.unlink()
            
            return {
                'status': 'success',
                'message': 'Successfully generated analogues',
                'predictions': predictions
            }
            
        except Exception as e:
            error_msg = f"Error in mol2mol generation: {str(e)}"
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg}


    def _get_intermediate_path(self, sample_id: str) -> Path:
        """Get path to intermediate file for a sample."""
        return self.intermediate_dir / f"{sample_id}_intermediate.json"


    def _load_or_create_intermediate(self, sample_id: str, context: Dict[str, Any] = None) -> Dict:
        """Load existing intermediate file or create new one from master data.
        """
        intermediate_path = self._get_intermediate_path(sample_id)
        
        # Try loading existing intermediate
        if intermediate_path.exists():
            with open(intermediate_path, 'r') as f:
                data = json.load(f)
                if 'molecule_data' not in data:
                    data['molecule_data'] = {}
                return data
                
        # If no intermediate exists, try getting data from master file
        master_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
        if master_path.exists():
            with open(master_path, 'r') as f:
                master_data = json.load(f)

                if sample_id in master_data:
                    # Create new intermediate with just this sample's data
                    intermediate_data = master_data[sample_id]
                    self._save_intermediate(sample_id, intermediate_data)
                    return intermediate_data
                    
        # If neither exists and we have context, create new intermediate
        if context:
            intermediate_data["molecule_data"] = context
            self._save_intermediate(sample_id, intermediate_data)
            return intermediate_data
            
        raise ValueError(f"No data found for sample {sample_id}")

    def _save_intermediate(self, sample_id: str, data: Dict):
        """Save data to intermediate file.
        
        Args:
            sample_id: ID of the sample to save data for
            data: Dictionary containing data to save
        """
        intermediate_path = self._get_intermediate_path(sample_id)
        with open(intermediate_path, 'w') as f:
            json.dump(data, f, indent=2)

    # async def _update_master_data(self, analogues_file: Path, molecule_id: str) -> None:
    #     """Update the master data file with generated molecular analogues.
        
    #     Args:
    #         analogues_file: Path to the file containing generated analogues
    #         molecule_id: ID of the molecule to update
            
    #     Raises:
    #         FileNotFoundError: If master data file doesn't exist
    #         ValueError: If analogues file is empty or invalid
    #     """
    #     try:
    #         self.logger.info(f"Starting master data update for molecule {molecule_id}")
    #         master_data_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
            
    #         if not master_data_path.exists():
    #             error_msg = f"Master data file not found at {master_data_path}. Please upload a CSV file first."
    #             self.logger.error(error_msg)
    #             raise FileNotFoundError(error_msg)
                
    #         self.logger.info(f"Reading existing master data from {master_data_path}")
    #         with open(master_data_path, 'r') as f:
    #             master_data = json.load(f)
            
    #         self.logger.info(f"Reading generated analogues from {analogues_file}")
    #         # Read generated analogues directly from file
    #         with open(analogues_file, 'r') as f:
    #             # Skip empty lines and strip whitespace
    #             generated_smiles = [line.strip() for line in f if line.strip()]
                
    #         self.logger.info(f"Found {len(generated_smiles)} generated SMILES")
            
    #         # Initialize molecule entry if it doesn't exist
    #         if molecule_id not in master_data:
    #             master_data[molecule_id] = {}
            
    #         # Store the predictions
    #         master_data[molecule_id]['mol2mol_predictions'] = generated_smiles
    #         self.logger.info(f"Added {len(generated_smiles)} predictions for molecule {molecule_id}")
            
    #         # Write updated data back to file
    #         self.logger.info("Writing updated master data")
    #         with open(master_data_path, 'w') as f:
    #             json.dump(master_data, f, indent=2)
    #         self.logger.info("Successfully updated master data file")
            
    #         # Delete the generated molecules file to prevent accidental reuse
    #         if analogues_file.exists():
    #             analogues_file.unlink()
    #             self.logger.info(f"Deleted generated molecules file: {analogues_file}")
                
    #     except Exception as e:
    #         self.logger.error(f"Error updating master data: {str(e)}", exc_info=True)
    #         raise

    # async def process_batch(self, molecules: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    #     """Process a batch of molecules.
        
    #     Args:
    #         molecules: List of dictionaries containing 'SMILES' and optionally 'sample_id'
            
    #     Returns:
    #         List of generation results for each molecule
    #     """
    #     results = []
    #     for mol in molecules:
    #         try:
    #             # Extract data
    #             smiles = mol['SMILES']
    #             sample_id = mol.get('sample_id')
                
    #             # Generate analogues
    #             result = await self.generate_analogues(smiles, sample_id)
    #             results.append(result)
                
    #         except Exception as e:
    #             results.append({
    #                 'status': 'error',
    #                 'message': f'Failed to process molecule: {str(e)}',
    #                 'smiles': mol.get('SMILES', 'unknown')
    #             })
                
    #     return results

    # async def process_all_molecules(self) -> Dict[str, Any]:
    #     """Process all molecules in the master data file for mol2mol predictions.
        
    #     Returns:
    #         Dictionary containing processing results
    #     """
    #     try:
    #         master_data_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
    #         if not master_data_path.exists():
    #             return {'status': 'error', 'message': 'Master data file not found'}
            
    #         # Read master data
    #         with open(master_data_path, 'r') as f:
    #             master_data = json.load(f)
            
    #         results = []
    #         # Process each molecule
    #         for molecule_id, molecule_data in master_data.items():
    #             if 'smiles' in molecule_data:
    #                 # Prepare molecule data for generation
    #                 mol_input = {
    #                     'SMILES': molecule_data['smiles'],
    #                     'sample_id': molecule_id,
    #                     'name': molecule_id
    #                 }
                    
    #                 # Generate analogues with unique filename
    #                 generation_result = await self.generate_analogues(mol_input['SMILES'], mol_input['sample_id'])
                    
    #                 if generation_result['status'] == 'success':
    #                     # Update master data with predictions using the output file path from generation
    #                     output_file = Path(generation_result['output_file'])
    #                     await self._update_master_data(output_file, molecule_id)
    #                     results.append({
    #                         'molecule_id': molecule_id,
    #                         'status': 'success'
    #                     })
    #                 else:
    #                     results.append({
    #                         'molecule_id': molecule_id,
    #                         'status': 'error',
    #                         'message': generation_result['message']
    #                     })
            
    #         return {
    #             'status': 'success',
    #             'message': f'Processed {len(results)} molecules',
    #             'results': results
    #         }
            
    #     except Exception as e:
    #         self.logger.error(f"Error processing molecules: {str(e)}")
    #         return {
    #             'status': 'error',
    #             'message': str(e)
    #         }


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/tools/nmr_simulation_tool.py ---
"""Tool for simulating NMR spectra from molecular structures using SGNN."""
import os
import logging
import shutil
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import sys
import ast
import subprocess
import json
import time
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.sbatch_utils import execute_sbatch, wait_for_job_completion
import asyncio
import os   

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # This will force the configuration even if logging was already configured
)
logger = logging.getLogger(__name__)

# Constants for paths and directories
BASE_DIR = Path(__file__).parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "agents" / "scripts"
SIMULATIONS_DIR = BASE_DIR / "_temp_folder"
INTERMEDIATE_DIR = SIMULATIONS_DIR / "intermediate_results"
SGNN_DATA_DIR = BASE_DIR / "_temp_folder"
SBATCH_SCRIPT = SCRIPTS_DIR / "sgnn_sbatch.sh"
LOCAL_SCRIPT = SCRIPTS_DIR / "sgnn_local.sh"
SGNN_INPUT_FILENAME = "current_molecule.csv"

# Constants for SGNN output processing
SGNN_OUTPUT_TYPES = ['1H', '13C', 'COSY', 'HSQC']
SGNN_OUTPUT_CHECK_INTERVAL = 5  # seconds
SGNN_OUTPUT_TIMEOUT = 300  # 5 minutes to wait for output files
SGNN_OUTPUT_PATTERN = "nmr_prediction_{type}.csv"  # e.g., nmr_prediction_1H.csv
SGNN_TIMEOUT = 300  # 5 minutes to wait for output files

class NMRSimulationTool:
    """Tool for simulating NMR spectra from molecular structures using SGNN."""
    
    def __init__(self):
        """Initialize the NMR simulation tool with required directories."""
        # Ensure required directories exist
        self.scripts_dir = SCRIPTS_DIR
        self.simulations_dir = SIMULATIONS_DIR
        self.sgnn_data_dir = SGNN_DATA_DIR
        self.intermediate_dir = INTERMEDIATE_DIR
        
        # Create directories if they don't exist
        self.simulations_dir.mkdir(exist_ok=True)
        self.scripts_dir.mkdir(exist_ok=True)
        self.intermediate_dir.mkdir(exist_ok=True, parents=True)  # parents=True to create parent dirs if needed
        
        # Validate script existence
        if not SBATCH_SCRIPT.exists():
            raise FileNotFoundError(f"Required SBATCH script not found at {SBATCH_SCRIPT}")
        if not LOCAL_SCRIPT.exists():
            raise FileNotFoundError(f"Required local script not found at {LOCAL_SCRIPT}")
            
        # Validate environment
        try:
            import torch
            if not torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID') is None:
                self.logger.warning("CUDA not available for local execution. SLURM execution will be forced.")
        except ImportError:
            self.logger.warning("PyTorch not found. Please ensure the SGNN environment is activated.")
        
        # Set up logger
        self.logger = logging.getLogger(__name__)

    async def _wait_for_sgnn_outputs(self, smiles: str) -> Dict[str, Path]:
        """Wait for SGNN outputs to be generated.
        
        Args:
            smiles: SMILES string of molecule
            
        Returns:
            Dictionary mapping NMR type to output file path
        """
        output_files = {}
        start_time = time.time()
        

        # Wait for new output files
        while True:
            # Check each type of NMR output
            for nmr_type in SGNN_OUTPUT_TYPES:
                output_file = self.sgnn_data_dir / SGNN_OUTPUT_PATTERN.format(type=nmr_type)
                if output_file.exists() and output_file not in output_files.values():
                    self.logger.info(f"Found valid output for {nmr_type} NMR")
                    output_files[nmr_type] = output_file
            
            # Check if we have all outputs
            if len(output_files) == len(SGNN_OUTPUT_TYPES):
                break
                
            # Check timeout
            if time.time() - start_time > SGNN_TIMEOUT:
                raise TimeoutError("Timeout waiting for SGNN outputs")
                
            time.sleep(0.1)
            
        return output_files

    async def _prepare_input_data(self, molecule_data: str, simulation_mode: str) -> pd.DataFrame:
        """Prepare input data for NMR simulation from master JSON file.
        
        Args:
            molecule_data: Path to master JSON file
            simulation_mode: Always 'batch' mode for efficiency
            
        Returns:
            DataFrame with SMILES and sample-id columns
        """
        if not os.path.exists(molecule_data):
            raise FileNotFoundError(f"Master JSON file not found: {molecule_data}")
            
        self.logger.info(f"Loading molecular data from master JSON: {molecule_data}")
        
        try:
            # Read master JSON file
            with open(molecule_data, 'r') as f:
                master_data = json.load(f)
                
            # Extract SMILES and sample IDs
            simulation_input = []
            for sample_id, data in master_data.items():
                if 'smiles' in data:  # Make sure we have SMILES data
                    simulation_input.append({
                        'SMILES': data['smiles'],
                        'sample-id': sample_id  # Use the exact sample ID from master data
                    })
                    
            if not simulation_input:
                raise ValueError("No valid molecules found in master JSON file")
                
            df = pd.DataFrame(simulation_input)
            self.logger.info(f"Prepared simulation input for {len(df)} molecules")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to prepare simulation input from JSON: {str(e)}")
            raise

    async def simulate_batch(self, master_data_path: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Simulate NMR spectrum for molecules in master JSON file.
        
        Args:
            master_data_path: Path to master JSON file containing molecular data
            context: Optional context including:
                - use_slurm: Whether to use SLURM (default: False)
            
        Returns:
            Dictionary containing simulation results
        """
        try:
            self.logger.info("Starting NMR simulation process")
            context = context or {}
                

                    
            # Always use batch mode
            simulation_mode = 'batch'
            
            # Check CUDA availability for local execution
            use_slurm = context.get('use_slurm', False)
            if not use_slurm:
                try:
                    import torch
                    if not torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID') is None:
                        self.logger.warning("CUDA not available. Switching to SLURM execution.")
                        use_slurm = True
                except ImportError:
                    self.logger.warning("PyTorch not found. Switching to SLURM execution.")
                    use_slurm = True
            
            # Prepare input data from master JSON
            try:
                df = await self._prepare_input_data(master_data_path, simulation_mode)
                self.logger.info(f"Prepared input data with {len(df)} molecules from master JSON")
            except Exception as e:
                raise ValueError(f"Failed to prepare input data from master JSON: {str(e)}")
            
            try:
                self.logger.info("Input validation")
                
                # Ensure temp directory exists
                self.sgnn_data_dir.mkdir(exist_ok=True)
                
                # Define target path for SGNN input
                sgnn_input_path = self.sgnn_data_dir / SGNN_INPUT_FILENAME
                
                # Copy input file to SGNN location with fixed name, replacing if exists
                self.logger.info(f"Copying input file to SGNN location: {sgnn_input_path}")
                if sgnn_input_path.exists():
                    self.logger.info(f"Removing existing file at {sgnn_input_path}")
                    sgnn_input_path.unlink()  # Explicitly remove existing file
                df.to_csv(sgnn_input_path, index=False)

                # Validate file content
                try:
                    df = pd.read_csv(sgnn_input_path)
                    if df.empty:
                        raise ValueError("Input CSV file is empty")
                except Exception as e:
                    raise ValueError(f"Invalid CSV file format: {str(e)}")
                
                self.logger.info("Input file processed successfully")
                
                # Create timestamp for unique output directory
                timestamp = datetime.now().strftime("%H%M%S")
                output_dir = self.sgnn_data_dir / f"nmr_output_{timestamp}"
                output_dir.mkdir(exist_ok=True)
                
                # Store original filename for later use (without extension)
                self.current_molecule_name = sgnn_input_path.stem
                
                # Determine execution mode
                if use_slurm:
                    # Execute using SLURM
                    self.logger.info("Running NMR simulation with SLURM")
                    # Pass arguments individually to execute_sbatch
                    self.logger.info(f"Running SLURM with input file: {sgnn_input_path}")
                    job_id = await execute_sbatch(str(SBATCH_SCRIPT), "--input_file", str(sgnn_input_path))
                    success = await wait_for_job_completion(job_id)
                    if not success:
                        self.logger.error("SLURM job failed")
                        return {
                            'status': 'error',
                            'message': 'SLURM job failed'
                        }
                else:
                    # Execute locally
                    self.logger.info("Running NMR simulation locally")
                    cmd = [str(LOCAL_SCRIPT), "--input_file", str(sgnn_input_path)]
                    self.logger.info(f"Running command: {' '.join(cmd)}")
                    try:
                        subprocess.run(cmd, check=True)
                    except subprocess.CalledProcessError as e:
                        self.logger.error(f"Local execution failed with error: {str(e)}")
                        return {
                            'status': 'error',
                            'message': f'Local execution failed: {str(e)}'
                        }
                
                self.logger.info("NMR simulation completed successfully")
                
                # Wait for and validate output files
                try:
                    output_files = await self._wait_for_sgnn_outputs(df['SMILES'].iloc[0])
                    self.logger.info(f"Found all required output files: {list(output_files.keys())}")
                    
                    # Compile results into single file
                    result_path = await self._compile_results(output_files)
                    
                    # Update master data with simulation results
                    # await self._update_master_data(result_path)

                    # Clean up temporary directories
                    try:
                        # Remove timestamp-based output directory
                        if output_dir.exists():
                            shutil.rmtree(output_dir)
                            self.logger.info(f"Cleaned up temporary output directory: {output_dir}")
                    except Exception as e:
                        self.logger.warning(f"Error during cleanup: {str(e)}")

                    return {
                        "status": "success",
                        "type": "nmr_prediction",
                        "data": {
                            "message": "NMR simulation completed and results compiled",
                            "result_file": str(result_path),
                            "output_files": {k: str(v) for k, v in output_files.items()}
                        }
                    }
                    
                except TimeoutError:
                    return {
                        'status': 'error',
                        'message': 'Timeout waiting for output files'
                    }
                except Exception as e:
                    return {
                        'status': 'error',
                        'message': f'Failed to process output files: {str(e)}'
                    }
            
            except Exception as e:
                return {
                    'status': 'error',
                    'message': f'NMR simulation failed: {str(e)}'
                }
                
        except Exception as e:
            self.logger.error(f"Error in NMR simulation: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    async def _compile_results(self, output_files: Dict[str, Path]) -> Path:
        try:
            input_file = self.sgnn_data_dir / SGNN_INPUT_FILENAME
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
                
            # Read and validate input file
            input_df = pd.read_csv(input_file)
            if input_df.empty:
                raise pd.errors.EmptyDataError("Input file is empty")

            # Read and validate each NMR prediction file
            for nmr_type, file_path in output_files.items():
                if not file_path.exists():
                    self.logger.warning(f"NMR prediction file not found: {file_path}")
                    continue
                    
                df = pd.read_csv(file_path)
                if df.empty:
                    self.logger.warning(f"NMR prediction file is empty: {file_path}")
                    continue
                    
                if 'shifts' not in df.columns or 'sample-id' not in df.columns:
                    self.logger.warning(f"Missing required columns in {nmr_type} NMR predictions")
                    continue
                    
                # Create a dictionary to store processed shifts by sample ID
                processed_shifts_dict = {}
                
                # Process each row
                for _, row in df.iterrows():
                    sample_id = row['sample-id']
                    shift_string = row['shifts']
                    
                    try:
                        # Convert string representation to actual list using ast.literal_eval
                        shift_list = ast.literal_eval(shift_string)
                        
                        # Handle different NMR types
                        if nmr_type == '1H':
                            # 1H NMR format: keep both shift and intensity as tuples
                            shift_values = [(float(tup[0]), float(tup[1])) for tup in shift_list]
                        elif nmr_type in ['COSY', 'HSQC']:
                            # COSY and HSQC format: keep correlation pairs as tuples
                            shift_values = [(float(tup[0]), float(tup[1])) for tup in shift_list]
                        else:
                            # 13C NMR format: direct list of float values
                            shift_values = [float(val) for val in shift_list]
                            
                        processed_shifts_dict[sample_id] = shift_values
                    except Exception as e:
                        self.logger.warning(f"Error processing shift value for {sample_id}: {str(e)}")
                        processed_shifts_dict[sample_id] = []

                # Add to input DataFrame with correct column name, using empty list for missing predictions
                column_name = f"{nmr_type}_NMR_sim" if nmr_type in ['1H', '13C'] else f"{nmr_type}_sim"
                # Create a new Series with the same index as input_df
                shifts_series = pd.Series(index=input_df.index, dtype=object)
                for idx, sample_id in enumerate(input_df['sample-id']):
                    shifts_series[idx] = processed_shifts_dict.get(sample_id, [])
                input_df[column_name] = shifts_series
                
            # Create output path in simulations directory
            self.simulations_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.simulations_dir / f"{self.current_molecule_name}_sim.csv"
        
            # Save combined results
            input_df.to_csv(output_path, index=False)
            self.logger.info(f"Saved compiled results to {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error compiling results: {str(e)}")
            raise ValueError(f"Failed to compile NMR prediction results: {str(e)}")

    # async def _update_master_data(self, simulation_results_file: Path) -> None:
    #     """Update the master data file with NMR simulation results.
        
    #     Args:
    #         simulation_results_file: Path to the CSV containing NMR simulation results
            
    #     The CSV file contains:
    #     - sample-id: Used to identify the molecule in the master JSON
    #     - NMR simulation columns (1H_NMR_sim, 13C_NMR_sim, COSY_sim, HSQC_sim)
            
    #     Raises:
    #         FileNotFoundError: If master data file doesn't exist
    #         ValueError: If simulation results file is empty or invalid
    #     """
    #     try:
    #         self.logger.info("Starting master data update from simulation results")
    #         master_data_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
            
    #         if not master_data_path.exists():
    #             error_msg = f"Master data file not found at {master_data_path}"
    #             self.logger.error(error_msg)
    #             raise FileNotFoundError(error_msg)
                
    #         if not simulation_results_file.exists():
    #             error_msg = f"Simulation results file not found at {simulation_results_file}"
    #             self.logger.error(error_msg)
    #             raise FileNotFoundError(error_msg)
                
    #         self.logger.info(f"Reading existing master data from {master_data_path}")
    #         with open(master_data_path, 'r') as f:
    #             master_data = json.load(f)
            
    #         self.logger.info(f"Reading NMR simulation results from {simulation_results_file}")
    #         simulation_df = pd.read_csv(simulation_results_file)
            
    #         if simulation_df.empty:
    #             raise ValueError("Simulation results file is empty")
                
    #         # NMR column mapping from CSV to JSON
    #         nmr_mapping = {
    #             '1H_NMR_sim': '1H_sim',
    #             '13C_NMR_sim': '13C_sim',
    #             'COSY_sim': 'COSY_sim',
    #             'HSQC_sim': 'HSQC_sim'
    #         }
            
    #         # Process each row in the simulation results
    #         for idx, row in simulation_df.iterrows():
    #             sample_id = row['sample-id']
    #             self.logger.info(f"Processing NMR data for sample: {sample_id}")
                
    #             # Create entry if it doesn't exist
    #             if sample_id not in master_data:
    #                 master_data[sample_id] = {'nmr_data': {}}
    #             elif 'nmr_data' not in master_data[sample_id]:
    #                 master_data[sample_id]['nmr_data'] = {}
                
    #             # Process each NMR type for this sample
    #             for csv_col, json_key in nmr_mapping.items():
    #                 if csv_col in simulation_df.columns:
    #                     try:
    #                         prediction_data = row[csv_col]
    #                         # Handle empty or missing predictions
    #                         if pd.isna(prediction_data):
    #                             master_data[sample_id]['nmr_data'][json_key] = []
    #                             self.logger.warning(f"No {json_key} predictions for sample {sample_id}")
    #                         else:
    #                             # Convert string representation to list if needed
    #                             if isinstance(prediction_data, str):
    #                                 prediction_data = ast.literal_eval(prediction_data)
    #                             master_data[sample_id]['nmr_data'][json_key] = prediction_data
    #                             self.logger.info(f"Added {json_key} predictions for sample {sample_id}")
    #                     except Exception as e:
    #                         self.logger.warning(f"Failed to process {csv_col} for sample {sample_id}: {str(e)}")
    #                         master_data[sample_id]['nmr_data'][json_key] = []
            
    #         # Write updated data back to file
    #         self.logger.info("Writing updated master data")
    #         with open(master_data_path, 'w') as f:
    #             json.dump(master_data, f, indent=2)
    #         self.logger.info("Successfully updated master data file")
            
    #         # Clean up simulation results file
    #         # if simulation_results_file.exists():
    #         #     simulation_results_file.unlink()
    #         #     self.logger.info(f"Deleted simulation results file: {simulation_results_file}")
                
    #     except Exception as e:
    #         self.logger.error(f"Error updating master data: {str(e)}", exc_info=True)
    #         raise

    async def simulate_nmr(self, sample_id: str, context: Dict[str, Any] = None) -> Dict:
        """Run NMR simulation for a molecule.
        
        Args:
            sample_id: ID of the sample to simulate
            context: Optional context data if not loading from master file
            
        Returns:
            Dictionary containing simulation results
        """
        self.logger.info(f"Selected sample_id: {sample_id}")

        # Delete any existing output files first
        for nmr_type in SGNN_OUTPUT_TYPES:
            output_file = self.sgnn_data_dir / SGNN_OUTPUT_PATTERN.format(type=nmr_type)
            if output_file.exists():
                self.logger.info(f"Removing existing output file for {nmr_type} NMR")
                output_file.unlink()

        # Load or create intermediate file
        intermediate_data = self._load_or_create_intermediate(sample_id, context)
        
        # Get SMILES from molecule data
        smiles = intermediate_data['molecule_data'].get('smiles')
        if not smiles:
            raise ValueError(f"No SMILES found for sample {sample_id}")
            
        # Canonicalize SMILES for consistent comparison
        smiles = self._canonicalize_smiles(smiles)
        
        # Check if we have all required NMR data
        self.logger.info(f"Checking NMR data: {intermediate_data['molecule_data'].get('nmr_data', {})}")
        nmr_data = intermediate_data['molecule_data'].get('nmr_data', {})
        
        # Check for simulation results (not experimental data)
        required_sims = ['1H_exp', '13C_exp', 'COSY_exp', 'HSQC_exp', '1H_sim', '13C_sim', 'COSY_sim', 'HSQC_sim']
        existing_sims = [key for key in required_sims if key in nmr_data]
        self.logger.info(f"Required simulations: {required_sims}")
        self.logger.info(f"Found simulations: {existing_sims}")
        
        if len(existing_sims) == len(required_sims):
            self.logger.info(f"All NMR simulations exist for sample {sample_id}")
            return {
                'status': 'success',
                'message': 'NMR simulations already exist',
                'predictions': nmr_data
            }
        else:
            missing_sims = set(required_sims) - set(existing_sims)
            self.logger.info(f"Missing simulations: {missing_sims}, will run simulation")
        
        # Create input file
        input_file = self.simulations_dir / SGNN_INPUT_FILENAME
        df = pd.DataFrame([{'SMILES': smiles, "sample-id": sample_id}])
        df.to_csv(input_file, index=False)
        
        # Run prediction locally
        try:
            self.logger.info("Running NMR simulation locally")
            self.logger.info(f"Using script: {LOCAL_SCRIPT}")
            self.logger.info(f"Input file: {input_file}")
            
            process = await asyncio.create_subprocess_exec(
                str(LOCAL_SCRIPT),
                f"--input_file={input_file}",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Capture output in real-time
            stdout, stderr = await process.communicate()
            stdout_str = stdout.decode() if stdout else ""
            stderr_str = stderr.decode() if stderr else ""
            
            # Log all output
            if stdout_str:
                self.logger.info(f"SGNN stdout:\n{stdout_str}")
            if stderr_str:
                self.logger.error(f"SGNN stderr:\n{stderr_str}")
            
            if process.returncode != 0:
                error_msg = f"Local execution failed with return code {process.returncode}"
                if stderr_str:
                    error_msg += f": {stderr_str}"
                self.logger.error(error_msg)
                return {'status': 'error', 'message': error_msg}
                
            self.logger.info("SGNN script completed successfully")
            
        except Exception as e:
            error_msg = f"Error during local execution: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {'status': 'error', 'message': error_msg}
        
        # Wait for output files to be generated
        try:
            self.logger.info("Waiting for output files...")
            output_files = await self._wait_for_sgnn_outputs(smiles)
            self.logger.info(f"Found output files: {list(output_files.keys())}")
        except TimeoutError as e:
            error_msg = str(e)
            self.logger.error(error_msg)
            # Check directory contents one last time
            if self.sgnn_data_dir.exists():
                self.logger.error(f"Final contents of output directory {self.sgnn_data_dir}:")
                for f in self.sgnn_data_dir.iterdir():
                    self.logger.error(f"  {f.name} ({f.stat().st_size} bytes)")
            return {'status': 'error', 'message': error_msg}
        
        # Process results and update intermediate file
        nmr_data = {}
        
        for nmr_type, output_file in output_files.items():
            try:
                df = pd.read_csv(output_file)
                self.logger.info(f"Processing {nmr_type} NMR data")
                self.logger.info(f"DataFrame columns: {df.columns.tolist()}")
                self.logger.info(f"DataFrame content:\n{df}")
                
                # Filter for current sample's data
                if 'SMILES' in df.columns:
                    current_sample_data = df[df['SMILES'] == smiles]
                    if current_sample_data.empty:
                        self.logger.error(f"No data found for SMILES {smiles} in {nmr_type} output")
                        continue
                else:
                    current_sample_data = df
                    
                self.logger.info(f"Filtered data for current sample:\n{current_sample_data}")
                
                if nmr_type == '1H':
                    # Convert string representation of shifts to actual list of [shift, intensity] pairs
                    shifts_str = current_sample_data['shifts'].iloc[0]
                    self.logger.info(f"Raw shifts string: {shifts_str}")
                    shifts_list = ast.literal_eval(shifts_str)
                    # Convert tuples to lists and ensure they're floats
                    shifts_list = [[float(shift), float(intensity)] for shift, intensity in shifts_list]
                    nmr_data['1H_sim'] = shifts_list
                    
                elif nmr_type == '13C':
                    # Convert string representation of shifts to list of floats
                    shifts_str = current_sample_data['shifts'].iloc[0]
                    self.logger.info(f"Raw shifts string: {shifts_str}")
                    shifts_list = ast.literal_eval(shifts_str)
                    # Convert to list of floats
                    shifts_list = [float(shift) for shift in shifts_list]
                    nmr_data['13C_sim'] = shifts_list
                    
                elif nmr_type == 'COSY':
                    # Convert string representation of shifts to list of [shift1, shift2] pairs
                    shifts_str = current_sample_data['shifts'].iloc[0]
                    self.logger.info(f"Raw shifts string: {shifts_str}")
                    shifts_list = ast.literal_eval(shifts_str)
                    # Convert tuples to lists and ensure they're floats
                    shifts_list = [[float(shift1), float(shift2)] for shift1, shift2 in shifts_list]
                    nmr_data['COSY_sim'] = shifts_list
                    
                elif nmr_type == 'HSQC':
                    # Convert string representation of shifts to list of [1H_shift, 13C_shift] pairs
                    shifts_str = current_sample_data['shifts'].iloc[0]
                    self.logger.info(f"Raw shifts string: {shifts_str}")
                    shifts_list = ast.literal_eval(shifts_str)
                    # Convert to list of lists and ensure they're floats
                    shifts_list = [[float(h_shift), float(c_shift)] for h_shift, c_shift in shifts_list]
                    nmr_data['HSQC_sim'] = shifts_list
                    
            except Exception as e:
                self.logger.error(f"Error processing {nmr_type} NMR data: {str(e)}")
                self.logger.error(f"Error details:", exc_info=True)
                continue
         
        if not nmr_data:
            raise ValueError("Failed to process any NMR simulation data")
        
        # Update intermediate file with NMR data
        intermediate_data['molecule_data']['nmr_data'].update(nmr_data)
        self._save_intermediate(sample_id, intermediate_data)
        
        return {
            'status': 'success',
            'message': 'Successfully simulated NMR spectra',
            'predictions': nmr_data
        }
        
        
    def _get_intermediate_path(self, sample_id: str) -> Path:
        """Get path to intermediate file for a sample."""
        return self.intermediate_dir / f"{sample_id}_intermediate.json"


    def _load_or_create_intermediate(self, sample_id: str, context: Dict[str, Any] = None) -> Dict:
        """Load existing intermediate file or create new one from master data.
        """
        intermediate_path = self._get_intermediate_path(sample_id)
        
        # Try loading existing intermediate
        if intermediate_path.exists():
            with open(intermediate_path, 'r') as f:
                data = json.load(f)
                if 'molecule_data' not in data:
                    data['molecule_data'] = {}
                return data
                
        # If no intermediate exists, try getting data from master file
        master_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
        if master_path.exists():
            with open(master_path, 'r') as f:
                master_data = json.load(f)

                if sample_id in master_data:
                    # Create new intermediate with just this sample's data
                    intermediate_data = master_data[sample_id]
                    self._save_intermediate(sample_id, intermediate_data)
                    return intermediate_data
                    
        # If neither exists and we have context, create new intermediate
        if context:
            intermediate_data["molecule_data"] = context
            self._save_intermediate(sample_id, intermediate_data)
            return intermediate_data
            
        raise ValueError(f"No data found for sample {sample_id}")

    def _save_intermediate(self, sample_id: str, data: Dict) -> None:
        """Save data to intermediate file."""
        intermediate_path = self._get_intermediate_path(sample_id)
        with open(intermediate_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _canonicalize_smiles(self, smiles: str) -> str:
        """Convert SMILES to canonical form for consistent comparison.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Canonical SMILES string
        """
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.logger.warning(f"Could not parse SMILES: {smiles}")
                return smiles
            return Chem.MolToSmiles(mol, canonical=True)
        except ImportError:
            self.logger.warning("RDKit not available, using raw SMILES")
            return smiles


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/tools/structure_visualization_tool.py ---
"""
Tool for generating and managing molecular structure visualizations.
"""
from __future__ import annotations 
from typing import Dict, Optional, Union, List, Any
from pathlib import Path
import logging
from rdkit import Chem
import os
import math
import numpy as np
from datetime import datetime
from .analysis_enums import DataSource, RankingMetric
from .data_extraction_tool import DataExtractionTool
from .stout_operations import STOUTOperations

# Set up logging
logger = logging.getLogger(__name__)

class StructureVisualizationTool:
    """Tool for generating and managing molecular structure visualizations."""

    def __init__(self):
        """Initialize the structure visualization tool."""
        self.logger = logger  # Use module-level logger
        self.stout_ops = STOUTOperations()

    def _format_experimental_data(self, exp_data: List, spectrum_type: str) -> str:
        """Format experimental NMR data based on spectrum type.
        
        Args:
            exp_data: List of experimental NMR data points
            spectrum_type: Type of NMR spectrum ('13C_exp', 'HSQC_exp', 'COSY_exp', or '1H_exp')
            
        Returns:
            Formatted string representation of the data
        """
        if not exp_data:
            return ""

        if spectrum_type == '13C_exp':  # 13C NMR has single values
            formatted_data = "13C NMR chemical shifts:\n"
            for shift in exp_data:
                formatted_data += f"{shift:.2f} ppm\n"
        elif spectrum_type in ['HSQC_exp', 'COSY_exp']:  # 2D NMR
            formatted_data = f"{spectrum_type} peaks (x, y coordinates):\n"
            for peak in exp_data:
                if isinstance(peak, (list, tuple)) and len(peak) >= 2:
                    formatted_data += f"({peak[0]:.2f} ppm, {peak[1]:.2f} ppm)\n"
        else:  # 1H NMR
            formatted_data = "1H NMR peaks (shift, intensity):\n"
            for peak in exp_data:
                if isinstance(peak, (list, tuple)) and len(peak) >= 2:
                    formatted_data += f"{peak[0]:.2f} ppm (intensity: {peak[1]:.1f})\n"
        
        return formatted_data

    async def _generate_individual_analysis_section(self, molecule_index: int, spectrum_type: str, formatted_data: str, smiles: str = None) -> str:
        """Generate analysis section for an individual molecule."""
        if spectrum_type != 'HSQC_exp':
            return ""  # Skip non-HSQC spectra
        
        iupac_name = "Not available"
        if smiles:
            iupac_result = await self.stout_ops.get_iupac_name(smiles)
            iupac_name = iupac_result.get('iupac_name', 'Not available')
            
        return f"""
                Molecule {molecule_index} Analysis:
                1. Structural Description:
                   - IUPAC Name: {iupac_name}
                   - Describe the overall molecular framework
                   - Identify key functional groups and their positions
                   - Note any distinctive structural features or patterns

                2. Expected HSQC Features:
                   - List the expected HSQC correlations based on structure
                   - Identify characteristic cross-peaks that should be present
                   - Note any unique HSQC patterns this structure should show

                3. Data Matching for Molecule {molecule_index}:
                   - Compare expected HSQC signals with experimental data:
                     {formatted_data}
                   - Identify which HSQC cross-peaks support this structure
                   - Note any missing or unexplained correlations
                """

    async def _generate_analysis_prompt(self, num_candidates: int, spectrum_type: str, formatted_data: str, smiles_list: List[str] = None) -> tuple[str, List[str]]:
        """Generate the complete analysis prompt for spectral evaluation.
        
        Args:
            num_candidates: Number of candidate molecules to analyze
            spectrum_type: Type of NMR spectrum being analyzed
            formatted_data: Formatted experimental data string
            smiles_list: Optional list of SMILES strings for the candidate molecules
            
        Returns:
            Tuple containing:
            - Complete analysis prompt string
            - List of IUPAC names for each molecule
        """
        # Generate individual analysis sections for each molecule
        individual_analysis_sections = []
        iupac_names = []
        for i in range(num_candidates):
            smiles = smiles_list[i] if smiles_list and i < len(smiles_list) else None
            section = await self._generate_individual_analysis_section(i+1, spectrum_type, formatted_data, smiles)
            individual_analysis_sections.append(section)
            
            # Get IUPAC name from the result
            if smiles:
                iupac_result = await self.stout_ops.get_iupac_name(smiles)
                iupac_names.append(iupac_result.get('iupac_name', 'Not available'))
            else:
                iupac_names.append('Not available')

        # Build the complete prompt
        prompt = f"""
                I'm showing you {num_candidates} candidate molecular structures (ranked left to right) and their experimental {spectrum_type} NMR data.

                Part 1: Individual Structure Analysis
                {' '.join(individual_analysis_sections)}

                Part 2: Comprehensive Comparative Analysis
                
                1. Detailed Structure Comparison:
                   a) Systematic Structural Analysis:
                      - For each molecule, provide a detailed breakdown of:
                         * Core scaffold identification and description
                         * Functional group positions and types
                         * Stereochemistry and conformational features
                      - Document exact atom indices for key features
                   
                   b) Comparative Feature Analysis:
                      - For each structural difference identified:
                         * Specify exact atom indices involved
                         * Describe the chemical environment changes
                         * Explain the potential impact on spectral properties
                      - Create a hierarchical list of differences, from most to least significant
                   
                   c) Common Elements Evaluation:
                      - Detail all shared structural motifs:
                         * Core frameworks
                         * Functional group patterns
                         * Stereochemical elements
                      - Explain how these commonalities support or challenge the structural assignments

                2. Evidence-Based Spectral Compatibility Analysis:
                   a) Detailed Ranking Justification:
                      - For each molecule, provide:
                         * Numerical score (1-10) for spectral match
                         * Specific peak assignments supporting the score
                         * Detailed explanation of any mismatches
                   
                   b) Critical Spectral Features:
                      - For each decisive spectral feature:
                         * Exact chemical shift values
                         * Coupling patterns and constants
                         * Correlation with structural elements
                         * Impact on structural validation
                   
                   c) Comparative Spectral Analysis:
                      - Create a feature-by-feature comparison:
                         * Chemical shift patterns
                         * Coupling relationships
                         * Through-space correlations
                      - Explain how each feature discriminates between candidates

                3. Comprehensive Confidence Evaluation:
                   a) Detailed Confidence Assessment:
                      - Provide a numerical confidence score (1-10)
                      - For each point affecting confidence:
                         * Specific evidence supporting the assessment
                         * Weight of the evidence (high/medium/low)
                         * Impact on overall structure determination
                   
                   b) Uncertainty Analysis:
                      - For each identified ambiguity:
                         * Exact location in the structure
                         * Nature of the uncertainty
                         * Impact on structure determination
                         * Potential alternative interpretations
                   
                   c) Data Gap Analysis:
                      - Identify missing experimental data:
                         * Specific experiments needed
                         * Expected information gain
                         * How it would resolve ambiguities
                      - Prioritize additional data needs

                Remember to:
                - Provide exact atom indices for all structural features discussed
                - Support each conclusion with specific spectral evidence
                - Quantify confidence levels for each assessment
                - Make explicit connections between structural features and spectral data
                - Present information in a clear, hierarchical format
                - Be thorough in documenting both supporting and contradicting evidence

                Part 3: Final Evaluation
                1. Structure Comparison:
                   - Compare key structural differences between all molecules
                   - Identify unique features in each candidate
                   - Note shared structural elements

                2. Spectral Compatibility Ranking:
                   - Rank structures from best to worst match with NMR data
                   - Provide specific evidence for each ranking
                   - Highlight decisive spectral features

                3. Confidence Assessment:
                   - Rate confidence (1-10) in your top choice
                   - Explain key factors in your decision
                   - Identify any remaining ambiguities
                   - Suggest additional data needed for confirmation
                """
        
        return prompt, iupac_names

    def _generate_overall_analysis_prompt(self, spectral_comparison: Dict[str, Any], top_candidates: List[Dict[str, Any]]) -> str:
        """Generate the complete overall analysis prompt focusing on HSQC analysis."""
        
        # Build individual molecule sections with HSQC analysis
        molecule_sections = []
        for i, candidate in enumerate(top_candidates):
            # Get HSQC scores and analysis from spectral comparison
            hsqc_data = None
            for spectrum_type, data in spectral_comparison.items():
                if spectrum_type == 'HSQC_exp' and 'candidates' in data:
                    hsqc_data = data['candidates'][i] if i < len(data['candidates']) else None
                    break
            
            section = f"""
            Molecule {i+1}:
            - IUPAC Name: {candidate.get('iupac_name', 'Not available')}
            - HSQC Score: {hsqc_data['score'] if hsqc_data and 'score' in hsqc_data else 'Not available'}
            - Structure Overview:
              * Describe the overall molecular framework
              * Note key functional groups and their positions
              * Highlight distinctive structural features that affect HSQC patterns
            """
            molecule_sections.append(section)

        return f"""
        Analyze the structural candidates based on their HSQC spectral matching, which is the most reliable indicator for structural matching in this analysis.

        Part 1: Individual Structure Analysis
        {' '.join(molecule_sections)}

        Part 2: HSQC-Based Analysis
        1. Structural Matching:
           - For each structure, evaluate:
             a) How well the structural features match the HSQC patterns (primary criterion)
             b) How well the predicted HSQC spectra match experimental data (shown by scores, lower is better)
           - Highlight any cases where HSQC scores contradict structural analysis
           - Explain how you resolved such contradictions, prioritizing logical structural consistency

        2. Confidence Assessment:
           - For each structure, provide:
             a) Final confidence score (1-10)
             b) Structural match confidence (how well structure explains HSQC patterns)
             c) HSQC match confidence (based on scores, lower is better)
           - Explain which factors were most influential in your scoring
           - Flag any concerning mismatches or contradictions

        3. Key Findings:
           - Identify the most diagnostic structural features supported by HSQC data
           - Highlight any discrepancies between structural analysis and HSQC scores
           - Discuss cases where:
             a) Low HSQC scores support structural analysis
             b) Low HSQC scores conflict with structural logic
             c) High HSQC scores might be acceptable due to strong structural evidence

        Remember: While HSQC scores provide valuable input (lower is better), they should not override clear structural evidence. 
        If a structure with slightly higher HSQC scores makes more chemical sense, this should be weighted more heavily 
        in the final analysis.
        """

    async def analyze_spectral_llm_evaluation(self,
                                            workflow_data: Dict[str, Any],
                                            context: Dict[str, Any],
                                            data_tool: DataExtractionTool,
                                            ranking_tool: 'CandidateRankingTool',
                                            llm_service: Any) -> Dict[str, Any]:
        """
        LLM-based evaluation of how well candidate structures match experimental NMR spectra.
        Uses vision capabilities to analyze structural features against spectral patterns.
        
        Args:
            workflow_data: Dictionary containing workflow data including molecule_data
            context: Context dictionary containing settings and state
            data_tool: Tool for extracting experimental data
            ranking_tool: Tool for ranking candidates
            llm_service: Service for LLM interactions
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            # Extract molecule_data from workflow_data
            logger.info(f"[analyze_spectral_llm_evaluation] workflow_data keys: {list(workflow_data.keys())}")
            if not workflow_data.get('molecule_data'):
                raise ValueError("No molecule_data found in workflow_data")
            
            molecule_data = workflow_data['molecule_data']
            sample_id = molecule_data['sample_id']

            # Determine data source and get ranking results
            # logger.info(f"[analyze_spectral_llm_evaluation] context: {context}")
            is_full_analysis = context.get('from_orchestrator', False)
            data_source = DataSource.INTERMEDIATE if is_full_analysis else DataSource.MASTER_FILE
            
            # Load data from appropriate source
            data = await data_tool.load_data(sample_id, data_source)
            # logger.info(f"[analyze_spectral_llm_evaluation] data keys: {list(data.keys())}")
            
            # Check if candidate ranking has been completed
            completed_steps = data.get('completed_analysis_steps', {})
            # logger.info(f"[analyze_spectral_llm_evaluation] completed_steps: {completed_steps}")
            candidate_ranking_completed = completed_steps.get('candidate_ranking', {})
            logger.info(f"[analyze_spectral_llm_evaluation] candidate_ranking_completed type: {type(candidate_ranking_completed)}, value: {candidate_ranking_completed}")
               
            if not candidate_ranking_completed:
                # Run candidate ranking if not completed
                # logger.info(f"Candidate ranking not found in {data_source.value}, running analysis...")
                candidate_ranking = await ranking_tool.analyze_candidates(
                        molecule_data=molecule_data,
                        sample_id=sample_id,
                        metric=RankingMetric.HSQC,  # Use HSQC for ranking
                        top_n=3,
                        include_reasoning=True
                    )
                
                # Reload data as it was updated by ranking tool
                data = await data_tool.load_data(sample_id, data_source)
            
            # Get ranking results which contain the image paths
            candidate_ranking = data.get('analysis_results', {}).get('candidate_ranking', {})
            if not candidate_ranking:
                raise ValueError("Candidate ranking results not found in data")
                
            # Extract image paths from ranking results
            mol_image_paths = []
            ranked_candidates = candidate_ranking.get('ranked_candidates', [])
            for candidate in ranked_candidates:
                image_path = candidate.get('structure_image')
                if not image_path:
                    raise ValueError(f"Structure image path not found for candidate rank {candidate.get('rank')}")
                mol_image_paths.append(image_path)
                
            # Get combined image path
            combined_image_path = candidate_ranking.get('combined_structure_image')
            if not combined_image_path:
                raise ValueError("Combined structure image path not found in ranking results")

            # Initialize results
            spectral_comparison = {}
            spectrum_types = ['HSQC_exp']
            # spectrum_types = ['HSQC_exp', 'COSY_exp', '1H_exp', '13C_exp']
            available_spectra = []

            # Check which spectra are available and get spectral comparison results
            for spectrum_type in spectrum_types:
                try:
                    exp_data = await data_tool.extract_experimental_nmr_data(
                        sample_id=molecule_data['sample_id'],
                        spectrum_type=spectrum_type,
                        source=DataSource.MASTER_FILE
                    )
                    if exp_data:
                        available_spectra.append(spectrum_type)
                except Exception as e:
                    logger.warning(f"Spectrum {spectrum_type} not available: {str(e)}")
                    continue

            if not available_spectra:
                raise ValueError("No experimental NMR data available for analysis")

            # Analyze each available spectrum type
            for spectrum_type in available_spectra:
                # Get experimental NMR data
                exp_data = await data_tool.extract_experimental_nmr_data(
                    sample_id=molecule_data['sample_id'],
                    spectrum_type=spectrum_type,
                    source=DataSource.MASTER_FILE
                )

                # Format the experimental data based on spectrum type
                formatted_data = self._format_experimental_data(exp_data, spectrum_type)
                if not formatted_data:
                    logger.warning(f"No data found for {spectrum_type}")
                    continue


                # Generate analysis prompt using helper functions
                analysis_prompt, iupac_names = await self._generate_analysis_prompt(
                    num_candidates=len(ranked_candidates),
                    spectrum_type=spectrum_type,
                    formatted_data=formatted_data,
                    smiles_list=[candidate['smiles'] for candidate in ranked_candidates]
                )

                # Get LLM analysis for this spectrum type (e.g., 3 times)
                analyses = []
                for _ in range(1): ########################### can run the analysis multiple times if needed
                    analysis = await llm_service.analyze_with_vision(
                        prompt=analysis_prompt,
                        image_path=str(combined_image_path),  # Pass single path string instead of list
                        model=context.get('model_choice', 'claude-3-5-sonnet')
                    )
                    analyses.append(analysis)
                spectral_comparison[spectrum_type] = {
                    'analyses': analyses,
                    'analysis_prompt': analysis_prompt,
                    'candidates': [{
                        'rank': candidate['rank'],
                        **{f"score_{spectrum_type.replace('_exp', '')}": candidate['scores'].get(spectrum_type.replace('_exp', ''))}
                    } for candidate in ranked_candidates],
                    'iupac_names': iupac_names
                }

            # # Final overall analysis comparing all spectrum types
            # overall_prompt = self._generate_overall_analysis_prompt(spectral_comparison, ranked_candidates)

            # # Use regular completion for final analysis (no vision needed)
            # overall_analysis = await llm_service.get_completion(
            #     message=overall_prompt,
            #     model=context.get('model_choice', 'claude-3-5-sonnet'),
            #     system="You are an expert in NMR spectroscopy analysis. Your task is to synthesize multiple spectral analyses into a clear, well-reasoned final evaluation."
            # )

            # Prepare final results
            evaluation_results = {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'spectral_llm_evaluation',
                'spectral_comparison': spectral_comparison,
                # 'overall_analysis': overall_analysis,
                # 'overall_analysis_prompt': overall_prompt,
                'candidates': ranked_candidates
            }

            # Store results in appropriate file
            if is_full_analysis:
                data['analysis_results']['spectral_llm_evaluation'] = evaluation_results
                if 'completed_analysis_steps' not in data:
                    data['completed_analysis_steps'] = {}
                data['completed_analysis_steps']['spectral_llm_evaluation'] = True
                await data_tool.save_data(data, molecule_data['sample_id'], DataSource.INTERMEDIATE)
            else:
                master_data = await data_tool.load_data(molecule_data['sample_id'], DataSource.MASTER_FILE)
                master_data['analysis_results']['spectral_llm_evaluation'] = evaluation_results
                if 'completed_analysis_steps' not in master_data:
                    master_data['completed_analysis_steps'] = {}
                master_data['completed_analysis_steps']['spectral_llm_evaluation'] = True
                await data_tool.save_data(master_data, molecule_data['sample_id'], DataSource.MASTER_FILE)

            return evaluation_results

        except Exception as e:
            logger.error(f"Error in spectral LLM evaluation: {str(e)}")
            raise

    def _create_temp_folder(self, sample_id: str) -> str:
        """Create a temporary folder for storing analysis files."""
        temp_folder = Path(f"temp/{sample_id}")
        temp_folder.mkdir(parents=True, exist_ok=True)
        return str(temp_folder)


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/tools/mmst_tool.py ---
"""Tool for predicting molecular structures using Multi-Modal Spectral Transformer."""
import os
import logging
import shutil
import pandas as pd
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess
import json
import pickle
import uuid
from rdkit import Chem  # Add RDKit for SMILES canonicalization
import time

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.sbatch_utils import execute_sbatch, wait_for_job_completion
from models.molecule import MoleculeHandler

# Constants for paths and directories
BASE_DIR = Path(__file__).parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "agents" / "scripts"
TEMP_DIR = BASE_DIR / "_temp_folder" / "mmst_temp"
SBATCH_SCRIPT = SCRIPTS_DIR / "mmst_sbatch.sh"
LOCAL_SCRIPT = SCRIPTS_DIR / "mmst_local.sh"

# Constants for MMST execution
MMST_OUTPUT_CHECK_INTERVAL = 5  # seconds
MMST_OUTPUT_TIMEOUT = 1800  # 30 minutes (longer timeout due to fine-tuning)
MMST_SUMMARY_FILE = 'mmst_final_results.json'  # New summary file name
MMST_INPUT_FILENAME = "mmst_input.csv"

MMST_SGNN_OUTPUT_DIR = TEMP_DIR / "sgnn_output"

class MMSTTool:
    """Tool for predicting molecular structures using Multi-Modal Spectral Transformer."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize MMST tool."""
        self.config_path = config_path
        self.temp_dir = TEMP_DIR
        self.start_time = time.time()
        self.intermediate_dir = BASE_DIR / "_temp_folder" / "intermediate_results"
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp-based run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = TEMP_DIR / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Created run directory: {self.run_dir}")

    def get_run_dir(self) -> Path:
        """Get the current run directory."""
        return self.run_dir

    def _get_work_dir(self, sample_id: Optional[str] = None) -> Path:
        """Get working directory for a sample within the current run directory."""
        return self.run_dir / (sample_id if sample_id else 'MOL_1')

    def get_sample_dirs(self, sample_id: str) -> dict:
        """Create and return dictionary of sample-specific directories within run directory.
        
        Args:
            sample_id: Sample ID string
            
        Returns:
            Dictionary containing paths for each subdirectory
        """
        sample_dir = self._get_work_dir(sample_id)
        dirs = {
            'sample': sample_dir,
            'models': sample_dir / 'models',
            'sgnn_output': sample_dir / 'sgnn_output',
            'experimental_data': sample_dir / 'experimental_data',
            'test_results': sample_dir / 'test_results'
        }
        
        # Create all directories
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        return dirs

    async def _prepare_input_file(self, smiles: str, molecule_id: str = None) -> Path:
        """Prepare input file for MMST.
        
        Args:
            smiles: SMILES string of reference molecule
            molecule_id: Optional molecule identifier
            
        Returns:
            Path to created input file
        """
        # Validate SMILES
        if not MoleculeHandler.validate_smiles(smiles):
            raise ValueError(f"Invalid SMILES string: {smiles}")
            
        # Prepare input data
        input_data = {
            'SMILES': [smiles],
            'sample-id': [molecule_id if molecule_id else 'MOL_1']
        }
        df = pd.DataFrame(input_data)
        
        # Create sample-specific directory
        sample_dir = self._get_work_dir(molecule_id)
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV in sample directory
        input_file = sample_dir / MMST_INPUT_FILENAME
        await asyncio.to_thread(df.to_csv, input_file, index=False)
        self.logger.info(f"Created input file at: {input_file}")
        
        return input_file

    async def _wait_for_output(self, sample_id: Optional[str] = None) -> Optional[Path]:
        """Wait for MMST output file to be generated and return its path."""
        # Use sample directory if provided
        work_dir = self._get_work_dir(sample_id)
        test_results_dir = work_dir / "test_results"  # Add test_results subdirectory
        output_file = test_results_dir / MMST_SUMMARY_FILE
        
        while True:
            # Check if output file exists and is valid
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r') as f:
                        mmst_results = json.load(f)
                    
                    # Validate the new JSON structure
                    if 'runs' in mmst_results and isinstance(mmst_results['runs'], list):
                        # Check if we have at least one run with required data
                        for run in mmst_results['runs']:
                            required_keys = ['exp_results_file', 'final_performance', 'model_save_path']
                            if all(key in run for key in required_keys):
                                # Verify that exp_results_file exists
                                exp_results_file = Path(run['exp_results_file'])
                                if exp_results_file.exists():
                                    return output_file  # Return the Path object
                                else:
                                    self.logger.warning(f"Experimental results file not found: {exp_results_file}")
                        
                        # If we get here, no valid run was found
                        self.logger.warning("No valid runs found in results file")
                    else:
                        self.logger.warning("Invalid results format: 'runs' array not found")
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid JSON in {output_file}")
                except Exception as e:
                    self.logger.warning(f"Error processing results file: {str(e)}")
            
            # Check timeout
            if time.time() - self.start_time > MMST_OUTPUT_TIMEOUT:
                self.logger.error("Timeout waiting for MMST output")
                return None
            
            time.sleep(MMST_OUTPUT_CHECK_INTERVAL)

    def _get_intermediate_path(self, sample_id: str) -> Path:
        """Get path to intermediate file for a sample."""
        return self.intermediate_dir / f"{sample_id}_intermediate.json"


    def _load_or_create_intermediate(self, sample_id: str, context: Dict[str, Any] = None) -> Dict:
        """Load existing intermediate file or create new one from master data.
        """
        intermediate_path = self._get_intermediate_path(sample_id)
        
        # Try loading existing intermediate
        if intermediate_path.exists():
            with open(intermediate_path, 'r') as f:
                data = json.load(f)
                if 'molecule_data' not in data:
                    data['molecule_data'] = {}
                return data
                
        # If no intermediate exists, try getting data from master file
        master_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
        if master_path.exists():
            with open(master_path, 'r') as f:
                master_data = json.load(f)

                if sample_id in master_data:
                    # Create new intermediate with just this sample's data
                    intermediate_data = master_data[sample_id]
                    self._save_intermediate(sample_id, intermediate_data)
                    return intermediate_data
                    
        # If neither exists and we have context, create new intermediate
        if context:
            intermediate_data["molecule_data"] = context
            self._save_intermediate(sample_id, intermediate_data)
            return intermediate_data
            
        raise ValueError(f"No data found for sample {sample_id}")

    def _save_intermediate(self, sample_id: str, data: Dict):
        """Save intermediate data to file.
        
        Args:
            sample_id: ID of the sample
            data: Data to save
        """
        intermediate_path = self._get_intermediate_path(sample_id)
        intermediate_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(intermediate_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _check_existing_predictions(self, molecule_id: str) -> Optional[Dict]:
        """Check if MMST predictions already exist for a given molecule ID.
        
        Args:
            molecule_id: ID of the molecule to check
            
        Returns:
            Dict containing prediction results if they exist, None otherwise
        """
        try:
            # Load intermediate data
            intermediate_data = self._load_or_create_intermediate(molecule_id)
            
            # Check if molecule exists and has MMST predictions
            if ('molecule_data' in intermediate_data and 
                'mmst_results' in intermediate_data['molecule_data']):
                return {
                    'status': 'success',
                    'message': 'Retrieved existing predictions',
                    'predictions': intermediate_data['molecule_data']['mmst_results']
                }
            return None
        except Exception:
            return None

    async def _process_mmst_results(self, final_results_file: Path) -> Dict:
        """Process MMST results from all runs and combine them."""
        try:
            with open(final_results_file, 'r') as f:
                final_results = json.load(f)
                self.logger.info(f"Number of runs in final_results: {len(final_results.get('runs', []))}")
                self.logger.info(f"Content of final_results: {json.dumps(final_results, indent=2)}")

            # Initialize combined results structure
            combined_results = {
                "mmst_results": {
                    "generated_analogues_target": {},
                    "generatedSmilesProbabilities": {},
                    "generated_molecules": [],
                    "performance": 0.0,
                    "model_info": {
                        "model_path": final_results['runs'][-1]['model_save_path'] if final_results.get('runs') else ""
                    },
                    "timestamp": datetime.now().isoformat(),
                    "status": "success",
                    "runs": final_results.get('runs', [])
                }
            }

            total_performance = 0
            unique_smiles_set = set()  # Track unique SMILES for deduplication
            molecule_data_dict = {}  # Track complete molecule data

            # Process each run's experimental results
            for run in final_results.get('runs', []):
                exp_results_file = run.get('exp_results_file')
                if not exp_results_file or not os.path.exists(exp_results_file):
                    self.logger.warning(f"Experimental results file not found: {exp_results_file}")
                    continue

                try:
                    with open(exp_results_file, 'r') as f:
                        run_data = json.load(f)
                    self.logger.info(f"Keys in run_data: {list(run_data.keys())}")
                    if "results" in run_data:
                        self.logger.info(f"Number of targets in results: {len(run_data['results'])}")
                        
                    if "results" in run_data:
                        # Process each target and its generated analogues
                        for target_smiles, analogues in run_data["results"].items():
                            # Canonicalize target SMILES
                            target_mol = Chem.MolFromSmiles(target_smiles)
                            if target_mol is None:
                                continue
                            canon_target = Chem.MolToSmiles(target_mol, canonical=True)
                            
                            if canon_target not in combined_results["mmst_results"]["generated_analogues_target"]:
                                combined_results["mmst_results"]["generated_analogues_target"][canon_target] = []
                                combined_results["mmst_results"]["generatedSmilesProbabilities"][canon_target] = []
                            
                            # Process and canonicalize each analogue
                            for analogue_data in analogues[0]:  # First level of nesting
                                try:
                                    # analogue_data = analogue_group[0]  # Second level - get the actual data
                                    smiles = analogue_data[0]
                                    mol = Chem.MolFromSmiles(smiles)
                                    if mol is None:
                                        continue
                                    canon_smiles = Chem.MolToSmiles(mol, canonical=True)
                                    
                                    # Only process if it's a new unique SMILES
                                    if canon_smiles not in unique_smiles_set:
                                        unique_smiles_set.add(canon_smiles)
                                        
                                        # Add to generated_analogues_target (only SMILES)
                                        if canon_smiles not in combined_results["mmst_results"]["generated_analogues_target"][canon_target]:
                                            combined_results["mmst_results"]["generated_analogues_target"][canon_target].append(canon_smiles)
                                        
                                        # Add probabilities
                                        probabilities = analogue_data[3] if isinstance(analogue_data[3], list) else analogue_data[3].tolist()
                                        if probabilities not in combined_results["mmst_results"]["generatedSmilesProbabilities"][canon_target]:
                                            combined_results["mmst_results"]["generatedSmilesProbabilities"][canon_target].append(probabilities)
                                        
                                        # Store complete molecule data
                                        molecule_data_dict[canon_smiles] = {
                                            "smiles": canon_smiles,
                                            "cosine_sim": float(analogue_data[1] if not isinstance(analogue_data[1], list) else analogue_data[1][0]),
                                            "dot_sim": float(analogue_data[2] if not isinstance(analogue_data[2], list) else analogue_data[2][0]),
                                            "probabilities": probabilities,
                                            "tanimoto_sim": float(analogue_data[4] if not isinstance(analogue_data[4], list) else analogue_data[4][0]),
                                            "HSQC_COSY_error": [float(x) if not isinstance(x, list) else float(x[0]) for x in analogue_data[5]]
                                        }
                                            
                                except Exception as e:
                                    self.logger.warning(f"Error processing SMILES {smiles}: {str(e)}")

                    # Add performance to average calculation
                    if "performance" in run_data:
                        total_performance += run_data["performance"]

                except Exception as e:
                    self.logger.warning(f"Error processing run results from {exp_results_file}: {str(e)}")
                    continue

            # Calculate average performance across all successful runs
            num_runs = len(final_results.get('runs', []))
            if num_runs > 0:
                combined_results["mmst_results"]["performance"] = total_performance / num_runs

            # Add the complete molecule data to generated_molecules
            combined_results["mmst_results"]["generated_molecules"] = list(molecule_data_dict.values())

            self.logger.info(f"Successfully combined results from {num_runs} runs")
            self.logger.info(f"Total unique molecules generated: {len(unique_smiles_set)}")
            
            return combined_results
        except Exception as e:
            self.logger.error(f"Error processing MMST results: {str(e)}")
            raise

    async def _execute_mmst_local(self, input_file: Path, molecule_id: Optional[str] = None) -> None:
        """Execute MMST prediction locally."""
        try:
            # Get sample directories
            sample_dirs = self.get_sample_dirs(molecule_id)
            
            # Create command with run directory
            cmd = [
                str(LOCAL_SCRIPT),
                f"--run_dir={str(self.run_dir)}",
                f"--input_csv={str(input_file)}",
                "--config_dir=/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/utils_MMT",
                f"--model_save_dir={str(sample_dirs['models'])}",
                f"--sgnn_gen_folder={str(sample_dirs['sgnn_output'])}",
                "--exp_data_path=/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/data/molecular_data/molecular_data.json"
            ]
            
            # Create a log file for this run
            log_file = sample_dirs['sample'] / f"mmst_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            self.logger.info(f"Logging MMST execution to: {log_file}")
            
            try:
                # Open log file for writing
                with open(log_file, 'w') as f:
                    f.write(f"=== MMST Execution Log ===\n")
                    f.write(f"Start Time: {datetime.now().isoformat()}\n")
                    f.write(f"Input CSV: {input_file}\n")
                    f.write(f"Output Dir: {self.temp_dir}\n\n")
                    f.write(f"Run Directory: {self.run_dir}\n\n")

                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Create async tasks to read stdout and stderr
                async def log_output(stream, prefix, log_file):
                    while True:
                        line = await stream.readline()
                        if not line:
                            break
                        line_str = line.decode().strip()
                        # Log to file
                        with open(log_file, 'a') as f:
                            f.write(f"{prefix}: {line_str}\n")
                        # Also log to console
                        if prefix == 'stdout':
                            self.logger.info(line_str)
                        else:
                            self.logger.warning(line_str)
                
                # Start logging tasks
                stdout_task = asyncio.create_task(log_output(process.stdout, 'stdout', log_file))
                stderr_task = asyncio.create_task(log_output(process.stderr, 'stderr', log_file))
                
                # Wait for process to complete and logging tasks to finish
                await process.wait()
                await stdout_task
                await stderr_task
                
                # Log completion status
                with open(log_file, 'a') as f:
                    f.write(f"\nEnd Time: {datetime.now().isoformat()}\n")
                    f.write(f"Return Code: {process.returncode}\n")
                
                if process.returncode != 0:
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                    error_msg = f"MMST execution failed. Check log file: {log_file}\n\nLast few lines:\n{log_content[-500:]}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
                    
            except Exception as e:
                error_msg = f"Failed to execute MMST script: {str(e)}"
                self.logger.error(error_msg)
                # Log the error
                with open(log_file, 'a') as f:
                    f.write(f"\nError occurred: {error_msg}\n")
                raise RuntimeError(error_msg)
            
            self.logger.info(f"MMST execution completed. Full log available at: {log_file}")
        
        except Exception as e:
            self.logger.error(f"Error executing MMST locally: {str(e)}")
            raise

    async def predict_structure(self, reference_smiles: str, molecule_id: str = None, context: Optional[Dict[str, Any]] = None) -> Dict:
        """Predict molecular structure using MMST.
        
        Args:
            reference_smiles: SMILES string of reference molecule
            molecule_id: Optional molecule identifier
            context: Additional context for prediction
            
        Returns:
            Dict containing status and results/error message
        """
        self.logger.info("Starting MMST structure prediction")
        
        try:
            # Check for existing predictions first
            if molecule_id:
                existing = self._check_existing_predictions(molecule_id)
                if existing:
                    self.logger.info(f"Found existing predictions for {molecule_id}")
                    return existing

            # Create run ID
            run_id = str(uuid.uuid4())
            self.logger.info(f"Starting MMST prediction with run ID: {run_id}")
            
            # Create sample directory structure
            sample_dirs = self.get_sample_dirs(molecule_id)
            
            # Prepare input file
            input_file = await self._prepare_input_file(reference_smiles, molecule_id)
            
            # Check if using SLURM
            use_slurm = False  # Default to local execution
            if context and context.get('use_slurm'):
                use_slurm = True
            
            # Check CUDA availability for local execution
            try:
                import torch
                if not torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID') is None:
                    self.logger.warning("CUDA not available. Switching to SLURM execution.")
                    use_slurm = True
            except ImportError:
                self.logger.warning("PyTorch not available. Switching to SLURM execution.")
                use_slurm = True
                
            if use_slurm:
                # Execute using SBATCH
                self.logger.info("Executing MMST prediction using SLURM")
                job_id = await execute_sbatch(
                    str(SBATCH_SCRIPT),
                    f"--input_csv={str(input_file)}",
                    f"--output_dir={str(self.temp_dir)}",
                    f"--model_save_dir={str(self.temp_dir / 'models')}"
                )
                
                # Wait for job completion
                await wait_for_job_completion(job_id)
            else:
                # Execute locally
                self.logger.info("Executing MMST prediction locally")
                await self._execute_mmst_local(input_file, molecule_id)
            
            # Wait for output file
            output_file = await self._wait_for_output(molecule_id)
            
            # Process MMST results
            results = await self._process_mmst_results(output_file)
            
            # Save results to intermediate file
            intermediate_data = self._load_or_create_intermediate(molecule_id, context)
            intermediate_data['molecule_data']['mmst_results'] = results['mmst_results']
            self._save_intermediate(molecule_id, intermediate_data)
            
            return {
                'status': 'success',
                'message': 'Successfully predicted structure',
                'predictions': results
            }
            
        except Exception as e:
            self.logger.error(f"Error in MMST prediction: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/tools/candidate_ranking_tool.py ---
"""
Tool for analyzing and ranking candidate molecules based on various NMR matching criteria.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
from rdkit import Chem
from rdkit.Chem import Descriptors
import os
import math
import numpy as np
from .analysis_enums import DataSource, RankingMetric
from .data_extraction_tool import DataExtractionTool
from .stout_operations import STOUTOperations

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RankingResult:
    """Structure for storing ranking results"""
    smiles: str
    analysis_type: str
    scores: Dict[str, float]
    nmr_data: Dict[str, Dict[str, Dict[str, float]]]
    rank: int
    iupac_name: Optional[str] = None
    reasoning: Optional[str] = None

class CandidateRankingTool:
    """Tool for ranking and analyzing candidate molecules."""

    def __init__(self, llm_service: Any = None):
        """Initialize the ranking tool."""
        self.llm_service = llm_service
        self.stout_ops = STOUTOperations()

    async def analyze_candidates(self, 
                           molecule_data: Dict,
                           sample_id: str,
                           metric: Union[str, RankingMetric] = RankingMetric.HSQC,
                           top_n: int = 3,
                           mw_tolerance: float = 1.0,  # Molecular weight tolerance in Da
                           include_reasoning: bool = True) -> Dict:
        """
        Analyze and rank candidate molecules based on NMR scores, filtering by molecular weight.
        
        Args:
            molecule_data: Dictionary containing molecule data and analysis results
            sample_id: Sample ID for storing results
            metric: Metric to use for ranking (default: HSQC)
            top_n: Number of top candidates to return
            mw_tolerance: Tolerance for molecular weight matching in Daltons
            include_reasoning: Whether to include LLM reasoning for ranking
            
        Returns:
            Dictionary containing ranking results and analysis
        """
        try:
            logger.info(f"[analyze_candidates] Starting candidate analysis for sample {sample_id}")
            logger.info(f"[analyze_candidates] Molecule data keys: {list(molecule_data.keys())}")
            logger.info(f"[analyze_candidates] Using metric: {metric}")
            
            # Get target molecular weight
            target_mw = molecule_data.get('molecular_weight')
            if target_mw is None:
                # If not provided, try to calculate from target SMILES
                target_smiles = molecule_data.get('smiles')
                if target_smiles:
                    mol = Chem.MolFromSmiles(target_smiles)
                    if mol:
                        target_mw = Descriptors.ExactMolWt(mol)
                        logger.info(f"[analyze_candidates] Calculated target molecular weight: {target_mw}")
            
            # Convert string metric to enum if needed
            if isinstance(metric, str):
                metric = RankingMetric(metric)
            
            # Extract candidate analysis results
            results = []
            analysis_types = ['forward_synthesis', 'mol2mol', 'MMST']
            
            logger.info(f"[analyze_candidates] Checking candidate_analysis for types: {analysis_types}")
            logger.info(f"[analyze_candidates] candidate_analysis present: {'candidate_analysis' in molecule_data}")
            if 'candidate_analysis' in molecule_data:
                logger.info(f"[analyze_candidates] Found analysis types: {list(molecule_data['candidate_analysis'].keys())}")
            
            for analysis_type in analysis_types:
                if analysis_type in molecule_data.get('candidate_analysis', {}):
                    molecules = molecule_data['candidate_analysis'][analysis_type].get('molecules', [])
                    logger.info(f"[analyze_candidates] Found {len(molecules)} molecules for {analysis_type}")
                    
                    for molecule in molecules:
                        if 'nmr_analysis' in molecule and 'matching_scores' in molecule['nmr_analysis']:
                            scores = molecule['nmr_analysis']['matching_scores']
                            logger.info(f"[analyze_candidates] Processing molecule with scores: {scores}")
                            
                            # Get IUPAC name for this molecule
                            iupac_result = await self.stout_ops.get_iupac_name(molecule['smiles'])
                            
                            # Calculate molecular weight for candidate
                            mol = Chem.MolFromSmiles(molecule['smiles'])
                            if mol:
                                candidate_mw = Descriptors.ExactMolWt(mol)
                                
                                # Only include candidates within molecular weight tolerance
                                if target_mw is None or abs(candidate_mw - target_mw) <= mw_tolerance:
                                    # Get the relevant score based on metric
                                    if metric == RankingMetric.OVERALL:
                                        relevant_score = scores.get('overall')
                                    else:
                                        relevant_score = scores.get('by_spectrum', {}).get(metric.value)
                                    
                                    if relevant_score is not None:
                                        # Get NMR data for each spectrum type
                                        spectra = molecule['nmr_analysis'].get('spectra_matching', {})
                                        nmr_data = { 
                                            'spectra': {
                                                '1H': spectra.get('1H', {}),
                                                '13C': spectra.get('13C', {}),
                                                'HSQC': spectra.get('HSQC', {}),
                                                'COSY': spectra.get('COSY', {})
                                            }
                                        }

                                        results.append({
                                        'smiles': molecule['smiles'],
                                        'iupac_name': iupac_result.get('iupac_name', 'Not available'),
                                        'analysis_type': analysis_type,
                                            'relevant_score': relevant_score,
                                            'molecular_weight': candidate_mw,
                                            'mw_diff': abs(candidate_mw - target_mw) if target_mw else None,
                                            'scores': {
                                                'overall': scores.get('overall'),
                                                '1H': scores.get('by_spectrum', {}).get('1H'),
                                                '13C': scores.get('by_spectrum', {}).get('13C'),
                                                'HSQC': scores.get('by_spectrum', {}).get('HSQC'),
                                                'COSY': scores.get('by_spectrum', {}).get('COSY')
                                            },
                                            'nmr_data': nmr_data,
                                        })
            
            logger.info(f"[analyze_candidates] Total candidates within MW tolerance: {len(results)}")
            
            # First sort by relevant score (lower is better)
            sorted_results = sorted(results, key=lambda x: x['relevant_score'])
            
            # Then filter by molecular weight tolerance
            filtered_results = [
                result for result in sorted_results 
                if target_mw is None  # If no target MW, keep all
                or result['mw_diff'] is None  # If couldn't calculate diff, keep
                or result['mw_diff'] <= mw_tolerance  # Keep if within tolerance
            ]
            
            logger.info(f"[analyze_candidates] Candidates within MW tolerance: {len(filtered_results)}")
            top_results = filtered_results[:top_n]
            logger.info(f"[analyze_candidates] Selected top {len(top_results)} candidates")
            
            # Add rankings and get LLM reasoning if requested
            ranked_results = []
            prompts = []
            for rank, result in enumerate(top_results, 1):
                ranking_result = RankingResult(
                    smiles=result['smiles'],
                    iupac_name=result.get('iupac_name'),
                    analysis_type=result['analysis_type'],
                    scores=result['scores'],
                    rank=rank,
                    nmr_data=result['nmr_data'],
                )
                
                if include_reasoning and self.llm_service:
                    # Generate reasoning using LLM
                    prompt = f"""
                    Analyze why this molecule ranked #{rank}:
                    SMILES: {result['smiles']}
                    IUPAC: {result.get('iupac_name', 'Not available')}
                    
                    HSQC NMR Error Score: {result['scores']['HSQC']:.6f}
                    
                    Explain the ranking focusing on the HSQC score, which has been shown to be the most reliable indicator 
                    for structural matching in this analysis.
                    """
                    # NMR Error Scores:
                    # Overall: {result['scores']['overall']:.6f}
                    # 1H NMR: {result['scores']['1H']:.6f}
                    # 13C NMR: {result['scores']['13C']:.6f}
                    # HSQC: {result['scores']['HSQC']:.6f}
                    # COSY: {result['scores']['COSY']:.6f}
                    reasoning = await self.llm_service.get_completion(
                        message=prompt,
                        model="claude-3-5-sonnet",   #### maybe make it flexible to adapt to the user input
                        system="You are an expert in NMR spectroscopy analysis. Analyze the molecule's ranking based on its NMR matching scores."
                    )
                    ranking_result.reasoning = reasoning
                
                ranked_results.append(ranking_result)
                prompts.append(prompt)
            
            # Prepare final results
            analysis_result = {
                'timestamp': datetime.now().isoformat(),
                'metric': metric.value,
                'top_n': top_n,
                'total_candidates': len(results),
                'ranked_candidates': [
                    {
                        'rank': r.rank,
                        'smiles': r.smiles,
                        'iupac_name': r.iupac_name,
                        'analysis_type': r.analysis_type,
                        'scores': r.scores,
                        'reasoning': r.reasoning,
                        'prompt': prompt,
                        'nmr_data': r.nmr_data,
                    } for r, prompt in zip(ranked_results, prompts)
                ]
            }
            
            # Store results in intermediate file
            data_tool = DataExtractionTool()
            intermediate_data = await data_tool.load_data(sample_id, DataSource.INTERMEDIATE)
            
            # Create analysis section if it doesn't exist
            if 'analysis_results' not in intermediate_data:
                intermediate_data['analysis_results'] = {}
            
            # Create completed_analysis_steps if it doesn't exist
            if 'completed_analysis_steps' not in intermediate_data:
                intermediate_data['completed_analysis_steps'] = {}
            
            # Store candidate ranking results
            intermediate_data['analysis_results']['candidate_ranking'] = analysis_result
            intermediate_data['completed_analysis_steps']['candidate_ranking'] = True
            
            # Save updated data
            await data_tool.save_data(intermediate_data, sample_id, DataSource.INTERMEDIATE)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in candidate analysis: {str(e)}")
            raise

    # async def suggest_ranking_metric(self, user_input: str) -> RankingMetric:
    #     """
    #     Use LLM to suggest the best ranking metric based on user input.
        
    #     Args:
    #         user_input: User's request or question
            
    #     Returns:
    #         RankingMetric enum value
    #     """
    #     if not self.llm_service:
    #         return RankingMetric.HSQC
            
    #     try:
    #         prompt = f"""
    #         Based on the user's request: "{user_input}"
            
    #         Which NMR spectrum type would be most appropriate for ranking molecules?
    #         Options are:
    #         - overall (combined score)
    #         - 1H (proton NMR)
    #         - 13C (carbon NMR)
    #         - HSQC
    #         - COSY
            
    #         Return ONLY ONE of these exact options without explanation or additional text.
            
    #         Examples:
    #         1. If the best option is proton NMR, reply with exactly: 1H
    #         2. If the best option is HSQC, reply with exactly: HSQC
            
    #         Your response should be a single word/option from the list above.
    #         """
            
    #         response = await self.llm_service.get_completion(
    #             message=prompt,
    #             model="claude-3-5-haiku",
    #             require_json=False
    #         )
    #         metric = response.strip().lower()
            
    #         # Map response to enum
    #         metric_map = {
    #             'overall': RankingMetric.OVERALL,
    #             '1h': RankingMetric.PROTON,
    #             '13c': RankingMetric.CARBON,
    #             'hsqc': RankingMetric.HSQC,
    #             'cosy': RankingMetric.COSY
    #         }
            
    #         return metric_map.get(metric, RankingMetric.OVERALL)
            
    #     except Exception as e:
    #         logger.error(f"Error suggesting ranking metric: {str(e)}")
    #         return RankingMetric.OVERALL

    async def analyze_top_candidates(self,
                                   workflow_data: Dict,
                                   data_tool: DataExtractionTool,
                                   ranking_tool: 'CandidateRankingTool',
                                   context: Dict = None) -> Dict:
        """Analyze top candidates and generate structure images."""
        try:
            # Get molecule data and sample ID
            molecule_data = workflow_data.get('molecule_data')
            if not molecule_data:
                raise ValueError("No molecule data found in workflow data")
            
            sample_id = molecule_data.get('sample_id')
            if not sample_id:
                raise ValueError("Sample ID not provided in molecule data")

            logger.info(f"[analyze_top_candidates] Processing sample_id: {sample_id}")

            # Get analysis folder from context
            analysis_run_folder = context.get('analysis_run_folder')
            if not analysis_run_folder:
                raise ValueError("Analysis run folder not provided in context")
            
            logger.info(f"[analyze_top_candidates] Using analysis folder: {analysis_run_folder}")
            
            # Create Top Candidates subfolder
            candidates_folder = Path(analysis_run_folder) / "top_candidates"
            candidates_folder.mkdir(exist_ok=True)
            logger.info(f"[analyze_top_candidates] Created candidates folder: {candidates_folder}")

            # Get data source
            is_full_analysis = context.get('from_orchestrator', False)
            data_source = DataSource.INTERMEDIATE if is_full_analysis else DataSource.MASTER_FILE
            logger.info(f"[analyze_top_candidates] Using data source: {data_source}")
            
            # Load data from appropriate source
            data = await data_tool.load_data(sample_id, data_source)
            logger.info(f"[analyze_top_candidates] Initial data keys: {list(data.keys())}")
            if 'analysis_results' in data:
                logger.info(f"[analyze_top_candidates] Initial analysis_results keys: {list(data.get('analysis_results', {}).keys())}")
            
            # Check if candidate ranking has been completed
            completed_steps = data.get('completed_analysis_steps', {})
            candidate_ranking_completed = completed_steps.get('candidate_ranking', {})
            logger.info(f"[analyze_top_candidates] Candidate ranking completed: {bool(candidate_ranking_completed)}")
               
            if not candidate_ranking_completed:
                # Run candidate ranking if not completed
                logger.info("[analyze_top_candidates] Running new candidate ranking")
                _ = await ranking_tool.analyze_candidates(      ### because it is stored in intermediate_data file
                        molecule_data=molecule_data,
                        sample_id=sample_id,
                        metric=RankingMetric.HSQC,
                        top_n=3,    ### top 3 candidates
                        mw_tolerance=5,     ### mw tolerance 
                        include_reasoning=True
                    )
                
                # Reload data as it was updated by ranking tool
                data = await data_tool.load_data(sample_id, data_source)
                logger.info(f"[analyze_top_candidates] Reloaded data after ranking. Keys: {list(data.keys())}")
            
            # Get ranking results which contain the image paths
            candidate_ranking = data.get('analysis_results', {}).get('candidate_ranking', {})
            if not candidate_ranking:
                logger.error(f"[analyze_top_candidates] Missing candidate_ranking in data structure. Available keys: {list(data.get('analysis_results', {}).keys())}")
                raise ValueError("Candidate ranking results not found in data")
            
            logger.info(f"[analyze_top_candidates] Candidate ranking keys: {list(candidate_ranking.keys())}")
            
            ranked_candidates = candidate_ranking.get('ranked_candidates', [])
            if not ranked_candidates:
                logger.error("[analyze_top_candidates] No ranked candidates found")
                raise ValueError("No top candidates found in ranking results")

            logger.info(f"[analyze_top_candidates] Found {len(ranked_candidates)} ranked candidates")

            # Generate structure images for top candidates
            mol_image_paths = []
            
            # Add structure images to existing ranked candidates
            for i, candidate in enumerate(ranked_candidates):
                mol_image_path = candidates_folder / f"candidate_{candidate['rank']}.png"
                logger.info(f"[analyze_top_candidates] Generating image {i+1}/{len(ranked_candidates)}: {mol_image_path}")
                
                mol_image_path = await self.generate_structure_image(
                    smiles=candidate['smiles'],
                    output_path=str(mol_image_path),
                    rotation_degrees=0,
                    font_size=12,
                    scale_factor=1.25,
                    show_indices=True
                )
                mol_image_paths.append(str(mol_image_path))
                
                # Add structure image path directly to candidate data
                candidate['structure_image'] = str(mol_image_path)
                logger.info(f"[analyze_top_candidates] Added image path to candidate {i+1}: {mol_image_path}")

            # Generate and add combined image to ranking results
            combined_image_path = candidates_folder / "combined_candidates.png"
            labels = [f"Rank {i+1}" for i in range(len(ranked_candidates))]
            logger.info(f"[analyze_top_candidates] Generating combined image: {combined_image_path}")
            
            combined_image_path = await self.combine_structure_images(
                image_paths=mol_image_paths,
                output_path=str(combined_image_path),
                labels=labels
            )
            
            # Add combined image to ranking results
            candidate_ranking['combined_structure_image'] = str(combined_image_path)
            logger.info(f"[analyze_top_candidates] Added combined image path: {combined_image_path}")

            # Update the data with modified ranking results that now include images
            data['analysis_results']['candidate_ranking'] = candidate_ranking
            
            # Log the final data structure before saving
            logger.info(f"[analyze_top_candidates] Final data structure keys: {list(data.keys())}")
            logger.info(f"[analyze_top_candidates] Final analysis_results keys: {list(data.get('analysis_results', {}).keys())}")
            logger.info(f"[analyze_top_candidates] Final candidate_ranking keys: {list(data.get('analysis_results', {}).get('candidate_ranking', {}).keys())}")
            
            # Verify image paths are present
            for i, candidate in enumerate(ranked_candidates):
                logger.info(f"[analyze_top_candidates] Candidate {i+1} image path: {candidate.get('structure_image')}")
            logger.info(f"[analyze_top_candidates] Combined image path: {candidate_ranking.get('combined_structure_image')}")

            # Save updated data to appropriate source
            await data_tool.save_data(data, sample_id, data_source)
            logger.info("[analyze_top_candidates] Saved updated data with image paths")

            return candidate_ranking

        except Exception as e:
            logger.error(f"Error in analyze_top_candidates: {str(e)}")
            logger.exception("Full traceback:")
            raise

    async def generate_structure_image(self, 
                                    smiles: str,
                                    output_path: str,
                                    rotation_degrees: float = 0,
                                    font_size: int = 10,
                                    scale_factor: float = 1.0,
                                    show_indices: bool = True,
                                    sample_id: Optional[str] = None) -> str:
        """Generate a 2D structure image from SMILES and save it to the specified path."""
        try:
            # Handle output path
            if not os.path.isabs(output_path):
                raise ValueError("output_path must be an absolute path")
            
            # Create the molecule and remove stereochemistry
            mol = Chem.MolFromSmiles(smiles)
            Chem.RemoveStereochemistry(mol)
            smiles = Chem.MolToSmiles(mol, canonical=True)
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")
            
            # Generate 2D coordinates
            from rdkit.Chem import rdDepictor
            rdDepictor.Compute2DCoords(mol)
            
            # Rotate the molecule if specified
            if rotation_degrees != 0:
                angle = math.radians(rotation_degrees)
                rotation_matrix = np.array([
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)]
                ])
                
                conf = mol.GetConformer()
                center_x = sum(conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())) / mol.GetNumAtoms()
                center_y = sum(conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())) / mol.GetNumAtoms()
                
                for i in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    x = pos.x - center_x
                    y = pos.y - center_y
                    new_x, new_y = rotation_matrix.dot([x, y])
                    conf.SetAtomPosition(i, (new_x + center_x, new_y + center_y, 0))
            
            # Add atom indices if requested
            if show_indices:
                for atom in mol.GetAtoms():
                    atom.SetProp("atomNote", str(atom.GetIdx()))
            
            # Set up drawing options with scaled size
            base_size = 2000
            scaled_size = int(base_size * scale_factor)
            rdDepictor.SetPreferCoordGen(True)
            from rdkit.Chem import Draw
            drawer = Draw.rdMolDraw2D.MolDraw2DCairo(scaled_size, scaled_size)
            
            # Set drawing options
            opts = drawer.drawOptions()
            opts.baseFontSize = font_size
            opts.atomLabelFontSize = font_size
            opts.scalingFactor = scale_factor
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Draw the molecule
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            drawer.WriteDrawingText(output_path)

            # Post-process the image
            from PIL import Image, ImageDraw, ImageFont
            with Image.open(output_path) as img:
                # Convert to RGBA
                img = img.convert('RGBA')
                
                # Convert white to transparent
                data = img.getdata()
                new_data = []
                for item in data:
                    # If it's white or nearly white, make it transparent
                    if item[0] > 250 and item[1] > 250 and item[2] > 250:
                        new_data.append((255, 255, 255, 0))
                    else:
                        new_data.append(item)
                
                img.putdata(new_data)
                
                # Get the bounding box of non-transparent content
                bbox = img.getbbox()
                
                if bbox:
                    # Crop to content
                    cropped = img.crop(bbox)
                    
                    # Calculate new size maintaining aspect ratio
                    target_size = 1024 if scale_factor == 1.0 else int(1024 * 1.25)
                    
                    # Calculate scaling factor to fit within target size
                    aspect_ratio = cropped.width / cropped.height
                    if aspect_ratio > 1:
                        new_width = target_size
                        new_height = int(target_size / aspect_ratio)
                    else:
                        new_height = target_size
                        new_width = int(target_size * aspect_ratio)
                    
                    # Resize the cropped image
                    resized = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Create new transparent image of target size
                    final_img = Image.new('RGBA', (target_size, target_size), (0, 0, 0, 0))
                    
                    # Center the resized image
                    paste_x = (target_size - new_width) // 2
                    paste_y = (target_size - new_height) // 2
                    
                    # Paste the resized image
                    final_img.paste(resized, (paste_x, paste_y))
                    
                    # Save the final image
                    final_img.save(output_path, "PNG", quality=100)

            logger.info(f"Generated structure image at: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error generating structure image: {str(e)}")
            raise

    async def combine_structure_images(self,
                                     image_paths: List[str],
                                     output_path: str,
                                     labels: Optional[List[str]] = None) -> str:
        """
        Combine multiple structure images horizontally into a single image.
        
        Args:
            image_paths: List of paths to structure images to combine
            output_path: Path where to save the combined image
            labels: Optional list of labels for each image
            
        Returns:
            Path to the combined image file
        """
        try:
            # Open all images
            from PIL import Image
            images = [Image.open(path) for path in image_paths]
            
            # Get dimensions
            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)
            max_height = max(heights)
            
            # Create new image with white background
            combined_image = Image.new('RGBA', (total_width, max_height), (255, 255, 255, 0))
            
            # Paste images side by side
            x_offset = 0
            for i, img in enumerate(images):
                combined_image.paste(img, (x_offset, 0))
                
                # Add label if provided
                if labels and i < len(labels):
                    from PIL import ImageDraw, ImageFont
                    draw = ImageDraw.Draw(combined_image)
                    font = ImageFont.load_default()
                    draw.text((x_offset + 10, 10), labels[i], fill='black', font=font)
                
                x_offset += img.size[0]
            
            # Save combined image
            combined_image.save(output_path, format='PNG')
            logger.info(f"Generated combined structure image at: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error combining structure images: {str(e)}")
            raise


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/tools/___molecular_visual_comparison_tool.py ---
"""
Tool for visual comparison of molecular structures using Claude 3.5 Sonnet's vision capabilities.
"""
from typing import Dict, Any, Optional, List
import logging
import anthropic
import json
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import base64
import io
import csv
from tqdm import tqdm
import asyncio
from ..tools.stout_tool import STOUTTool  # Import STOUTTool

class MolecularVisualComparisonTool:
    """Tool for comparing molecular structures visually using AI vision analysis."""

    def __init__(self):
        """Initialize the molecular visual comparison tool."""
        self.logger = logging.getLogger(__name__)
        self.api_call_delay = 1.0  # Delay between API calls in seconds
        self.max_retries = 3  # Maximum number of API call retries
        self.batch_save_interval = 10  # Save partial results every N molecules
        self.stout_tool = STOUTTool()  # Initialize STOUTTool

    def _smiles_to_image(self, smiles: str, size: tuple = (800, 800)) -> bytes:
        """Convert SMILES to a PNG image.
        
        Args:
            smiles: SMILES string of the molecule
            size: Tuple of (width, height) for the image
            
        Returns:
            bytes: PNG image data
        """
        try:
            # Convert SMILES to RDKit molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")
            
            # Generate 2D coordinates for the molecule
            AllChem.Compute2DCoords(mol)
            
            # Create the drawing object
            img = Draw.MolToImage(mol, size=size)
            
            # Convert PIL image to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            return img_bytes.getvalue()
            
        except Exception as e:
            self.logger.error(f"Error converting SMILES to image: {str(e)}")
            raise

    def _encode_image(self, image_data: bytes) -> str:
        """Encode image data to base64.
        
        Args:
            image_data: Raw image data in bytes
            
        Returns:
            str: Base64 encoded image string
        """
        return base64.b64encode(image_data).decode('utf-8')

    def _prepare_prompt(self, comparison_type: str, molecule_names: Dict[str, str]) -> str:
        """Prepare the prompt for visual comparison.
        
        Args:
            comparison_type: Type of comparison ('guess_vs_starting' or 'guess_vs_target')
            molecule_names: Dictionary of molecule names and their SMILES
            
        Returns:
            str: Formatted prompt for Claude
        """
        base_prompt = """Analyze the molecular structures shown in the images and provide a detailed comparison focusing on:

1. Structural Similarity Assessment:
   - Core structure similarities and differences
   - Functional group analysis
   - Spatial arrangement comparison

2. Chemical Properties Comparison:
   - Functional group modifications
   - Bond type changes
   - Potential reactivity differences

3. Overall Assessment:
   - Similarity score (0-100%)
   - Pass/Fail evaluation (Pass if similarity > 70%)
   - Confidence in the assessment (0-100%)

4. Detailed Explanation:
   - Key structural differences
   - Chemical implications of the differences
   - Reasoning for the similarity score

Format the response as a JSON object with the following structure:
{
    "similarity_score": float,  # 0-100
    "pass_fail": string,       # "PASS" or "FAIL"
    "confidence": float,       # 0-100
    "analysis": {
        "structural_comparison": string,
        "chemical_properties": string,
        "key_differences": list[string],
        "explanation": string
    }
}"""

        if comparison_type == 'guess_vs_starting':
            specific_prompt = f"\nCompare the guess molecule with the starting materials (which may be multiple molecules separated by dots). Focus on whether the guess molecule could reasonably be derived from these starting materials."
        else:  # guess_vs_target
            specific_prompt = f"\nCompare the guess molecule with the target molecule. Focus on whether they represent the same or very similar chemical structures."

        return base_prompt + specific_prompt

    def _validate_csv(self, csv_path: str) -> bool:
        """Validate CSV file structure and content.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            bool: True if valid, raises ValueError if invalid
        """
        try:
            if not Path(csv_path).exists():
                raise ValueError(f"CSV file not found: {csv_path}")
                
            df = pd.read_csv(csv_path)
            
            # Check required columns
            if 'SMILES' not in df.columns:
                raise ValueError("CSV must contain a 'SMILES' column")
            
            # Check for empty dataframe
            if df.empty:
                raise ValueError("CSV file is empty")
            
            # Validate SMILES strings
            invalid_smiles = []
            for idx, smiles in enumerate(df['SMILES']):
                if not isinstance(smiles, str) or not Chem.MolFromSmiles(smiles):
                    invalid_smiles.append(f"Row {idx + 1}: {smiles}")
            
            if invalid_smiles:
                raise ValueError(f"Invalid SMILES strings found:\n" + "\n".join(invalid_smiles))
            
            return True
            
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty")
        except pd.errors.ParserError:
            raise ValueError("Invalid CSV file format")
        except Exception as e:
            raise ValueError(f"CSV validation error: {str(e)}")

    def _read_smiles_csv(self, csv_path: str) -> List[str]:
        """Read SMILES strings from a CSV file.
        
        Args:
            csv_path: Path to CSV file containing SMILES strings
            
        Returns:
            List[str]: List of SMILES strings
        """
        try:
            # Validate CSV first
            self._validate_csv(csv_path)
            
            # Read validated CSV
            df = pd.read_csv(csv_path)
            return df['SMILES'].tolist()
            
        except Exception as e:
            self.logger.error(f"Error reading SMILES from CSV: {str(e)}")
            raise

    def _write_batch_results(self, results: List[Dict], output_path: str):
        """Write batch comparison results to a CSV file.
        
        Args:
            results: List of comparison results
            output_path: Path to write CSV file
        """
        try:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            self.logger.info(f"Batch results written to {output_path}")
        except Exception as e:
            self.logger.error(f"Error writing batch results: {str(e)}")
            raise

    async def compare_structures(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Compare molecular structures visually using Claude's vision capabilities."""
        try:
            if 'comparison_type' not in input_data:
                raise ValueError("Comparison type not specified")

            comparison_type = input_data['comparison_type']
            
            # Handle batch processing
            if comparison_type in ['batch_vs_target', 'batch_vs_starting']:
                guess_smiles_list = self._read_smiles_csv(input_data['guess_smiles_csv'])
                
                # Create output directory
                output_dir = Path(context['run_dir']) / "batch_results"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                batch_results = []
                failed_comparisons = []
                
                for i, guess_smiles in enumerate(tqdm(guess_smiles_list, desc="Processing molecules")):
                    try:
                        # Prepare single comparison input
                        single_input = {
                            'comparison_type': 'guess_vs_target' if comparison_type == 'batch_vs_target' else 'guess_vs_starting',
                            'guess_smiles': guess_smiles
                        }
                        
                        if comparison_type == 'batch_vs_target':
                            single_input['target_smiles'] = input_data['target_smiles']
                        else:
                            single_input['starting_materials_smiles'] = input_data['starting_materials_smiles']
                        
                        # Add rate limiting delay
                        if i > 0:
                            await asyncio.sleep(self.api_call_delay)
                        
                        # Perform single comparison
                        result = await self._compare_single(single_input, context)
                        
                        # Add SMILES to result for reference
                        result['guess_smiles'] = guess_smiles
                        batch_results.append(result)
                        
                        # Save partial results periodically
                        if (i + 1) % self.batch_save_interval == 0:
                            partial_output = output_dir / f"partial_results_{i + 1}.csv"
                            self._write_batch_results(batch_results, str(partial_output))
                            

                    except Exception as e:
                        self.logger.error(f"Error processing molecule {i + 1}: {str(e)}")
                        failed_comparisons.append({
                            'index': i + 1,
                            'smiles': guess_smiles,
                            'error': str(e)
                        })
                
                # Write final results
                output_path = output_dir / "comparison_results.csv"
                self._write_batch_results(batch_results, str(output_path))
                
                # Write failed comparisons if any
                if failed_comparisons:
                    failed_path = output_dir / "failed_comparisons.json"
                    with open(failed_path, 'w') as f:
                        json.dump(failed_comparisons, f, indent=2)
                
                return {
                    "status": "success",
                    "type": "batch_comparison",
                    "results": batch_results,
                    "output_file": str(output_path),
                    "failed_comparisons": failed_comparisons if failed_comparisons else None,
                    "total_processed": len(batch_results),
                    "total_failed": len(failed_comparisons)
                }
            
            # Handle single comparison
            return await self._compare_single(input_data, context)
            
        except Exception as e:
            self.logger.error(f"Error in compare_structures: {str(e)}")
            return {
                "status": "error",
                "type": "comparison_error",
                "error": str(e)
            }

    async def _compare_single(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Compare a single pair of molecular structures."""
        try:
            self.logger.info("Starting single molecule comparison")
            self.logger.debug(f"Input data: {input_data}")
            
            # Validate API key presence
            if 'anthropic_api_key' not in context:
                self.logger.error("Missing Anthropic API key in context")
                raise ValueError("Anthropic API key not found in context")

            # Convert SMILES to images
            try:
                self.logger.info("Converting guess SMILES to image")
                guess_image = self._smiles_to_image(input_data['guess_smiles'])
                if guess_image is None:
                    self.logger.error(f"Failed to generate image for guess SMILES: {input_data['guess_smiles']}")
                    raise ValueError("Failed to generate image from guess SMILES")
            except Exception as e:
                self.logger.error(f"Error converting guess SMILES to image: {str(e)}")
                raise

            try:
                self.logger.info("Converting comparison SMILES to image")
                if input_data['comparison_type'] == 'guess_vs_starting':
                    second_image = self._smiles_to_image(input_data['starting_materials_smiles'])
                    if second_image is None:
                        self.logger.error(f"Failed to generate image for starting materials SMILES: {input_data['starting_materials_smiles']}")
                        raise ValueError("Failed to generate image from starting materials SMILES")
                else:  # guess_vs_target
                    second_image = self._smiles_to_image(input_data['target_smiles'])
                    if second_image is None:
                        self.logger.error(f"Failed to generate image for target SMILES: {input_data['target_smiles']}")
                        raise ValueError("Failed to generate image from target SMILES")
            except Exception as e:
                self.logger.error(f"Error converting comparison SMILES to image: {str(e)}")
                raise

            # Encode images
            try:
                self.logger.info("Encoding molecule images")
                guess_image_encoded = self._encode_image(guess_image)
                second_image_encoded = self._encode_image(second_image)
            except Exception as e:
                self.logger.error(f"Error encoding molecule images: {str(e)}")
                raise

            # Prepare prompt
            try:
                self.logger.info("Preparing comparison prompt")
                prompt = self._prepare_prompt(
                    input_data['comparison_type'],
                    {"guess": input_data['guess_smiles']}
                )
            except Exception as e:
                self.logger.error(f"Error preparing comparison prompt: {str(e)}")
                raise

            # Prepare API request
            try:
                self.logger.info("Preparing API request")
                request_body = {
                    "model": "claude-3-sonnet-20240229",
                    "max_tokens": 2048,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": guess_image_encoded
                                }
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": second_image_encoded
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }]
                }
            except Exception as e:
                self.logger.error(f"Error preparing API request: {str(e)}")
                raise

            # Make API call
            try:
                self.logger.info("Making API call to Claude")
                headers = {
                    "x-api-key": context['anthropic_api_key'],
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }

                client = anthropic.Anthropic(api_key=context['anthropic_api_key'])
                response = await self._make_api_call(client, request_body, headers)
                self.logger.debug(f"API response received: {response[:200]}...")  # Log first 200 chars
            except Exception as e:
                self.logger.error(f"Error making API call: {str(e)}")
                raise

            # Parse and validate response
            try:
                self.logger.info("Parsing API response")
                analysis_result = json.loads(response)
                self.logger.debug(f"Parsed analysis result: {analysis_result}")
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse API response as JSON: {str(e)}")
                self.logger.debug(f"Raw response: {response}")
                analysis_result = self._format_error_response(response)

            self.logger.info("Single molecule comparison completed successfully")
            return {
                "status": "success",
                "type": "single_comparison",
                "analysis": analysis_result
            }

        except Exception as e:
            self.logger.error(f"Error in _compare_single: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "type": "comparison_error",
                "error": str(e)
            }

    async def _make_api_call(self, client: anthropic.Anthropic, request_body: dict, headers: dict, retry_count: int = 0) -> str:
        """Make API call to Claude with retry mechanism."""
        try:
            self.logger.debug(f"Making API call with request body: {request_body}")
            # Create message using the async client
            response = client.messages.create(
                model=request_body["model"],
                max_tokens=request_body["max_tokens"],
                messages=request_body["messages"]
            )
            
            # Extract the text content from the response
            if response and response.content and len(response.content) > 0:
                return response.content[0].text
            else:
                raise ValueError("Empty or invalid response from API")
            
        except Exception as e:
            if retry_count < self.max_retries:
                wait_time = (retry_count + 1) * 2  # Exponential backoff
                self.logger.warning(f"API call failed, retrying in {wait_time}s... (Attempt {retry_count + 1}/{self.max_retries})")
                await asyncio.sleep(wait_time)
                return await self._make_api_call(client, request_body, headers, retry_count + 1)
            else:
                self.logger.error(f"API call failed after {self.max_retries} retries: {str(e)}")
                raise

    def _format_error_response(self, raw_response: str) -> Dict[str, Any]:
        """Format error response when JSON parsing fails.
        
        Args:
            raw_response: Raw response string from API
            
        Returns:
            Dict containing formatted error response
        """
        return {
            "similarity_score": 0.0,
            "pass_fail": "FAIL",
            "confidence": 0.0,
            "analysis": {
                "structural_comparison": "Error in analysis",
                "chemical_properties": "Error in analysis",
                "key_differences": ["Error processing response"],
                "explanation": f"Failed to parse API response: {raw_response[:200]}..."
            }
        }


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/tools/threshold_calculation_tool.py ---
"""Tool for calculating thresholds using retrosynthesis, NMR simulation, and peak matching."""
import os
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from .nmr_simulation_tool import NMRSimulationTool
from .retro_synthesis_tool import RetrosynthesisTool
from .peak_matching_tool import EnhancedPeakMatchingTool

class ThresholdCalculationTool:
    """Tool for calculating thresholds using multiple specialized tools."""
    
    def __init__(self):
        """Initialize the ThresholdCalculationTool."""
        self.retro_tool = RetrosynthesisTool()
        self.nmr_tool = NMRSimulationTool()
        self.peak_matching_tool = EnhancedPeakMatchingTool()
        self.master_data_path = Path(__file__).parent.parent.parent / "data" / "molecular_data" / "molecular_data.json"
        self.temp_dir = Path(__file__).parent.parent.parent / "_temp_folder"
        self.peak_matching_dir = self.temp_dir / "peak_matching"  # Updated to use correct path
        self.intermediate_dir = self.temp_dir / "intermediate_results"
        self.intermediate_dir.mkdir(exist_ok=True)
        self.peak_matching_dir.mkdir(exist_ok=True)  # Ensure directory exists
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)  # Set logger level to DEBUG
        
        
    def _get_intermediate_path(self, sample_id: str) -> Path:
        """Get path to intermediate file for a sample."""
        return self.intermediate_dir / f"{sample_id}_intermediate.json"


    def _load_or_create_intermediate(self, sample_id: str, context: Dict[str, Any] = None) -> Dict:
        """Load existing intermediate file or create new one from master data.
        """
        intermediate_path = self._get_intermediate_path(sample_id)
        
        # Try loading existing intermediate
        if intermediate_path.exists():
            with open(intermediate_path, 'r') as f:
                data = json.load(f)
                if 'molecule_data' not in data:
                    data['molecule_data'] = {}
                return data
                
        # If no intermediate exists, try getting data from master file
        master_path = Path(__file__).parent.parent.parent / "data" / "molecular_data" / "molecular_data.json"
        if master_path.exists():
            with open(master_path, 'r') as f:
                master_data = json.load(f)

                if sample_id in master_data:
                    # Create new intermediate with just this sample's data
                    intermediate_data = master_data[sample_id]
                    self._save_intermediate(sample_id, intermediate_data)
                    return intermediate_data
                    
        # If neither exists and we have context, create new intermediate
        if context:
            intermediate_data["molecule_data"] = context
            self._save_intermediate(sample_id, intermediate_data)
            return intermediate_data
            
        raise ValueError(f"No data found for sample {sample_id}")

    def _save_intermediate(self, sample_id: str, data: Dict):
        """Save data to intermediate file.
        
        Args:
            sample_id: ID of the sample to save
            data: Data to save to intermediate file
        """
        path = self._get_intermediate_path(sample_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    async def calculate_threshold(self, sample_id: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Calculate threshold using retrosynthesis and NMR simulation pipeline.
        
        Args:
            sample_id: ID of the sample to calculate thresholds for
            context: Optional context information for the calculation
            
        Returns:
            Dict containing the calculation results or error information
        """
        try:
            # Load or create intermediate data
            self.logger.info(f"Loading intermediate data for sample {sample_id}...")
            try:
                intermediate_data = self._load_or_create_intermediate(sample_id, context)
                self.logger.info("Successfully loaded intermediate data")
            except ValueError as e:
                self.logger.error(f"Failed to load intermediate data: {str(e)}")
                return {'status': 'error', 'message': str(e)}
            
            # Check if we already have threshold results
            if 'threshold_data' in intermediate_data:
                self.logger.info(f"Retrieved cached threshold results for sample {sample_id}")
                return {'status': 'success', 'data': intermediate_data['threshold_data']}

            # Get SMILES from intermediate data
            target_smiles = intermediate_data['molecule_data'].get('smiles')
            if not target_smiles:
                error_msg = f"No SMILES found in intermediate data for sample {sample_id}"
                self.logger.error(error_msg)
                return {'status': 'error', 'message': error_msg}

            self.logger.info(f"Found SMILES for sample_id {sample_id}: {target_smiles}")
            
            try:
                # Step 1: Run retrosynthesis prediction
                self.logger.info(f"Starting retrosynthesis prediction for SMILES: {target_smiles}")
                
                try:
                    # Run retrosynthesis prediction
                    self.logger.info("Calling retrosynthesis prediction tool")
                    retro_result = await self.retro_tool.predict_retrosynthesis(
                        target_smiles,
                        context={'use_slurm': False}  # Force local execution
                    )
                    self.logger.info(f"Retrosynthesis prediction result status: {retro_result.get('status', 'unknown')}")
                    
                    # Reload molecular data to get updated predictions
                    self.logger.info("Reloading molecular data after retrosynthesis")
                    all_data = self._load_molecular_data()
                    
                    # Check if retrosynthesis results were stored for this molecule
                    if sample_id in all_data:
                        sample_data = all_data[sample_id]
                        starting_materials = sample_data.get('starting_smiles', [])
                        if starting_materials:  # Check if list is not empty
                            starting_material_smiles = starting_materials[0]
                            self.logger.info(f"Found starting material in master data: {starting_material_smiles}")
                        else:
                            self.logger.warning("No starting materials found in master data (empty list)")
                    else:
                        self.logger.warning(f"Sample {sample_id} not found in master data")
                        
                except Exception as e:
                    self.logger.error(f"Error in retrosynthesis prediction or data retrieval: {str(e)}")

                # If no starting material found in master data, use target molecule itself
                if not starting_material_smiles:
                    self.logger.info("Using target molecule as starting material for threshold calculation")
                    starting_material_smiles = target_smiles
                
                # Step 2: Run NMR simulation for target and starting materials
                self.logger.info(f"Running NMR simulation for target ({target_smiles}) and starting materials ({starting_material_smiles})")
                
                # Create a temporary JSON file for this specific simulation
                temp_json_path = self.temp_dir / f"temp_simulation_{sample_id}.json"
                
                # Split starting materials (they're separated by '.')
                starting_materials = starting_material_smiles.split('.')
                
                # Prepare simulation data structure - keep it simple with just required fields
                simulation_data = {}
                
                # Add target molecule
                simulation_data[f"{sample_id}_target"] = {
                    "smiles": target_smiles,
                    "sample_id": f"{sample_id}_target"
                }
                
                # Add starting materials
                for idx, start_mat in enumerate(starting_materials):
                    simulation_data[f"{sample_id}_starting_{idx}"] = {
                        "smiles": start_mat,
                        "sample_id": f"{sample_id}_starting_{idx}"
                    }
                
                # Write temporary JSON
                self.logger.debug(f"Writing simulation data to {temp_json_path}")
                self.logger.debug(f"Simulation data structure: {json.dumps(simulation_data, indent=2)}")
                with open(temp_json_path, 'w') as f:
                    json.dump(simulation_data, f, indent=2)
                
                try:
                    # Run simulation
                    self.logger.info("Starting NMR simulation")
                    sim_result = await self.nmr_tool.simulate(
                        str(temp_json_path),
                        context={'use_slurm': False}
                    )
                    self.logger.debug(f"NMR simulation result: {json.dumps(sim_result, indent=2)}")
                    
                    if sim_result['status'] != 'success':
                        error_msg = f"NMR simulation failed: {sim_result.get('message', 'Unknown error')}"
                        self.logger.error(error_msg)
                        return {
                            'status': 'error',
                            'message': error_msg
                        }
                    
                    # Load master data to get simulation results
                    self.logger.debug("Loading master data to retrieve simulation results")
                    master_data = self._load_molecular_data()

                    # Initialize threshold calculation data
                    self.logger.debug("Initializing threshold calculation data structure")
                    
                    # Get target simulation data from master data
                    target_key = f"{sample_id}_target"
                    self.logger.debug(f"Looking for target simulation data with key: {target_key}")
                    if target_key in master_data and 'nmr_data' in master_data[target_key]:
                        threshold_calc_data = {
                            'target_simulation': master_data[target_key]['nmr_data'],
                            'starting_material_simulations': [],
                            'calculation_timestamp': datetime.now().isoformat()
                        }
                        self.logger.debug(f"Found target simulation data: {json.dumps(threshold_calc_data['target_simulation'], indent=2)}")
                    else:
                        self.logger.warning(f"No NMR data found for target key {target_key}")
                        return {
                            'status': 'error',
                            'message': f"No NMR data found for target key {target_key}"
                        }
                    
                    # Get starting material simulation data from master data
                    for idx in range(len(starting_materials)):
                        start_key = f"{sample_id}_starting_{idx}"
                        if start_key in master_data and 'nmr_data' in master_data[start_key]:
                            threshold_calc_data['starting_material_simulations'].append(
                                master_data[start_key]['nmr_data']
                            )
                    
                    # Clean up temporary entries
                    keys_to_remove = [f"{sample_id}_target"] + [f"{sample_id}_starting_{i}" for i in range(len(starting_materials))]
                    for key in keys_to_remove:
                        master_data.pop(key, None)
                    
                    # Save master data once
                    with open(self.master_data_path, 'w') as f:
                        json.dump(master_data, f, indent=2)
                    
                finally:
                    # Clean up temporary file
                    if temp_json_path.exists():
                        temp_json_path.unlink()
                
                # Step 3: Run peak matching between target and starting material peaks
                self.logger.info("Starting peak matching process")
                self.logger.debug(f"Peak matching input - Target NMR: {json.dumps(threshold_calc_data['target_simulation'], indent=2)}")
                self.logger.debug(f"Peak matching input - Starting Material NMR: {json.dumps(threshold_calc_data['starting_material_simulations'][0], indent=2)}")
                
                # Get peaks from threshold calculation data
                target_peaks = threshold_calc_data.get('target_simulation', {})
                starting_material_peaks = threshold_calc_data.get('starting_material_simulations', [])
                
                # Check if we have valid peaks
                if not target_peaks:
                    return {
                        'status': 'error',
                        'message': 'No target peaks found for peak matching'
                    }
                if not starting_material_peaks:
                    return {
                        'status': 'error',
                        'message': 'No starting material peaks found for peak matching'
                    }

                def format_peaks(peaks, spectrum_type):
                    """Format peaks into the required structure for peak matching."""
                    if not peaks:  # Handle empty peaks list
                        if spectrum_type in ['1H', '13C']:
                            return {'shifts': [], 'Intensity': []}
                        else:  # 2D spectra (HSQC, COSY)
                            return {'F1 (ppm)': [], 'F2 (ppm)': [], 'Intensity': []}
                        
                    if spectrum_type == '1H':
                        # peaks is a list of [shift, intensity] lists
                        shifts = [float(p[0]) for p in peaks]
                        intensities = [float(p[1]) for p in peaks]
                        return {
                            'shifts': shifts,
                            'Intensity': intensities
                        }
                    elif spectrum_type == '13C':
                        # peaks is a list of shift values
                        shifts = [float(p) for p in peaks]
                        return {
                            'shifts': shifts,
                            'Intensity': [1.0] * len(shifts)
                        }
                    else:  # 2D spectra (HSQC, COSY)
                        # peaks is a list of [f1, f2] lists
                        f1_shifts = [float(p[0]) for p in peaks]
                        f2_shifts = [float(p[1]) for p in peaks]
                        return {
                            'F1 (ppm)': f1_shifts,
                            'F2 (ppm)': f2_shifts,
                            'Intensity': [1.0] * len(peaks)
                        }

                # Format NMR data for peak matching
                target_nmr = {}
                for spectrum_type, peaks in target_peaks.items():
                    if spectrum_type in ['1H_sim', '13C_sim', 'COSY_sim', 'HSQC_sim']:
                        target_nmr[spectrum_type[:-4]] = format_peaks(peaks, spectrum_type[:-4])

                starting_material_nmr = []
                for sm in starting_material_peaks:
                    sm_nmr = {}
                    for spectrum_type, peaks in sm.items():
                        if spectrum_type in ['1H_sim', '13C_sim', 'COSY_sim', 'HSQC_sim']:
                            sm_nmr[spectrum_type[:-4]] = format_peaks(peaks, spectrum_type[:-4])
                    starting_material_nmr.append(sm_nmr)

                # Extract specific NMR types
                self.logger.debug("Checking available spectra types")
                available_spectra = []
                for st in ['1H', '13C', 'HSQC', 'COSY']:
                    if st in target_nmr and st in starting_material_nmr[0]:
                        if st in ['1H', '13C']:
                            if len(target_nmr[st]['shifts']) > 0 and len(starting_material_nmr[0][st]['shifts']) > 0:
                                available_spectra.append(st)
                                self.logger.debug(f"Found valid {st} spectrum")
                        else:  # 2D spectra (HSQC, COSY)
                            if len(target_nmr[st]['F1 (ppm)']) > 0 and len(starting_material_nmr[0][st]['F1 (ppm)']) > 0:
                                available_spectra.append(st)
                                self.logger.debug(f"Found valid {st} spectrum")
                
                self.logger.info(f"Available spectra for peak matching: {available_spectra}")
                
                # Run peak matching between target and starting material peaks
                match_result = await self.peak_matching_tool.process(
                    {
                        'peaks1': target_nmr,  
                        'peaks2': starting_material_nmr[0]  # Taking only the first set
                    },
                    context={
                        'matching_mode': 'hung_dist_nn',
                        'error_type': 'sum',
                        'spectra': available_spectra
                    }
                )
                
                # Read results from the hardcoded location
                results_path = self.peak_matching_dir / 'current_run' / 'results.json'
                
                with open(results_path) as f:
                    peak_matching_results = json.load(f)
                
                if peak_matching_results['status'] == 'success':
                    # Format peak matching data for storage
                    self.logger.debug("Formatting peak matching results for storage")
                    peak_matching_data = {
                        'status': 'success',
                        'spectrum_errors': {},
                        'spectra': peak_matching_results['data']['results']
                    }
                    
                    # Extract spectrum errors
                    self.logger.debug("Extracting spectrum errors")
                    for spectrum_type, result in peak_matching_results['data']['results'].items():
                        if result['status'] == 'success':
                            error_val = result['overall_error']
                            peak_matching_data['spectrum_errors'][spectrum_type] = error_val
                            self.logger.debug(f"Error for {spectrum_type}: {error_val}")
                    
                    # Store peak matching results in master data for both target and starting material
                    all_data = self._load_molecular_data()
                    
                    # Store peak matching data for target molecule
                    if sample_id in all_data:
                        if 'exp_sim_peak_matching' not in all_data[sample_id]:
                            all_data[sample_id]['exp_sim_peak_matching'] = {}
                        all_data[sample_id]['exp_sim_peak_matching']['starting_material_comparison'] = peak_matching_data
                    
                    # Store peak matching data for starting material
                    for idx, sm_smiles in enumerate(starting_materials):
                        sm_id = f"{sample_id}_starting_{idx}"
                        if sm_id in all_data:
                            if 'exp_sim_peak_matching' not in all_data[sm_id]:
                                all_data[sm_id]['exp_sim_peak_matching'] = {}
                            all_data[sm_id]['exp_sim_peak_matching']['target_comparison'] = peak_matching_data
                    
                    # Save updated master data
                    self.logger.info("Saving updated master data with threshold results")
                    self._save_molecular_data(all_data)
                    
                    # Extract spectrum thresholds
                    spectrum_thresholds = {}
                    for spectrum_type, result in peak_matching_results['data']['results'].items():
                        if result['status'] == 'success':
                            spectrum_thresholds[spectrum_type] = result['overall_error']

                    # Calculate overall threshold
                    overall_threshold = sum(spectrum_thresholds.values()) / len(spectrum_thresholds)
                    self.logger.info(f"Calculated overall threshold: {overall_threshold}")
                    self.logger.debug(f"Individual spectrum thresholds: {json.dumps(spectrum_thresholds, indent=2)}")

                    # Extract just the matched_peaks from the results
                    simplified_results = {}
                    raw_results = peak_matching_results['data']['results']
                    for spectrum_type in raw_results:
                        simplified_results[spectrum_type] = {
                            'matched_peaks': raw_results[spectrum_type]['matched_peaks']
                        }

                    # Store threshold results in master data
                    threshold_data = {
                        'status': 'success',
                        'calculation_timestamp': datetime.now().isoformat(),
                        'type': 'peaks_vs_peaks',
                        'matching_mode': 'hung_dist_nn',
                        'error_type': 'sum',
                        'overall_threshold': overall_threshold,
                        'spectrum_thresholds': spectrum_thresholds,
                        'starting_material': starting_material_smiles,
                        'spectra': simplified_results  # Store only the matched peaks data
                    }
                    
                    all_data[sample_id]['threshold_data'] = threshold_data
                    self._save_molecular_data(all_data)
                    
                    return threshold_data
                    
                else:
                    return {
                        'status': 'error',
                        'message': f"Peak matching failed: {peak_matching_results.get('error', 'Unknown error')}"
                    }
                    
            except Exception as e:
                self.logger.error(f"Error in threshold calculation: {str(e)}")
                raise
                
        except Exception as e:
            self.logger.error(f"Threshold calculation failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/tools/spectral_comparison_tool.py ---
"""
Tool for comparing spectral data between candidates and experimental data.
"""
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
from datetime import datetime
from .data_extraction_tool import DataExtractionTool, DataSource
from .stout_operations import STOUTOperations
from .analysis_enums import DataSource

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpectralComparisonTool:
    """Tool for analyzing and comparing spectral data between candidates."""
    
    def __init__(self, llm_service=None):
        """Initialize the spectral comparison tool."""
        self.llm_service = llm_service
        self.data_tool = DataExtractionTool()
        self.stout_ops = STOUTOperations()
        
    async def analyze_spectral_comparison(self, 
                                  workflow_data: Dict[str, Any], 
                                  context: Dict[str, Any],
                                  data_tool: Optional[DataExtractionTool] = None,
                                  spectral_tool: Optional['SpectralComparisonTool'] = None,
                                  llm_service: Optional[Any] = None) -> Dict[str, Any]:
        """
        Analyze spectral data across different types and candidates.
        
        Args:
            workflow_data: Dictionary containing molecular data and analysis results
            context: Additional context for the analysis, including:
                - num_candidates: Number of top candidates to analyze (default: 2)
            data_tool: Optional DataExtractionTool instance. If not provided, uses self.data_tool
            spectral_tool: Optional SpectralComparisonTool instance. If not provided, uses self
            llm_service: Optional LLM service instance. If not provided, uses self.llm_service
            
        Returns:
            Dictionary containing spectral comparison results
        """
        try:
            # Use provided tools/services or fallback to instance ones
            data_tool = data_tool or self.data_tool
            spectral_tool = spectral_tool 
            llm_service = llm_service or self.llm_service
            
            # Extract top candidates from previous analysis
            molecule_data = workflow_data["molecule_data"]
            # Get number of candidates from context, default to 2 if not specified
            num_candidates = context.get('num_candidates', 2)
            logger.info(f"Analyzing top {num_candidates} candidates from context")
            
            candidates = await self._get_top_candidates(molecule_data, num_candidates)
            if not candidates:
                return {
                    'type': 'error',
                    'content': 'No candidates found for spectral comparison'
                }
            
            # Analyze each candidate's spectral data
            candidate_analyses = []
            for candidate in candidates:
                analysis = await self._analyze_candidate_spectra(candidate, molecule_data)
                candidate_analyses.append(analysis)
            
            # Store results in intermediate file
            try:
                sample_id = molecule_data.get('sample_id')
                if sample_id:
                    # Load existing data first
                    try:
                        existing_data = await data_tool.load_data(sample_id, DataSource.INTERMEDIATE)
                    except FileNotFoundError:
                        existing_data = {
                            'analysis_results': {},
                            'completed_analysis_steps': {}
                        }

                    # Update with new analysis
                    existing_data['analysis_results']['spectral_analysis'] = {
                        'type': 'structure_peak_correlation',
                        'candidate_analyses': [analysis for analysis in candidate_analyses],
                    }
                    
                    # Mark spectral analysis as completed
                    existing_data['completed_analysis_steps']['spectral_analysis'] = True
                    
                    await data_tool.save_data(
                        existing_data, 
                        sample_id, 
                        DataSource.INTERMEDIATE
                    )
            except Exception as e:
                logger.error(f"Error saving spectral analysis to intermediate file: {str(e)}")
            
            # Add error handling for LLM service
            if all(not analysis.get('analysis_text') for analysis in candidate_analyses):
                logger.warning("No valid LLM analyses were generated. Check API key configuration.")
                
            return {
                'type': 'spectral_comparison',
                'content': {
                    'spectral_analysis': {
                        'candidate_analyses': [
                            {
                                'candidate_id': candidate.get('id'),
                                'smiles': candidate.get('smiles'),
                                'analysis': analysis
                            }
                            for candidate, analysis in zip(candidates, candidate_analyses)
                        ]
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error in spectral comparison: {str(e)}")
            return {
                'type': 'error',
                'content': str(e)
            }
            
    async def _get_top_candidates(self, molecule_data: Dict[str, Any], num_candidates: int = 2) -> List[Dict[str, Any]]:
        """Extract top candidates from previous analysis results.
        
        Args:
            molecule_data: Dictionary containing the sample_id and other molecular data
            num_candidates: Number of top candidates to return (default: 2)
            
        Returns:
            List of top N candidate molecules with their data
        """
        try:
            sample_id = molecule_data.get('sample_id')
            if not sample_id:
                logger.error("No sample_id found in molecule_data")
                return []
            
            # Try to load from intermediate file first, then master file
            for data_source in [DataSource.INTERMEDIATE, DataSource.MASTER_FILE]:
                logger.info(f"Attempting to load data from {data_source.value}")
                data = await self.data_tool.load_data(sample_id, data_source)
                
                if data and 'analysis_results' in data:
                    analysis_results = data['analysis_results']
                    logger.info(f"Found analysis_results with keys: {list(analysis_results.keys())}")
                    
                    if 'candidate_ranking' in analysis_results:
                        ranked_candidates = analysis_results['candidate_ranking'].get('ranked_candidates', [])
                        if ranked_candidates: 
                            logger.info(f"Found {len(ranked_candidates)} ranked candidates in {data_source.value}, returning top {num_candidates}")
                            return ranked_candidates[:num_candidates]  # Get top N candidates
            
            logger.warning(f"No ranked candidates found for sample {sample_id}")
            return []
            
        except Exception as e:
            logger.error(f"Error getting top candidates: {str(e)}")
            return []
            
    async def _analyze_candidate_spectra(self, candidate: Dict[str, Any], molecule_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze spectral data for a single candidate using the existing matching data.
        """
        try:
            # Extract the NMR data that already contains matching information
            nmr_data = candidate.get('nmr_data', {}).get('spectra', {})
            smiles = candidate.get('smiles')
            logger.info(f"Analyzing NMR data for candidate with SMILES: {smiles}")
            
            if not nmr_data:
                logger.warning(f"No NMR data found for candidate with SMILES: {smiles}")
                return {
                    'error': 'No NMR data available for analysis',
                    'candidate_id': candidate.get('rank'),
                    'smiles': smiles
                }
            
            # Analyze each spectrum type
            spectrum_analyses = {}
            valid_analyses = 0
            
            for spectrum_type in [ 'HSQC']: #['1H', '13C', 'HSQC', 'COSY']:
                if spectrum_type in nmr_data:
                    analysis = await self._analyze_matched_spectrum(
                        nmr_data[spectrum_type],
                        spectrum_type,
                        smiles,
                        candidate
                    )
                    # print("_____analysis")
                    # print(analysis)
                    # Check if analysis was successful
                    if analysis and analysis.get('structural_analysis', {}).get('analysis_text'):
                        valid_analyses += 1
                    # logger.info(f"___{spectrum_type} analysis: {analysis}")  # Comment out noisy logging
                    
                    spectrum_analyses[f"{spectrum_type}_analysis"] = analysis
            
            # Log analysis results
            if valid_analyses == 0:
                logger.warning(f"No valid spectral analyses generated for candidate with SMILES: {smiles}")
            else:
                logger.info(f"Generated {valid_analyses} valid spectral analyses for candidate with SMILES: {smiles}")
            
            return {
                'candidate_id': candidate.get('rank'),  # Using rank as ID
                'smiles': smiles,
                'iupac_name': candidate.get('iupac_name'),
                'HSQC_score': candidate.get('scores', {}).get('HSQC'),
                'spectrum_analyses': spectrum_analyses,
            }
            
        except Exception as e:
            logger.error(f"Error analyzing candidate spectra: {str(e)}")
            return {
                'error': str(e),
                'candidate_id': candidate.get('rank'),
                'smiles': candidate.get('smiles')
            }
            
    async def _analyze_matched_spectrum(self, spectrum_data: Dict[str, Any], spectrum_type: str, smiles: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze matched peaks between experimental and predicted spectra.
        
        Args:
            spectrum_data: Dictionary containing matched peak data
            spectrum_type: Type of NMR spectrum (1H, 13C, HSQC, COSY)
            smiles: SMILES string of the molecule
            candidate: Dictionary containing candidate molecule data including structure image
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # print("spectrum_data_____")
            # print(spectrum_data)
            # Extract matched peaks from spectrum data
            spectrum1 = spectrum_data.get('spectrum1', [])
            spectrum2 = spectrum_data.get('spectrum2', [])
            
            # Determine if this is a 2D spectrum
            is_2d = spectrum_type in ['HSQC', 'COSY']
            
            # Collect matching information
            peak_matches = []
            for peak1, peak2 in zip(spectrum1, spectrum2):
                if is_2d:
                    match_info = {
                        'experimental': {
                            'F1': peak1['F1 (ppm)'],  # Direct dictionary access
                            'F2': peak1['F2 (ppm)']
                        },
                        'predicted': {
                            'F1': peak2['F1 (ppm)'],
                            'F2': peak2['F2 (ppm)']
                        },
                        'error': peak1['Error'],
                        'atom_index': peak1['atom_index']
                    }
                else:
                    match_info = {
                        'experimental': {
                            'shift': peak1['shifts'],
                            'intensity': peak1['intensity']
                        },
                        'predicted': {
                            'shift': peak2['shifts'],
                            'intensity': peak2['intensity']
                        },
                        'error': peak1['Error'],
                        'atom_index': peak1['atom_index']
                    }
                peak_matches.append(match_info)
            
            # Get IUPAC name from intermediate data
            iupac_result = await self.stout_ops.get_iupac_name(smiles)

            # Format peak matches for prompt
            formatted_peaks = self._format_peak_matches(peak_matches, is_2d)
            
            # Create vision prompt
            prompt = f"""
First, I will provide a detailed structural description of the molecule shown in the image:
    
1. Molecular Structure Description:
   - IUPAC Name: {iupac_result.get('iupac_name', 'Not available')}
   - Describe the complete molecular structure systematically, starting from a core feature
   - Note the connectivity and spatial arrangement of all atoms
   - Include atom numbering/labels as shown in the image
   - Identify key functional groups and structural motifs
   - Describe any notable stereochemistry or conformational features
   - Ensure the description is very detailed to allow reconstruction of the structure
   - Explain how the structure corresponds to its IUPAC name, particularly focusing on the parts relevant to the NMR analysis

2. Evaluate how well this candidate molecule's simulated {spectrum_type} NMR spectrum matches the experimental data.

The image shows the proposed molecular structure with numbered atoms. These numbers correspond to the peak match data below, which compares experimental vs simulated chemical shifts.

SMILES: {smiles}

Peak Match Data (Experimental vs Simulated):
{formatted_peaks}

Please analyze and provide a detailed evaluation:

1. Overall Match Quality:
   - Is this a good overall match between experimental and simulated spectra?
   - What is your confidence level in this structure based on the spectral match?

2. Region-by-Region Analysis:
   - Which regions show excellent agreement? 
   - Which regions show concerning deviations? 
   - Are the deviations systematic or random?

3. Structural Validation:
   - Do the well-matched regions confirm key structural features?
   - Do any mismatches suggest potential structural errors?
   - Are there specific substructures that should be reconsidered?

4. Detailed Chemical Environment Analysis:
   Please analyze each point thoroughly, providing clear explanations and reasoning:

   a) Electronic Effects Analysis:
      - Examine each significant deviation systematically
      - For each deviation, consider and explain:
         * Electron-withdrawing/donating effects from nearby groups
         * Resonance effects and their expected impact
         * Inductive effects and their influence
         * How these effects compare to similar known structures
      - Explain why these effects support or contradict the proposed structure

   b) Structural Motif Evaluation:
      - Systematically analyze each structural feature present:
         * Aromatic systems: Ring current effects and their expected impact
         * Heteroatoms: Their influence on neighboring atoms
         * Functional groups: Their characteristic effects on chemical shifts
      - Compare observed vs. expected effects for each motif
      - Identify any inconsistencies and explain their significance

   c) Conformational Analysis:
      - Consider possible conformers of the structure
      - Explain how different conformations might affect:
         * Chemical shifts
         * Coupling patterns
         * Through-space interactions
      - Assess whether conformational flexibility could explain any discrepancies

Comprehensive Conclusion:
1. Structure Validity Assessment:
   - Provide a detailed evaluation of structural correctness
   - Support your conclusion with specific evidence from each analysis section
   - Explain the relative importance of each piece of evidence

2. Detailed Structural Issues:
   a) Problematic Atom Analysis:
      - List each atom index showing significant deviation
      - For each atom:
         * Quantify the deviation
         * Explain the expected vs. observed chemical environment
         * Provide specific reasoning for the mismatch

   b) Regional Analysis:
      - Identify and describe problematic molecular regions
      - For each region:
         * Explain the specific spectral mismatches
         * Analyze the chemical environment
         * Discuss potential alternative structures

   c) Comprehensive Reasoning:
      - Connect all observations into a coherent explanation
      - Explain the chemical principles behind each conclusion
      - Address any apparent contradictions in the data

3. Improvement Suggestions:
   - Propose specific structural modifications
   - For each modification:
      * Explain the chemical reasoning
      * Predict the expected spectral changes
      * Discuss potential improvements to the match

Final Evaluation:
Provide a thorough assessment of whether the experimental NMR data supports the proposed structure. Consider all evidence presented above and explain any uncertainties or limitations in the analysis.

Remember to:
- Support each conclusion with specific data points
- Explain your reasoning in clear, logical steps
- Consider alternative explanations for discrepancies
- Highlight both supporting and contradicting evidence
- Be explicit about confidence levels in each conclusion"""
            
            # logger.info("____prompt____")
            # logger.info(prompt)
            # Use vision capabilities for analysis
            try:
                analysis_result = await self.llm_service.analyze_with_vision(
                    prompt=prompt,
                    image_path=candidate['structure_image'],
                    model="claude-3-5-sonnet",  # Using better vision model
                    system=("You are an expert in NMR spectroscopy and structural analysis."
                           "Analyze how well the predicted NMR peaks match the experimental data, "
                           "focusing on structural features and chemical environments.")
                )
                
                # Ensure we have a valid analysis result
                if not isinstance(analysis_result, dict):
                    logger.warning(f"Invalid analysis result format: {type(analysis_result)}")
                    logger.warning(f"analysis_result: {analysis_result}")
                    analysis_result = {
                        'analysis_text': ''}
                
                return {
                    'type': spectrum_type,
                    'reasoning': analysis_result.get('analysis_text', ''),
                    'formatted_peaks': formatted_peaks,
                    'prompt': prompt,
                    'peak_matches': peak_matches,
                } 
                
            except Exception as e:
                logger.error(f"Error in vision analysis: {str(e)}")
                return {
                    'type': spectrum_type,
                    'reasoning': "",
                    'formatted_peaks': formatted_peaks,
                    'prompt': prompt,
                    'peak_matches': peak_matches,
                }
            
        except Exception as e:
            logger.error(f"Error analyzing matched spectrum {spectrum_type}: {str(e)}")
            return {}
            
    def _format_peak_matches(self, peak_matches: List[Dict[str, Any]], is_2d: bool) -> str:
        """Format peak matches in a chemically meaningful way."""
        formatted = []
        for match in peak_matches:
            exp = match['experimental']
            pred = match['predicted']
            error = match['error']
            atom_index = match['atom_index']

            if is_2d:
                # Format 2D NMR data
                formatted.append(
                    f"Atom {atom_index}: \n"
                    f"  Experimental: F1 δ {exp['F1']:.2f} ppm, F2 δ {exp['F2']:.2f} ppm\n"
                    f"  Predicted: F1 δ {pred['F1']:.2f} ppm, F2 δ {pred['F2']:.2f} ppm\n"
                    f"  Error: {error:.3f}"
                )
            else:
                # Format 1D NMR data
                formatted.append(
                    f"Atom {atom_index}: \n"
                    f"  Experimental: δ {exp['shift']:.2f} ppm, Intensity: {exp['intensity']:.2f}\n"
                    f"  Predicted: δ {pred['shift']:.2f} ppm, Intensity: {pred['intensity']:.2f}\n"
                    f"  Error: {error:.4f}"
                )
        return "\n".join(formatted)

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/tools/candidate_analyzer_tool.py ---
"""Tool for analyzing candidate molecules from various prediction sources."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
from rdkit import Chem
import os
import pandas as pd
import ast

from .peak_matching_tool import EnhancedPeakMatchingTool
from .nmr_simulation_tool import NMRSimulationTool

class CandidateAnalyzerTool:
    """Tool for analyzing and scoring candidate molecules."""

    def __init__(self, analysis_type: str = None):
        """Initialize the candidate analyzer tool.
        
        Args:
            analysis_type: Type of analysis to perform ('forward', 'mol2mol', 'mmst', or None for all)
        """
        self.logger = logging.getLogger(__name__)
        self.peak_matcher = EnhancedPeakMatchingTool()
        self.nmr_tool = NMRSimulationTool()
        self.analysis_type = analysis_type
        
        # Set up paths
        self.base_dir = Path(__file__).parent.parent.parent
        self.master_data_path = self.base_dir / "data" / "molecular_data" / "molecular_data.json"
        self.temp_dir = self.base_dir / "_temp_folder"
        self.intermediate_dir = self.temp_dir / "intermediate_results"
        
        # Create all necessary directories at initialization
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.intermediate_dir.mkdir(exist_ok=True)

    def _get_intermediate_path(self, sample_id: str) -> Path:
        """Get path to intermediate file for a sample."""
        return self.intermediate_dir / f"{sample_id}_intermediate.json"


    def _load_or_create_intermediate(self, sample_id: str, context: Dict[str, Any] = None) -> Dict:
        """Load existing intermediate file or create new one from master data.
        """
        intermediate_path = self._get_intermediate_path(sample_id)
        
        # Try loading existing intermediate
        if intermediate_path.exists():
            with open(intermediate_path, 'r') as f:
                data = json.load(f)
                if 'molecule_data' not in data:
                    data['molecule_data'] = {}
                return data
                
        # If no intermediate exists, try getting data from master file
        master_path = self.base_dir / "data" / "molecular_data" / "molecular_data.json"
        if master_path.exists():
            with open(master_path, 'r') as f:
                master_data = json.load(f)

                if sample_id in master_data:
                    # Create new intermediate with just this sample's data
                    intermediate_data = master_data[sample_id]
                    self._save_intermediate(sample_id, intermediate_data)
                    return intermediate_data
                    
        # If neither exists and we have context, create new intermediate
        if context:
            intermediate_data["molecule_data"] = context
            self._save_intermediate(sample_id, intermediate_data)
            return intermediate_data
            
        raise ValueError(f"No data found for sample {sample_id}")

    def _save_intermediate(self, sample_id: str, data: Dict):
        """Save data to intermediate file."""
        path = self._get_intermediate_path(sample_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_molecular_data(self) -> Dict:
        """Load the molecular data from JSON file."""
        if self.master_data_path.exists():
            with open(self.master_data_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_molecular_data(self, data: Dict):
        """Save the molecular data to JSON file."""
        self.master_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.master_data_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_intermediate_results(self, sample_id: str, data: Dict[str, Any]) -> Path:
        """Save intermediate results for a sample to a temporary file.
        
        Args:
            sample_id: ID of the sample being processed
            data: Data to save for this sample
            
        Returns:
            Path to the saved intermediate file
        """
        # Create intermediate directory if it doesn't exist
        intermediate_dir = self.temp_dir / "intermediate_results"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # Log directory creation status
        if intermediate_dir.exists():
            self.logger.info(f"Intermediate directory exists: {intermediate_dir}")
        else:
            self.logger.error(f"Failed to create intermediate directory: {intermediate_dir}")
            raise RuntimeError(f"Could not create intermediate directory: {intermediate_dir}")
    
        # Use a consistent filename for this sample
        intermediate_file = intermediate_dir / f"{sample_id}_intermediate.json"
        
        # Create parent directories for the file if needed
        intermediate_file.parent.mkdir(parents=True, exist_ok=True)
    
        # Save the data
        try:
            with open(intermediate_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Successfully saved intermediate results to: {intermediate_file}")
        except Exception as e:
            self.logger.error(f"Failed to save intermediate results: {str(e)}")
            raise
        
        return intermediate_file

    def _combine_intermediate_results(self, intermediate_files: List[Path]) -> None:
        """Combine all intermediate results into the master data file.
        
        Args:
            intermediate_files: List of paths to intermediate result files
        """
        # Load current master data
        master_data = self._load_molecular_data()
        
        # Process each intermediate file
        for file_path in intermediate_files:
            with open(file_path, 'r') as f:
                sample_data = json.load(f)
                
            # Update master data with this sample's results
            for sample_id, data in sample_data.items():
                if sample_id not in master_data:
                    master_data[sample_id] = {}
                master_data[sample_id].update(data)
        
        # Save updated master data
        self._save_molecular_data(master_data)
        
        # Clean up intermediate files
        for file_path in intermediate_files:
            file_path.unlink()

    def _canonicalize_smiles(self, smiles: str) -> str:
        """Canonicalize SMILES string using RDKit.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Canonicalized SMILES string or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.logger.warning(f"Invalid SMILES string: {smiles}")
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            self.logger.error(f"Error canonicalizing SMILES {smiles}: {str(e)}")
            return None

    def _collect_unique_predictions(
        self,
        forward_predictions: List[Dict[str, Any]]
        ) -> Dict[str, Dict[str, Any]]:
        """Collect and deduplicate predictions based on canonicalized SMILES.
        
        Args:
            forward_predictions: List of prediction dictionaries
            
        Returns:
            Dictionary mapping canonicalized SMILES to their prediction info
        """
        unique_predictions: Dict[str, Dict[str, Any]] = {}
        
        # Sort predictions by log_likelihood to process most likely predictions first
        sorted_predictions = sorted(
            forward_predictions,
            key=lambda x: x.get("log_likelihood", float("-inf")),
            reverse=True
        )
        
        for prediction in sorted_predictions:
            if "all_predictions" not in prediction or not prediction["all_predictions"]:
                self.logger.warning("Skipping prediction with no all_predictions data")
                continue
                
            starting_material = prediction.get("starting_material", "")
            log_likelihood = prediction.get("log_likelihood", None)
            
            for smiles in prediction["all_predictions"]:
                canon_smiles = self._canonicalize_smiles(smiles)
                if not canon_smiles:
                    continue
                    
                # Only store the first (most likely) prediction for each unique molecule
                if canon_smiles not in unique_predictions:
                    unique_predictions[canon_smiles] = {
                        "starting_material": starting_material,
                        "log_likelihood": log_likelihood
                    }
                
        return unique_predictions

    async def _process_forward_synthesis(
        self,
        forward_predictions: List[Dict[str, Any]],
        sample_data: Dict[str, Any]
        ) -> Dict[str, Any]:
        """Process forward synthesis predictions."""
        self.logger.info(f"Starting forward synthesis processing with {len(forward_predictions)} predictions")
        self.logger.info(f"Sample data keys available: {list(sample_data.keys())}")
        
        result = {
            "tool_version": "1.0",
            "molecules": []
        }

        # First collect and deduplicate all predictions
        self.logger.info("Collecting and deduplicating predictions...")
        unique_predictions = self._collect_unique_predictions(forward_predictions)
        self.logger.info(f"Found {len(unique_predictions)} unique molecules after deduplication")
        
        # Try to get NMR predictions, but continue even if they fail
        nmr_predictions = {}
        try:
            self.logger.info("Starting batch NMR predictions...")
            nmr_predictions = await self._batch_process_nmr_predictions(
                    unique_predictions,
                    sample_data.get('sample_id', 'unknown')  # Pass sample_id here
                )            # self.logger.info(f"NMR predictions: {json.dumps(nmr_predictions, indent=2)}")

            self.logger.info(f"Completed NMR predictions for {len(nmr_predictions)} molecules")
        except Exception as e:
            self.logger.warning(f"NMR predictions failed: {str(e)}. Continuing without NMR data.")
        
        # Process each unique molecule
        self.logger.info("Starting analysis of individual molecules...")
        for canon_smiles, prediction_info in unique_predictions.items():
            self.logger.info(f"Analyzing molecule with SMILES: {canon_smiles}")
            molecule_result = await self._analyze_single_molecule(
                canon_smiles,  # Use canonicalized SMILES
                sample_data,
                source_info={
                    "source": "forward_synthesis",
                    "starting_material": prediction_info["starting_material"],
                    "log_likelihood": prediction_info["log_likelihood"]
                    },
                nmr_predictions=nmr_predictions.get(canon_smiles)  # Pass pre-computed predictions
                )
            if molecule_result:
                self.logger.info(f"Successfully analyzed molecule: {canon_smiles}")
                result["molecules"].append(molecule_result)
            else:
                self.logger.warning(f"Analysis failed for molecule: {canon_smiles}")

        self.logger.info(f"Processed {len(forward_predictions)} predictions into {len(result['molecules'])} unique molecules")
        return result

    async def _batch_process_nmr_predictions(
        self,
        unique_predictions: Dict[str, Dict[str, Any]],
        sample_id: str
        ) -> Dict[str, Dict[str, Any]]:
        """Process NMR predictions for a batch of unique molecules.
        
        Args:
            unique_predictions: Dictionary mapping canonicalized SMILES to their prediction info
            sample_id: ID of the sample being analyzed
            
        Returns:
            Dictionary mapping canonicalized SMILES to their NMR prediction data
        """
        predictions_data = {}
        failed_molecules = []
        
        try:
            # Create a temporary master JSON file for batch processing
            temp_master_data = {}
            prediction_keys = {}  # Map canonical SMILES to their temporary keys
            
            # Process each unique molecule with a properly structured sample ID
            for idx, (canon_smiles, prediction_info) in enumerate(unique_predictions.items(), start=1):
                try:
                    # Validate SMILES before adding to batch
                    if not self._canonicalize_smiles(canon_smiles):
                        self.logger.warning(f"Invalid SMILES string: {canon_smiles}")
                        failed_molecules.append((canon_smiles, "Invalid SMILES"))
                        continue
                        
                    # Create unique sample ID based on source and index
                    #source = prediction_info.get("starting_material", prediction_info.get("parent_smiles", "unknown"))
                    temp_key = f"pred_{idx}_{sample_id}"
                    
                    # Store mapping for later retrieval
                    prediction_keys[canon_smiles] = temp_key
                    
                    # Add molecule data with proper structure
                    temp_master_data[temp_key] = {
                        "smiles": canon_smiles,
                        "sample_id": temp_key
                    }
                except Exception as e:
                    self.logger.error(f"Error processing molecule {canon_smiles}: {str(e)}")
                    failed_molecules.append((canon_smiles, str(e)))
                    continue
            
            if not temp_master_data:
                self.logger.warning("No valid molecules to process")
                return predictions_data
            
            # Write temporary master data to file
            temp_master_path = self.temp_dir / "temp_master.json"

            with open(temp_master_path, "w") as f:
                json.dump(temp_master_data, f, indent=2)
            
            # Debug: Print the first few entries of temp_master_data
            temp_master_str = json.dumps(temp_master_data, indent=2)
            self.logger.info(f"First 500 characters of temp_master_data:\n{temp_master_str[:500]}")
            
            try:
                # Initialize NMR simulation tool
                nmr_tool = NMRSimulationTool()
                
                # Run batch NMR prediction
                # self.logger.info(f"Starting NMR prediction with {len(unique_predictions)} molecules")
                # self.logger.info(f"First 500 characters of temp_master_data:\n{temp_master_str[:500]}")
                # self.logger.info(f"Path of temp_master_path file: {temp_master_path}")

                result = await nmr_tool.simulate_batch(  
                    str(temp_master_path),
                    context={"use_slurm": False}
                )
                
                # self.logger.info(f"NMR prediction result: {result}")
                
                if result["status"] != "success":
                    self.logger.info(f"Batch NMR prediction failed: {result.get('message', 'Unknown error')}")
                    return predictions_data
                
                # Read the compiled results file
                result_file = result['data']['result_file']
                self.logger.info(f"Reading NMR predictions from {result_file}")
                try:
                    df = pd.read_csv(result_file)
                    # print("_---------------------------------------------------------")
                    # print(df)
                    # Process each molecule's predictions
                    for canon_smiles, temp_key in prediction_keys.items():
                        try:
                            # Find the row for this molecule
                            row = df[df['sample-id'] == temp_key]
                            if not row.empty:
                                # Extract NMR predictions
                                nmr_data = {}
                                
                                # Parse 1H NMR data
                                if '1H_NMR_sim' in row:
                                    try:
                                        peaks_str = row['1H_NMR_sim'].iloc[0]
                                        peaks = ast.literal_eval(peaks_str) if peaks_str else []
                                        nmr_data['1H_sim'] = peaks
                                    except Exception as e:
                                        self.logger.error(f"Error parsing 1H NMR data: {str(e)}")
                                        nmr_data['1H_sim'] = []

                                # Parse 13C NMR data
                                if '13C_NMR_sim' in row:
                                    try:
                                        peaks_str = row['13C_NMR_sim'].iloc[0]
                                        peaks = ast.literal_eval(peaks_str) if peaks_str else []
                                        nmr_data['13C_sim'] = peaks
                                    except Exception as e:
                                        self.logger.error(f"Error parsing 13C NMR data: {str(e)}")
                                        nmr_data['13C_sim'] = []

                                # Parse COSY data
                                if 'COSY_sim' in row:
                                    try:
                                        peaks_str = row['COSY_sim'].iloc[0]
                                        peaks = ast.literal_eval(peaks_str) if peaks_str else []
                                        nmr_data['COSY_sim'] = peaks
                                    except Exception as e:
                                        self.logger.error(f"Error parsing COSY data: {str(e)}")
                                        nmr_data['COSY_sim'] = []

                                # Parse HSQC data
                                if 'HSQC_sim' in row:
                                    try:
                                        peaks_str = row['HSQC_sim'].iloc[0]
                                        peaks = ast.literal_eval(peaks_str) if peaks_str else []
                                        nmr_data['HSQC_sim'] = peaks
                                    except Exception as e:
                                        self.logger.error(f"Error parsing HSQC data: {str(e)}")
                                        nmr_data['HSQC_sim'] = []
                                predictions_data[canon_smiles] = nmr_data
                                self.logger.info(f"Found NMR predictions for {canon_smiles}")
                            else:
                                self.logger.warning(f"No predictions found for {canon_smiles} in results file")
                                failed_molecules.append((canon_smiles, "No predictions in results file"))
                        except Exception as e:
                            self.logger.error(f"Error processing predictions for {canon_smiles}: {str(e)}")
                            failed_molecules.append((canon_smiles, str(e)))
                            
                except Exception as e:
                    self.logger.error(f"Error reading results file: {str(e)}")
                    return predictions_data
                
            finally:
                # Clean up temporary file
                if temp_master_path.exists():
                    temp_master_path.unlink()
            
        except Exception as e:
            self.logger.error(f"Error in batch NMR prediction: {str(e)}")
            # Don't raise - return partial results if any
        
        # Log summary of failures
        if failed_molecules:
            self.logger.warning(f"Failed to process {len(failed_molecules)} molecules:")
            for smiles, error in failed_molecules:
                self.logger.warning(f"  - {smiles}: {error}")
        
        self.logger.info(f"Successfully processed {len(predictions_data)} out of {len(unique_predictions)} molecules")
        return predictions_data

    async def _process_mol2mol(
        self,
        mol2mol_results: Dict[str, Any],
        sample_data: Dict[str, Any]
        ) -> Dict[str, Any]:
        """Process mol2mol predictions.
        
        Args:
            mol2mol_results: Dictionary containing mol2mol results with generated analogues
            sample_data: Sample data containing NMR experimental data and other metadata
        """
        self.logger.info("Starting mol2mol processing")
        self.logger.debug(f"Input mol2mol_results structure: {mol2mol_results}")
        self.logger.debug(f"Sample data keys: {list(sample_data.keys())}")
        
        result = {
            "tool_version": "1.0",
            "molecules": []
        }

        # First collect and deduplicate all predictions
        unique_predictions = {}
        generated_analogues = mol2mol_results.get("generated_analogues_target", {})
        self.logger.info(f"Found {len(generated_analogues)} target SMILES in generated_analogues_target")
        
        for target_smiles, analogues in generated_analogues.items():
            self.logger.debug(f"Processing analogues for target SMILES: {target_smiles}")
            self.logger.debug(f"Number of analogues: {len(analogues)}")
            
            for analogue_smiles in analogues:
                canon_smiles = self._canonicalize_smiles(analogue_smiles)
                if not canon_smiles:
                    self.logger.warning(f"Failed to canonicalize SMILES: {analogue_smiles}")
                    continue
                    
                if canon_smiles not in unique_predictions:
                    unique_predictions[canon_smiles] = {
                        "parent_smiles": target_smiles,
                        "source": "mol2mol"
                    }
        
        self.logger.info(f"Collected {len(unique_predictions)} unique predictions after canonicalization")
        
        # Batch process NMR predictions for all unique molecules
        self.logger.info("Starting NMR predictions batch processing")
        nmr_predictions = await self._batch_process_nmr_predictions(
                unique_predictions,
                sample_data.get('sample_id', 'unknown')
            )
        self.logger.info(f"Completed NMR predictions for {len(nmr_predictions)} molecules")

        # Process each unique molecule
        self.logger.info("Processing individual molecules")
        for canon_smiles, prediction_info in unique_predictions.items():
            self.logger.debug(f"Analyzing molecule: {canon_smiles}")
            molecule_result = await self._analyze_single_molecule(
                canon_smiles,
                sample_data,
                source_info=prediction_info,
                nmr_predictions=nmr_predictions.get(canon_smiles)
            )
            if molecule_result:
                result["molecules"].append(molecule_result)
            else:
                self.logger.warning(f"No result produced for molecule: {canon_smiles}")

        self.logger.info(f"Completed processing {len(unique_predictions)} unique mol2mol analogues")
        self.logger.info(f"Final result contains {len(result['molecules'])} molecules")
        return result

    async def _process_mmst(
        self,
        mmst_results: Dict[str, Any],
        sample_data: Dict[str, Any]
        ) -> Dict[str, Any]:
        """Process MMST predictions.
        
        Args:
            mmst_results: Dictionary containing MMST results with generated analogues
            sample_data: Sample data containing NMR experimental data and other metadata
        """
        result = {
            "tool_version": "1.0",
            "molecules": []
        }

        # Print MMST results
        # self.logger.info(f"MMST results: {json.dumps(mmst_results, indent=2)}")

        # First collect and deduplicate all predictions
        unique_predictions = {}
        for target_smiles, analogues in mmst_results.get("generated_analogues_target", {}).items():
            for analogue_smiles in analogues:
                canon_smiles = self._canonicalize_smiles(analogue_smiles)
                if not canon_smiles:
                    continue
                    
                if canon_smiles not in unique_predictions:
                    unique_predictions[canon_smiles] = {
                        "parent_smiles": target_smiles,
                        "source": "mmst"
                    }
        
        # Batch process NMR predictions for all unique molecules
        nmr_predictions = await self._batch_process_nmr_predictions(
                unique_predictions,
                sample_data.get('sample_id', 'unknown')
            )

        # Process each unique molecule
        for canon_smiles, prediction_info in unique_predictions.items():
            molecule_result = await self._analyze_single_molecule(
                canon_smiles,
                sample_data,
                source_info=prediction_info,
                nmr_predictions=nmr_predictions.get(canon_smiles)
            )
            if molecule_result:
                result["molecules"].append(molecule_result)

        self.logger.info(f"Processed {len(unique_predictions)} unique MMST predictions")
        return result


    async def _process_structure_generator(
        self,
        structure_predictions: List[Dict[str, Any]],
        experimental_data: Dict[str, Any]
        ) -> Dict[str, Any]:
        """Process structure generator predictions."""
        result = {
            "tool_version": "1.0",
            "molecules": []
        }

        for prediction in structure_predictions:
            molecule_result = await self._analyze_single_molecule(
                prediction["smiles"],
                experimental_data,
                source_info={
                    "method": prediction.get("method", "unknown"),
                    "starting_smiles": prediction.get("starting_point", "")
                }
            )
            if molecule_result:
                result["molecules"].append(molecule_result)

        return result

    async def _analyze_single_molecule(
        self,
        smiles: str,
        experimental_data: Dict[str, Any],
        source_info: Dict[str, Any],
        nmr_predictions: Optional[Dict[str, Any]] = None
        ) -> Optional[Dict[str, Any]]:
        """Analyze a single molecule against experimental data.
        
        Args:
            smiles: SMILES string of the molecule to analyze
            experimental_data: Dictionary containing experimental NMR data under 'nmr_data' key
            source_info: Information about how this molecule was generated
            nmr_predictions: Optional pre-computed NMR predictions for this molecule
        """
        # Get intermediate file path from experimental data
        # intermediate_file = Path(experimental_data.get("_intermediate_file_path"))
        # if not intermediate_file.exists():
        #     raise ValueError(f"Intermediate file not found: {intermediate_file}")

        try:
            self.logger.info(f"Starting analysis for molecule: {smiles}")
            self.logger.info(f"Available experimental data keys: {list(experimental_data.keys())}")
            self.logger.info(f"Source info: {source_info}")
             
            # Extract experimental NMR data
            nmr_data = experimental_data.get("nmr_data", {})
            if not nmr_data:
                self.logger.warning(f"No experimental NMR data found for molecule {smiles}")
                return None
                
            # Log available NMR data
            self.logger.info(f"Available experimental NMR data keys: {list(nmr_data.keys())}")
            if nmr_predictions:
                self.logger.info(f"Available predicted NMR data keys: {list(nmr_predictions.keys())}")
            
            experimental_spectra = {}
            predicted_spectra = {}
            
            # Format experimental peaks for each spectrum type
            for spectrum_type in ["1H", "13C", "HSQC", "COSY"]:
                exp_key = f"{spectrum_type}_exp"
                if exp_key in nmr_data and nmr_data[exp_key]:
                    try:
                        self.logger.info(f"Processing experimental {spectrum_type} peaks:")
                        # self.logger.info(f"Raw peaks: {nmr_data[exp_key]}")
                        
                        formatted_peaks = self._format_peaks(
                            nmr_data[exp_key], 
                            spectrum_type
                        )
                        if formatted_peaks:
                            experimental_spectra[spectrum_type] = formatted_peaks
                    except Exception as e:
                        self.logger.error(f"Error formatting experimental peaks for {spectrum_type}: {str(e)}")
                        continue

            # self.logger.info(f"nmr_predictions {nmr_predictions}")

            # Format predicted peaks for each spectrum type
            if nmr_predictions:
                for spectrum_type in ["1H", "13C", "HSQC", "COSY"]:
                    pred_key = f"{spectrum_type}_sim"  # Predicted data uses _sim suffix
                    # exp_key = f"{spectrum_type}_exp"  # Experimental data uses _exp suffix
                    
                    if pred_key in nmr_predictions and nmr_predictions[pred_key]:
                        try:
                            formatted_peaks = self._format_peaks(
                                nmr_predictions[pred_key],
                                spectrum_type
                            )
                            if formatted_peaks:
                                predicted_spectra[spectrum_type] = formatted_peaks
                        except Exception as e:
                            self.logger.error(f"Error formatting predicted peaks for {spectrum_type}: {str(e)}")
                            continue

            # Determine which spectra are available for matching
            available_spectra = []
            for st in ["1H", "13C", "HSQC", "COSY"]:
                # Both experimental and predicted data use base spectrum type
                if st in experimental_spectra and st in predicted_spectra:
                    # For 1D spectra (1H, 13C)
                    if st in ["1H", "13C"]:
                        if (len(experimental_spectra[st]['shifts']) > 0 and 
                            len(predicted_spectra[st]['shifts']) > 0):
                            available_spectra.append(st)
                    # For 2D spectra (HSQC, COSY)
                    else:
                        if (len(experimental_spectra[st]['F1 (ppm)']) > 0 and 
                            len(predicted_spectra[st]['F1 (ppm)']) > 0):
                            available_spectra.append(st)

            # Check if we have any spectra to analyze
            if not available_spectra:
                self.logger.error(f"No matching spectra available for analysis of {smiles}")
                self.logger.error(f"Experimental spectra keys: {list(experimental_spectra.keys())}")
                return None

            if not experimental_spectra or not predicted_spectra:
                self.logger.error(f"Missing experimental or predicted spectra for {smiles}")
                return None

            # Perform peak matching between experimental and predicted spectra
            peak_matching_result = await self.peak_matcher.process(
                {
                'peaks1': experimental_spectra,
                'peaks2': predicted_spectra
                },
                context={
                    'matching_mode': 'hung_dist_nn',
                    'error_type': 'sum',
                    'spectra': available_spectra
                }
                        )

            # self.logger.info(f"peak_matching_result {peak_matching_result}")
                        
            # After getting peak_matching_result
            if not peak_matching_result or 'status' not in peak_matching_result or peak_matching_result['status'] != 'success':
                self.logger.error(f"Peak matching returned invalid result for {smiles}")
                return None

            # Extract data from the result structure
            result_data = peak_matching_result.get('data', {})
            if not result_data or 'results' not in result_data:
                self.logger.error(f"Missing results data for {smiles}")
                return None

            # Process results for each spectrum type
            spectrum_errors = {}
            matched_peaks = {}

            for spectrum_type, spectrum_result in result_data['results'].items():
                if spectrum_result['status'] == 'success':
                    # Store the overall error for this spectrum
                    spectrum_errors[spectrum_type] = spectrum_result['overall_error']
                    
                    # Store the matched peaks
                    if 'matched_peaks' in spectrum_result:
                        matched_peaks[spectrum_type] = {
                            'spectrum1': spectrum_result['matched_peaks']['spectrum1'],
                            'spectrum2': spectrum_result['matched_peaks']['spectrum2'],
                            'spectrum1_orig': spectrum_result['original_data']['spectrum1'],
                            'spectrum2_orig': spectrum_result['original_data']['spectrum2']
                        }

            # Calculate overall score (lower error = better match)
            if spectrum_errors:
                overall_score = sum(spectrum_errors.values()) / len(spectrum_errors)
            else:
                self.logger.error(f"No spectrum errors found for {smiles}")
                return None

            results = {
                "smiles": smiles,
                "generation_info": source_info,
                "nmr_analysis": {
                    "spectra_matching": matched_peaks,
                    "matching_scores": {
                        "overall": overall_score,
                        "by_spectrum": spectrum_errors
                    }
                }
            }
  
            # In _analyze_single_molecule:
            # try:
            #     # Determine source type from source_info
            #     source_type = source_info.get("source")
            #     if source_type not in ["forward_synthesis", "mol2mol"]:
            #         raise ValueError(f"Invalid or missing source type: {source_type}. Must be either 'forward_synthesis' or 'mol2mol'")
                    
            #     if not sample_id:
            #         raise ValueError("Missing sample_id in experimental data")

            #     self.logger.error(f"experimental_data {experimental_data}")
            #     self._save_peak_matching_results(smiles, results, source_type, experimental_data.get("sample_id"), intermediate_file)
            # except Exception as e:
            #     self.logger.error(f"Error saving results for {smiles}: {str(e)}")

            return results

        except Exception as e:
            self.logger.error(f"Error analyzing molecule {smiles}: {str(e)}")
            return None

    # def _save_peak_matching_results(self, smiles: str, results: Dict[str, Any], source_type: str, sample_id: str, intermediate_file: Path):
    #     """Save peak matching results for a molecule.
        
    #     Args:
    #         smiles: Canonicalized SMILES string of the molecule
    #         results: Dictionary containing peak matching results and analysis
    #         source_type: Type of analysis ('forward_synthesis' or 'mol2mol')
    #         sample_id: ID of the sample being analyzed
    #         intermediate_file: Path to the intermediate file containing sample data
    #     """
    #     # Load current sample data from intermediate file
    #     with open(intermediate_file, 'r') as f:
    #         current_data = json.load(f)
    #         sample_data = current_data[sample_id]
        
    #     # Initialize or get sample data structure
    #     if "candidate_analysis" not in sample_data:
    #         sample_data["candidate_analysis"] = {}
    #     if source_type not in sample_data["candidate_analysis"]:
    #         sample_data["candidate_analysis"][source_type] = {
    #             "tool_version": "1.0",
    #             "molecules": []
    #         }
        
    #     # Add timestamp to results
    #     results["timestamp"] = datetime.now().isoformat()
        
    #     # Get the correct section for this sample
    #     target_section = sample_data["candidate_analysis"][source_type]["molecules"]
        
    #     # Look for existing molecule entry
    #     molecule_entry = next(
    #         (mol for mol in target_section if mol.get("smiles") == smiles),
    #         None
    #     )
        
    #     if molecule_entry is None:
    #         # Create new molecule entry
    #         molecule_entry = {
    #             "smiles": smiles,
    #             "peak_matching_results": results
    #         }
    #         target_section.append(molecule_entry)
    #     else:
    #         # Update existing molecule entry
    #         molecule_entry["peak_matching_results"] = results
        
    #     # Save updated data back to intermediate file
    #     current_data[sample_id] = sample_data
    #     with open(intermediate_file, 'w') as f:
    #         json.dump(current_data, f, indent=2)
    
    #     self.logger.info(f"Saved peak matching results for molecule {smiles} in sample {sample_id} under {source_type}")

    def _format_peaks(self, peaks: List[Any], spectrum_type: str) -> Dict[str, Any]:
        """Format peaks into the required structure for peak matching.
        
        Args:
            peaks: List of peaks from NMR data
            spectrum_type: Type of spectrum ('1H', '13C', 'HSQC', 'COSY', or with _sim/_exp suffix)
             
        Returns:
            Dictionary with formatted peak data
        """
        # Extract base spectrum type by removing _sim or _exp suffix if present
        base_type = spectrum_type.split('_')[0]
        
        # self.logger.info(f"Formatting peaks for spectrum type: {spectrum_type} (base type: {base_type})")
        # self.logger.info(f"Input peaks: {peaks}")
        
        if not peaks:  # Handle empty peaks list
            if base_type in ['1H', '13C']:
                return {'shifts': [], 'Intensity': []}
            else:  # 2D spectra (HSQC, COSY)
                return {'F1 (ppm)': [], 'F2 (ppm)': [], 'Intensity': []}
            
        try:
            if base_type == '1H':
                # peaks is a list of [shift, intensity] lists
                shifts = [float(p[0]) for p in peaks]
                intensities = [float(p[1]) for p in peaks]
                formatted = {
                    'shifts': shifts,
                    'Intensity': intensities
                }
            elif base_type == '13C':
                # peaks is a list of shift values
                shifts = [float(p) for p in peaks]
                formatted = {
                    'shifts': shifts,
                    'Intensity': [1.0] * len(shifts)
                }
            else:  # 2D spectra (HSQC, COSY)
                # peaks is a list of [f1, f2] lists
                f1_shifts = [float(p[0]) for p in peaks]
                f2_shifts = [float(p[1]) for p in peaks]
                formatted = {
                    'F1 (ppm)': f1_shifts,
                    'F2 (ppm)': f2_shifts,
                    'Intensity': [1.0] * len(peaks)
                }
            
            #self.logger.info(f"Formatted peaks: {formatted}")
            return formatted
            
        except (ValueError, IndexError) as e:
            self.logger.error(f"Error formatting peaks for {spectrum_type} spectrum: {str(e)}")
            self.logger.error(f"Problematic peaks data: {peaks}")
            if base_type in ['1H', '13C']:
                return {'shifts': [], 'Intensity': []}
            else:  # 2D spectra (HSQC, COSY)
                return {'F1 (ppm)': [], 'F2 (ppm)': [], 'Intensity': []}

    def _calculate_overall_score(self, peak_matching_result: Dict[str, Any]) -> float:
        """Calculate overall score from peak matching results."""
        # Implement scoring logic based on peak matching results
        # This is a placeholder - implement actual scoring logic
        return 0.0

    def _extract_spectrum_scores(self, peak_matching_result: Dict[str, Any]) -> Dict[str, float]:
        """Extract individual spectrum scores from peak matching results."""
        # Implement logic to extract scores for each spectrum type
        # This is a placeholder - implement actual extraction logic
        return {
            "1H": 0.0,
            "13C": 0.0,
            "HSQC": 0.0,
            "COSY": 0.0
        }

    def _generate_summary(self, candidates: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for all candidates."""
        summary = {
            "total_candidates": 0,
            "by_source": {},
            "top_candidates": []
        }

        # Calculate statistics
        for source, data in candidates.items():
            num_candidates = len(data.get("molecules", []))
            summary["by_source"][source] = num_candidates
            summary["total_candidates"] += num_candidates

        # Generate top candidates list (sorted by overall score)
        all_candidates = []
        for source, data in candidates.items():
            for molecule in data.get("molecules", []):
                all_candidates.append({
                    "smiles": molecule["smiles"],
                    "source": source,
                    "overall_score": molecule["nmr_analysis"]["matching_scores"]["overall"]
                })

        # Sort by overall score and take top N
        all_candidates.sort(key=lambda x: x["overall_score"], reverse=True)
        summary["top_candidates"] = all_candidates[:10]  # Top 10 candidates

        return summary

    async def process(
        self,
        molecular_data: Dict,
        context: Optional[Dict] = None
        ) -> Dict[str, Any]:
        """Process molecular data to analyze candidates.
        
        Args:
            molecular_data: Dictionary containing molecular data or sample_id
            context: Optional context dictionary
        """
        try:
            # Get sample_id from molecular_data or context
            sample_id = None
            if isinstance(molecular_data, dict):
                # Try to get sample_id directly from molecular_data
                sample_id = molecular_data.get('sample_id')
            
            if not sample_id and context and 'current_molecule' in context:
                sample_id = context['current_molecule'].get('sample_id')
            
            if not sample_id:
                self.logger.error("No sample_id found in molecular_data or context")
                return {
                    'status': 'error',
                    'error': 'No sample_id found in input data'
                }
                
            # Load intermediate data
            try:
                intermediate_data = self._load_or_create_intermediate(sample_id, context.get('current_molecule') if context else None)
                molecule_data = intermediate_data.get('molecule_data', {})
            except ValueError as e:
                self.logger.error(f"Error loading intermediate data: {str(e)}")
                return {
                    'status': 'error',
                    'error': f'Failed to load data: {str(e)}'
                }
            
            # Initialize result structure
            result = {
                "candidates": {},
                "sample_id": sample_id
            }
            
            try:
                # Process predictions based on analysis type
                if self.analysis_type == 'forward':
                    if "forward_predictions" in molecule_data:
                        self.logger.info(f"Processing forward synthesis predictions for {sample_id}")
                        self.logger.info(f"Number of predictions: {len(molecule_data['forward_predictions'])}")
                        
                        result["candidates"]["forward_synthesis"] = await self._process_forward_synthesis(
                            molecule_data["forward_predictions"],
                            molecule_data
                        )
                        
                        # Update and save results
                        if "candidate_analysis" not in molecule_data:
                            molecule_data["candidate_analysis"] = {}
                        molecule_data["candidate_analysis"]["forward_synthesis"] = result["candidates"]["forward_synthesis"]
                        
                elif self.analysis_type == 'mol2mol':
                    if "mol2mol_results" in molecule_data and molecule_data['mol2mol_results'].get('status') == 'success':
                        self.logger.info(f"Processing mol2mol predictions for {sample_id}")
                        
                        result["candidates"]["mol2mol"] = await self._process_mol2mol(
                            molecule_data["mol2mol_results"],
                            molecule_data
                        )
                        
                        # Update and save results
                        if "candidate_analysis" not in molecule_data:
                            molecule_data["candidate_analysis"] = {}
                        molecule_data["candidate_analysis"]["mol2mol"] = result["candidates"]["mol2mol"]
                        
                elif self.analysis_type == 'mmst':
                    if "mmst_results" in molecule_data and molecule_data['mmst_results'].get('status') == 'success':
                        self.logger.info(f"Processing MMST predictions for {sample_id}")
                        
                        result["candidates"]["mmst"] = await self._process_mmst(
                            molecule_data["mmst_results"],
                            molecule_data
                        )
                        
                        # Update and save results
                        if "candidate_analysis" not in molecule_data:
                            molecule_data["candidate_analysis"] = {}
                        molecule_data["candidate_analysis"]["mmst"] = result["candidates"]["mmst"]
                
                # Save updated data if any changes were made
                if result["candidates"]:
                    intermediate_data['molecule_data'] = molecule_data
                    self._save_intermediate(sample_id, intermediate_data)
                    
                    # Generate summary
                    result["summary"] = self._generate_summary(result["candidates"])
                    result["status"] = "success"
                else:
                    result["status"] = "no_candidates"
                    result["message"] = f"No candidates to analyze for {self.analysis_type} prediction type"
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error in candidate analysis: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        except Exception as e:
            self.logger.error(f"main: {str(e)}")
            return {
            "status": "error",
            "error": str(e)
                }


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/tools/stout_operations.py ---
"""
Simple STOUT operations module for converting SMILES to IUPAC names.
"""
import uuid
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class STOUTOperations:
    """Simple handler for SMILES to IUPAC name conversion."""
    
    def __init__(self):
        """Initialize STOUT operations."""
        self.base_path = Path(__file__).parent.parent.parent
        self.temp_dir = self.base_path / "_temp_folder/stout"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.script_path = Path(__file__).parent.parent / "scripts" / "stout_local.sh"

    async def get_iupac_name(self, smiles: str) -> Dict[str, Any]:
        """Convert a single SMILES string to IUPAC name.
        
        Args:
            smiles: SMILES string to convert
            
        Returns:
            Dictionary with status and either IUPAC name or error message
        """
        job_id = uuid.uuid4().hex
        input_file = self.temp_dir / f"input_{job_id}.txt"
        output_file = self.temp_dir / f"output_{job_id}.json"
        
        try:
            # Write SMILES to input file
            input_file.write_text(smiles)
            
            # Run conversion script
            subprocess.run(
                [str(self.script_path), str(input_file), str(output_file), 'forward'],
                check=True,
                timeout=35
            )
            
            # Read and parse result
            result = json.loads(output_file.read_text())
            if result['status'] == 'success':
                return {
                    'status': 'success',
                    'iupac_name': result['result']
                }
            else:
                return {
                    'status': 'error',
                    'error': result.get('error', 'Unknown conversion error')
                }
                
        except subprocess.TimeoutExpired:
            return {
                'status': 'error',
                'error': 'Conversion timed out after 35 seconds'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
        finally:
            # Cleanup temp files
            input_file.unlink(missing_ok=True)
            output_file.unlink(missing_ok=True)

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/tools/retro_synthesis_tool.py ---
"""Tool for generating retrosynthesis predictions using Chemformer."""
from math import log
import os
import logging
import json
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Dict, Any, Optional, List
import pandas as pd
import shutil

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.sbatch_utils import execute_sbatch, wait_for_job_completion
import asyncio

# Constants for paths and directories
BASE_DIR = Path(__file__).parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "agents" / "scripts"
RETRO_DIR = BASE_DIR / "_temp_folder" / "retro_output"
TEMP_DIR = BASE_DIR / "_temp_folder"

# Constants for Chemformer execution
RETRO_OUTPUT_CHECK_INTERVAL = 5  # seconds
RETRO_OUTPUT_TIMEOUT = 600  # 10 minutes
RETRO_OUTPUT_PATTERN = "retro_predictions.csv"  # Will be formatted with timestamp
RETRO_INPUT_FILENAME = "retro_targets.txt"  # Input filename

SBATCH_SCRIPT = SCRIPTS_DIR / "chemformer_retro_sbatch.sh"
LOCAL_SCRIPT = SCRIPTS_DIR / "chemformer_retro_local.sh"

class RetrosynthesisTool:
    """Tool for generating retrosynthesis predictions using Chemformer."""
    
    def __init__(self):
        """Initialize the Retrosynthesis tool with required directories."""
        # Ensure required directories exist
        self.scripts_dir = SCRIPTS_DIR
        self.retro_dir = RETRO_DIR
        self.temp_dir = TEMP_DIR
        
        # Create directories if they don't exist
        self.retro_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Add intermediate results directory
        self.intermediate_dir = self.temp_dir / "intermediate_results"
        self.intermediate_dir.mkdir(exist_ok=True)
        
        # Validate script existence
        if not SBATCH_SCRIPT.exists():
            raise FileNotFoundError(f"Required SBATCH script not found at {SBATCH_SCRIPT}")
        if not LOCAL_SCRIPT.exists():
            raise FileNotFoundError(f"Required local script not found at {LOCAL_SCRIPT}")

        # Validate environment
        try:
            import torch
            if not torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID') is None:
                logging.warning("CUDA not available for local execution. SLURM execution will be forced.")
        except ImportError:
            logging.warning("PyTorch not found. Please ensure the chemformer environment is activated.")

    async def _prepare_input_from_context(self, context: Dict[str, Any]) -> Path:
        """Prepare input file from context.
        
        Args:
            context: Dictionary containing molecule data and flags
            
        Returns:
            Path to the created input file
        """
        try:
            # Initialize variables
            smiles_list = []
            input_file = self.temp_dir / RETRO_INPUT_FILENAME
            
            # Get molecules from master JSON file
            master_data_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
            if master_data_path.exists():
                with open(master_data_path, 'r') as f:
                    master_data = json.load(f)
                # Extract all SMILES from master data
                for molecule_id, molecule_data in master_data.items():
                    if 'smiles' in molecule_data:
                        smiles = self._normalize_smiles(molecule_data['smiles'])
                        if smiles:
                            smiles_list.append(smiles)
                logging.info(f"Extracted {len(smiles_list)} molecules from master JSON")
            else:
                logging.warning("Master JSON file not found")
            
            # If no molecules found in master JSON, try context as fallback
            if not smiles_list and context.get('current_molecule'):
                current_molecule = context['current_molecule']
                if isinstance(current_molecule, dict) and 'SMILES' in current_molecule:
                    smiles = self._normalize_smiles(current_molecule['SMILES'])
                    if smiles:
                        smiles_list.append(smiles)
                elif isinstance(current_molecule, str):
                    smiles = self._normalize_smiles(current_molecule)
                    if smiles:
                        smiles_list.append(smiles)
            
            # Validate SMILES list
            if not smiles_list:
                raise ValueError("No valid SMILES found in master JSON or context")
            
            # Remove duplicates while preserving order
            smiles_list = list(dict.fromkeys(smiles_list))
            
            # Write SMILES to input file
            with open(input_file, 'w') as f:
                for smiles in smiles_list:
                    f.write(f"{smiles}\n")
            
            #logging.info(f"Prepared input file with {len(smiles_list)} molecules")
            return input_file
            
        except Exception as e:
            logging.error(f"Error preparing input file: {str(e)}")
            raise

    def _normalize_smiles(self, smiles: str) -> str:
        """Normalize SMILES string to canonical form."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
        except ImportError:
            logging.warning("RDKit not available for SMILES normalization")
        return smiles.strip()

    async def _check_starting_materials_exist(self, smiles_list: List[str], intermediate_data: Dict) -> Dict[str, bool]:
        """Check if starting materials already exist in the intermediate data.
        
        Args:
            smiles_list: List of SMILES strings to check
            intermediate_data: The loaded intermediate data containing molecule info and results
            
        Returns:
            Dictionary mapping SMILES to boolean indicating if starting materials exist
        """
        try:
            # Initialize result dictionary
            result = {smiles: False for smiles in smiles_list}
            
            # Check if molecule has starting materials in intermediate
            # if ('step_outputs' in intermediate_data and 
            #     'retrosynthesis' in intermediate_data['step_outputs']):
            #     retro_data = intermediate_data['step_outputs']['retrosynthesis']
            #     if retro_data.get('status') == 'success' and 'predictions' in retro_data:
            #         for smiles in smiles_list:
            #             norm_smiles = self._normalize_smiles(smiles)
            #             # Check if this SMILES has predictions in intermediate
            #             for pred in retro_data['predictions']:
            #                 if self._normalize_smiles(pred.get('target_smiles', '')) == norm_smiles:
            #                     result[smiles] = True
            #                     break
            
            # Also check if starting materials were provided in the original molecule data
            if 'molecule_data' in intermediate_data and 'starting_materials' in intermediate_data['molecule_data']:
                molecule_smiles = intermediate_data['molecule_data'].get('smiles', '')
                if molecule_smiles:
                    norm_molecule_smiles = self._normalize_smiles(molecule_smiles)
                    for smiles in smiles_list:
                        if self._normalize_smiles(smiles) == norm_molecule_smiles:
                            result[smiles] = True
            
            return result
            
        except Exception as e:
            logging.error(f"Error checking starting materials: {str(e)}")
            return {smiles: False for smiles in smiles_list}

    async def _update_master_data(self, predictions_df: pd.DataFrame) -> None:
        """Update the master data file with retrosynthesis predictions.
        
        Args:
            predictions_df: DataFrame containing retrosynthesis predictions with columns:
                          'target_smiles', 'predicted_smiles', 'all_predictions', 'all_log_likelihoods'
        """
        try:
            # Path to master data file
            master_data_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
            
            if not master_data_path.exists():
                logging.error("Master data file not found")
                return
            
            # Read current master data
            with open(master_data_path, 'r') as f:
                master_data = json.load(f)

            # logging.info(f"Loaded master data with {len(master_data)} samples")
            # Create mapping of SMILES to predictions
            smiles_to_predictions = {}
            for _, row in predictions_df.iterrows():
                target = self._normalize_smiles(row['target_smiles'])
                
                # Get all predictions and their log likelihoods
                all_preds = [p.strip() for p in row['all_predictions'].split(';') if p.strip()]
                all_logs = [float(l.strip()) for l in row['all_log_likelihoods'].split(';') if l.strip()]
                
                # Create list of (prediction, log_likelihood) pairs
                pred_pairs = list(zip(all_preds, all_logs))
                
                # Sort by log likelihood (highest first) and normalize SMILES
                pred_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Get unique predictions (using normalized SMILES)
                seen = set()
                unique_preds = []
                for pred, _ in pred_pairs:
                    norm_pred = self._normalize_smiles(pred)
                    if norm_pred not in seen:
                        seen.add(norm_pred)
                        unique_preds.append(norm_pred)
                
                # Take top 5 unique predictions
                if unique_preds:
                    smiles_to_predictions[target] = unique_preds[:]

            # logging.info(f"Loaded smiles_to_predictions {smiles_to_predictions}")

            # Update master data
            updated = False
            for sample_id, sample_data in master_data.items():
                if 'smiles' in sample_data:
                    target_smiles = self._normalize_smiles(sample_data['smiles'])
                    if target_smiles in smiles_to_predictions:
                        sample_data['starting_smiles'] = smiles_to_predictions[target_smiles]
                        updated = True
                        #logging.info(f"Updated starting materials for sample {sample_id}")
            # logging.info(f"Loaded master_data {master_data}")
            if updated:
                # Write updated data back to file
                with open(master_data_path, 'w') as f:
                    json.dump(master_data, f, indent=2)
                logging.info("Successfully updated master data with retrosynthesis predictions")
            else:
                logging.warning("No matching SMILES found in master data")
                
        except Exception as e:
            logging.error(f"Error updating master data: {str(e)}")
            raise
 
    async def predict_retrosynthesis(self, molecule_data: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Predict retrosynthesis for a given molecule.
        
        Args:
            sample_id: ID of the sample to predict retrosynthesis for
            context: Optional context data
            
        Returns:
            Dict containing prediction results
        """
        try:
            # Get sample_id from context or molecule_data
            sample_id = None
            if context and 'sample_id' in context:
                sample_id = context['sample_id']
            elif isinstance(molecule_data, dict) and 'sample_id' in molecule_data:
                sample_id = molecule_data['sample_id']

            if not sample_id:
                raise ValueError("No sample_id provided in context or molecule_data")

            # Load or create intermediate file
            intermediate_data = self._load_or_create_intermediate(sample_id, molecule_data)

            
            # Check if starting materials already exist
            if ('molecule_data' in intermediate_data and 
                'starting_smiles' in intermediate_data['molecule_data'] and
                intermediate_data['molecule_data']['starting_smiles']):
                logging.info(f"Starting materials already exist for sample {sample_id}")
                return {
                    'status': 'success', 
                    'message': 'Starting materials already exist',
                    'predictions': intermediate_data['molecule_data']['starting_smiles']
                }
            
            # Get SMILES from molecule data          
            smiles = intermediate_data['molecule_data'].get('smiles')
            
            # Create input file with single SMILES
            input_file = self.temp_dir / f"{sample_id}_input.txt"
            with open(input_file, 'w') as f:
                f.write(f"{smiles}\n")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.retro_dir / RETRO_OUTPUT_PATTERN.format(timestamp)
            
            logging.info(f"Running retrosynthesis prediction for molecule {smiles}")
            
            # Run prediction locally
            try:
                logging.info("Running retrosynthesis locally")
                process = await asyncio.create_subprocess_exec(
                    str(LOCAL_SCRIPT),
                    f"--input_file={input_file}",
                    f"--output_file={output_file}",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    error_msg = f"Local execution failed with return code {process.returncode}: {stderr.decode()}"
                    logging.error(error_msg)
                    return {'status': 'error', 'message': error_msg}
                
            except Exception as e:
                error_msg = f"Error during local execution: {str(e)}"
                logging.error(error_msg)
                return {'status': 'error', 'message': error_msg}
            
            # Wait for output file to be generated
            if not await self._wait_for_output(output_file):
                error_msg = f"Timeout waiting for output file at {output_file}"
                logging.error(error_msg)
                return {'status': 'error', 'message': error_msg}
            
            # Read predictions from output file
            try:
                predictions_df = pd.read_csv(output_file)
                
                # Extract starting materials from all_predictions
                starting_materials = []
                for _, row in predictions_df.iterrows():
                    if 'all_predictions' in row:
                        # Split all predictions by semicolon and add to starting materials
                        all_preds = row['all_predictions'].split(';')
                        starting_materials.extend(all_preds)
                
                # Remove duplicates while preserving order
                seen = set()
                starting_materials = [x for x in starting_materials if not (x in seen or seen.add(x))]
                
                # Update intermediate file with starting materials
                intermediate_data['molecule_data']['starting_smiles'] = starting_materials
                self._save_intermediate(sample_id, intermediate_data)
            
                # Return full prediction data for the tool response
                return {
                    'status': 'success',
                    'message': 'Successfully generated retrosynthesis predictions',
                    'predictions': predictions_df.to_dict('records')
                }
                
            except Exception as e:
                    error_msg = f"Error reading predictions from output file: {str(e)}"
                    logging.error(error_msg)
                    return {'status': 'error', 'message': error_msg}
                
        except Exception as e:
            error_msg = f"Unexpected error in retrosynthesis prediction: {str(e)}"
            logging.error(error_msg)
            raise

    async def _wait_for_output(self, output_file: Path, timeout: int = RETRO_OUTPUT_TIMEOUT) -> bool:
        """Wait for the output file to be generated."""
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            if output_file.exists():
                return True
            await asyncio.sleep(RETRO_OUTPUT_CHECK_INTERVAL)
        return False

    def _get_intermediate_path(self, sample_id: str) -> Path:
        """Get path to intermediate file for a sample."""
        return self.intermediate_dir / f"{sample_id}_intermediate.json"


    def _load_or_create_intermediate(self, sample_id: str, context: Dict[str, Any] = None) -> Dict:
        """Load existing intermediate file or create new one from master data.
        """
        intermediate_path = self._get_intermediate_path(sample_id)
        
        # Try loading existing intermediate
        if intermediate_path.exists():
            with open(intermediate_path, 'r') as f:
                data = json.load(f)
                if 'molecule_data' not in data:
                    data['molecule_data'] = {}
                return data
                
        # If no intermediate exists, try getting data from master file
        master_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
        if master_path.exists():
            with open(master_path, 'r') as f:
                master_data = json.load(f)

                if sample_id in master_data:
                    # Create new intermediate with just this sample's data
                    intermediate_data = master_data[sample_id]
                    self._save_intermediate(sample_id, intermediate_data)
                    return intermediate_data
                    
        # If neither exists and we have context, create new intermediate
        if context:
            intermediate_data["molecule_data"] = context
            self._save_intermediate(sample_id, intermediate_data)
            return intermediate_data
            
        raise ValueError(f"No data found for sample {sample_id}")
        
    def _save_intermediate(self, sample_id: str, data: Dict):
        """Save data to intermediate file.
        
        Args:
            sample_id: ID of the sample to save
            data: Data to save to intermediate file
        """
        path = self._get_intermediate_path(sample_id)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/tools/analysis_enums.py ---
"""
Shared enums for analysis tools.
"""
from enum import Enum

class DataSource(Enum):
    MASTER_FILE = "master_file"
    INTERMEDIATE = "intermediate"

class RankingMetric(Enum):
    """Available metrics for ranking candidates"""
    OVERALL = "overall"
    PROTON = "1H"
    CARBON = "13C"
    HSQC = "HSQC"
    COSY = "COSY"


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/tools/peak_matching_tool.py ---
"""Tool for peak matching between NMR spectra with support for various input formats."""
import os
import logging
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import sys
import pandas as pd
import asyncio

# Constants for paths and directories
BASE_DIR = Path(__file__).parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "agents" / "scripts"
TEMP_DIR = BASE_DIR / "_temp_folder"
PEAK_MATCHING_DIR = TEMP_DIR / "peak_matching"
LOCAL_SCRIPT = SCRIPTS_DIR / "peak_matching_local.sh"

# Constants for peak matching
SUPPORTED_SPECTRA = ['1H', '13C', 'HSQC', 'COSY']
OUTPUT_CHECK_INTERVAL = 5  # seconds
OUTPUT_TIMEOUT = 300  # 5 minutes
RESULTS_FILENAME = "results.json"

# Constants for timeouts and retries
SUBPROCESS_TIMEOUT = 300  # 5 minutes
RESULTS_WAIT_TIMEOUT = 60  # 1 minute
SUBPROCESS_CHECK_INTERVAL = 0.5  # 500ms
RESULTS_CHECK_INTERVAL = 1.0  # 1 second
MAX_RETRIES = 3

# Constants for peak matching configuration
SUPPORTED_MATCHING_MODES = ['hung_dist_nn', 'euc_dist_all']  # Supported peak matching strategies
SUPPORTED_ERROR_TYPES = ['sum', 'avg']  # Supported error calculation methods
DEFAULT_MATCHING_MODE = 'hung_dist_nn'
DEFAULT_ERROR_TYPE = 'avg'

"""
Future enhancements for peak matching configuration:
- Add intensity weighting for peak matching
- Add distance normalization options
- Add maximum distance thresholds
- Add weighted averaging for error calculation
- Add configurable parameters for each matching mode
"""

class EnhancedPeakMatchingTool:
    """Tool for comparing NMR peak lists using external Python environment."""
    
    def __init__(self):
        """Initialize the peak matching tool with required directories."""
        self.scripts_dir = SCRIPTS_DIR
        self.temp_dir = TEMP_DIR
        self.intermediate_dir = self.temp_dir / "intermediate_results"
        self.intermediate_dir.mkdir(exist_ok=True)
        self.peak_matching_dir = PEAK_MATCHING_DIR
        
        # Create directories if they don't exist
        self.temp_dir.mkdir(exist_ok=True)
        self.scripts_dir.mkdir(exist_ok=True)
        self.peak_matching_dir.mkdir(exist_ok=True)
        
        # Validate script existence
        if not LOCAL_SCRIPT.exists():
            raise FileNotFoundError(f"Required local script not found at {LOCAL_SCRIPT}")
        
        # Configure logging
        log_file = PEAK_MATCHING_DIR / 'enhanced_tool.log'
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"EnhancedPeakMatchingTool initialized. Log file: {log_file}")


    def _validate_simulation_data(self, data: Dict) -> bool:
        """Check if required simulation data exists in NMR data"""
        if 'molecule_data' not in data:
            return False
        
        mol_data = data['molecule_data']
        if 'nmr_data' not in mol_data:
            return False
        
        required_keys = ['1H_sim', '13C_sim', 'HSQC_sim', 'COSY_sim']
        nmr_data = mol_data['nmr_data']
        
        for key in required_keys:
            if key not in nmr_data:
                self.logger.warning(f"Missing required simulation data: {key}")
                return False
            if not nmr_data[key]:  # Check if data exists
                self.logger.warning(f"Empty simulation data for: {key}")
                return False
                
        return True
        
        
    def _get_intermediate_path(self, sample_id: str) -> Path:
        """Get path to intermediate file for a sample."""
        return self.intermediate_dir / f"{sample_id}_intermediate.json"


    def _load_or_create_intermediate(self, sample_id: str, context: Dict[str, Any] = None) -> Dict:
        """Load existing intermediate file or create new one from master data.
        """
        intermediate_path = self._get_intermediate_path(sample_id)
        
        # Try loading existing intermediate
        if intermediate_path.exists():
            with open(intermediate_path, 'r') as f:
                data = json.load(f)
                if 'molecule_data' not in data:
                    data['molecule_data'] = {}
                return data
                
        # If no intermediate exists, try getting data from master file
        master_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
        if master_path.exists():
            with open(master_path, 'r') as f:
                master_data = json.load(f)

                if sample_id in master_data:
                    # Create new intermediate with just this sample's data
                    intermediate_data = master_data[sample_id]
                    self._save_intermediate(sample_id, intermediate_data)
                    return intermediate_data
                    
        # If neither exists and we have context, create new intermediate
        if context:
            intermediate_data["molecule_data"] = context
            self._save_intermediate(sample_id, intermediate_data)
            return intermediate_data
            
        raise ValueError(f"No data found for sample {sample_id}")

    def _save_intermediate(self, sample_id: str, data: Dict):
        """Save data to intermediate file"""
        intermediate_path = self.intermediate_dir / f"{sample_id}_intermediate.json"
        with open(intermediate_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _prepare_peak_matching_input(self, mol_data: Dict) -> Dict:
        """Prepare input data for peak matching from molecule data"""
        if 'nmr_data' not in mol_data:
            raise ValueError("No NMR data found in molecule data")
        
        nmr_data = mol_data['nmr_data']
        
        # Format peaks for 1D NMR (1H)
        def format_1d_peaks(peaks):
            if not peaks:
                return {'shifts': [], 'intensities': []}
            # Each peak is a list of [shift, intensity]
            return {
                'shifts': [peak[0] for peak in peaks],
                #'intensities': [peak[1] for peak in peaks]
                'intensities': [1 for peak in peaks] # Constant intensity
            }
        
        # Format peaks for 2D NMR (HSQC, COSY)
        def format_2d_peaks(peaks):
            if not peaks:
                return {'F2 (ppm)': [], 'F1 (ppm)': []}
            # Each peak is a list of [f2, f1]
            return {
                'F2 (ppm)': [peak[0] for peak in peaks],
                'F1 (ppm)': [peak[1] for peak in peaks]
            }
        
        # Format peaks for 13C NMR
        def format_13c_peaks(peaks):
            if not peaks:
                return {'shifts': [], 'intensities': []}
            # 13C peaks are just a list of shifts
            return {
                'shifts': peaks,
                'intensities': [1.0] * len(peaks)  # Constant intensity for 13C
            }
    
        # Prepare input data structure
        input_data = {
            'spectra': SUPPORTED_SPECTRA,
            'simulated_data': {
                'h1': format_1d_peaks(nmr_data.get('1H_sim', [])),
                'c13': format_13c_peaks(nmr_data.get('13C_sim', [])),
                'hsqc': format_2d_peaks(nmr_data.get('HSQC_sim', [])),
                'cosy': format_2d_peaks(nmr_data.get('COSY_sim', []))
            },
            'experimental_data': {
                'h1': format_1d_peaks(nmr_data.get('1H_exp', [])),
                'c13': format_13c_peaks(nmr_data.get('13C_exp', [])),
                'hsqc': format_2d_peaks(nmr_data.get('HSQC_exp', [])),
                'cosy': format_2d_peaks(nmr_data.get('COSY_exp', []))
            }
        }
    
        return input_data

    async def _wait_for_results(self, run_dir: Path, context: Dict) -> Dict:
        """Wait for and validate results file.
        
        Args:
            run_dir: Directory containing results
            context: Additional context dictionary
            
        Returns:
            Dictionary containing results
            
        Raises:
            TimeoutError: If results not found within timeout period
            ValueError: If results file is invalid
        """
        start_time = datetime.now()
        results_path = run_dir / RESULTS_FILENAME
        
        while not results_path.exists():
            if (datetime.now() - start_time).total_seconds() > RESULTS_WAIT_TIMEOUT:
                error_msg = f"Timeout waiting for results file after {RESULTS_WAIT_TIMEOUT} seconds"
                self.logger.error(error_msg)
                # Check for error file
                error_file = run_dir / 'error.log'
                if error_file.exists():
                    with open(error_file) as f:
                        error_content = f.read()
                    error_msg += f"\nError log content:\n{error_content}"
                raise TimeoutError(error_msg)
            await asyncio.sleep(RESULTS_CHECK_INTERVAL)
        
        # Validate results file
        with open(results_path) as f:
            results = json.load(f)
            return results
            
    def _has_existing_results(self, intermediate_data: Dict, comparison_type: str = 'peaks_vs_peaks') -> bool:
        """Check if peak matching results already exist for the given comparison type."""
        if 'molecule_data' not in intermediate_data or \
           'peak_matching_results' not in intermediate_data['molecule_data'] or \
           'comparisons' not in intermediate_data['molecule_data']['peak_matching_results']:
            return False
            
        # Determine category and subcategory
        if comparison_type == 'peaks_vs_peaks':
            category = 'simulation'
            subcategory = 'exp_vs_sim'
        elif comparison_type == 'smiles_vs_peaks':
            # This will be determined by context in _save_results
            return False  # Always run for SMILES comparisons
        else:
            category = 'custom'
            subcategory = comparison_type
            
        comparisons = intermediate_data['molecule_data']['peak_matching_results']['comparisons']
        return (category in comparisons and 
                subcategory in comparisons[category] and
                comparisons[category][subcategory]['status'] == 'success')

    def _get_existing_results(self, intermediate_data: Dict, comparison_type: str = 'peaks_vs_peaks') -> Dict:
        """Get existing peak matching results for the given comparison type."""
        if comparison_type == 'peaks_vs_peaks':
            category = 'simulation'
            subcategory = 'exp_vs_sim'
        else:
            category = 'custom'
            subcategory = comparison_type
            
        results = intermediate_data['molecule_data']['peak_matching_results']['comparisons'][category][subcategory]
        return {
            'status': 'success',
            'message': f'Peak matching results already exist for {category}/{subcategory}',
            'data': {
                'type': comparison_type,
                'results': results['results'],
                # 'matching_mode': results['metadata']['matching_mode'],
                # 'error_type': results['metadata']['error_type'],
                # 'spectra': results['metadata']['spectra']
            }
        }

    def _save_results(self, intermediate_data: Dict, results: Dict) -> None:
        """Save peak matching results to intermediate file with structured format."""
        # Initialize peak_matching_results if it doesn't exist
        if 'peak_matching_results' not in intermediate_data['molecule_data']:
            intermediate_data['molecule_data']['peak_matching_results'] = {
                'comparisons': {}
            }
        
        # Get the comparison type from results
        comparison_type = results.get('data', {}).get('type', 'unknown')
        
        # Determine the category and subcategory based on comparison type
        if comparison_type == 'peaks_vs_peaks':
            category = 'simulation'
            subcategory = 'exp_vs_sim'
        elif comparison_type == 'smiles_vs_peaks':
            # Check context to determine if it's mol2mol or mmst
            if 'mol2mol' in str(results.get('source', '')).lower():
                category = 'structure_candidates'
                subcategory = 'mol2mol'
            elif 'mmst' in str(results.get('source', '')).lower():
                category = 'structure_candidates'
                subcategory = 'mmst'
            else:
                category = 'custom'
                subcategory = 'smiles_peaks'
        else:
            category = 'custom'
            subcategory = comparison_type
        
        # Initialize category if it doesn't exist
        if category not in intermediate_data['molecule_data']['peak_matching_results']['comparisons']:
            intermediate_data['molecule_data']['peak_matching_results']['comparisons'][category] = {}
        
        # Save results in the appropriate category
        intermediate_data['molecule_data']['peak_matching_results']['comparisons'][category][subcategory] = {
            'status': results['status'],
            'timestamp': datetime.now().isoformat(),
            'results': results.get('data', {}).get('results', {}),
            # 'metadata': {
            #     'matching_mode': results.get('data', {}).get('matching_mode', ''),
            #     'error_type': results.get('data', {}).get('error_type', ''),
            #     'spectra': results.get('data', {}).get('spectra', [])
            # }
        }
        
        # # Update last_updated timestamp
        # intermediate_data['molecule_data']['peak_matching_results']['metadata']['last_updated'] = \
        #     datetime.now().isoformat()
        
        # Save to file
        self._save_intermediate(intermediate_data['molecule_data']['sample_id'], intermediate_data)

    def _prepare_exp_vs_sim_input(self, intermediate_data: Dict) -> Dict:
        """Prepare input data for experimental vs simulated comparison."""
        if 'molecule_data' not in intermediate_data or 'nmr_data' not in intermediate_data['molecule_data']:
            raise ValueError("No NMR data found in intermediate data")
            
        nmr_data = intermediate_data['molecule_data']['nmr_data']
         
        # Format peaks for 1D NMR (1H)
        def format_1d_peaks(peaks):
            if not peaks:
                return {'shifts': [], 'intensities': []}
            # Each peak is a list of [shift, intensity]
            return {
                'shifts': [peak[0] for peak in peaks],
                'intensities': [1.0 for peak in peaks]  # Constant intensity
            }
        
        # Format peaks for 2D NMR (HSQC, COSY)
        def format_2d_peaks(peaks):
            if not peaks:
                return {'F2 (ppm)': [], 'F1 (ppm)': []}
            # Each peak is a list of [f2, f1]
            return {
                'F2 (ppm)': [peak[0] for peak in peaks],
                'F1 (ppm)': [peak[1] for peak in peaks]
            }
        
        # Format peaks for 13C NMR
        def format_13c_peaks(peaks):
            if not peaks:
                return {'shifts': [], 'intensities': []}
            # 13C peaks are just a list of shifts
            return {
                'shifts': peaks,
                'intensities': [1.0] * len(peaks)  # Constant intensity for 13C
            }
    
        # Format experimental and simulated data
        peaks1 = {
            '1H': format_1d_peaks(nmr_data.get('1H_exp', [])),
            '13C': format_13c_peaks(nmr_data.get('13C_exp', [])),
            'HSQC': format_2d_peaks(nmr_data.get('HSQC_exp', [])),
            'COSY': format_2d_peaks(nmr_data.get('COSY_exp', []))
        }
        
        peaks2 = {
            '1H': format_1d_peaks(nmr_data.get('1H_sim', [])),
            '13C': format_13c_peaks(nmr_data.get('13C_sim', [])),
            'HSQC': format_2d_peaks(nmr_data.get('HSQC_sim', [])),
            'COSY': format_2d_peaks(nmr_data.get('COSY_sim', []))
        }
    
        return {
            'type': 'peaks_vs_peaks',
            'spectra': SUPPORTED_SPECTRA,
            'matching_mode': 'hung_dist_nn',
            'error_type': 'sum',
            'peaks1': peaks1,  # Experimental peaks
            'peaks2': peaks2   # Simulated peaks
        }

    def _prepare_smiles_comparison_input(self, context: Dict) -> Dict:
        """Prepare input data for SMILES vs SMILES comparison."""
        input_data = context.get('input_data', {})
        if 'smiles1' not in input_data or 'smiles2' not in input_data:
            raise ValueError("Missing SMILES data for comparison")
            
        return {
            'type': 'smiles_vs_smiles',
            'spectra': SUPPORTED_SPECTRA,
            'matching_mode': 'hung_dist_nn',
            'error_type': 'sum',
            'smiles1': input_data['smiles1'],
            'smiles2': input_data['smiles2']
        }

    def _prepare_smiles_peaks_input(self, context: Dict) -> Dict:
        """Prepare input data for SMILES vs peaks comparison."""
        input_data = context.get('input_data', {})
        if 'smiles' not in input_data or 'peaks' not in input_data:
            raise ValueError("Missing SMILES or peaks data for comparison")
            
        return {
            'type': 'smiles_vs_peaks',
            'spectra': SUPPORTED_SPECTRA,
            'matching_mode': 'hung_dist_nn',
            'error_type': 'sum',
            'smiles': input_data['smiles'],
            'peaks': input_data['peaks']
        }

    def _prepare_peaks_csv_input(self, context: Dict) -> Dict:
        """Prepare input data for peaks vs SMILES CSV comparison."""
        input_data = context.get('input_data', {})
        if 'peaks' not in input_data or 'smiles_csv' not in input_data:
            raise ValueError("Missing peaks or SMILES CSV data for comparison")
            
        return {
            'type': 'peaks_vs_smiles_csv',
            'spectra': SUPPORTED_SPECTRA,
            'matching_mode': 'hung_dist_nn',
            'error_type': 'sum',
            'peaks': input_data['peaks'],
            'smiles_csv': input_data['smiles_csv']
        }

    async def _run_peak_matching(self, input_data: Dict) -> Dict:
        """Run peak matching script with prepared input data."""
        try:
            # Create run directory
            run_dir = self.peak_matching_dir / 'current_run'
            if run_dir.exists():
                self.logger.info(f"Cleaning up previous run directory {run_dir}")
                for file in run_dir.glob('*'):
                    file.unlink()
            run_dir.mkdir(exist_ok=True)
            
            # Save input data
            data_path = run_dir / 'input_data.json'
            with open(data_path, 'w') as f:
                json.dump(input_data, f, indent=2)
            
            # Execute peak matching script
            self.logger.info("Executing peak matching script")
            cmd = [str(LOCAL_SCRIPT), str(data_path)]
            
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    env = os.environ.copy()
                    env['PYTHONPATH'] = str(BASE_DIR)
                    
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    try:
                        stdout, stderr = await asyncio.wait_for(
                            process.communicate(),
                            timeout=SUBPROCESS_TIMEOUT
                        )
                        
                        if process.returncode != 0:
                            error_msg = f"Peak matching script failed with code {process.returncode}"
                            if stderr:
                                error_msg += f"\nError output:\n{stderr.decode()}"
                            raise RuntimeError(error_msg)
                        
                        # Get results
                        return await self._wait_for_results(run_dir, {})
                        
                    except asyncio.TimeoutError:
                        process.kill()
                        raise TimeoutError(f"Peak matching script timed out after {SUBPROCESS_TIMEOUT} seconds")
                        
                except Exception as e:
                    retries += 1
                    if retries >= MAX_RETRIES:
                        self.logger.error(f"Failed after {MAX_RETRIES} attempts: {str(e)}")
                        return {'status': 'error', 'message': str(e)}
                    self.logger.warning(f"Attempt {retries} failed: {str(e)}. Retrying...")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            self.logger.error(f"Error running peak matching: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    async def process_peaks(self, sample_id: str, context: Optional[Dict] = None) -> Dict:
        """Process peak matching with different modes"""
        try:
            # Load intermediate data
            intermediate_data = self._load_or_create_intermediate(sample_id, context)
            
            # Check if results exist
            if self._has_existing_results(intermediate_data):
                return self._get_existing_results(intermediate_data)
                
            # Determine comparison mode
            comparison_mode = self._determine_comparison_mode(context)
            
            # Prepare input data based on mode
            if comparison_mode == 'default':
                input_data = self._prepare_exp_vs_sim_input(intermediate_data)
            elif comparison_mode == 'smiles_vs_smiles':
                input_data = self._prepare_smiles_comparison_input(context)
            elif comparison_mode == 'smiles_vs_peaks':
                input_data = self._prepare_smiles_peaks_input(context)
            elif comparison_mode == 'peaks_vs_smiles_csv':
                input_data = self._prepare_peaks_csv_input(context)
            else:
                raise ValueError(f"Unsupported comparison mode: {comparison_mode}")
                
            # Run peak matching
            results = await self._run_peak_matching(input_data)
            
            # Save results if successful
            if results['status'] == 'success':
                self._save_results(intermediate_data, results)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error in peak matching: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _determine_comparison_mode(self, context: Optional[Dict]) -> str:
        """Determine the comparison mode based on context."""
        if not context:
            return 'default'
            
        input_data = context.get('input_data', {})
        
        if 'comparison_mode' in input_data:
            return input_data['comparison_mode']
            
        if 'smiles1' in input_data and 'smiles2' in input_data:
            return 'smiles_vs_smiles'
        elif 'smiles' in input_data and 'peaks' in input_data:
            return 'smiles_vs_peaks'
        elif 'peaks' in input_data and 'smiles_csv' in input_data:
            return 'peaks_vs_smiles_csv'
            
        return 'default'

#---------------------------------------------------------------------------------

    async def _prepare_input_data(
        self,
        input_data: Dict,
        run_dir: Path,
        context: Dict
    ) -> Dict:
        """Prepare input data for peak matching script.
        
        Args:
            input_data: Input data dictionary containing:
                - Required: One of the following combinations:
                    * smiles1, smiles2 for SMILES vs SMILES comparison
                    * smiles, peaks for SMILES vs peaks comparison
                    * peaks1, peaks2 for peaks vs peaks comparison
                    * peaks, smiles_csv for peaks vs SMILES CSV comparison
                    * reference_smiles, smiles_csv for SMILES vs SMILES CSV comparison
                - Optional:
                    * matching_mode: Peak matching strategy ('hung_dist_nn' or 'euc_dist_all')
                    * error_type: Error calculation method ('sum' or 'avg')
                    * spectra: List of spectrum types to compare
            run_dir: Directory for this run
            context: Additional context dictionary
            
        Returns:
            Dictionary with prepared data and paths
        """
        try:
            self.logger.info("Preparing input data")
            
            # Validate peak matching configuration
            matching_mode = input_data.get('matching_mode', context.get('matching_mode', DEFAULT_MATCHING_MODE))
            if matching_mode not in SUPPORTED_MATCHING_MODES:
                raise ValueError(f"Unsupported matching mode: {matching_mode}. Must be one of {SUPPORTED_MATCHING_MODES}")
                
            error_type = input_data.get('error_type', context.get('error_type', DEFAULT_ERROR_TYPE))
            if error_type not in SUPPORTED_ERROR_TYPES:
                raise ValueError(f"Unsupported error type: {error_type}. Must be one of {SUPPORTED_ERROR_TYPES}")
            
            # Determine input type and prepare data
            if 'smiles1' in input_data and 'smiles2' in input_data:
                data_type = 'smiles_vs_smiles'
                prepared_data = {
                    'type': data_type,
                    'smiles1': input_data['smiles1'],
                    'smiles2': input_data['smiles2']
                }
            elif 'smiles' in input_data and 'peaks' in input_data:
                data_type = 'smiles_vs_peaks'
                prepared_data = {
                    'type': data_type,
                    'smiles': input_data['smiles'],
                    'peaks': input_data['peaks']
                }
            elif 'peaks1' in input_data and 'peaks2' in input_data:
                data_type = 'peaks_vs_peaks'
                prepared_data = {
                    'type': data_type,
                    'peaks1': input_data['peaks1'],
                    'peaks2': input_data['peaks2']
                }
            elif 'peaks' in input_data and 'smiles_csv' in input_data:
                data_type = 'peaks_vs_smiles_csv'
                prepared_data = {
                    'type': data_type,
                    'peaks': input_data['peaks'],
                    'smiles_csv': input_data['smiles_csv']
                }
            elif 'reference_smiles' in input_data and 'smiles_csv' in input_data:
                data_type = 'smiles_vs_smiles_csv'
                prepared_data = {
                    'type': data_type,
                    'reference_smiles': input_data['reference_smiles'],
                    'smiles_csv': input_data['smiles_csv']
                }
            else:
                raise ValueError("Invalid input data format")

            # Add context information
            prepared_data.update({
                'spectra': context.get('spectra', SUPPORTED_SPECTRA),
                'matching_mode': context.get('matching_mode', 'hung_dist_nn'),
                'error_type': context.get('error_type', 'sum')
            })
            
            # Note: The following configuration options are planned for future implementation:
            # config = {
            #     'matching': {
            #         'mode': matching_mode,
            #         'parameters': {
            #             'hung_dist_nn': {
            #                 'max_distance': 0.1,  # Maximum distance for peak matching
            #                 'use_intensity': True  # Whether to consider peak intensities
            #             },
            #             'euc_dist_all': {
            #                 'threshold': 0.1,  # Distance threshold for considering peaks as matching
            #                 'normalize': True  # Whether to normalize distances
            #             }
            #         }
            #     },
            #     'error': {
            #         'type': error_type,
            #         'parameters': {
            #             'sum': {},  # No additional parameters needed
            #             'avg': {
            #                 'weighted': True  # Whether to weight by peak intensities
            #             }
            #         }
            #     }
            # }
            
            # Save prepared data
            data_path = run_dir / 'input_data.json'
            with open(data_path, 'w') as f:
                json.dump(prepared_data, f, indent=2)
            return {
                'status': 'success',
                'data_type': data_type,
                'data_path': str(data_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing input data: {str(e)}")
            raise


    async def process(
        self,
        input_data: Dict,
        context: Optional[Dict] = None
    ) -> Dict:
        """Process peak matching request."""
        self.logger.info("Starting peak matching tool process")
        context = context or {}
        
        # Create run directory
        run_dir = self.peak_matching_dir / 'current_run'
        if run_dir.exists():
            self.logger.info("Cleaning up previous run directory")
            for file in run_dir.glob('*'):
                file.unlink()
        run_dir.mkdir(exist_ok=True)
        self.logger.info(f"Created run directory: {run_dir}")
        
        retries = 0
        while retries < MAX_RETRIES:
            try:
                # Prepare input data
                self.logger.info("Preparing input data")
                prep_result = await self._prepare_input_data(input_data, run_dir, context)
                if prep_result['status'] != 'success':
                    self.logger.error(f"Input data preparation failed: {prep_result}")
                    return prep_result
                self.logger.info("Input data preparation successful")
                
                # Execute peak matching script
                self.logger.info("Executing peak matching script")
                cmd = [str(LOCAL_SCRIPT), str(prep_result['data_path'])]
                self.logger.info(f"Command: {' '.join(cmd)}")
                
                try:
                    env = os.environ.copy()
                    env['PYTHONPATH'] = str(BASE_DIR)
                    self.logger.info(f"Environment: PYTHONPATH={env['PYTHONPATH']}")
                    
                    # Start subprocess with timeout
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    try:
                        stdout, stderr = await asyncio.wait_for(
                            process.communicate(),
                            timeout=SUBPROCESS_TIMEOUT
                        )
                        
                        if process.returncode != 0:
                            error_msg = f"Peak matching script failed with code {process.returncode}"
                            if stderr:
                                error_msg += f"\nError output:\n{stderr.decode()}"
                            raise RuntimeError(error_msg)
                        
                        # Wait for and return results
                        return await self._wait_for_results(run_dir, {})
                        
                    except asyncio.TimeoutError:
                        process.kill()
                        raise TimeoutError(f"Peak matching script timed out after {SUBPROCESS_TIMEOUT} seconds")
                        
                except (subprocess.SubprocessError, OSError) as e:
                    error_msg = f"Error executing peak matching script: {str(e)}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
                    
            except Exception as e:
                retries += 1
                if retries >= MAX_RETRIES:
                    self.logger.error(f"Failed after {MAX_RETRIES} attempts: {str(e)}")
                    raise
                self.logger.warning(f"Attempt {retries} failed: {str(e)}. Retrying...")
                await asyncio.sleep(1)  # Wait before retry

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/tools/__init__.py ---



--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/tools/forward_synthesis_tool.py ---
"""Tool for generating forward synthesis predictions using Chemformer."""
import os
import logging
import shutil
import pandas as pd
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import json

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.sbatch_utils import execute_sbatch, wait_for_job_completion

# Constants for paths and directories
BASE_DIR = Path(__file__).parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "agents" / "scripts"
FORWARD_DIR = BASE_DIR / "_temp_folder" / "forward_output"
TEMP_DIR = BASE_DIR / "_temp_folder"
SBATCH_SCRIPT = SCRIPTS_DIR / "chemformer_forward_sbatch.sh"
LOCAL_SCRIPT = SCRIPTS_DIR / "chemformer_forward_local.sh"

# Constants for Chemformer execution
FORWARD_OUTPUT_CHECK_INTERVAL = 5  # seconds
FORWARD_OUTPUT_TIMEOUT = 600  # 10 minutes
FORWARD_OUTPUT_PATTERN = "forward_predictions_{}.csv"  # Will be formatted with timestamp
FORWARD_INPUT_FILENAME = "forward_reactants.txt"  # Input filename

class ForwardSynthesisTool:
    """Tool for generating forward synthesis predictions using Chemformer."""
    
    def __init__(self):
        """Initialize the Forward Synthesis tool with required directories."""
        # Ensure required directories exist
        self.scripts_dir = SCRIPTS_DIR
        self.forward_dir = FORWARD_DIR
        
        # Create directories if they don't exist
        self.forward_dir.mkdir(exist_ok=True)
        
        # Add intermediate results directory
        self.temp_dir = TEMP_DIR
        self.intermediate_dir = self.temp_dir / "intermediate_results"
        self.intermediate_dir.mkdir(exist_ok=True)
        
        # Validate script existence
        if not SBATCH_SCRIPT.exists():
            raise FileNotFoundError(f"Required SBATCH script not found at {SBATCH_SCRIPT}")
        if not LOCAL_SCRIPT.exists():
            raise FileNotFoundError(f"Required local script not found at {LOCAL_SCRIPT}")

        # Validate environment
        try:
            import torch
            if not torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID') is None:
                logging.warning("CUDA not available for local execution. SLURM execution will be forced.")
        except ImportError:
            logging.warning("PyTorch not found. Please ensure the chemformer environment is activated.")

    # def _normalize_smiles(self, smiles: str) -> str:
    #     """Normalize SMILES string to canonical form."""
    #     try:
    #         from rdkit import Chem
    #         # Handle disconnected structures by splitting on '.'
    #         parts = smiles.split('.')
    #         normalized_parts = []
    #         for part in parts:
    #             mol = Chem.MolFromSmiles(part.strip())
    #             if mol is not None:
    #                 normalized_parts.append(Chem.MolToSmiles(mol, canonical=True))
    #         if normalized_parts:
    #             return '.'.join(normalized_parts)
    #     except ImportError:
    #         logging.warning("RDKit not available for SMILES normalization")
    #     except Exception as e:
    #         logging.warning(f"Error normalizing SMILES {smiles}: {str(e)}")
    #     return smiles.strip()

    # def _prepare_input_from_context(self, context: Dict[str, Any]) -> tuple[Path, Dict[str, list]]:
    #     """Prepare input file from context and return mapping of predictions to source molecules.
        
    #     Args:
    #         context: Dictionary containing molecule data and flags
            
    #     Returns:
    #         Tuple of:
    #         - Path to the created input file
    #         - Dictionary mapping molecule IDs to their starting material indices in the input file
    #     """
    #     try:
    #         # Initialize variables
    #         smiles_list = []
    #         input_file = self.temp_dir / FORWARD_INPUT_FILENAME
    #         molecule_mapping = {}  # Maps molecule IDs to their starting material indices
    #         current_index = 0
            
    #         # Get molecules from master JSON file
    #         master_data_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
    #         if master_data_path.exists():
    #             with open(master_data_path, 'r') as f:
    #                 master_data = json.load(f)
    #             # Extract starting materials from master data
    #             for molecule_id, molecule_data in master_data.items():
    #                 if 'starting_smiles' in molecule_data:
    #                     start_idx = current_index
    #                     for starting_smiles in molecule_data['starting_smiles']:
    #                         if isinstance(starting_smiles, list):
    #                             for smiles in starting_smiles:
    #                                 smiles_list.append(smiles.strip())
    #                                 current_index += 1
    #                         elif isinstance(starting_smiles, str):
    #                             smiles_list.append(starting_smiles.strip())
    #                             current_index += 1
    #                     # Store the range of indices for this molecule's starting materials
    #                     if current_index > start_idx:
    #                         molecule_mapping[molecule_id] = (start_idx, current_index)
    #             # logging.info(f"Extracted {len(smiles_list)} starting materials from master JSON")
    #         else:
    #             logging.warning("Master JSON file not found")
            
    #         # If no starting materials found in master JSON, try context as fallback
    #         if not smiles_list and context.get('current_molecule'):
    #             current_molecule = context['current_molecule']
    #             if isinstance(current_molecule, dict):
    #                 # Try to get starting materials from current molecule
    #                 if 'starting_smiles' in current_molecule:
    #                     materials = current_molecule['starting_smiles']
    #                     if isinstance(materials, list):
    #                         for material in materials:
    #                             smiles_list.append(material.strip())
    #                     elif isinstance(materials, str):
    #                         smiles_list.append(materials.strip())
    #             elif isinstance(current_molecule, str):
    #                 # If it's just a SMILES string, use it as is
    #                 smiles_list.append(current_molecule.strip())
            
    #         # Validate SMILES list
    #         if not smiles_list:
    #             raise ValueError("No valid starting materials found in master JSON or context")
            
    #         # Write SMILES to input file
    #         with open(input_file, 'w') as f:
    #             for smiles in smiles_list:
    #                 f.write(f"{smiles}\n")
            
    #         # logging.info(f"Prepared input file with {len(smiles_list)} starting materials")
    #         return input_file, molecule_mapping
            
    #     except Exception as e:
    #         logging.error(f"Error preparing input file: {str(e)}")
    #         raise

    # async def _update_master_data(self, predictions_df: pd.DataFrame, molecule_mapping: Dict[str, tuple]) -> None:
    #     """Update the master data file with forward synthesis predictions.
        
    #     Args:
    #         predictions_df: DataFrame containing forward synthesis predictions
    #         molecule_mapping: Dictionary mapping molecule IDs to their prediction indices
    #     """
    #     try:
    #         master_data_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
    #         if not master_data_path.exists():
    #             logging.error("Master data file not found")
    #             return
             
    #         # Read current master data
    #         with open(master_data_path, 'r') as f:
    #             master_data = json.load(f)
            
    #         # Convert predictions to list for easier processing
    #         predictions_list = predictions_df.to_dict('records')
            
    #         updated = False
    #         # Update each molecule's data with forward synthesis predictions
    #         for molecule_id, (start_idx, end_idx) in molecule_mapping.items():
    #             if molecule_id in master_data:
    #                 # Get all predictions for this molecule's starting materials
    #                 molecule_predictions = predictions_list[start_idx:end_idx]
                    
    #                 # Initialize forward_predictions if not exists
    #                 if 'forward_predictions' not in master_data[molecule_id]:
    #                     master_data[molecule_id]['forward_predictions'] = []
                    
    #                 # Process predictions for each starting material
    #                 for pred in molecule_predictions:
    #                     # Create prediction entry exactly matching CSV format
    #                     prediction = {
    #                         'starting_material': pred['target_smiles'],  
    #                         'predicted_smiles': pred['predicted_smiles'],
    #                         'log_likelihood': float(pred['log_likelihood']),
    #                         'all_predictions': pred['all_predictions'].split(';'),
    #                         'all_log_likelihoods': [float(l.strip()) for l in pred['all_log_likelihoods'].split(';')]
    #                     }
                        
    #                     # Check if this prediction already exists
    #                     exists = False
    #                     for existing_pred in master_data[molecule_id]['forward_predictions']:
    #                         if (existing_pred['starting_material'] == prediction['starting_material'] and
    #                             existing_pred['predicted_smiles'] == prediction['predicted_smiles']):
    #                             exists = True
    #                             break
                        
    #                     if not exists:
    #                         master_data[molecule_id]['forward_predictions'].append(prediction)
    #                         updated = True
    #                         # logging.info(f"Added forward prediction for sample {molecule_id} starting material {pred['target_smiles']}")
            
    #         if updated:
    #             # Write updated data back to file
    #             with open(master_data_path, 'w') as f:
    #                 json.dump(master_data, f, indent=2)
    #             logging.info("Successfully updated master data with forward synthesis predictions")
    #         else:
    #             logging.warning("No new predictions to add to master data")
                
    #     except Exception as e:
    #         logging.error(f"Error updating master data: {str(e)}")
    #         raise

    async def predict_forward_synthesis(self, molecule_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            # Get sample_id from context or molecule_data
            sample_id = None
            if context and 'sample_id' in context:
                sample_id = context['sample_id']
            elif isinstance(molecule_data, dict) and 'sample_id' in molecule_data:
                sample_id = molecule_data['sample_id']
            
            if not sample_id:
                raise ValueError("No sample_id provided in context or molecule_data")

            # Load or create intermediate file
            intermediate_data = self._load_or_create_intermediate(sample_id, molecule_data)
            
            # Check if forward synthesis predictions already exist
            if ('molecule_data' in intermediate_data and 
                'forward_predictions' in intermediate_data['molecule_data'] and
                intermediate_data['molecule_data']['forward_predictions']):
                logging.info(f"Forward synthesis predictions already exist for sample {sample_id}")
                return {
                    'status': 'success',
                    'message': 'Forward synthesis predictions already exist',
                    'predictions': intermediate_data['molecule_data']['forward_predictions']
                }

            # Get starting materials from molecule data
            if not isinstance(intermediate_data.get('molecule_data', {}), dict):
                intermediate_data['molecule_data'] = {}
            
            starting_smiles = intermediate_data['molecule_data'].get('starting_smiles')
            if not starting_smiles:
                starting_smiles = molecule_data.get('starting_smiles')  # Try getting from original molecule_data
                if starting_smiles:
                    intermediate_data['molecule_data']['starting_smiles'] = starting_smiles
                else:
                    raise ValueError("No starting materials found in molecule data")
            
            # Create input file with starting materials
            input_file = self.temp_dir / f"{sample_id}_input.txt"
            with open(input_file, 'w') as f:
                for smiles in starting_smiles:
                    f.write(f"{smiles}\n")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.forward_dir / FORWARD_OUTPUT_PATTERN.format(timestamp)
            
            # Check CUDA availability for local execution
            use_slurm = context.get('use_slurm', False)
            if not use_slurm:
                try:
                    import torch
                    if not torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID') is None:
                        logging.warning("CUDA not available. Switching to SLURM execution.")
                        use_slurm = True
                except ImportError:
                    logging.warning("PyTorch not found. Switching to SLURM execution.")
                    use_slurm = True
            
            logging.info(f"Running forward synthesis prediction using {'SLURM' if use_slurm else 'local'} execution")
            
            if use_slurm:
                try:
                    # Execute using SLURM
                    logging.info("Running forward synthesis with SLURM")
                    job_id = await execute_sbatch(
                        str(SBATCH_SCRIPT),
                        f"--input_file={input_file}",
                        f"--output_file={output_file}"
                    )
                    logging.info(f"SLURM job submitted with ID: {job_id}")
                    
                    success = await wait_for_job_completion(job_id)
                    if not success:
                        logging.error("SLURM job failed during execution")
                        return {
                            'status': 'error',
                            'message': 'SLURM job failed during execution'
                        }
                    
                    # Add a small delay to ensure file is fully written
                    logging.info("SLURM job completed, waiting for file system sync...")
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    logging.error(f"Error during SLURM execution: {str(e)}")
                    return {
                        'status': 'error',
                        'message': f'Error during SLURM execution: {str(e)}'
                    }
                
                # Check if the output file exists and has content
                try:
                    if not output_file.exists():
                        error_msg = f"Output file not found at {output_file} after SLURM job completion"
                        logging.error(error_msg)
                        return {
                            'status': 'error',
                            'message': error_msg
                        }
                    
                    file_size = output_file.stat().st_size
                    logging.info(f"Output file exists with size: {file_size} bytes")
                    
                    if file_size == 0:
                        error_msg = "Output file is empty after SLURM job completion"
                        logging.error(error_msg)
                        return {
                            'status': 'error',
                            'message': error_msg
                        }
                except Exception as e:
                    logging.error(f"Error checking output file: {str(e)}")
                    return {
                        'status': 'error',
                        'message': f'Error checking output file: {str(e)}'
                    }
            else:
                # Execute locally
                try:
                    logging.info("Running forward synthesis locally")
                    process = await asyncio.create_subprocess_exec(
                        str(LOCAL_SCRIPT),
                        f"--input_file={input_file}",
                        f"--output_file={output_file}",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode != 0:
                        error_msg = f"Local execution failed with return code {process.returncode}: {stderr.decode()}"
                        logging.error(error_msg)
                        return {'status': 'error', 'message': error_msg}
                        
                except Exception as e:
                    error_msg = f"Error during local execution: {str(e)}"
                    logging.error(error_msg)
                    return {'status': 'error', 'message': error_msg}
            
            # Wait for output file
            if not await self._wait_for_output(output_file):
                error_msg = f"Timeout waiting for output file at {output_file}"
                logging.error(error_msg)
                return {'status': 'error', 'message': error_msg}
            
            # Read predictions from output file
            try:
                predictions_df = pd.read_csv(output_file)
                
                # Process predictions maintaining the same structure
                forward_predictions = []
                for _, row in predictions_df.iterrows():
                    # Get all predictions and their log likelihoods
                    all_preds = [p.strip() for p in row['all_predictions'].split(';') if p.strip()]
                    all_logs = [float(l.strip()) for l in row['all_log_likelihoods'].split(';') if l.strip()]
                    
                    prediction = {
                        'starting_material': row['target_smiles'],
                        'predicted_smiles': row['predicted_smiles'],
                        'log_likelihood': float(row['log_likelihood']),
                        'all_predictions': all_preds,
                        'all_log_likelihoods': all_logs
                    }
                    forward_predictions.append(prediction)
                
                # Store predictions directly in molecule_data
                intermediate_data['molecule_data']['forward_predictions'] = forward_predictions
                
                # Save to intermediate file
                self._save_intermediate(sample_id, intermediate_data)
                
                # Clean up temporary files
                if input_file.exists():
                    input_file.unlink()
                if output_file.exists():
                    output_file.unlink()
                
                return {
                    'status': 'success',
                    'message': 'Successfully generated forward synthesis predictions',
                    'predictions': forward_predictions
                }
                
            except Exception as e:
                error_msg = f"Error reading predictions from output file: {str(e)}"
                logging.error(error_msg)
                return {'status': 'error', 'message': error_msg}
                
        except Exception as e:
            error_msg = f"Unexpected error in forward synthesis prediction: {str(e)}"
            logging.error(error_msg)
            raise

    async def _wait_for_output(self, output_file: Path, timeout: int = FORWARD_OUTPUT_TIMEOUT) -> bool:
        """Wait for the output file to be generated."""
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            if output_file.exists():
                return True
            await asyncio.sleep(FORWARD_OUTPUT_CHECK_INTERVAL)
        return False
        

    def _get_intermediate_path(self, sample_id: str) -> Path:
        """Get path to intermediate file for a sample."""
        return self.intermediate_dir / f"{sample_id}_intermediate.json"


    def _load_or_create_intermediate(self, sample_id: str, context: Dict[str, Any] = None) -> Dict:
        """Load existing intermediate file or create new one from master data.
        """
        intermediate_path = self._get_intermediate_path(sample_id)
        
        # Try loading existing intermediate
        if intermediate_path.exists():
            with open(intermediate_path, 'r') as f:
                data = json.load(f)
                if 'molecule_data' not in data:
                    data['molecule_data'] = {}
                return data
                
        # If no intermediate exists, try getting data from master file
        master_path = BASE_DIR / "data" / "molecular_data" / "molecular_data.json"
        if master_path.exists():
            with open(master_path, 'r') as f:
                master_data = json.load(f)

                if sample_id in master_data:
                    # Create new intermediate with just this sample's data
                    intermediate_data = master_data[sample_id]
                    self._save_intermediate(sample_id, intermediate_data)
                    return intermediate_data
                    
        # If neither exists and we have context, create new intermediate
        if context:
            intermediate_data["molecule_data"] = context
            self._save_intermediate(sample_id, intermediate_data)
            return intermediate_data
            
        raise ValueError(f"No data found for sample {sample_id}")

    def _save_intermediate(self, sample_id: str, data: Dict):
        """Save data to intermediate file.
        
        Args:
            sample_id: ID of the sample to save
            data: Data to save to intermediate file
        """
        path = self._get_intermediate_path(sample_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/tools/data_extraction_tool.py ---
"""
Tool for extracting and managing molecular data from both master and intermediate files.
"""
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
import logging
from enum import Enum
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Enum for different data sources"""
    MASTER_FILE = "master"
    INTERMEDIATE = "intermediate"

class DataExtractionTool:
    """Tool for extracting molecular and spectral data from various sources."""

    def __init__(self):
        """Initialize the data extraction tool."""
        self.base_path = Path(__file__).parent.parent.parent
        
        # Set up file paths
        self.master_file_path = self.base_path / "data" / "molecular_data" / "molecular_data.json"
        self.intermediate_dir = self.base_path / "_temp_folder" / "intermediate_results"

    def _get_intermediate_path(self, sample_id: str) -> Path:
        """Get path to intermediate file for a sample."""
        return self.intermediate_dir / f"{sample_id}_intermediate.json"

    async def load_data(self, 
                       sample_id: Optional[str] = None, 
                       source: DataSource = DataSource.MASTER_FILE) -> Dict:
        """
        Load molecular data from either master file or intermediate file.
        
        Args:
            sample_id: Sample ID to load. Required for master file, optional for intermediate.
            source: Source to load data from (master or intermediate file).
            
        Returns:
            Dictionary containing molecular data with standardized structure.
        """
        try:
            if source == DataSource.MASTER_FILE:
                if not sample_id:
                    raise ValueError("sample_id is required for master file access")
                    
                if not self.master_file_path.exists():
                    raise FileNotFoundError(f"Master file not found at {self.master_file_path}")
                    
                with open(self.master_file_path, 'r') as f:
                    master_data = json.load(f)
                    
                if sample_id not in master_data:
                    raise KeyError(f"Sample {sample_id} not found in master file")
                    
                data = master_data[sample_id]
                
            else:  # INTERMEDIATE
                if not sample_id:
                    raise ValueError("sample_id is required for intermediate file access")
                    
                intermediate_path = self._get_intermediate_path(sample_id)
                if not intermediate_path.exists():
                    raise FileNotFoundError(f"Intermediate file not found at {intermediate_path}")
                    
                with open(intermediate_path, 'r') as f:
                    data = json.load(f)

            # Ensure analysis_results key exists at top level
            if 'analysis_results' not in data:
                data['analysis_results'] = {}
                
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {source.value}: {str(e)}")
            raise

    async def save_data(self,
                       data: Dict,
                       sample_id: str,
                       source: DataSource = DataSource.INTERMEDIATE) -> None:
        """
        Save data back to the source file.
        
        Args:
            data: Data to save
            sample_id: Sample ID to save data for
            source: Source to save data to
        """
        try:
            # Ensure data has the correct structure
            if not isinstance(data, dict):
                data = {'molecule_data': data}
            elif 'molecule_data' not in data:
                data = {'molecule_data': data.copy()}
                
            if source == DataSource.MASTER_FILE:
                if not self.master_file_path.exists():
                    self.master_file_path.parent.mkdir(parents=True, exist_ok=True)
                    master_data = {}
                else:
                    with open(self.master_file_path, 'r') as f:
                        master_data = json.load(f)
                
                master_data[sample_id] = data
                
                with open(self.master_file_path, 'w') as f:
                    json.dump(master_data, f, indent=2)
            
            else:  # INTERMEDIATE
                intermediate_path = self._get_intermediate_path(sample_id)
                intermediate_path.parent.mkdir(parents=True, exist_ok=True)
                with open(intermediate_path, 'w') as f:
                    json.dump(data, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Error saving data to {source.value}: {str(e)}")
            raise

    async def extract_experimental_nmr_data(self, 
                                  sample_id: str, 
                                  spectrum_type: str,
                                  source: DataSource = DataSource.MASTER_FILE) -> Dict:
        """
        Extract NMR data for a specific spectrum type.
        
        Args:
            sample_id: Sample ID to extract data for
            spectrum_type: Type of spectrum (e.g., 'HSQC', 'COSY', '1H', '13C')
            source: Source to load data from
            
        Returns:
            Dictionary containing NMR spectral data
        """
        try:
            data = await self.load_data(sample_id, source)
            molecule_data = data.get('molecule_data', data)  # Handle both old and new format
            
            if 'nmr_data' not in molecule_data:
                return {}
                
            # Map common spectrum type variations
            spectrum_map = {
                '1h': '1H_exp',
                '13c': '13C_exp',
                'hsqc': 'HSQC_exp',
                'cosy': 'COSY_exp'
            }
            
            spectrum_key = spectrum_map.get(spectrum_type.lower(), spectrum_type)
            return molecule_data['nmr_data'].get(spectrum_key, {})
            
        except Exception as e:
            logger.error(f"Error extracting NMR data: {str(e)}")
            return {}

    async def extract_top_candidates(self, 
                                   sample_id: str,
                                   n: int = 3,
                                   sort_by: str = 'hsqc_score',
                                   source: DataSource = DataSource.MASTER_FILE) -> List[Dict]:
        """
        Extract top N candidates based on scoring criteria.
        
        Args:
            sample_id: Sample ID to extract data for
            n: Number of top candidates to extract
            sort_by: Score type to sort by ('hsqc_score', 'overall_score', etc.)
            source: Source to load data from
            
        Returns:
            List of top N candidates with their data
        """
        data = await self.load_data(sample_id, source)
        
        try:
            if 'candidates' not in data:
                raise KeyError("No candidate data found in molecular data")
                
            candidates = data['candidates']
            
            # Sort candidates by specified score
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x.get('scores', {}).get(sort_by, float('-inf')),
                reverse=True  # Higher score is better
            )
            
            # Store analysis results
            analysis_result = {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'candidate_ranking',
                'parameters': {
                    'n': n,
                    'sort_by': sort_by
                },
                'results': {
                    'top_candidates': sorted_candidates[:n],
                    'ranking_criteria': sort_by
                }
            }
            
            data['analysis_results']['candidate_ranking'] = analysis_result
            await self.save_data(data, sample_id, source)
            
            return sorted_candidates[:n]
            
        except Exception as e:
            logger.error(f"Error extracting top candidates: {str(e)}")
            raise

    async def extract_reaction_data(self, 
                                  sample_id: str,
                                  source: DataSource = DataSource.MASTER_FILE) -> Dict:
        """
        Extract reaction-related data including starting materials.
        
        Args:
            sample_id: Sample ID to extract data for
            source: Source to load data from
            
        Returns:
            Dictionary containing reaction data
        """
        data = await self.load_data(sample_id, source)
        
        try:
            reaction_data = {
                'starting_material': data.get('starting_material'),
                'target_molecule': data.get('target_molecule'),
                'predicted_products': data.get('predicted_products', []),
                'reaction_type': data.get('reaction_type'),
                'reaction_conditions': data.get('reaction_conditions')
            }
            
            # Store analysis results
            analysis_result = {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'reaction_data_extraction',
                'results': reaction_data
            }
            
            data['analysis_results']['reaction_analysis'] = analysis_result
            await self.save_data(data, sample_id, source)
            
            return reaction_data
            
        except Exception as e:
            logger.error(f"Error extracting reaction data: {str(e)}")
            raise

    async def extract_analysis_results(self,
                                     sample_id: str,
                                     analysis_type: Optional[str] = None,
                                     source: DataSource = DataSource.MASTER_FILE) -> Dict:
        """
        Extract previous analysis results.
        
        Args:
            sample_id: Sample ID to extract data for
            analysis_type: Specific type of analysis to extract (optional)
            source: Source to load data from
            
        Returns:
            Dictionary containing analysis results
        """
        data = await self.load_data(sample_id, source)
        
        try:
            if 'analysis_results' not in data:
                return {}
                
            results = data['analysis_results']
            
            if analysis_type:
                return results.get(analysis_type, {})
            
            return results
            
        except Exception as e:
            logger.error(f"Error extracting analysis results: {str(e)}")
            raise


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/tools/final_analysis_tool.py ---
"""
Tool for performing final comprehensive analysis of molecule candidates.
"""
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import json
import ast
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import re
from .analysis_enums import DataSource, RankingMetric
from .data_extraction_tool import DataExtractionTool
import json as json_module  # Alias the json module to avoid name conflicts
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinalAnalysisTool:
    """Tool for performing final comprehensive analysis of molecule candidates."""
    
    def __init__(self, llm_service: Any = None):
        """Initialize the final analysis tool."""
        self.llm_service = llm_service
        self.data_tool = DataExtractionTool()
        
    async def analyze_final_results(self, workflow_data: Dict[str, Any], 
                                  context: Dict[str, Any],
                                  data_tool: Optional[DataExtractionTool] = None,
                                  llm_service: Optional[Any] = None) -> Dict[str, Any]:
        """
        Perform final comprehensive analysis of all available results.
        
        Args:
            data: Dictionary containing molecular data and analysis results
            context: Additional context including previous analysis results
            data_tool: Optional DataExtractionTool instance, uses self.data_tool if not provided
            llm_service: Optional LLM service instance, uses self.llm_service if not provided
            
        Returns:
            Dictionary containing final analysis results
        """
        try:
            logger.info("Starting final comprehensive analysis")
            
            # Initialize tools
            data, sample_id, molecule_data = await self._initialize_analysis(workflow_data, context, data_tool, llm_service)
            
            # Extract data using helper methods
            try:
                final_ranked_results = self._extract_ranked_candidates(data.get('analysis_results', {}))
                candidate_reasonings = self._extract_candidate_reasonings(data.get('analysis_results', {}))
                spectral_llm_reasonings = self._extract_spectral_llm_reasonings(data.get('analysis_results', {}))
            except Exception as e:
                logger.error(f"Error during data extraction: {str(e)}")
                raise

            # Get the overall LLM analysis
            llm_eval = data.get('analysis_results', {}).get('spectral_llm_evaluation', {})
            overall_llm_analysis = llm_eval.get('overall_analysis', 'No overall analysis available')
            logger.info(f"Found overall LLM analysis: {bool(overall_llm_analysis != 'No overall analysis available')}")

            # Generate comprehensive analysis using LLM
            if self.llm_service and final_ranked_results:
                logger.info("Starting comprehensive LLM analysis")
                
                # Prepare target molecule info
                target_info = {
                    'smiles': molecule_data.get('smiles'),
                    'molecular_weight': molecule_data.get('molecular_weight', ""),
                    'molecular_formula': molecule_data.get('molecular_formula', ""),
                    'experimental_data': molecule_data.get('experimental_data', {})
                }
                logger.info(f"Target molecule info prepared: {target_info}")
                
                # Generate prompt using helper methods
                try:
                    candidate_sections = self._generate_candidate_sections(final_ranked_results, candidate_reasonings)
                    analysis_prompt = self._generate_analysis_prompt(
                        target_info,
                        overall_llm_analysis,
                        spectral_llm_reasonings,
                        candidate_sections,
                        final_ranked_results
                    )
                    logger.info("Successfully generated analysis prompt")
                except Exception as e:
                    logger.error(f"Error generating prompt: {str(e)}")
                    raise
                
                try:
                    # Get model configurations
                    model_configs = self._get_model_configs()
                    
                    # Initialize variables to store responses
                    model_results = {}
                    
                    # Get analysis from each model
                    for model in ['claude', 'deepseek', 'gemini', 'o3']:
                        raw_response, results, reasoning, thinking = await self._get_model_analysis(model, analysis_prompt, model_configs[model])
                        model_results[f'{model}_results'] = {
                            'raw_response': raw_response,
                            'content': results,
                            'reasoning_content': reasoning,
                            'thinking': thinking,
                            'analysis_prompt': analysis_prompt
                        }
                    logger.info("ölkj")
                    # Create final analysis output with all model data
                    final_analysis = self._create_final_analysis(sample_id, molecule_data, final_ranked_results, model_results, analysis_prompt, model_configs)
                    logger.info("asdfasd")

                    # Save final analysis to data
                    try:
                        # Create completed_analysis_steps if it doesn't exist
                        if 'completed_analysis_steps' not in data:
                            data['completed_analysis_steps'] = {}
                        
                        # Ensure analysis_results exists
                        if 'analysis_results' not in data:
                            data['analysis_results'] = {}
                        
                        # Store analysis results and mark as completed
                        data['analysis_results']['final_analysis'] = final_analysis
                        data['completed_analysis_steps']['final_analysis'] = True
                        
                        await self.data_tool.save_data(
                            data,
                            sample_id,
                            DataSource.INTERMEDIATE
                        )
                        logger.info(f"Successfully saved final analysis for sample {sample_id}")
                    except Exception as e:
                        logger.error(f"Error saving final analysis: {str(e)}")
                        raise
                    
                    return {
                        'type': 'success',
                        'content': final_analysis,
                    }
                
                except Exception as e:
                    logger.error(f"Error during LLM request: {str(e)}")
                    # Don't raise here, continue with what we have
                    print(final_ranked_results)
                    # Prepare final output with whatever we have
                    final_analysis = {
                        'timestamp': datetime.now().isoformat(),
                        'sample_id': sample_id,
                        'target_info': {
                            'smiles': molecule_data.get('smiles'),
                            'molecular_weight': molecule_data.get('molecular_weight')
                        },
                        'analyzed_candidates': [
                            {
                                'smiles': result['smiles'],
                                'rank': result['rank'],
                                'molecular_weight': result['molecular_weight'],
                                'scores': result['scores'],
                                'confidence_score': result['confidence_score'],
                                'reasoning': result['reasoning'],
                                'llm_analysis': result.get('llm_analysis', {})
                            }
                            for result in final_ranked_results
                        ],
                        'llm_responses': {
                            'analysis_prompt': analysis_prompt,
                            'claude': {
                                'raw_response': model_results.get('claude_results', {}).get('raw_response'),
                                'parsed_results': model_results.get('claude_results', {}).get('content')
                            },
                            'deepseek': {
                                'raw_response': model_results.get('deepseek_results', {}).get('raw_response'),
                                'reasoning_content': model_results.get('deepseek_results', {}).get('reasoning_content'),
                                'parsed_results': model_results.get('deepseek_results', {}).get('content')
                            },
                            'gemini': {
                                'raw_response': model_results.get('gemini_results', {}).get('raw_response'),
                                'parsed_results': model_results.get('gemini_results', {}).get('content')
                            },
                            'o3': {
                                'raw_response': model_results.get('o3_results', {}).get('raw_response'),
                                'parsed_results': model_results.get('o3_results', {}).get('content')
                            }
                        },
                        'metadata': {
                            'num_candidates': len(final_ranked_results),
                            'analysis_types_used': [],
                            'models_used': [],
                            'description': 'Comprehensive final analysis of all candidates based on available evidence',
                            'analysis_status': 'partial_failure',
                            'analysis_prompt': analysis_prompt
                        }
                    }

                    # Save final analysis to data
                    try:
                        # Create completed_analysis_steps if it doesn't exist
                        if 'completed_analysis_steps' not in data:
                            data['completed_analysis_steps'] = {}
                        
                        # Ensure analysis_results exists
                        if 'analysis_results' not in data:
                            data['analysis_results'] = {}
                        
                        # Store analysis results and mark as completed
                        data['analysis_results']['final_analysis'] = final_analysis
                        data['completed_analysis_steps']['final_analysis'] = True
                        
                        await self.data_tool.save_data(
                            data,
                            sample_id,
                            DataSource.INTERMEDIATE
                        )
                        logger.info(f"Successfully saved final analysis for sample {sample_id}")
                    except Exception as e:
                        logger.error(f"Error saving final analysis: {str(e)}")
                        raise
                    
                    return {
                        'type': 'success',
                        'content': final_analysis,
                    }
                    
        except Exception as e:
            logger.error(f"Error in final analysis: {str(e)}")
            return {
                'type': 'error',
                'content': str(e),
            }

    async def analyze_with_deepseek(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """Analyze using DeepSeek model via Azure endpoint."""
        try:
            thinking, content  = await self.llm_service.query_deepseek_azure(prompt, system_prompt)
            
            return {
                'model': 'deepseek-r1',
                'content': content,
                'thinking': thinking
            }
        except Exception as e:
            logger.error(f"Error in DeepSeek analysis: {str(e)}")
            return {
                'model': 'deepseek-r1',
                'content': str(e),
                'thinking': 'Error occurred during analysis'
            }

    async def _initialize_analysis(self, workflow_data: Dict[str, Any], context: Dict[str, Any],
                                 data_tool: Optional[DataExtractionTool] = None,
                                 llm_service: Optional[Any] = None) -> Tuple[Dict, str, Dict]:
        """Initialize analysis by setting up tools and extracting basic data."""
        data_tool = data_tool or self.data_tool
        if data_tool is None:
            raise ValueError("No data_tool provided and self.data_tool is None")
            
        llm_service = llm_service or self.llm_service
        if llm_service is None:
            raise ValueError("No llm_service provided and self.llm_service is None")

        # Extract molecule data and sample_id
        molecule_data = workflow_data.get('molecule_data', {})
        sample_id = (
            molecule_data.get('sample_id') or 
            workflow_data.get('sample_id') or 
            context.get('sample_id')
        )
        
        if not sample_id:
            raise ValueError("sample_id is required but not found in any data source")
            
        is_full_analysis = context.get('from_orchestrator', False)
        data_source = DataSource.INTERMEDIATE if is_full_analysis else DataSource.MASTER_FILE
        
        # Load data
        data = await data_tool.load_data(sample_id, data_source)
        
        return data, sample_id, molecule_data

    def _get_model_configs(self) -> Dict[str, Dict[str, str]]:
        """Get configuration for all LLM models."""
        base_system_prompt = ('You are an expert chemist specializing in structure elucidation and '
                            'spectral analysis. Analyze molecular candidates based on all available '
                            'evidence and provide detailed scientific assessments.')
        
        return {
            'claude': {'model': 'claude-3-5-sonnet', 'system': base_system_prompt},
            'deepseek': {'model': "DeepSeek-R1", 'system': base_system_prompt},
            'gemini': {'model': 'gemini-thinking', 'system': base_system_prompt},
            'o3': {'model': 'o3-mini', 'system': base_system_prompt}
        }
        #'deepseek': {'model': 'deepseek-reasoner', 'system': base_system_prompt},

    async def _get_model_analysis(self, model: str, analysis_prompt: str, 
                                model_config: Dict[str, str]) -> Tuple[Optional[str], Optional[Dict]]:
        """Get analysis from a specific model."""
        try:
            if model == 'deepseek':
                response_dict = await self.analyze_with_deepseek(analysis_prompt, model_config['system'])
                raw_response = response_dict.get('content', '')
                thinking = response_dict.get('thinking', '')
                results, reasoning = self._process_model_response(raw_response, model)
                logger.info(f"DeepSeek analysis (first 100 chars): {raw_response[:100]}")
                return raw_response, results, reasoning, thinking
            else:
                thinking =""
                raw_response = await self.llm_service.get_completion(
                    message=analysis_prompt,
                    max_tokens=2000,
                    model=model_config['model'],
                    system=model_config['system']
                )
                results, reasoning = self._process_model_response(raw_response, model)
                logger.info(f"{model.capitalize()} analysis (first 100 chars): {raw_response[:100]}")
                return raw_response, results, reasoning, thinking
        except Exception as e:
            logger.error(f"Error getting {model} analysis: {str(e)}")
            return None, None, None, None

    def _create_candidate_analysis(self, candidate: Dict[str, Any], model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Create analysis for a single candidate including all model results."""
        llm_analysis = {}
        for model in ['claude', 'deepseek', 'gemini', 'o3']:
            specific_model_results = model_results.get(f'{model}_results', {})
            model_content = specific_model_results.get('content', {})
            candidates = model_content.get('candidates', [])

            # Log candidate analysis
            logger.info(f"Analyzing candidate: {candidate.get('smiles', 'Unknown SMILES')}")
            logger.info(f"Candidate rank: {candidate.get('rank', 'Unknown')}")
            logger.info(f"Candidate confidence score: {candidate.get('confidence_score', 0.0)}")
            
            matching_candidate = next(
                (c for c in candidates if c.get('smiles') == candidate.get('smiles')), 
                {}
            )
            logger.info(f"matching_candidate: {matching_candidate}")
            logger.info(f"matching_candidate: {[c for c in candidates]}")

            llm_analysis[model] = {
                'confidence_score': matching_candidate.get('confidence_score'),
                'reasoning': matching_candidate.get('reasoning')
            }
            logger.info(f"llm_analysis: {llm_analysis}")

        # # Log LLM analysis results
        # for model, analysis in llm_analysis.items():
        #     logger.info(f"{model.capitalize()} confidence score: {analysis.get('confidence_score', 'N/A')}")
        #     logger.debug(f"{model.capitalize()} reasoning: {analysis.get('reasoning', 'N/A')[:100]}...")
            
        return {
            'smiles': candidate.get('smiles', ''),
            'rank': candidate.get('rank',"N/A"),
            'molecular_weight': candidate.get('molecular_weight', "N/A"),
            'scores': candidate.get('scores', {}),  
            'confidence_score': candidate.get('confidence_score', "N/A"),
            'reasoning': candidate.get('reasoning', ''),
            'llm_analysis': llm_analysis
        }

    def _create_final_analysis(self, sample_id: str, molecule_data: Dict[str, Any],
                             final_ranked_results: List[Dict], model_results: Dict[str, Dict],
                             analysis_prompt: str, model_configs: Dict[str, Dict]) -> Dict[str, Any]:
        """Create the final analysis dictionary with all results."""
        logger.info("-___________dsd_______________-")

        return {
            'timestamp': datetime.now().isoformat(),
            'sample_id': sample_id,
            'target_info': {
                'smiles': molecule_data.get('smiles'),
                'molecular_weight': molecule_data.get('molecular_weight')
            },
            'analyzed_candidates': [
                self._create_candidate_analysis(result, model_results)
                for result in final_ranked_results
            ],
            'llm_responses': {
                model: {
                    'raw_response': model_results.get(f'{model}_results', {}).get('raw_response'),
                    'parsed_results': model_results.get(f'{model}_results', {}).get('content'),
                    'thinking': model_results.get(f'{model}_results', {}).get('thinking'),
                    'reasoning_content': model_results.get(f'{model}_results', {}).get('reasoning_content'),
                    'analysis_prompt': model_results.get(f'{model}_results', {}).get('analysis_prompt'),
                    'config': model_configs[model]
                }
                for model in ['claude', 'deepseek', 'gemini', 'o3']
            },
            'metadata': {
                'num_candidates': len(final_ranked_results),
                'analysis_types_used': list(model_results.keys()),
                'models_used': [config['model'] for config in model_configs.values()],
                'description': 'Comprehensive final analysis of all candidates based on available evidence',
                'analysis_status': 'complete' if any(model_results.values()) else 'partial_failure',
            }
        }

    def _extract_ranked_candidates(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract and format ranked candidates from analysis results.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            List of formatted candidate results
        """
        try:
            logger.info("Starting extraction of ranked candidates")
            final_results = []
            
            # First try spectral LLM evaluation results
            spectral_llm_eval = analysis_results.get('spectral_llm_evaluation', {})
            if not spectral_llm_eval:
                logger.warning("No spectral LLM evaluation results found")
                return []
                
            logger.info(f"Found spectral LLM evaluation with keys: {list(spectral_llm_eval.keys())}")
            candidates = spectral_llm_eval.get('candidates', [])
            logger.info(f"Found {len(candidates)} candidates in spectral LLM evaluation")
            
            # Log full candidate data for debugging
            for i, candidate in enumerate(candidates):
                logger.info(f"Candidate {i} full data: {candidate}")
                logger.debug(f"Processing candidate with keys: {list(candidate.keys())}")
                smiles = candidate.get('smiles')
                if not smiles:
                    logger.warning("Candidate missing SMILES, skipping")
                    continue
                
                # Initialize molecular properties
                mol_weight = None
                formula = None
                
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        mol_weight = Descriptors.ExactMolWt(mol)
                        formula = rdMolDescriptors.CalcMolFormula(mol)
                except Exception as e:
                    logger.warning(f"Failed to calculate molecular properties for {smiles}: {str(e)}")
                
                result = {
                    'smiles': smiles,  
                    'rank': candidate.get('rank'),
                    'molecular_weight': mol_weight,
                    'formula': formula,
                    'scores': candidate.get('scores', {}),  
                    'spectral_analysis': {},
                    'confidence_score': 0.0,
                    'reasoning': candidate.get('reasoning', ''),
                    'iupac_name': candidate.get('iupac_name', 'Not available')
                }
                logger.info(f"Created result dict for candidate {i}: {result}")
                final_results.append(result)
                
            logger.info(f"Successfully extracted {len(final_results)} candidate results")
            # logger.info(f"Final results full data: {final_results}")
            return final_results
        except Exception as e:
            logger.error(f"Error extracting ranked candidates: {str(e)}")
            raise

    def _extract_candidate_reasonings(self, analysis_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Extract spectral analysis reasonings for each candidate.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            Dictionary mapping candidate IDs to their spectral reasonings
        """
        try:
            logger.info("Starting extraction of candidate reasonings")
            spectral_analysis = analysis_results.get('spectral_analysis', {})
            logger.info(f"Found spectral analysis with keys: {list(spectral_analysis.keys())}")
            
            candidate_analyses = spectral_analysis.get('candidate_analyses', {})
            candidate_reasonings = {}
            
            for analysis in candidate_analyses:
                candidate_id = analysis.get('candidate_id')
                smiles = analysis.get('smiles')
                if not candidate_id or not smiles:
                    logger.warning(f"Skipping analysis due to missing candidate_id or SMILES")
                    continue
                    
                candidate_data = candidate_reasonings[candidate_id] = {}
                logger.debug(f"Extracting reasonings for candidate {candidate_id}")
                
                spectrum_analyses = analysis.get('spectrum_analyses', {})
                candidate_data['spectral_reasonings'] = {
                    'HSQC': spectrum_analyses.get('HSQC_analysis', {}).get('reasoning', ''),
                    '1H': spectrum_analyses.get('1H_analysis', {}).get('reasoning', ''),
                    '13C': spectrum_analyses.get('13C_analysis', {}).get('reasoning', ''),
                    'COSY': spectrum_analyses.get('COSY_analysis', {}).get('reasoning', '')
                }
            
            logger.info(f"Successfully extracted reasonings for {len(candidate_reasonings)} candidates")
            return candidate_reasonings
        except Exception as e:
            logger.error(f"Error extracting candidate reasonings: {str(e)}")
            raise

    def _extract_spectral_llm_reasonings(self, analysis_results: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Extract LLM reasonings for each spectrum type.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            Dictionary mapping spectrum types to their LLM analysis
        """
        try:
            logger.info("Starting extraction of spectral LLM reasonings")
            spectral_llm_eval = analysis_results.get('spectral_llm_evaluation', {})
            logger.info(f"Found spectral LLM evaluation with keys: {list(spectral_llm_eval.keys())}")
            
            spectral_comparison = spectral_llm_eval.get('spectral_comparison', {})
            spectral_llm_reasonings = {}
            
            for spectrum_type in ['1H', '13C', 'HSQC', 'COSY']:
                spectrum_key = f"{spectrum_type}_exp"
                if spectrum_key in spectral_comparison:
                    spectrum_analyses = spectral_comparison[spectrum_key].get('analyses', [])
                    if spectrum_analyses:
                        spectral_llm_reasonings[spectrum_type] = {
                            'analysis_text': spectrum_analyses[0].get('analysis_text', ''),
                            'analysis_type': spectral_llm_eval.get('analysis_type', '')
                        }
                        logger.debug(f"Extracted LLM reasoning for {spectrum_type}")
                else:
                    logger.warning(f"No analysis found for spectrum type: {spectrum_type}")
            
            logger.info(f"Successfully extracted LLM reasonings for {len(spectral_llm_reasonings)} spectrum types")
            return spectral_llm_reasonings
        except Exception as e:
            logger.error(f"Error extracting spectral LLM reasonings: {str(e)}")
            raise

    def _generate_candidate_sections(self, final_results: List[Dict[str, Any]], 
                                   candidate_reasonings: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Generate detailed analysis sections for each candidate.
        
        Args:
            final_results: List of candidate results with scores and metadata
            candidate_reasonings: Dictionary mapping candidate IDs to their spectral reasonings
            
        Returns:
            List of formatted analysis sections for each candidate
        """
        try:
            logger.info(f"Starting generation of candidate analysis sections with {len(final_results)} results")
            logger.info(f"Final results full data: {final_results}")
            logger.info(f"Candidate reasonings full data: {candidate_reasonings}")
            
            if not final_results:
                logger.error("No final results provided")
                return []
                
            candidate_sections = []
            
            for i, result in enumerate(final_results):
                logger.info(f"Processing result {i} with data: {result}")
                
                # Get candidate reasoning using index+1 (since candidate_reasonings starts at 1)
                candidate_data = candidate_reasonings.get(i + 1, {
                    'spectral_reasonings': {'HSQC': 'No analysis available'}
                })
                logger.info(f"Using reasoning for candidate {i + 1}")
                
                # Create analysis section
                try:
                    section = f""" 
                    Candidate {result.get('rank', 'Unknown')} Analysis:
                    IUPAC Name: {result.get('iupac_name', 'Not available')}
                    SMILES: {result.get('smiles', 'Not available')}
                    Molecular Weight: {result.get('molecular_weight', 'Not available')}
                    
                    Step 1. NMR Error Analysis:
                    - HSQC Error: {result.get('scores', {}).get('HSQC', 'N/A')}
                    
                    Step 2. Individual Spectral Analyses:
                    1. HSQC Analysis:
                    {candidate_data['spectral_reasonings'].get('HSQC', 'No analysis available')}
                    """
                    candidate_sections.append(section)
                    logger.info(f"Added analysis section for candidate {result.get('rank', 'Unknown')}")
                except Exception as e:
                    logger.error(f"Error creating section for candidate {i}: {str(e)}")
                    continue
                
            logger.info(f"Generated {len(candidate_sections)} candidate analysis sections")
            return candidate_sections
        except Exception as e:
            logger.error(f"Error generating candidate sections: {str(e)}")
            raise

    def _generate_analysis_prompt(self, target_info: Dict[str, Any],
                                overall_llm_analysis: str,
                                spectral_llm_reasonings: Dict[str, Dict[str, str]],
                                candidate_sections: List[str],
                                final_ranked_results: List[Dict[str, Any]]) -> str:
        """
        Generate the complete analysis prompt for LLM processing.
        
        Args:
            target_info: Dictionary containing target molecule information
            overall_llm_analysis: Overall LLM analysis text
            spectral_llm_reasonings: Dictionary mapping spectrum types to their LLM analysis
            candidate_sections: List of formatted candidate analysis sections
            final_ranked_results: List of candidate results with scores and metadata
            
        Returns:
            Formatted analysis prompt string
        """
        try:
            logger.info("Generating complete analysis prompt")
            prompt = f"""
            You are tasked with making a final determination of the most likely correct molecular structure based on all available spectral and analytical evidence. Your analysis must be extremely thorough and systematic.
            
            Target Molecule Information:
            - Target Molecular Weight: {target_info.get('molecular_weight', 'Not available')}
            - Target Formula: {target_info.get('formula', 'Not available')}

            Candidate Information:
            {'\n'.join([f"Candidate {i+1}:"
                       f"\n- SMILES: {cand.get('smiles', 'Not available')}"
                       f"\n- Molecular Weight: {cand.get('molecular_weight', 'Not available')}"
                       f"\n- Formula: {cand.get('formula', 'Not available')}"
                       for i, cand in enumerate(final_ranked_results)])}
        
            Overall Spectral Analyses:
            1. HSQC Overall Analysis:
            {spectral_llm_reasonings.get('HSQC', {}).get('analysis_text', 'No analysis available')}
            Detailed Candidate Analyses:
            {'\n\n'.join(candidate_sections)}
            
            IMPORTANT: Provide a thorough analysis for EACH candidate structure in the processed list, followed by a clear final recommendation.
            Your response must end with a JSON result in the exact format shown below.
            Do not include any text after the JSON.
            For each candidate structure:
            1. Analyze all available spectral data
            2. Compare predicted vs experimental NMR shifts
            3. Evaluate structural features and their compatibility with data
            4. Consider molecular weight and other physical properties
            5. Assess data quality and potential issues
            
            Then synthesize all analyses to select the best candidate.
            
            CRITICAL: The end of your response MUST follow this EXACT JSON structure:

            JSON_RESULT = {{
                "candidates": [
                    {{
                        "smiles": "<SMILES string of this specific candidate>",
                        "confidence_score": <float between 0-1>,
                        "molecular_weight": <float>,
                        "reasoning": "Thorough evidence-based analysis for THIS SPECIFIC candidate addressing:
                                    - Detailed spectral analysis results
                                    - NMR shift comparisons and deviations
                                    - Structural feature evaluation
                                    - Molecular property matches/mismatches
                                    - Supporting and contradicting evidence
                                    Explain each point with specific data references.",
                        "data_quality_issues": {{
                            "title": "Brief description of quality concerns for this candidate",
                            "description": "Detailed explanation of ALL identified issues",
                            "impact": "high/medium/low",
                            "atom_index": <int between 0-50>
                        }}
                    }},
                    {{
                        "smiles": "<SMILES string of another candidate>",
                        "confidence_score": <float between 0-1>,
                        "molecular_weight": <float>,
                        "reasoning": "Thorough evidence-based analysis for THIS SPECIFIC candidate addressing:
                                    - Detailed spectral analysis results
                                    - NMR shift comparisons and deviations
                                    - Structural feature evaluation
                                    - Molecular property matches/mismatches
                                    - Supporting and contradicting evidence
                                    Explain each point with specific data references.",
                        "data_quality_issues": {{
                            "title": "Brief description of quality concerns for this candidate",
                            "description": "Detailed explanation of ALL identified issues",
                            "impact": "high/medium/low",
                            "atom_index": <int between 0-50>
                        }}
                    }}
                ],
                "final_recommendation": {{
                    "best_smiles": "<SMILES of the winning candidate>",
                    "overall_confidence": <float between 0-1>,
                    "molecular_weight_match": <boolean>,
                    "explanation": "Comprehensive justification for selecting this candidate:
                                  - Compare and contrast with other candidates
                                  - Highlight decisive factors in selection
                                  - Address any contradictions or uncertainties
                                  - Explain confidence level assessment
                                  - Discuss any remaining concerns"
                }}
            }}"""
                                    
            # 2. COSY Overall Analysis:
            # {spectral_llm_reasonings.get('COSY', {}).get('analysis_text', 'No analysis available')}
            
            # 3. 1H NMR Overall Analysis:
            # {spectral_llm_reasonings.get('1H', {}).get('analysis_text', 'No analysis available')}
            
            # 4. 13C NMR Overall Analysis:
            # {spectral_llm_reasonings.get('13C', {}).get('analysis_text', 'No analysis available')}

            logger.info("Successfully generated analysis prompt")
            return prompt
        except Exception as e:
            logger.error(f"Error generating analysis prompt: {str(e)}")
            raise

    def _process_model_response(self, raw_response: str, model_type: str) -> Tuple[Dict[str, Any], str]:
        """Process raw model response and extract both JSON result and reasoning."""
        try:
            result = self._extract_model_json_result(raw_response, model_type)
            return result['json_content'], result['reasoning_content']
        except Exception as e:
            logger.error(f"Error processing model response: {str(e)}")
            return {}, ""

    # def _extract_model_json_result(self, raw_text: str, model_type: str) -> Dict[str, Any]:
    #     """
    #     Extract JSON result from different model outputs based on their specific formats.

    #     Returns:
    #         Dictionary containing:
    #             - json_content: The parsed JSON content (or raw JSON substring if parsing fails)
    #             - reasoning_content: The reasoning text preceding the JSON
    #     """
    #     try:
    #         if not raw_text:
    #             logger.warning(f"Empty response from {model_type} model")
    #             return {'json_content': {}, 'reasoning_content': ''}

    #         if model_type == 'gemini':
    #             # Extract content between ```json and ```
    #             json_marker = "```json"
    #             reasoning_content = ''
    #             if json_marker in raw_text:
    #                 json_start = raw_text.find(json_marker)
    #                 reasoning_content = raw_text[:json_start].strip()
    #                 json_start += len(json_marker)
    #                 json_end = raw_text.find("```", json_start)
    #                 if json_end != -1:
    #                     json_content = raw_text[json_start:json_end].strip()
    #                     try:
    #                         return {
    #                             'json_content': json_module.loads(json_content),
    #                             'reasoning_content': reasoning_content
    #                         }
    #                     except (ValueError, SyntaxError) as e:
    #                         logger.warning(f"Failed to parse {model_type} JSON: {e}")
    #                         return {
    #                             'json_content': json_content,
    #                             'reasoning_content': reasoning_content
    #                         }
    #             return {'json_content': raw_text.strip(), 'reasoning_content': ''}

    #         elif model_type in ['o3', 'claude']:
    #             marker = "JSON_RESULT ="
    #             marker_index = raw_text.find(marker)
    #             if marker_index != -1:
    #                 # Everything before the marker is the reasoning content
    #                 reasoning_content = raw_text[:marker_index].strip()
    #                 remaining_text = raw_text[marker_index + len(marker):].strip()

    #                 # Find the beginning of the JSON object
    #                 start_brace_index = remaining_text.find("{")
    #                 if start_brace_index == -1:
    #                     logger.warning(f"No JSON object found in {model_type} response")
    #                     return {"reasoning_content": reasoning_content, "json_content": None}

    #                 # Use a counter to capture the complete JSON block (support nested braces)
    #                 brace_count = 0
    #                 end_index = None
    #                 for i, char in enumerate(remaining_text[start_brace_index:]):
    #                     if char == '{':
    #                         brace_count += 1
    #                     elif char == '}':
    #                         brace_count -= 1
    #                         if brace_count == 0:
    #                             end_index = start_brace_index + i + 1
    #                             break

    #                 if end_index is None:
    #                     logger.warning(f"No matching closing brace found in {model_type} response")
    #                     json_str = remaining_text[start_brace_index:]
    #                 else:
    #                     json_str = remaining_text[start_brace_index:end_index]

    #                 # Remove invalid control characters and clean up JSON string
    #                 json_str_clean = re.sub(r'[\x00-\x1f]+', " ", json_str)
                    
    #                 # First try to parse with json.loads after converting JSON literals
    #                 json_str_clean = (json_str_clean
    #                     .replace("True", "true")
    #                     .replace("False", "false")
    #                     .replace("None", "null")
    #                     .replace("'", '"'))
                    
    #                 try:
    #                     parsed_json = json.loads(json_str_clean)
    #                     logger.debug(f"Successfully parsed {model_type} JSON using json.loads")
    #                     return {
    #                         'json_content': parsed_json,
    #                         'reasoning_content': reasoning_content
    #                     }
    #                 except json.JSONDecodeError as e:
    #                     logger.warning(f"json.loads failed: {e}")
    #                     # Fallback: try ast.literal_eval with Python literals
    #                     python_literal = (json_str_clean
    #                         .replace("true", "True")
    #                         .replace("false", "False")
    #                         .replace("null", "None"))
    #                     try:
    #                         parsed_json = ast.literal_eval(python_literal)
    #                         logger.debug(f"Successfully parsed {model_type} JSON using ast.literal_eval")
    #                         return {
    #                             'json_content': parsed_json,
    #                             'reasoning_content': reasoning_content
    #                         }
    #                     except Exception as e2:
    #                         logger.warning(f"ast.literal_eval failed: {e2}")
    #                         return {
    #                             'json_content': json_str_clean,
    #                             'reasoning_content': reasoning_content
    #                         }
    #             else:
    #                 # Fallback: try a case-insensitive search for "json"
    #                 lower_text = raw_text.lower()
    #                 fallback_marker = "json"
    #                 fallback_index = lower_text.find(fallback_marker)
    #                 if fallback_index != -1:
    #                     reasoning_content = raw_text[:fallback_index].strip()
    #                     json_content = raw_text[fallback_index:].strip()
    #                     return {"reasoning_content": reasoning_content, "json_content": json_content}
    #                 else:
    #                     # If no marker is found, treat the entire text as reasoning
    #                     return {"reasoning_content": raw_text.strip(), "json_content": None}
    #         elif model_type == 'deepseek':
    #             try:
    #                 return {
    #                     'json_content': json_module.loads(raw_text),
    #                     'reasoning_content': ''
    #                 }
    #             except (ValueError, SyntaxError):
    #                 json_start = raw_text.find('{')
    #                 json_end = raw_text.rfind('}') + 1
    #                 if json_start != -1 and json_end > json_start:
    #                     reasoning_content = raw_text[:json_start].strip()
    #                     try:
    #                         return {
    #                             'json_content': json_module.loads(raw_text[json_start:json_end]),
    #                             'reasoning_content': reasoning_content
    #                         }
    #                     except (ValueError, SyntaxError) as e:
    #                         logger.warning(f"Failed to parse {model_type} JSON: {e}")
    #                         return {
    #                             'json_content': raw_text,
    #                             'reasoning_content': raw_text.strip()
    #                         }
    #                 return {'json_content': {}, 'reasoning_content': raw_text.strip()}

    #         logger.warning(f"No valid JSON format found for {model_type} model")
    #         return {'json_content': {}, 'reasoning_content': raw_text.strip()}

    #     except Exception as e:
    #         logger.warning(f"Failed to extract JSON for {model_type} model: {e}")
    #         return {'json_content': {}, 'reasoning_content': raw_text.strip()}

    def _extract_model_json_result(self, raw_text: str, model_type: str) -> Dict[str, Any]:
        """
        Extract JSON result from different model outputs based on their specific formats.
        
        Args:
            raw_text: Raw text output from the model
            model_type: Type of model ('gemini', 'claude', 'o3', 'deepseek')
            
        Returns:
            Dictionary containing:
                - json_content: The parsed JSON content (or raw JSON substring if parsing fails)
                - reasoning_content: The reasoning text preceding the JSON
        """
        if not raw_text:
            logger.warning(f"Empty response from {model_type} model")
            return {'json_content': {}, 'reasoning_content': ''}

        def clean_json_string(json_str: str) -> str:
            """Helper to clean and normalize JSON string."""
            # Remove invalid control characters
            cleaned = re.sub(r'[\x00-\x1f]+', " ", json_str)
            
            # Normalize boolean and null values
            cleaned = (cleaned
                    .replace("True", "true")
                    .replace("False", "false")
                    .replace("None", "null"))
                    
            # Remove any trailing commas before closing braces/brackets
            cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
            
            return cleaned.strip()

        def find_json_boundaries(text: str) -> Tuple[int, int]:
            """Find the start and end indices of the outermost JSON object."""
            start = text.find('{')
            if start == -1:
                return -1, -1
                
            brace_count = 0
            end = -1
            
            for i, char in enumerate(text[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
                        
            return start, end

        try:
            if model_type == 'gemini':
                json_marker = "```json"
                if json_marker in raw_text:
                    json_start = raw_text.find(json_marker) + len(json_marker)
                    json_end = raw_text.find("```", json_start)
                    
                    if json_end != -1:
                        reasoning_content = raw_text[:raw_text.find(json_marker)].strip()
                        json_str = raw_text[json_start:json_end].strip()
                        
                        try:
                            return {
                                'json_content': json.loads(clean_json_string(json_str)),
                                'reasoning_content': reasoning_content
                            }
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse Gemini JSON: {e}")
                            return {
                                'json_content': json_str,
                                'reasoning_content': reasoning_content
                            }
                            
                # Try to find JSON without markers as fallback
                start, end = find_json_boundaries(raw_text)
                if start != -1 and end != -1:
                    return {
                        'json_content': raw_text[start:end],
                        'reasoning_content': raw_text[:start].strip()
                    }
                    
                return {'json_content': raw_text.strip(), 'reasoning_content': ''}

            elif model_type in ['o3', 'claude']:
                marker = "JSON_RESULT ="
                marker_index = raw_text.find(marker)
                
                if marker_index == -1:
                    # Try case-insensitive "json" as fallback
                    marker_index = raw_text.lower().find("json")
                    if marker_index != -1:
                        marker_len = 4  # len("json")
                    else:
                        # No marker found, look for raw JSON
                        start, end = find_json_boundaries(raw_text)
                        if start != -1 and end != -1:
                            return {
                                'json_content': raw_text[start:end],
                                'reasoning_content': raw_text[:start].strip()
                            }
                        return {'json_content': {}, 'reasoning_content': raw_text.strip()}
                else:
                    marker_len = len(marker)

                reasoning_content = raw_text[:marker_index].strip()
                json_text = raw_text[marker_index + marker_len:].strip()
                
                # Find and extract the JSON object
                start, end = find_json_boundaries(json_text)
                if start != -1 and end != -1:
                    json_str = clean_json_string(json_text[start:end])
                    
                    # Try parsing with json.loads
                    try:
                        return {
                            'json_content': json.loads(json_str),
                            'reasoning_content': reasoning_content
                        }
                    except json.JSONDecodeError:
                        # Try ast.literal_eval as fallback
                        try:
                            python_str = (json_str
                                .replace("true", "True")
                                .replace("false", "False")
                                .replace("null", "None"))
                            return {
                                'json_content': ast.literal_eval(python_str),
                                'reasoning_content': reasoning_content
                            }
                        except (ValueError, SyntaxError) as e:
                            logger.warning(f"Both parsing methods failed: {e}")
                            return {
                                'json_content': json_str,
                                'reasoning_content': reasoning_content
                            }
                
                return {'json_content': json_text, 'reasoning_content': reasoning_content}

            elif model_type == 'deepseek':
                try:
                    return {
                        'json_content': json.loads(clean_json_string(raw_text)),
                        'reasoning_content': ''
                    }
                except json.JSONDecodeError:
                    start, end = find_json_boundaries(raw_text)
                    if start != -1 and end != -1:
                        json_str = clean_json_string(raw_text[start:end])
                        try:
                            return {
                                'json_content': json.loads(json_str),
                                'reasoning_content': raw_text[:start].strip()
                            }
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse Deepseek JSON: {e}")
                            return {
                                'json_content': raw_text[start:end],
                                'reasoning_content': raw_text[:start].strip()
                            }
                    return {'json_content': {}, 'reasoning_content': raw_text.strip()}

            logger.warning(f"Unsupported model type: {model_type}")
            return {'json_content': {}, 'reasoning_content': raw_text.strip()}

        except Exception as e:
            logger.error(f"Unexpected error extracting JSON for {model_type} model: {e}")
            return {'json_content': {}, 'reasoning_content': raw_text.strip()}

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/memory/conversation.py ---
"""
Conversation history management system.
"""

class ConversationMemory:
    """Manages conversation history and context."""
    pass


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/memory/knowledge_base.py ---
"""
Knowledge base for storing and retrieving analysis results and workflow data.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import json
import sqlite3
import logging


class KnowledgeBase:
    """
    Persistent storage for analysis results and workflow data.
    Implements a SQLite-based storage with JSON serialization for complex data.
    """
    
    def __init__(self):
        """Initialize knowledge base with SQLite storage."""
        # Set up storage directory in temp/memory
        self.base_dir = Path(__file__).parent.parent.parent / 'temp' / 'memory' / 'knowledge'
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = self.base_dir / 'knowledge.db'
        self._initialize_db()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self._initialize_logging()
    
    def _initialize_logging(self):
        """Initialize logging system."""
        self.logger.setLevel(logging.INFO)
        
        # Create log directory
        log_dir = self.base_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'knowledge_base_{timestamp}.log'
        
        # Add file handler
        file_handler = logging.FileHandler(str(log_file))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def _initialize_db(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Create results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    workflow_type TEXT NOT NULL,
                    step_type TEXT NOT NULL,
                    data JSON NOT NULL,
                    confidence REAL,
                    metadata JSON
                )
            """)
            
            # Create workflow_context table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_context (
                    workflow_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    workflow_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    context JSON NOT NULL,
                    metadata JSON
                )
            """)
            
            # Create index on workflow_id and timestamp
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_results_workflow 
                ON results(workflow_id, timestamp)
            """)
            
            conn.commit()
    
    def add_results(self, workflow_id: str, workflow_type: str, 
                   step_type: str, data: Dict, confidence: float = None,
                   metadata: Dict = None) -> int:
        """
        Add analysis results to the knowledge base.
        
        Args:
            workflow_id: ID of the workflow that generated the results
            workflow_type: Type of workflow (e.g., STARTING_MATERIAL)
            step_type: Type of step that generated the results
            data: Results data (will be JSON serialized)
            confidence: Optional confidence score for the results
            metadata: Optional metadata about the results
            
        Returns:
            ID of the inserted result record
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO results 
                    (timestamp, workflow_id, workflow_type, step_type, data, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    workflow_id,
                    workflow_type,
                    step_type,
                    json.dumps(data),
                    confidence,
                    json.dumps(metadata) if metadata else None
                ))
                
                result_id = cursor.lastrowid
                conn.commit()
                
                self.logger.info(f"Added results for workflow {workflow_id}, step {step_type}")
                return result_id
                
        except Exception as e:
            self.logger.error(f"Failed to add results: {str(e)}")
            raise
    
    def update_workflow_context(self, workflow_id: str, workflow_type: str,
                              status: str, context: Dict, metadata: Dict = None):
        """
        Update workflow context in the knowledge base.
        
        Args:
            workflow_id: ID of the workflow
            workflow_type: Type of workflow
            status: Current workflow status
            context: Current workflow context
            metadata: Optional workflow metadata
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO workflow_context
                    (workflow_id, timestamp, workflow_type, status, context, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    workflow_id,
                    datetime.now().isoformat(),
                    workflow_type,
                    status,
                    json.dumps(context),
                    json.dumps(metadata) if metadata else None
                ))
                
                conn.commit()
                
                self.logger.info(f"Updated context for workflow {workflow_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to update workflow context: {str(e)}")
            raise
    
    def get_workflow_results(self, workflow_id: str) -> List[Dict]:
        """Get all results for a specific workflow."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM results 
                    WHERE workflow_id = ?
                    ORDER BY timestamp ASC
                """, (workflow_id,))
                
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    result['data'] = json.loads(result['data'])
                    if result['metadata']:
                        result['metadata'] = json.loads(result['metadata'])
                    results.append(result)
                
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to get workflow results: {str(e)}")
            raise
    
    def get_workflow_context(self, workflow_id: str) -> Optional[Dict]:
        """Get current context for a specific workflow."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM workflow_context 
                    WHERE workflow_id = ?
                """, (workflow_id,))
                
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    result['context'] = json.loads(result['context'])
                    if result['metadata']:
                        result['metadata'] = json.loads(result['metadata'])
                    return result
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get workflow context: {str(e)}")
            raise
    
    def get_similar_results(self, workflow_type: str, data_pattern: Dict,
                          min_confidence: float = 0.0) -> List[Dict]:
        """
        Find similar results based on workflow type and data pattern.
        
        Args:
            workflow_type: Type of workflow to search
            data_pattern: Dictionary of key-value pairs to match in results
            min_confidence: Minimum confidence score for results
            
        Returns:
            List of matching result records
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Convert pattern to JSON string fragments for LIKE matching
                patterns = []
                for key, value in data_pattern.items():
                    patterns.append(f'"%{key}": "{value}%"')
                
                # Build query with pattern matching
                query = """
                    SELECT * FROM results 
                    WHERE workflow_type = ?
                    AND confidence >= ?
                """
                params = [workflow_type, min_confidence]
                
                for pattern in patterns:
                    query += f" AND data LIKE ?"
                    params.append(pattern)
                
                cursor.execute(query, params)
                
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    result['data'] = json.loads(result['data'])
                    if result['metadata']:
                        result['metadata'] = json.loads(result['metadata'])
                    results.append(result)
                
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to get similar results: {str(e)}")
            raise
    
    def size(self) -> Dict[str, int]:
        """Get current size of the knowledge base."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Get counts from both tables
                cursor.execute("SELECT COUNT(*) FROM results")
                results_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM workflow_context")
                context_count = cursor.fetchone()[0]
                
                return {
                    'results': results_count,
                    'contexts': context_count
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get knowledge base size: {str(e)}")
            raise


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/memory/__init__.py ---



--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/_agents_descriptions_/MMST_imports.py ---

# Standard library imports
import json
import os
import random
import glob
import pickle
import sys
import os
from argparse import Namespace
from collections import defaultdict
from types import SimpleNamespace

# Third-party imports
## Data processing and scientific computing
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.autonotebook import tqdm

## Machine learning and data visualization
import matplotlib.pyplot as plt
#import seaborn as sns
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import train_test_split

## PyTorch and related libraries
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

## RDKit for cheminformatics
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, MolFromSmiles, MolToSmiles, Descriptors
from rdkit.Chem.Descriptors import MolWt

## Miscellaneous
from IPython.display import HTML, SVG

# Local imports
#sys.path.append('/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer')
# Dynamically add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# Chemprop imports
#sys.path.append("/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/chemprop-IR")
chemprop_ir_path = os.path.join(project_root, 'chemprop_IR')

if chemprop_ir_path not in sys.path:
    sys.path.append(chemprop_ir_path)

import utils_MMT.MT_functions_v15_4 as mtf
import utils_MMT.run_batch_gen_val_MMT_v15_4 as rbgvm
import utils_MMT.clustering_visualization_v15_4 as cv
import utils_MMT.plotting_v15_4 as pt
import utils_MMT.execution_function_v15_4 as ex
import utils_MMT.ir_simulation_v15_4 as irs
import utils_MMT.helper_functions_pl_v15_4 as hf
import utils_MMT.mmt_result_test_functions_15_4 as mrtf
import utils_MMT.data_generation_v15_4 as dl


from chemprop.train import make_predictions
from chemprop.parsing import modify_predict_args


# Setting up environment
torch.cuda.device_count()



def parse_arguments(hyperparameters):
    parsed_args = {key: val[0] if isinstance(val, (list, tuple)) else val for key, val in hyperparameters.items()}
    return Namespace(**parsed_args)


# def load_json_dics():
#     with open('/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/itos.json', 'r') as f:
#         itos = json.load(f)
#     with open('/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/stoi.json', 'r') as f:
#         stoi = json.load(f)
 
#     with open('/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/stoi_MF.json', 'r') as f:
#         stoi_MF = json.load(f)
#     with open('/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/itos_MF.json', 'r') as f:
#         itos_MF = json.load(f)    
#     return itos, stoi, stoi_MF, itos_MF
# itos, stoi, stoi_MF, itos_MF = load_json_dics()
# rand_num = str(random.randint(1, 10000000))
 


def load_json_dics():
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    
    # Build paths relative to the script's location
    base_path = os.path.abspath(os.path.join(script_dir, '../..'))
    
    itos_path = os.path.join(base_path, 'itos.json')
    stoi_path = os.path.join(base_path, 'stoi.json')
    stoi_MF_path = os.path.join(base_path, 'stoi_MF.json')
    itos_MF_path = os.path.join(base_path, 'itos_MF.json')

    # Load JSON files
    with open(itos_path, 'r') as f:
        itos = json.load(f)
    with open(stoi_path, 'r') as f:
        stoi = json.load(f)
    with open(stoi_MF_path, 'r') as f:
        stoi_MF = json.load(f)
    with open(itos_MF_path, 'r') as f:
        itos_MF = json.load(f)
    
    return itos, stoi, stoi_MF, itos_MF

# Example usage
itos, stoi, stoi_MF, itos_MF = load_json_dics()
rand_num = str(random.randint(1, 10000000))

 
def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f)
 
def load_config(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None    

 
def save_updated_config(config, path):
    config_dict = vars(config)
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=4)
 
 
# def load_configs():
#     # Load IR config
#     ir_config_path = '/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/_ISAK/Runfolder/ir_config_V8.json'
#     IR_config_dict = load_config(ir_config_path)
#     if IR_config_dict is None:
#         raise FileNotFoundError(f"IR config file not found at {ir_config_path}")
#     IR_config = parse_arguments(IR_config_dict)
#     modify_predict_args(IR_config)
    
#     # Load main config
#     config_path = '/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/_ISAK/Runfolder/config_V8.json'
#     config_dict = load_config(config_path)
#     if config_dict is None:
#         raise FileNotFoundError(f"Main config file not found at {config_path}")
#     config = parse_arguments(config_dict)
 
#     return IR_config, config

def load_configs():
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    
    # Build paths relative to the script's location
    base_path = os.path.abspath(os.path.join(script_dir, ''))
    
    ir_config_path = os.path.join(base_path, 'ir_config_V8.json')
    config_path = os.path.join(base_path, 'config_V8.json')
    
    # Load IR config
    IR_config_dict = load_config(ir_config_path)
    if IR_config_dict is None:
        raise FileNotFoundError(f"IR config file not found at {ir_config_path}")
    IR_config = parse_arguments(IR_config_dict)
    modify_predict_args(IR_config)
    
    # Load main config
    config_dict = load_config(config_path)
    if config_dict is None:
        raise FileNotFoundError(f"Main config file not found at {config_path}")
    config = parse_arguments(config_dict)
 
    return IR_config, config

# IR_config, config = load_config()
# config = parse_arguments(config)
# rand_num = random.randint(0, 1000000)
# itos, stoi, stoi_MF, itos_MF = load_json_dics()

def plot_first_smiles(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    # Extract the first SMILES string
    first_smiles = df['SMILES'].iloc[0]
    # Generate the molecule from SMILES
    mol = Chem.MolFromSmiles(first_smiles)
    # Draw the molecule
    img = Draw.MolToImage(mol)
    # Display the image
    img.show()

def sim_and_display():
    print("sim_and_display")
    #import IPython; IPython.embed();

    itos, stoi, stoi_MF, itos_MF = load_json_dics()
    IR_config, config = load_configs()
    config.csv_SMI_targets = config.SGNN_csv_gen_smi #smi_file_path
    #config.SGNN_csv_gen_smi =  config["SGNN_csv_gen_smi"] #smi_file_path
    config = ex.clean_dataset(config)
    print("\033[1m\033[31mThis is: simulate_syn_data\033[0m")
    config = ex.gen_sim_aug_data(config, IR_config) 

    config.csv_1H_path_display = config.csv_1H_path_SGNN
    config.csv_13C_path_display = config.csv_13C_path_SGNN
    config.csv_HSQC_path_display = config.csv_HSQC_path_SGNN
    config.csv_COSY_path_display = config.csv_COSY_path_SGNN
    config.IR_data_folder_display = config.IR_data_folder
    ##########################################################
    ### this is where you can get the spectra for plotting ###
    ##########################################################
    #plot_first_smiles(config.csv_1H_path_SGNN)
    save_updated_config(config, config.config_path)
    return config

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/_agents_descriptions_/IC_MMST.py ---


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/orchestrator/__init__.py ---
"""
Orchestration Agent module for managing structure elucidation workflows.
"""

from .orchestrator import OrchestrationAgent

__all__ = ['OrchestrationAgent']


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/orchestrator/workflow_definitions.py ---
"""
Workflow definitions for structure elucidation.
"""

from typing import Dict, List, Optional, NamedTuple
from enum import Enum

class WorkflowType(Enum):
    MULTIPLE_TARGETS = "multiple_targets"
    STARTING_MATERIAL = "starting_material"
    TARGET_ONLY = "target_only"
    SPECTRAL_ONLY = "spectral_only"

class WorkflowStep(NamedTuple):
    """Represents a single step in a workflow"""
    keyword: str  # Unique identifier for validation
    command: str  # Command to execute
    description: str  # Human-readable description
    requires: List[List[str]]  # List of requirement groups, where each group represents OR conditions

# Define workflow steps
WORKFLOW_STEPS = {
    'threshold_calculation': WorkflowStep(
        keyword='threshold_calculation',
        command='Calculate dynamic thresholds for spectral data analysis',
        description='Calculate spectral analysis thresholds',
        requires=[]
    ),
    'retrosynthesis': WorkflowStep(
        keyword='retrosynthesis',
        command='Run retrosynthesis analysis on target structure',
        description='Perform retrosynthesis analysis',
        requires=[]
    ),
    'mol2mol': WorkflowStep(
        keyword='mol2mol',
        command='Run mol2mol to generate similar molecule analogs',
        description='Generate similar molecule analogs to the target molecule',
        requires=[]
    ),
    'nmr_simulation': WorkflowStep(
        keyword='nmr_simulation',
        command='Calculate simulated NMRs for the target structure',
        description='Generate simulated NMR spectra',
        requires=[]
    ),
    'peak_matching': WorkflowStep(
        keyword='peak_matching',
        command='Perform direct peak matching between target structure and experimental HSQC, COSY, 13C, and 1H NMR data',
        description='Match peaks between simulated and experimental spectra',
        requires=[['nmr_simulation']]
    ),
    'forward_prediction': WorkflowStep(
        keyword='forward_prediction',
        command='Run forward prediction on retrosynthesis products',
        description='Predict products from retrosynthesis',
        requires=[]
    ),
    'forward_candidate_analysis': WorkflowStep(
        keyword='forward_candidate_analysis',
        command='Analyze and score candidate molecules specifically from forward synthesis predictions using NMR data matching',
        description='Analyze candidates from forward synthesis prediction',
        requires=[['forward_prediction']]
    ),
    'mol2mol_candidate_analysis': WorkflowStep(
        keyword='mol2mol_candidate_analysis',
        command='Analyze and score candidate molecules specifically from mol2mol analogues using NMR data matching',
        description='Analyze candidates from mol2mol analogues',
        requires=[['mol2mol']]
    ),
    'mmst_candidate_analysis': WorkflowStep(
        keyword='mmst_candidate_analysis',
        command='Analyze and score candidate molecules specifically from MMST predictions using NMR data matching',
        description='Analyze candidates from MMST prediction',
        requires=[['mmst']]
    ),
    'candidate_analysis': WorkflowStep(
        keyword='candidate_analysis',
        command='Analyze and score candidate molecules from all prediction sources',
        description='Analyze candidates using NMR data',
        requires=[['forward_candidate_analysis'], ['mol2mol_candidate_analysis'], ['mmst_candidate_analysis']]
    ),
    'visual_comparison': WorkflowStep(
        keyword='visual_comparison',
        command='Generate visual comparison for best matching structures',
        description='Create visual comparisons',
        requires=[['threshold_calculation'], ['peak_matching'], ['candidate_analysis']]
    ),
    'mmst': WorkflowStep(
        keyword='mmst',
        command='Run MMST to predict molecular structure from NMR data',
        description='Predict molecular structure using MMST',
        requires=[['threshold_calculation']]
    ),
    'analysis': WorkflowStep(
        keyword='analysis',
        command='Perform comprehensive analysis of molecular data and generate interpretable results',
        description='Analyze molecular data using LLM and specialized tools',
        requires=[['candidate_analysis'], ['visual_comparison']]
    )
}

# Define workflow sequences
WORKFLOW_SEQUENCES = {
    WorkflowType.TARGET_ONLY: [
        # WORKFLOW_STEPS['threshold_calculation'],
        WORKFLOW_STEPS['retrosynthesis'],
        WORKFLOW_STEPS['nmr_simulation'],
        WORKFLOW_STEPS['peak_matching'],
        WORKFLOW_STEPS['forward_prediction'],
        WORKFLOW_STEPS['forward_candidate_analysis'],
        WORKFLOW_STEPS['mol2mol'],
        WORKFLOW_STEPS['mol2mol_candidate_analysis'],
        WORKFLOW_STEPS['mmst'],
        WORKFLOW_STEPS['mmst_candidate_analysis'],
        # WORKFLOW_STEPS['analysis']
    ],
    WorkflowType.STARTING_MATERIAL: [
        # WORKFLOW_STEPS['threshold_calculation'],
        # WORKFLOW_STEPS['nmr_simulation'],
        WORKFLOW_STEPS['forward_prediction'], # need to add different logic for experimental data comparison?
        WORKFLOW_STEPS['forward_candidate_analysis'],
        WORKFLOW_STEPS['mol2mol'],
        WORKFLOW_STEPS['mol2mol_candidate_analysis'],
        WORKFLOW_STEPS['mmst'],
        WORKFLOW_STEPS['mmst_candidate_analysis'],
        WORKFLOW_STEPS['candidate_analysis'],
        WORKFLOW_STEPS['analysis']
    ],
    WorkflowType.MULTIPLE_TARGETS: [
        WORKFLOW_STEPS['threshold_calculation'],
        WORKFLOW_STEPS['retrosynthesis'],
        WORKFLOW_STEPS['nmr_simulation'],
        WORKFLOW_STEPS['peak_matching'],
        WORKFLOW_STEPS['forward_prediction'],
        WORKFLOW_STEPS['forward_candidate_analysis'],
        WORKFLOW_STEPS['mol2mol'],
        WORKFLOW_STEPS['mol2mol_candidate_analysis'],
        WORKFLOW_STEPS['mmst'],
        WORKFLOW_STEPS['mmst_candidate_analysis'],
        WORKFLOW_STEPS['candidate_analysis'],
        WORKFLOW_STEPS['analysis']
    ],
    WorkflowType.SPECTRAL_ONLY: [
        WORKFLOW_STEPS['threshold_calculation'],
        WORKFLOW_STEPS['peak_matching'],
        WORKFLOW_STEPS['analysis']
    ]
}

def determine_workflow_type(data: Dict) -> WorkflowType:
    """Determine which workflow to use based on input data columns.
    
    The workflow type is determined by the presence of specific fields:
    - SMILES_list with multiple entries -> MULTIPLE_TARGETS
    - starting_material with non-empty SMILES -> STARTING_MATERIAL
    - SMILES present -> TARGET_ONLY
    - None of the above -> SPECTRAL_ONLY
    
    Note: For starting material workflow, the starting_smiles must contain actual SMILES data,
    not just an empty list or placeholder.
    
    Args:
        data: Dictionary containing either molecule_data directly or wrapped in molecule_data key
    """
    # Extract molecule_data if it exists, otherwise use data directly
    molecule_data = data.get('molecule_data', data)
    
    # Check if we have a SMILES field directly in the data
    has_smiles = ('SMILES' in molecule_data) or ('smiles' in molecule_data)
    
    # Check for SMILES_list first (multiple targets)
    if 'SMILES_list' in molecule_data and isinstance(molecule_data['SMILES_list'], list) and len(molecule_data['SMILES_list']) > 1:
        return WorkflowType.MULTIPLE_TARGETS
    
    # Check for starting material with actual content
    starting_smiles_key = next((key for key in molecule_data.keys() if key.lower().startswith('starting_smiles')), None)
    if starting_smiles_key:
        starting_smiles = molecule_data[starting_smiles_key]
        # Check if starting_smiles contains actual data
        if isinstance(starting_smiles, str) and starting_smiles.strip():
            return WorkflowType.STARTING_MATERIAL
        elif isinstance(starting_smiles, list) and any(isinstance(s, str) and s.strip() for s in starting_smiles):
            return WorkflowType.STARTING_MATERIAL
        elif isinstance(starting_smiles, dict) and any(isinstance(s, str) and s.strip() for s in starting_smiles.values()):
            return WorkflowType.STARTING_MATERIAL
    
    # Check for SMILES (single target)
    if has_smiles:
        return WorkflowType.TARGET_ONLY
        
    # Default to spectral only if no structure information is available
    return WorkflowType.SPECTRAL_ONLY

def get_workflow_steps(workflow_type: WorkflowType) -> List[WorkflowStep]:
    """Get the sequence of workflow steps for a workflow type."""
    return WORKFLOW_SEQUENCES.get(workflow_type, [])


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/orchestrator/orchestrator_backup.py ---
"""
Orchestration Agent for managing structure elucidation workflows.

This module implements the high-level orchestration logic for analyzing chemical structure data
and coordinating analysis workflows through the Coordinator Agent using LLM-generated commands.
"""
 
import logging
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import traceback
import uuid

from services.llm_service import LLMService
from ..coordinator.coordinator import CoordinatorAgent
from ..base.base_agent import BaseAgent
from .workflow_definitions import determine_workflow_type, get_workflow_steps, WorkflowType, WorkflowStep

class OrchestrationAgent(BaseAgent):
    """Agent responsible for orchestrating the structure elucidation workflow."""

    def __init__(self, llm_service: LLMService, coordinator=None):
        """Initialize orchestrator with required services."""
        capabilities = [
            "Workflow generation",
            "Process coordination",
            "Tool execution",
            "Error handling"
        ]
        super().__init__("Orchestration Agent", capabilities)
        self.llm_service = llm_service
        self.coordinator = coordinator
        self.tool_agent = coordinator.tool_agent if coordinator else None
        self.logger = self._setup_logger()

        # Set path to molecular data file using relative path from orchestrator location
        self.molecular_data_file = Path(__file__).parent.parent.parent / "data" / "molecular_data" / "molecular_data.json"
        
        # Create log directory if it doesn't exist
        log_dir = Path(__file__).parent.parent.parent / 'temp' / 'memory' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def load_molecule_data(self, sample_id: str, initial_data: Dict = None) -> Dict:
        """Load fresh molecule data from molecular data file for a specific sample.
        
        Args:
            sample_id: The ID of the sample to load
            initial_data: Initial molecule data to store if molecular data file doesn't exist
        """
        try:
            if not self.molecular_data_file.exists():
                self.logger.error(f"Molecular data file not found at {self.molecular_data_file}")
                return None
                
            with open(self.molecular_data_file, 'r') as f:
                molecular_data = json.load(f)
                
            # Get molecule data directly using sample_id as key
            if sample_id in molecular_data:
                self.logger.info(f"[load_molecule_data] Successfully loaded fresh data for sample {sample_id}")
                return molecular_data[sample_id]
                    
            self.logger.error(f"[load_molecule_data] No data found for sample {sample_id}")
            return None
            
        except Exception as e:
            self.logger.error(f"[load_molecule_data] Error loading molecule data: {str(e)}")
            return None

    async def process_molecule(self, molecule_data: Dict, user_input: str = None, context: Dict = None) -> Dict:
        """Process a single molecule through the structure elucidation workflow."""
        try:
            if not self.tool_agent:
                raise RuntimeError("Tool agent not initialized")
                
            # Use workflow type from context
            workflow_type = WorkflowType(context.get('workflow_type'))
            workflow_steps = get_workflow_steps(workflow_type)
            self.logger.info(f"[process_molecule] Using {len(workflow_steps)} workflow steps for workflow type: {workflow_type.value}")
            
            # Initialize context if not provided
            context = context or {}
            
            # Create temporary directory for this run
            run_id = str(uuid.uuid4())
            run_dir = Path("_temp_folder") / "structure_elucidation" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            context['run_dir'] = str(run_dir)
            context['run_id'] = run_id
            
            # Get initial sample_id
            sample_id = molecule_data.get('sample_id')
            if not sample_id:
                raise ValueError("No sample_id found in molecule_data")
            
            # Track workflow results and step completion
            workflow_data = {
                'predictions': [],
                'matches': [],
                'plots': [],
                'completed_steps': {},
                'step_outputs': {},
                'candidate_analysis': {}  # Add new key for combined candidate analysis
            }
            
            # Execute each step in sequence
            for idx, step in enumerate(workflow_steps, 1):
                try:
                    self.logger.info(f"[process_molecule] Step {idx}/{len(workflow_steps)}: {step.description}")
                    
                    # Load fresh molecule data before each step
                    fresh_molecule_data = self.load_molecule_data(sample_id)
                    if fresh_molecule_data is None:
                        error_msg = f"Failed to load fresh molecule data for step {idx}"
                        self.logger.error(f"[process_molecule] {error_msg}")
                        return {
                            'type': 'error',
                            'content': error_msg,
                            'metadata': {
                                'agent': 'ORCHESTRATOR',
                                'confidence': 0.0,
                                'reasoning': 'Failed to load fresh molecule data'
                            }
                        }
                    
                    # Update context with fresh molecule data and any existing candidate analysis
                    context['current_molecule'] = fresh_molecule_data
                    if 'candidate_analysis' in fresh_molecule_data:
                        workflow_data['candidate_analysis'].update(fresh_molecule_data['candidate_analysis'])
                    
                    # Validate prerequisites based on step requirements
                    if not self._validate_step_prerequisites(step, workflow_data):
                        error_msg = f"Prerequisites not met for step {idx}: {step.description}"
                        self.logger.error(f"[process_molecule] {error_msg}")
                        return {
                            'type': 'error',
                            'content': error_msg,
                            'metadata': {
                                'agent': 'ORCHESTRATOR',
                                'confidence': 0.0,
                                'reasoning': 'Step prerequisites not met'
                            }
                        }
                    
                    # Add current step to context
                    context['current_step'] = step
                    
                    # Execute the step command
                    result = await self.tool_agent.process(step.command, context=context)
                    
                    # Validate step output
                    if not self._validate_step_output(step, result):
                        error_msg = f"Step {idx} failed validation: {step.description}"
                        self.logger.error(f"[process_molecule] {error_msg}")
                        return {
                            'type': 'error',
                            'content': error_msg,
                            'metadata': {
                                'agent': 'ORCHESTRATOR',
                                'confidence': 0.0,
                                'reasoning': 'Step output validation failed'
                            }
                        }
                    
                    # Store step results
                    if result.get('type') == 'success':
                        self.logger.info(f"[process_molecule] Step {idx} completed successfully")
                        workflow_data['completed_steps'][step.keyword] = True
                        workflow_data['step_outputs'][step.keyword] = result.get('content', {})
                        
                        # Handle molecular data updates from tools
                        if 'molecular_data' in result.get('content', {}):
                            self.logger.info(f"[process_molecule] Updating molecular data from tool results")
                            # Update the master molecular data file
                            with open(self.molecular_data_file, 'r') as f:
                                master_data = json.load(f)
                            # self.logger.info(f"[process_molecule] Tool results: {result}")
                            # self.logger.info(f"[process_molecule] Content: {result.get('content', {})}")
                            self.logger.info(f"[process_molecule] Molecular data: {result.get('content', {}).get('molecular_data', {})}")
                            
                            master_data.update(result['content']['molecular_data'])
                            with open(self.molecular_data_file, 'w') as f:
                                json.dump(master_data, f, indent=2)
                            self.logger.info(f"[process_molecule] Updated master molecular data file {master_data}")
                    else:
                        error_msg = f"Step {idx} returned error: {result.get('content', 'Unknown error')}"
                        self.logger.info(f"[process_molecule] {error_msg}")
                        return {
                            'type': 'error',
                            'content': error_msg,
                            'metadata': {
                                'agent': 'ORCHESTRATOR',
                                'confidence': 0.0,
                                'reasoning': 'Step execution failed'
                            }
                        }
                    
                    # Update context with latest results
                    context['workflow_data'] = workflow_data
                    
                except Exception as e:
                    self.logger.error(f"[process_molecule] Error executing step {idx} '{step.description}': {str(e)}")
                    return {
                        'type': 'error',
                        'content': str(e),
                        'metadata': {
                            'agent': 'ORCHESTRATOR',
                            'confidence': 0.0,
                            'reasoning': f'Failed to execute step {idx}: {step.description}'
                        }
                    }
            
            # Log final workflow statistics
            self.logger.info(f"[process_molecule] Workflow completed successfully:")
            self.logger.info(f"[process_molecule] - Completed steps: {len(workflow_data['completed_steps'])}/{len(workflow_steps)}")
            self.logger.info(f"[process_molecule] - Total predictions: {len(workflow_data['predictions'])}")
            self.logger.info(f"[process_molecule] - Total matches: {len(workflow_data['matches'])}")
            self.logger.info(f"[process_molecule] - Total plots: {len(workflow_data['plots'])}")
            
            # Clean up intermediate files
            # try:
            #     import shutil
            #     intermediate_results_dir = Path("_temp_folder") / "intermediate_results"
            #     if intermediate_results_dir.exists():
            #         shutil.rmtree(intermediate_results_dir)
            #         self.logger.info(f"[process_molecule] Cleaned up intermediate results directory: {intermediate_results_dir}")
                
            #     # Also clean up the run directory
            #     if run_dir.exists():
            #         shutil.rmtree(run_dir)
            #         self.logger.info(f"[process_molecule] Cleaned up run directory: {run_dir}")
            # except Exception as e:
            #     self.logger.warning(f"[process_molecule] Error during cleanup: {str(e)}")
            
            return {
                'type': 'success',
                'content': workflow_data,
                'metadata': {
                    'agent': 'ORCHESTRATOR',
                    'confidence': 1.0,
                    'reasoning': f'Successfully processed molecule through {workflow_type.value} workflow'
                }
            }
            
        except Exception as e:
            self.logger.error(f"[process_molecule] Error in process_molecule: {str(e)}")
            return {
                'type': 'error',
                'content': str(e),
                'metadata': {
                    'agent': 'ORCHESTRATOR',
                    'confidence': 0.0,
                    'reasoning': 'Failed to process molecule'
                }
            }

    def _validate_step_prerequisites(self, step: WorkflowStep, workflow_data: Dict) -> bool:
        """Validate that prerequisites are met for a given step."""
        # Empty requires list means no prerequisites
        if not step.requires:
            return True
            
        # Get current molecule data from context
        molecule_data = workflow_data.get('current_molecule', {})
            
        # Check each requirement group (OR conditions)
        for requirement_group in step.requires:
            # For each group, at least one requirement must be met
            requirements_met = False
            
            for req in requirement_group:
                # Check if step was completed in current workflow
                if workflow_data['completed_steps'].get(req, False):
                    requirements_met = True
                    break
                    
                # Check if data is available in molecule data
                if req == 'forward_prediction' and molecule_data.get('forward_predictions'):
                    requirements_met = True
                    break
                elif req == 'mol2mol' and molecule_data.get('mol2mol_results', {}).get('status') == 'success':
                    requirements_met = True
                    break
                elif req == 'nmr_simulation' and any(f'{type}_sim' in molecule_data.get('nmr_data', {}) 
                                                   for type in ['1H', '13C', 'HSQC', 'COSY']):
                    requirements_met = True
                    break
                elif req == 'peak_matching' and molecule_data.get('exp_sim_peak_matching', {}).get('status') == 'success':
                    requirements_met = True
                    break
                    
            if not requirements_met:
                self.logger.error(f"[process_molecule] Missing prerequisites for step '{step.keyword}'. Need one of: {requirement_group}")
                return False
                
        return True

    def _validate_step_output(self, step: WorkflowStep, result: Dict) -> bool:
        """Validate the output of a workflow step."""
        if result.get('type') != 'success':
            return False
            
        content = result.get('content', {})
        
        # Validate based on step keyword
        if step.keyword == 'error_thresholds':
            if not content or 'threshold_data' not in content:
                return False
            threshold_data = content['threshold_data']
            
            # Just check if status is success
            return threshold_data.get('status') == 'success'
        elif step.keyword == 'retrosynthesis':
            return 'predictions' in content
        elif step.keyword == 'nmr_simulation':
            # Check if NMR simulation data exists in master data format
            if not content or 'status' not in content:
                return False
            if content['status'] != 'success':
                return False
            if 'data' not in content:
                return False
            return True
        elif step.keyword == 'peak_matching':
            # Check for proper response format and success status
            if not content or 'exp_sim_peak_matching' not in content:
                return False
            peak_results = content['exp_sim_peak_matching']
            return peak_results.get('status') == 'success'
        elif step.keyword == 'mol2mol':
            # Check if mol2mol generation was successful
            if not content or 'status' not in content:
                return False
            return content['status'] == 'success'
        elif step.keyword == 'visual_comparison':
            return 'plots' in content
            
        return True  # No specific validation for other steps

    async def process(self, message: str, context: Dict = None) -> Dict:
        """Process an orchestration request."""
        try:
            self.logger.info(f"[process] Starting orchestration with context: {context}")
            self.logger.info(f"[process] Message: {message}")
            model_choice = context.get('model_choice', 'gemini-flash')
            processing_mode = context.get('processing_mode', 'batch')  # Default to batch
            
            # Load molecular data
            with open(self.molecular_data_file, 'r') as f:
                molecular_data = json.load(f)
            self.logger.info(f"[process] Loaded {len(molecular_data)} molecules")
            
            results = []
            total_molecules = len(molecular_data)
            successful = 0
            failed = 0

            # If we have a current molecule in context and single mode, process just that one
            if context and 'current_molecule' in context and processing_mode == 'single':
                current_molecule = context['current_molecule']
                sample_id = current_molecule.get('sample_id', 'unknown')
                self.logger.info(f"[process] Processing single molecule from context: {sample_id}")
                
                # Load full molecule data from JSON file if available
                if sample_id in molecular_data:
                    current_molecule = molecular_data[sample_id]  # Use complete data from JSON
                elif 'workflow_type' not in current_molecule:
                    # If molecule is not in JSON and doesn't have workflow type, determine it
                    workflow_type = determine_workflow_type(current_molecule)
                    current_molecule['workflow_type'] = workflow_type.value
                
                try:
                    # Use stored workflow type from molecule data
                    workflow_type = WorkflowType(current_molecule.get('workflow_type', WorkflowType.SPECTRAL_ONLY.value))
                    molecule_context = {
                        **(context or {}),
                        'workflow_type': workflow_type.value  # Add workflow type to context
                    }
                    result = await self.process_molecule(current_molecule, context=molecule_context)
                    results.append(result)
                    if result['type'] == 'success':
                        successful += 1
                    else:
                        failed += 1
                    self.logger.info(f"[process] Current molecule processed with status: {result['type']}")
                except Exception as e:
                    self.logger.error(f"[process] Error processing current molecule: {str(e)}")
                    failed += 1
                    results.append({
                        'type': 'error',
                        'content': str(e),
                        'metadata': {
                            'agent': 'ORCHESTRATOR',
                            'confidence': 0.0,
                            'reasoning': 'Failed to process molecule'
                        }
                    })
                total_molecules = 1
            else:
                # Process all molecules from master JSON
                for molecule in molecular_data.values():
                    try:
                        self.logger.info(f"[process] Processing molecule {molecule.get('sample_id', 'unknown')}")
                        # Use stored workflow type from molecule data
                        workflow_type = WorkflowType(molecule.get('workflow_type', WorkflowType.SPECTRAL_ONLY.value))
                        molecule_context = {
                            **(context or {}),
                            'current_molecule': molecule,
                            'workflow_type': workflow_type.value  # Add workflow type to context
                        }
                        result = await self.process_molecule(molecule, context=molecule_context)
                        results.append(result)
                        
                        if result['type'] == 'success':
                            successful += 1
                        else:
                            failed += 1
                             
                        self.logger.info(f"[process] Molecule {molecule.get('sample_id', 'unknown')} processed with status: {result['type']}")
                    except Exception as e:
                        self.logger.error(f"[process] Error processing molecule {molecule.get('sample_id', 'unknown')}: {str(e)}")
                        failed += 1
                        results.append({
                            'type': 'error',
                            'content': str(e),
                            'metadata': {
                                'agent': 'ORCHESTRATOR',
                                'confidence': 0.0,
                                'reasoning': 'Failed to process molecule'
                            }
                        })
                    
            response = {
                'total_molecules': total_molecules,
                'successful': successful,
                'failed': failed,
                'results': results
            }
            self.logger.info(f"[process] Orchestration complete. Success: {successful}, Failed: {failed}")
            return response

        except Exception as e:
            self.logger.error(f"[process] Fatal error in orchestration: {str(e)}")
            self.logger.error(f"[process] Traceback: {traceback.format_exc()}")
            raise


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/orchestrator/orchestrator.py ---
"""
Orchestration Agent for managing structure elucidation workflows.

This module implements the high-level orchestration logic for analyzing chemical structure data
and coordinating analysis workflows through the Coordinator Agent using LLM-generated commands.
"""
 
import logging
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import traceback

from services.llm_service import LLMService
from ..coordinator.coordinator import CoordinatorAgent
from ..base.base_agent import BaseAgent
from .workflow_definitions import determine_workflow_type, get_workflow_steps, WorkflowType, WorkflowStep

class OrchestrationAgent(BaseAgent):
    """Agent responsible for orchestrating the structure elucidation workflow."""

    def __init__(self, llm_service: LLMService, coordinator=None):
        """Initialize orchestrator with required services."""
        capabilities = [
            "Workflow generation",
            "Process coordination",
            "Tool execution",
            "Error handling",
            "Analysis coordination"
        ]
        super().__init__("Orchestration Agent", capabilities)
        self.llm_service = llm_service
        self.coordinator = coordinator
        self.tool_agent = coordinator.tool_agent if coordinator else None
        self.analysis_agent = coordinator.analysis_agent if coordinator else None
        self.logger = self._setup_logger()

        # Set path to molecular data file using relative path from orchestrator location
        self.molecular_data_file = Path(__file__).parent.parent.parent / "data" / "molecular_data" / "molecular_data.json"
        
        # Create log directory if it doesn't exist
        log_dir = Path(__file__).parent.parent.parent / '_temp_folder' / 'memory' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def load_molecule_data(self, sample_id: str, initial_data: Dict = None) -> Dict:
        """Load fresh molecule data from molecular data file for a specific sample.
        
        Args:
            sample_id: The ID of the sample to load
            initial_data: Initial molecule data to store if molecular data file doesn't exist
        """
        try:
            if not self.molecular_data_file.exists():
                self.logger.error(f"Molecular data file not found at {self.molecular_data_file}")
                return None
                
            with open(self.molecular_data_file, 'r') as f:
                molecular_data = json.load(f)
                
            # Get molecule data directly using sample_id as key
            if sample_id in molecular_data:
                molecule_data = molecular_data[sample_id]
                # Set intermediate file path
                intermediate_path = self._get_intermediate_path(sample_id)
                molecule_data['_intermediate_file_path'] = str(intermediate_path)
                # Save initial data to intermediate file
                self._save_intermediate_data(sample_id, molecule_data)
                self.logger.info(f"[load_molecule_data] Successfully loaded fresh data for sample {sample_id}")
                return molecule_data
                    
            self.logger.error(f"[load_molecule_data] No data found for sample {sample_id}")
            return None
            
        except Exception as e:
            self.logger.error(f"[load_molecule_data] Error loading molecule data: {str(e)}")
            return None

    def _get_intermediate_path(self, sample_id: str) -> Path:
        """Get the path to the intermediate results file for a sample."""
        intermediate_dir = Path(__file__).parent.parent.parent / '_temp_folder' / 'intermediate_results'
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"[_get_intermediate_path] Using intermediate directory: {intermediate_dir}")
        return intermediate_dir / f"{sample_id}_intermediate.json"

    def _save_intermediate_data(self, sample_id: str, data: Dict) -> None:
        """Save data to the intermediate file."""
        filepath = self._get_intermediate_path(sample_id)
        self.logger.info(f"[_save_intermediate_data] Saving intermediate data to {filepath}")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_intermediate_data(self, sample_id: str) -> Dict:
        """Load data from the intermediate file."""
        filepath = self._get_intermediate_path(sample_id)
        if not filepath.exists():
            return None
            
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.logger.info(f"[_load_intermediate_data] Loaded intermediate data from {filepath}")
        return data
    
    def _update_master_data(self) -> None:
        """Update master molecular data file with workflow results."""
        try:
            with open(self.molecular_data_file, 'r') as f:
                master_data = json.load(f)
            
            # Get path to intermediate results directory using the same path as _get_intermediate_path
            intermediate_dir = Path(__file__).parent.parent.parent / '_temp_folder' / 'intermediate_results'
            self.logger.info(f"[_update_master_data] Checking intermediate directory: {intermediate_dir}")
            
            # Read all intermediate files and update master data
            updated_count = 0
            for intermediate_file in intermediate_dir.glob('*_intermediate.json'):
                try:
                    with open(intermediate_file, 'r') as f:
                        intermediate_data = json.load(f)
                    
                    # Extract sample_id from molecule_data
                    if 'molecule_data' in intermediate_data:
                        sample_id = intermediate_data['molecule_data'].get('sample_id')
                        if sample_id:
                            self.logger.info(f"[_update_master_data] Updating master data for sample {sample_id} from {intermediate_file.name}")
                            
                            # Create or update the sample entry in master data
                            if sample_id not in master_data:
                                master_data[sample_id] = {}
                            
                            # Update each top-level key from intermediate data
                            for key in intermediate_data.keys():
                                master_data[sample_id][key] = intermediate_data[key]
                            
                            updated_count += 1
                            self.logger.info(f"[_update_master_data] Updated keys for sample {sample_id}: {list(intermediate_data.keys())}")
                    else:
                        self.logger.warning(f"[_update_master_data] No molecule_data found in {intermediate_file.name}")
                        
                except Exception as e:
                    self.logger.error(f"[_update_master_data] Error processing intermediate file {intermediate_file}: {str(e)}")
                    continue
            
            # Write updated master data back to file
            with open(self.molecular_data_file, 'w') as f:
                json.dump(master_data, f, indent=2)
            self.logger.info(f"[_update_master_data] Updated master molecular data file with {updated_count} molecules from intermediate files")
        
        except Exception as e:
            self.logger.error(f"[_update_master_data] Error updating master molecular data file: {str(e)}")
            

    async def process_molecule(self, molecule: Dict, user_input: str = None, context: Dict = None) -> Dict:
        """Process a single molecule through the structure elucidation workflow."""
        try:
            # Add detailed logging of input molecule structure
            self.logger.debug(f"[process_molecule] Input molecule structure: {json.dumps(molecule, indent=2)}")
            self.logger.info(f"[process_molecule] Molecule keys: {list(molecule.keys())}")
            
            if not self.tool_agent:
                raise RuntimeError("Tool agent not initialized")
            if not self.analysis_agent and 'analysis' in context.get('workflow_type', ''):
                raise RuntimeError("Analysis agent not initialized")

            # Use workflow type from context
            workflow_type = WorkflowType(context.get('workflow_type'))
            workflow_steps = get_workflow_steps(workflow_type)
            self.logger.info(f"[process_molecule] workflow_steps: {workflow_steps}")

            self.logger.info(f"[process_molecule] Using {len(workflow_steps)} workflow steps for workflow type: {workflow_type.value}")
            context = context or {}

            # Get sample_id
            sample_id = molecule.get('sample_id')
            self.logger.debug(f"[process_molecule] Attempting to get sample_id: {sample_id}")
            if not sample_id:
                # Try alternate location if sample_id not found at root
                sample_id = molecule.get('molecule_data', {}).get('sample_id')
                self.logger.debug(f"[process_molecule] Tried alternate location for sample_id: {sample_id}")
            if not sample_id:
                raise ValueError("No sample_id found in molecule or molecule_data")

            # Set intermediate file path
            intermediate_path = self._get_intermediate_path(sample_id)
            self.logger.debug(f"[process_molecule] Setting up intermediate path: {intermediate_path}")
            
            # Validate molecule_data exists
            if "molecule_data" not in molecule:
                self.logger.error(f"[process_molecule] molecule_data missing from input structure: {list(molecule.keys())}")
                raise KeyError("molecule_data not found in molecule structure")
            
            # Set intermediate file path
            intermediate_path = self._get_intermediate_path(sample_id)
            molecule["molecule_data"]['_intermediate_file_path'] = str(intermediate_path)
            
            # Save initial data to intermediate file
            workflow_progress = {
                'molecule_data': molecule["molecule_data"],  # Include the full molecule data
                'completed_steps': {},
                # 'step_outputs': {},
            }
            self._save_intermediate_data(sample_id, workflow_progress)
            
            # Execute each step in sequence
            for idx, step in enumerate(workflow_steps, 1):
                try:
                    self.logger.info(f"[process_molecule] Step {idx}/{len(workflow_steps)}: {step.description}")
                    self.logger.info(f"[process_molecule] Step keyword {idx}/{len(workflow_steps)}: {step.keyword}")
                    
                    # Load latest intermediate data
                    workflow_data = self._load_intermediate_data(sample_id)
                    self.logger.info(f"[process_molecule] Loaded workflow data keys: {list(workflow_data.keys())}")
                    
                    # Update context with current molecule data
                    context['current_molecule'] = workflow_data['molecule_data']
                    
                    # Execute the step command based on step type
                    if step.keyword == 'analysis':
                        # Use analysis agent for analysis steps
                        self.logger.info(f"[process_molecule] Executing analysis step: {step.command}")
                        self.logger.info(f"[process_molecule] Executing analysis keyword: {step.keyword}")

                        result = await self.analysis_agent.process_all({
                                'task_input': {
                                    'command': step.command,
                                    'workflow_data': workflow_data,
                                    # 'step_outputs': workflow_data['step_outputs']
                                },
                                'context': context
                            })
                    else:
                        # Use tool agent for other steps
                        result = await self.tool_agent.process(step.command, context=context)
                    
                    if result.get('type') == 'success':
                        workflow_data = self._load_intermediate_data(sample_id)
                        self.logger.info(f"[process_molecule] Step {idx} completed successfully")
                        workflow_data['completed_steps'][step.keyword] = True
                        # workflow_data['step_outputs'][step.keyword] = result.get('content', {})
                        self._save_intermediate_data(sample_id, workflow_data)

                    else:
                        error_msg = f"Step {idx} failed: {result.get('content', 'Unknown error')}"
                        self.logger.error(f"[process_molecule] {error_msg}")
                        return {
                            'type': 'error',
                            'content': error_msg,
                            'metadata': {
                                'agent': 'ORCHESTRATOR',
                                'confidence': 0.0,
                                'reasoning': 'Step execution failed'
                            }
                        }
                    
                except Exception as e:
                    self.logger.error(f"[process_molecule] Error executing step {idx} '{step.description}': {str(e)}")
                    return {
                        'type': 'error',
                        'content': str(e),
                        'metadata': {
                            'agent': 'ORCHESTRATOR',
                            'confidence': 0.0,
                            'reasoning': f'Failed to execute step {idx}: {step.description}'
                        }
                    }
            
            # Log final workflow statistics
            self.logger.info(f"[process_molecule] Workflow completed successfully:")
            self.logger.info(f"[process_molecule] - Completed steps: {len(workflow_data['completed_steps'])}/{len(workflow_steps)}")
            
                        # Clean up intermediate files
            # try:
            #     import shutil
            #     intermediate_results_dir = Path("_temp_folder") / "intermediate_results"
            #     if intermediate_results_dir.exists():
            #         shutil.rmtree(intermediate_results_dir)
            #         self.logger.info(f"[process_molecule] Cleaned up intermediate results directory: {intermediate_results_dir}")
                
            #     # Also clean up the run directory
            #     if run_dir.exists():
            #         shutil.rmtree(run_dir)
            #         self.logger.info(f"[process_molecule] Cleaned up run directory: {run_dir}")
            # except Exception as e:
            #     self.logger.warning(f"[process_molecule] Error during cleanup: {str(e)}")
            # 
            # # Update master data file with results
            try:
                self._update_master_data()
                self.logger.info("[process_molecule] Successfully updated master data file")
            except Exception as e:
                self.logger.error(f"[process_molecule] Error updating master data file: {str(e)}")
                return {
                    'type': 'error',
                    'content': f"Workflow completed but failed to update master data: {str(e)}",
                    'metadata': {
                        'agent': 'ORCHESTRATOR',
                        'confidence': 0.0,
                        'reasoning': 'Failed to update master data file'
                    }
                }

            return {
                'type': 'success',
                'content': workflow_data,
                'metadata': {
                    'agent': 'ORCHESTRATOR',
                    'confidence': 1.0,
                    'reasoning': f'Successfully processed molecule through {workflow_type.value} workflow'
                }
            }
            
        except Exception as e:
            self.logger.error(f"[process_molecule] Error in process_molecule: {str(e)}")
            return {
                'type': 'error',
                'content': str(e),
                'metadata': {
                    'agent': 'ORCHESTRATOR',
                    'confidence': 0.0,
                    'reasoning': 'Failed to process molecule'
                }
            }

    async def process(self, message: str, context: Dict = None) -> Dict:
        """Process an orchestration request."""
        try:
            self.logger.info(f"[process] Starting orchestration with context: {context}")
            self.logger.info(f"[process] Message: {message}")
            
            # Add logging for molecular data file loading
            self.logger.debug(f"[process] Loading molecular data from: {self.molecular_data_file}")
            try:
                with open(self.molecular_data_file, 'r') as f:
                    molecular_data = json.load(f)
                    self.logger.debug(f"[process] Molecular data file structure: {json.dumps({k: list(v.keys()) for k, v in molecular_data.items()}, indent=2)}")
            except Exception as e:
                self.logger.error(f"[process] Error loading molecular data file: {str(e)}")
                raise
            
            model_choice = context.get('model_choice', 'gemini-flash')
            processing_mode = context.get('processing_mode', 'batch')  # Default to batch
            
            # Load molecular data
            with open(self.molecular_data_file, 'r') as f:
                molecular_data = json.load(f)
            self.logger.info(f"[process] Loaded {len(molecular_data)} molecules")
            
            results = []
            total_molecules = len(molecular_data)
            successful = 0
            failed = 0

            # If we have a current molecule in context and single mode, process just that one
            if context and 'current_molecule' in context and processing_mode == 'single':
                current_molecule = context['current_molecule']
                sample_id = current_molecule.get('sample_id', 'unknown')
                self.logger.info(f"[process] Processing single molecule from context: {sample_id}")
                
                # Load full molecule data from JSON file if available
                if sample_id in molecular_data:
                    current_molecule = molecular_data[sample_id]  # Use complete data from JSON
                elif 'workflow_type' not in current_molecule:
                    # If molecule is not in JSON and doesn't have workflow type, determine it
                    workflow_type = determine_workflow_type(current_molecule)
                    current_molecule['workflow_type'] = workflow_type.value
                    
                try:
                    # Use stored workflow type from molecule data
                    workflow_type = WorkflowType(current_molecule.get('workflow_type', WorkflowType.SPECTRAL_ONLY.value))
                    molecule_context = {
                        **(context or {}),
                        'workflow_type': workflow_type.value  # Add workflow type to context
                    }
                    result = await self.process_molecule(current_molecule, context=molecule_context)
                    results.append(result)
                    if result['type'] == 'success':
                        successful += 1
                    else:
                        failed += 1
                    self.logger.info(f"[process] Current molecule processed with status: {result['type']}")
                except Exception as e:
                    self.logger.error(f"[process] Error processing current molecule: {str(e)}")
                    failed += 1
                    results.append({
                        'type': 'error',
                        'content': str(e),
                        'metadata': {
                            'agent': 'ORCHESTRATOR',
                            'confidence': 0.0,
                            'reasoning': 'Failed to process molecule'
                        }
                    })
                total_molecules = 1
            else:
                                # Process all molecules from master JSON
                for current_molecule in molecular_data.values():
                    try:
                        self.logger.info(f"[process] Processing molecule {current_molecule.get('sample_id', 'unknown')}")
                        
                        # Determine or use stored workflow type
                        if 'workflow_type' not in current_molecule:
                            workflow_type = determine_workflow_type(current_molecule["molecule_data"])
                            self.logger.info(f"[process] Determined workflow type: {workflow_type.value}")
                            current_molecule["molecule_data"]['workflow_type'] = workflow_type.value
                        else:
                            workflow_type = WorkflowType(current_molecule['workflow_type'])
                            self.logger.info(f"[process] Using stored workflow type: {workflow_type.value}")
                        
                        molecule_context = {
                            **(context or {}),
                            'current_molecule': current_molecule["molecule_data"],
                            'workflow_type': workflow_type.value  # Add workflow type to context
                        }
                            
                        result = await self.process_molecule(current_molecule, context=molecule_context)
                        results.append(result)
                        
                        if result['type'] == 'success':
                            successful += 1
                        else:
                            failed += 1
                             
                        self.logger.info(f"[process] Molecule {current_molecule.get('sample_id', 'unknown')} processed with status: {result['type']}")
                    except Exception as e:
                        self.logger.error(f"[process] Error processing molecule {current_molecule.get('sample_id', 'unknown')}: {str(e)}")
                        failed += 1
                        results.append({
                            'type': 'error',
                            'content': str(e),
                            'metadata': {
                                'agent': 'ORCHESTRATOR',
                                'confidence': 0.0,
                                'reasoning': 'Failed to process molecule'
                            }
                        })
                    
            response = {
                'total_molecules': total_molecules,
                'successful': successful,
                'failed': failed,
                'results': results
            }
            
            # Update master molecular data file with results
            self._update_master_data()
            
            self.logger.info(f"[process] Orchestration complete. Success: {successful}, Failed: {failed}")
            return response

        except Exception as e:
            self.logger.error(f"[process] Fatal error in orchestration: {str(e)}")
            self.logger.error(f"[process] Traceback: {traceback.format_exc()}")
            raise


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/orchestrator/.ipynb_checkpoints/workflow_definitions-checkpoint.py ---
"""
Workflow definitions for structure elucidation.
"""

from typing import Dict, List, Optional, NamedTuple
from enum import Enum

class WorkflowType(Enum):
    MULTIPLE_TARGETS = "multiple_targets"
    STARTING_MATERIAL = "starting_material"
    TARGET_ONLY = "target_only"
    SPECTRAL_ONLY = "spectral_only"

class WorkflowStep(NamedTuple):
    """Represents a single step in a workflow"""
    keyword: str  # Unique identifier for validation
    command: str  # Command to execute
    description: str  # Human-readable description
    requires: List[List[str]]  # List of requirement groups, where each group represents OR conditions

# Define workflow steps
WORKFLOW_STEPS = {
    'threshold_calculation': WorkflowStep(
        keyword='threshold_calculation',
        command='Calculate dynamic thresholds for spectral data analysis',
        description='Calculate spectral analysis thresholds',
        requires=[]
    ),
    'retrosynthesis': WorkflowStep(
        keyword='retrosynthesis',
        command='Run retrosynthesis analysis on target structure',
        description='Perform retrosynthesis analysis',
        requires=[]
    ),
    'mol2mol': WorkflowStep(
        keyword='mol2mol',
        command='Run mol2mol to generate similar molecule analogs',
        description='Generate similar molecule analogs to the target molecule',
        requires=[]
    ),
    'nmr_simulation': WorkflowStep(
        keyword='nmr_simulation',
        command='Calculate simulated NMRs for the target structure',
        description='Generate simulated NMR spectra',
        requires=[]
    ),
    'peak_matching': WorkflowStep(
        keyword='peak_matching',
        command='Perform direct peak matching between target structure and experimental HSQC, COSY, 13C, and 1H NMR data',
        description='Match peaks between simulated and experimental spectra',
        requires=[['nmr_simulation']]
    ),
    'forward_prediction': WorkflowStep(
        keyword='forward_prediction',
        command='Run forward prediction on retrosynthesis products',
        description='Predict products from retrosynthesis',
        requires=[]
    ),
    'candidate_analysis': WorkflowStep(
        keyword='candidate_analysis',
        command='Analyze and score candidate molecules from all prediction sources',
        description='Analyze candidates using NMR data',
        requires=[['forward_prediction', 'mol2mol']]
    ),
    'visual_comparison': WorkflowStep(
        keyword='visual_comparison',
        command='Generate visual comparison for best matching structures',
        description='Create visual comparisons',
        requires=[['threshold_calculation'], ['peak_matching'], ['candidate_analysis']]
    ),
    'mmst': WorkflowStep(
        keyword='mmst',
        command='Run MMST to predict molecular structure from NMR data',
        description='Predict molecular structure using MMST',
        requires=[['threshold_calculation']]
    ),
    'analysis': WorkflowStep(
        keyword='analysis',
        command='Perform comprehensive analysis of molecular data and generate interpretable results',
        description='Analyze molecular data using LLM and specialized tools',
        requires=[['candidate_analysis'], ['visual_comparison']]
    )
}

# Define workflow sequences
WORKFLOW_SEQUENCES = {
    WorkflowType.TARGET_ONLY: [
        # WORKFLOW_STEPS['threshold_calculation'],
        WORKFLOW_STEPS['retrosynthesis'],
        WORKFLOW_STEPS['nmr_simulation'],
        WORKFLOW_STEPS['peak_matching'],
        # WORKFLOW_STEPS['forward_prediction'],
        # WORKFLOW_STEPS['candidate_analysis'],
        # WORKFLOW_STEPS['mol2mol'],
        # WORKFLOW_STEPS['candidate_analysis'],
        # WORKFLOW_STEPS['mmst'],
        # WORKFLOW_STEPS['candidate_analysis'],
       # WORKFLOW_STEPS['analysis']
    ],
    WorkflowType.STARTING_MATERIAL: [
        # WORKFLOW_STEPS['threshold_calculation'],
        # WORKFLOW_STEPS['nmr_simulation'],
        WORKFLOW_STEPS['forward_prediction'], # need to add different logic for experimental data comparison?
        WORKFLOW_STEPS['candidate_analysis'],
        WORKFLOW_STEPS['mol2mol'],
        WORKFLOW_STEPS['candidate_analysis'],
        WORKFLOW_STEPS['mmst'],
        WORKFLOW_STEPS['candidate_analysis'],
        WORKFLOW_STEPS['analysis']
    ],
    WorkflowType.MULTIPLE_TARGETS: [
        WORKFLOW_STEPS['threshold_calculation'],
        WORKFLOW_STEPS['retrosynthesis'],
        WORKFLOW_STEPS['nmr_simulation'],
        WORKFLOW_STEPS['peak_matching'],
        WORKFLOW_STEPS['forward_prediction'],
        WORKFLOW_STEPS['mol2mol'],
        WORKFLOW_STEPS['candidate_analysis'],
        WORKFLOW_STEPS['analysis']
    ],
    WorkflowType.SPECTRAL_ONLY: [
        WORKFLOW_STEPS['threshold_calculation'],
        WORKFLOW_STEPS['peak_matching'],
        WORKFLOW_STEPS['analysis']
    ]
}

def determine_workflow_type(data: Dict) -> WorkflowType:
    """Determine which workflow to use based on input data columns.
    
    The workflow type is determined by the presence of specific fields:
    - SMILES_list with multiple entries -> MULTIPLE_TARGETS
    - starting_material with non-empty SMILES -> STARTING_MATERIAL
    - SMILES present -> TARGET_ONLY
    - None of the above -> SPECTRAL_ONLY
    
    Note: For starting material workflow, the starting_smiles must contain actual SMILES data,
    not just an empty list or placeholder.
    
    Args:
        data: Dictionary containing either molecule_data directly or wrapped in molecule_data key
    """
    # Extract molecule_data if it exists, otherwise use data directly
    molecule_data = data.get('molecule_data', data)
    
    # Check if we have a SMILES field directly in the data
    has_smiles = ('SMILES' in molecule_data) or ('smiles' in molecule_data)
    
    # Check for SMILES_list first (multiple targets)
    if 'SMILES_list' in molecule_data and isinstance(molecule_data['SMILES_list'], list) and len(molecule_data['SMILES_list']) > 1:
        return WorkflowType.MULTIPLE_TARGETS
    
    # Check for starting material with actual content
    starting_smiles_key = next((key for key in molecule_data.keys() if key.lower().startswith('starting_smiles')), None)
    if starting_smiles_key:
        starting_smiles = molecule_data[starting_smiles_key]
        # Check if starting_smiles contains actual data
        if isinstance(starting_smiles, str) and starting_smiles.strip():
            return WorkflowType.STARTING_MATERIAL
        elif isinstance(starting_smiles, list) and any(isinstance(s, str) and s.strip() for s in starting_smiles):
            return WorkflowType.STARTING_MATERIAL
        elif isinstance(starting_smiles, dict) and any(isinstance(s, str) and s.strip() for s in starting_smiles.values()):
            return WorkflowType.STARTING_MATERIAL
    
    # Check for SMILES (single target)
    if has_smiles:
        return WorkflowType.TARGET_ONLY
        
    # Default to spectral only if no structure information is available
    return WorkflowType.SPECTRAL_ONLY

def get_workflow_steps(workflow_type: WorkflowType) -> List[WorkflowStep]:
    """Get the sequence of workflow steps for a workflow type."""
    return WORKFLOW_SEQUENCES.get(workflow_type, [])


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/coordinator/coordinator.py ---
"""
Main agent coordinator/orchestrator for managing agent interactions.
"""
from typing import Dict, Any, List, Optional, Tuple
from ..base import BaseAgent
from services.llm_service import LLMService
import json
import traceback
from enum import Enum, auto

class AgentType(Enum):
    """Available agent types in the system."""
    MOLECULE_PLOT = auto()
    NMR_PLOT = auto()
    TEXT_RESPONSE = auto()
    TOOL_USE = auto()  # Added tool agent type
    ORCHESTRATION = auto()  # Added orchestration agent type
    ANALYSIS = auto()  # Added analysis agent type

class CoordinatorAgent:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.agents: Dict[AgentType, BaseAgent] = {}
        
        # Initialize tool agent
        from ..specialized.tool_agent import ToolAgent
        self.tool_agent = ToolAgent(llm_service)
        self.tools = self.tool_agent.tools
        
        # Initialize analysis agent
        from ..specialized.analysis_agent import AnalysisAgent
        self.analysis_agent = AnalysisAgent(llm_service)
        
        # Register agents
        self.add_agent(AgentType.TOOL_USE, self.tool_agent)
        self.add_agent(AgentType.ANALYSIS, self.analysis_agent)
        
        # Define agent descriptions for LLM
        self._init_agent_descriptions()
        
    def set_orchestration_agent(self, orchestration_agent):
        """Set the orchestration agent and register it."""
        self.orchestration_agent = orchestration_agent
        self.add_agent(AgentType.ORCHESTRATION, orchestration_agent)
        
    def _init_agent_descriptions(self):
        """Initialize agent descriptions with string keys for JSON serialization."""
        self.agent_descriptions = {
            "MOLECULE_PLOT": {
                "name": "Molecule Plot",
                "description": "Specialized agent for handling molecular structure visualization requests",
                "capabilities": [
                    "2D/3D molecular structure visualization",
                    "Chemical structure drawing",
                    "Molecule rendering",
                ],
                "keywords": [
                    "molecule", "structure", "2D", "3D", 
                    "draw", "show", "visualize",
                    "can you show", "can you display",
                    "i want to see molecule"
                ]
            },
            "NMR_PLOT": {
                "name": "NMR Plot",
                "description": "Specialized agent for handling NMR spectroscopic data visualization and analysis",
                "capabilities": [
                    "1D/2D NMR spectrum visualization",
                    "HSQC plot generation",
                    "COSY correlation analysis",
                    "Peak analysis and integration",
                    "Chemical shift visualization"
                ],
                "keywords": [
                    "NMR", "spectrum", "HSQC", "COSY", 
                    "chemical shift", "proton", "carbon", 
                    "correlation", "peak", "integration",
                    "1h", "13c"
                ]
            },
            "ORCHESTRATION": {
                "name": "Workflow Orchestrator",
                "description": "Specialized agent for managing and executing structure elucidation workflows",
                "capabilities": [
                    "Structure elucidation workflow execution",
                    "Starting material workflow processing",
                    "Target structure workflow processing",
                    "Spectral-only workflow processing",
                    "Multi-step workflow coordination",
                    "Result validation and confidence scoring",
                    "Dynamic error threshold calculation",
                    "Forward and retrosynthesis prediction"
                ],
                "keywords": [
                    "analyze structure", "elucidate", "workflow",
                    "process spectra", "starting material",
                    "target structure", "analyze spectra",
                    "run analysis", "process workflow",
                    "structure analysis", "confidence score",
                    "run structure elucidation", "elucidation workflow",
                    "structure determination"
                ]
            },
            "TOOL_USE": {
                "name": "Tool Agent",
                "description": "Specialized agent for managing and coordinating various chemical prediction tools and simulations",
                "capabilities": [
                    "Retrosynthesis prediction and analysis",
                    "Forward synthesis prediction",
                    "Starting material identification",
                    "Reaction pathway prediction",
                    "Tool selection and coordination",
                    "NMR spectrum simulation",
                    "Analysis tool management",
                    "Simulation execution",
                    "Tool-specific task routing",
                    "Threshold calculation for NMR spectra",
                    "Peak matching and comparison",
                    "MMST structure improvement cycles",
                    "Multi-modal spectral analysis",
                    "Molecular analogue generation and scoring"
                ],
                "keywords": [
                    "starting material", "retrosynthesis", "predict starting materials",
                    "calculate starting material", "find starting materials",
                    "forward synthesis", "predict products", "reaction prediction",
                    "reaction pathway", "synthetic route", "synthesis steps",
                    "simulate", "prediction", "tool", "analysis",
                    "run", "execute", "predict", "generate",
                    "use tool", "run simulation", "analyze",
                    "nmr spectrum", "hsqc", "cosy",
                    "chemical shift", "proton", "carbon",
                    "threshold", "calculate threshold", "error threshold",
                    "peak matching", "compare peaks", "match spectra", "match peaks",
                    "run peak matching", "run nmr peak matching",
                    "improve structure", "structure improvement", "mmst",
                    "improvement cycle", "improve", "optimize structure",
                    "structure optimization", "molecular optimization"
                ]
            },
            "TEXT_RESPONSE": {
                "name": "Text Response",
                "description": "Default conversation handler for all queries that don't specifically require specialized agents",
                "capabilities": [
                    "Natural language understanding and conversation",
                    "General question answering on any topic",
                    "Chemistry-related discussions not requiring visualization",
                    "Contextual responses and clarifications",
                    "Information retrieval and explanation",
                    "Casual conversation and greetings",
                    "Fallback handling when other agents are not confident"
                ],
                "keywords": [
                    "what", "how", "why", "explain", 
                    "tell me", "describe", "help",
                    "question", "answer", "information", "hello",
                    "hi", "hey", "thanks", "thank you",
                    "can you", "could you", "please",
                    "discuss", "talk about", "understand"
                ]
            },
            "ANALYSIS": {
                "name": "Analysis Agent",
                "description": "Specialized agent for comprehensive analysis of molecular data and generation of interpretable results",
                "capabilities": [
                    "NMR data analysis and interpretation",
                    "Structure-spectrum correlation analysis",
                    "Visual molecular comparison analysis",
                    "LLM-based data interpretation",
                    "Data aggregation and reporting",
                    "Analysis coordination",
                    "Result explanation and summarization",
                    "Confidence scoring and validation",
                    "Discrepancy identification and explanation",
                    "Structure validation through spectral data"
                ],
                "keywords": [
                    "analyze data", "interpret results", "explain analysis",
                    "compare structures", "validate structure",
                    "analyze spectra", "interpret spectra",
                    "explain differences", "summarize results",
                    "explain discrepancies", "correlation analysis",
                    "confidence analysis", "structure validation",
                    "data interpretation", "comprehensive analysis",
                    "detailed analysis", "explain findings"
                ]
            }
        }
        
    def add_agent(self, agent_type: AgentType, agent: BaseAgent) -> None:
        """Add an agent to the coordinator."""
        if not isinstance(agent_type, AgentType):
            raise ValueError(f"Invalid agent type: {agent_type}")
        self.agents[agent_type] = agent

    async def process_message(self, user_input: str, model_choice: str = "claude-3-5-haiku") -> Dict[str, Any]:
        """Process an incoming user message and coordinate the appropriate agent response.
        
        This is the main entry point for processing messages in the system, used by both
        the chat interface and the orchestrator.
        
        Args:
            user_input: The user's input message
            model_choice: The selected model to use for processing (optional)
            
        Returns:
            Dict containing the response and any additional data
        """
        try:
            # Select appropriate agent based on input
            agent_type, confidence_agent, reasoning_agent, processing_mode = await self._select_agent(user_input, model_choice)
            
            # If confidence is too low, use TEXT_RESPONSE agent instead
            if confidence_agent <= 0.5:
                print(f"\n[Coordinator] Low confidence ({confidence_agent*100:.0f}%), falling back to TEXT_RESPONSE agent")
                print(f"[Coordinator] Original reasoning: {reasoning_agent}")
                agent_type = AgentType.TEXT_RESPONSE
                confidence_agent = 1.0  # Set high confidence for text response
                reasoning_agent = f"Falling back to general text response agent due to low confidence in specialized agents. Original reasoning: {reasoning_agent}"
            
            # Log the selected agent and confidence
            print(f"\n[Coordinator] Selected {agent_type.name} agent with {confidence_agent*100:.0f}% confidence")
            # print(f"[Coordinator] Reasoning: {reasoning_agent}")
            
            agent = self.agents.get(agent_type)
            if not agent:
                return {
                    "type": "error",
                    "content": f"No agent available for type: {agent_type.name}",
                    "metadata": {
                        "agent": "TEXT_RESPONSE",
                        "confidence": 0.0,
                        "reasoning": "Requested agent is not available in the system"
                    }
                }
                
            # Import socketio here to avoid circular import
            from core import socketio
            
            # Send start message
            socketio.emit('message', {
                'type': 'info',
                'content': f"🔄 Starting {agent_type.name.lower().replace('_', ' ')} task..."
            })

            # Get current molecule context
            from handlers.molecule_handler import get_current_molecule
            current_molecule = get_current_molecule()
            
            # Build context with molecule data
            context = {}
            if current_molecule:
                context['current_molecule'] = current_molecule
                context["processing_mode"] = processing_mode
                context["model_choice"] = model_choice
            print(f"[Coordinator] context: {context.keys()}")

            # Process the message with the selected agent and context
            response = await agent.process(user_input, context=context)
            # print(f"[Coordinator] Selected agent: {agent_type.name}, Confidence: {confidence_agent:.2f}")
            # print(f"[Coordinator] Agent response: {response}")

            # Handle tool responses
            if isinstance(response, dict) and response.get("type") == "tool_error":
                return {
                    "type": "error",
                    "content": response.get("content", "There is an error"),
                    "metadata": {
                        "agent": agent_type.name,
                        "confidence": response.get("confidence", 0.0),
                        "reasoning": response.get("reasoning", "No reasoning provided")
                    }
                }
            
            # Handle clarification responses
            if isinstance(response, dict) and response.get("type") == "clarification":
                return {
                    "type": "clarification",
                    "content": response.get("content", "Clarification needed"),
                    "metadata": {
                        "agent": agent_type.name,
                        "confidence": response.get("confidence", 0.0),
                        "reasoning": response.get("reasoning", "No reasoning provided")
                    }
                }
            
            # Preserve the original response type and structure for plot responses
            if isinstance(response, dict) and response.get("type") == "plot":
                response["metadata"] = {
                    "agent": agent_type.name,
                    "confidence": response.get("confidence", confidence_agent),
                    "reasoning": response.get("reasoning", reasoning_agent)
                }
                return response

            # Preserve the original response type and structure for plot responses
            if isinstance(response, dict) and response.get("type") == "molecule_plot":
                # print(f"[Coordinator] Response molecule_plot: {response}")
                response["metadata"] = {
                    "agent": agent_type.name,
                    "confidence": response.get("confidence", confidence_agent),
                    "reasoning": response.get("reasoning", reasoning_agent)
                }
                return response                
            
            # For non-dict responses, wrap them in a standard format
            return {
                "type": "text_response",
                "content": response,
                "metadata": {
                    "agent": agent_type.name,
                    "confidence": confidence_agent,
                    "reasoning": reasoning_agent
                }
            }
            
        except Exception as e:
            traceback.print_exc()
            return {
                "type": "error",
                "content": f"Error processing message: {str(e)}",
                "metadata": {
                    "agent": "TEXT_RESPONSE",
                    "confidence": 0.0,
                    "reasoning": f"An error occurred while processing the request: {str(e)}"
                }
            }
            
    async def process(self, message: str, model_choice: str = None, context: Dict = None) -> Dict:
        """Legacy method for backward compatibility. Use process_message instead."""
        return await self.process_message(message, model_choice)

    def _create_agent_selection_prompt(self, message: str) -> str:
        """Create a prompt for the LLM to select an appropriate agent."""
        agent_info = json.dumps(self.agent_descriptions, indent=2)
        
        prompt = f"""IMPORTANT: You must respond with ONLY a JSON object. No other text or explanations.

        Task: Given the following user message and available agents, determine:
        1. The most appropriate agent to handle the request
        2. Whether to process a single molecule or batch of molecules

        User Message: "{message}"

        Available Agents:
        {agent_info}

        You must respond with EXACTLY this JSON format, nothing else:
        {{
            "agent_type": "MOLECULE_PLOT or NMR_PLOT or TEXT_RESPONSE or TOOL_USE or ORCHESTRATION or ANALYSIS",
            "confidence": <float between 0 and 1>,
            "processing_mode": "single" or "batch",
            "reasoning": "<one sentence explaining why this agent and mode were chosen. If confidence is low, explain why and suggest a clearer way to phrase the request>"
        }}

        For example, a valid response would be:
        {{
            "agent_type": "NMR_PLOT",
            "confidence": 0.95,
            "processing_mode": "single",
            "reasoning": "Request specifically mentions HSQC spectrum visualization for a specific molecule which is a core capability of the NMR Plot agent"
        }}

        Or for a batch processing example:
        {{
            "agent_type": "TOOL_USE",
            "confidence": 0.85,
            "processing_mode": "batch",
            "reasoning": "Request asks to calculate thresholds for all molecules in the dataset, requiring batch processing through the tool agent"
        }}

        Or for a low confidence example:
        {{
            "agent_type": "TEXT_RESPONSE",
            "confidence": 0.3,
            "reasoning": "Request is too vague to determine specific visualization needs - consider specifying if you want to see a molecule structure, NMR spectrum, or get information about a specific topic"
        }}
        """
        
        # print("\n[Coordinator] Generated prompt for agent selection:")
        # print("----------------------------------------")
        # print(prompt)
        # print("----------------------------------------")
        
        return prompt

    async def _select_agent(self, message: str, model_choice: str) -> Tuple[AgentType, float, str, str]:
        """
        Select the most appropriate agent based on message content.
        
        Args:
            message: The user message
            model_choice: The LLM model to use for processing
            
        Returns:
            Tuple of (AgentType, confidence_score, reasoning, processing_mode)
        """
        # Prepare the prompt for agent selection
        prompt = self._create_agent_selection_prompt(message)
        
        try:
            print("\n[Coordinator] Requesting agent selection from LLM...")
            # print(f"[Coordinator] Input message: {message}")
            print(f"[Coordinator] Using model: {model_choice}")
            
            # Get LLM response with JSON validation
            response = await self.llm_service.get_completion(
                prompt, 
                model=model_choice,
                require_json=True,  # Enable JSON validation
                max_retries=3  # Allow up to 3 retries
            )
            
            print("\n[Coordinator] Raw LLM response for agent selection:")
            # print("----------------------------------------")
            # # print(response)
            # print("----------------------------------------")
            
            # Handle error responses
            if response.startswith("Error in LLM completion:"):
                raise ValueError(response)
            
            # Parse JSON response
            try:
                result = json.loads(response)
                print("\n[Coordinator] Parsed agent selection result:")
                print(json.dumps(result, indent=2))
            except json.JSONDecodeError as e:
                print(f"[Coordinator] Invalid JSON response: {response}")
                raise ValueError(f"Failed to parse JSON response: {str(e)}")
            
            # Convert string agent type to enum
            agent_type_str = result["agent_type"].upper()  # Ensure uppercase
            if agent_type_str not in ["MOLECULE_PLOT", "NMR_PLOT", "TEXT_RESPONSE", "TOOL_USE", "ORCHESTRATION", "ANALYSIS"]:
                raise ValueError(f"Invalid agent type: {agent_type_str}")
                
            agent_type = AgentType[agent_type_str]
            confidence = float(result["confidence"])
            reasoning = str(result.get("reasoning", "No reasoning provided"))
            processing_mode = str(result.get("processing_mode", "single")).lower()
            
            if processing_mode not in ["single", "batch"]:
                processing_mode = "single"  # Default to single if invalid
            
            return agent_type, confidence, reasoning, processing_mode
            
        except Exception as e:
            print(f"[Coordinator] Error selecting agent: {str(e)}")
            return AgentType.TEXT_RESPONSE, 0.0, f"Error selecting agent: {str(e)}", "single"

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/coordinator/__init__.py ---
"""
Coordinator Agent module for managing agent interactions.
"""

from .coordinator import CoordinatorAgent, AgentType

__all__ = ['CoordinatorAgent', 'AgentType']


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/scripts/chemformer_forward_script.py ---
#!/usr/bin/env python
from molbart.models import Chemformer
import hydra
import omegaconf
import pandas as pd
import sys
import os
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Run forward synthesis predictions")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input text file with reactant SMILES")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for predictions")
    parser.add_argument("--n_beams", type=int, default=50, help="Number of beams for beam search")
    parser.add_argument("--n_unique_beams", type=int, default=-1, help="Number of unique beams to return. Use -1 for no limit")
    return parser.parse_args()

def create_config(args):
    """Create Chemformer configuration."""
    config = {
        'data_path': args.input_file,
        'vocabulary_path': args.vocab_path,
        'model_path': args.model_path,
        "n_unique_beams": None if args.n_unique_beams == -1 else args.n_unique_beams,
        'task': 'forward_prediction',
        'output_sampled_smiles': args.output_file,
        'batch_size': args.batch_size,
        'n_beams': args.n_beams,
        'n_gpus': 1 if torch.cuda.is_available() else 0,
        'train_mode': 'eval',
        'model_type': 'bart',
        'datamodule': ['SynthesisDataModule'],
        "device": "cuda" if torch.cuda.is_available() else "cpu",        

    }
    return OmegaConf.create(config)

def write_predictions(smiles, log_lhs, target_smiles, output_file):
    """Write predictions to CSV file."""
    try:
        # Debug logging
        # logger.info(f"Number of predictions: {len(smiles)}")
        # logger.info(f"Shape of first prediction: {np.array(smiles[0]).shape if smiles else 'empty'}")
        # logger.info(f"Shape of first log_lhs: {np.array(log_lhs[0]).shape if log_lhs else 'empty'}")
        # logger.info(f"Number of target smiles: {len(target_smiles)}")
        
        # Create DataFrame
        predictions_df = pd.DataFrame({
            'target_smiles': target_smiles,
            'predicted_smiles': [s[0].item() if isinstance(s, (list, np.ndarray)) and len(s) > 0 else '' for s in smiles],
            'log_likelihood': [float(l[0]) if isinstance(l, (list, np.ndarray)) and len(l) > 0 else 0.0 for l in log_lhs],
            'all_predictions': [';'.join(map(str, s)) if isinstance(s, (list, np.ndarray)) else '' for s in smiles],
            'all_log_likelihoods': [';'.join(map(str, l)) if isinstance(l, (list, np.ndarray)) else '' for l in log_lhs]
        })
        
        # # Debug logging
        # logger.info(f"DataFrame shape: {predictions_df.shape}")
        # logger.info(f"DataFrame columns: {predictions_df.columns}")
        # if len(predictions_df) > 0:
        #     logger.info("First row of predictions:")
        #     logger.info(predictions_df.iloc[0].to_dict())
        
        predictions_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error writing predictions: {str(e)}")
        raise

def main():
    """Main function to run forward synthesis predictions."""
    try:
        # Parse arguments
        args = parse_arguments()
        logger.info("Creating Chemformer configuration")
        
        # Create config
        config = create_config(args)
        logger.info(f"Configuration created: {config}")
        
        # Initialize model
        logger.info("Initializing Chemformer model")
        chemformer = Chemformer(config)
        
        # Run prediction
        logger.info("Running predictions")
        smiles, log_lhs, target_smiles = chemformer.predict(dataset='full')
        
        # Save predictions
        write_predictions(smiles, log_lhs, target_smiles, args.output_file)
        logger.info(f"Predictions completed. Results saved to {args.output_file}")
        
    except Exception as e:
        logger.error(f"Error in forward synthesis pipeline: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/scripts/mmst_script.py ---
"""
MMST script for structure prediction workflow.
This script implements the improvement cycle for molecular structure prediction.
"""

# Standard library imports
import os
import sys
import json
import pickle
import time
import logging
import datetime
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Any
import pickle

# Third-party imports
import pandas as pd
import numpy as np
import torch

# Third-party imports
import random
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import Draw, MolFromSmiles

# Add necessary directories to Python path
# Get all relevant paths
script_dir = Path(__file__).resolve().parent  # scripts directory
llm_dir = script_dir.parent.parent  # LLM_Structure_Elucidator directory
mmt_dir = llm_dir.parent  # MMT_explainability directory
mol_opt_dir = mmt_dir / "deep-molecular-optimization"  # deep-molecular-optimization directory

# Clear any existing paths that might conflict
sys.path = [p for p in sys.path if not any(str(d) in p for d in [script_dir, llm_dir, mmt_dir, mol_opt_dir])]

# Add directories to path in the correct order
sys.path.insert(0, str(mol_opt_dir))  # First priority for models.dataset
sys.path.insert(0, str(mmt_dir))      # Second priority for utils_MMT
sys.path.insert(0, str(script_dir))   # Top priority for imports_MMST

print("Python paths added in order:")
print(f"1. Script dir: {script_dir}")
print(f"2. MMT dir: {mmt_dir}")
print(f"3. Mol-opt dir: {mol_opt_dir}")
print("\nFull Python path:")
for p in sys.path[:5]:  # Show first 5 paths
    print(f"  {p}")

# Now import the modules after path setup
from utils_MMT.execution_function_v15_4 import *
from utils_MMT.mmt_result_test_functions_15_4 import *
import utils_MMT.data_generation_v15_4 as dg

# Import the modules
from imports_MMST import (
    # Local utilities
    mtf, ex, mrtf, hf,
    # Helper functions
    parse_arguments, load_config, save_updated_config,
    load_configs, load_json_dics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_molecular_data(json_path: str) -> Dict:
    """Load molecular data from JSON file."""
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return {}

def test_pretrained_model(config: Dict, stoi: Dict, itos: Dict, stoi_MF: Dict) -> Tuple[float, Dict, Dict]:
    """Test the pre-trained model performance.
    
    Args:
        config: Configuration dictionary
        stoi: String to index mapping
        itos: Index to string mapping
        stoi_MF: String to index mapping for molecular fingerprints
        
    Returns:
        Tuple containing:
        - Performance score
        - Results dictionary
        - Greedy results dictionary
    """
    logger.info("Testing pre-trained model performance...")
    

    # Read input CSV
    input_df = pd.read_csv(config.input_csv)
    if len(input_df) != 1:
        raise ValueError("Input CSV must contain exactly one sample")
    
    # Verify sample-id column exists
    if 'sample-id' not in input_df.columns:
        raise ValueError("Input CSV must have a 'sample-id' column")
    
    # Create SGNN input directory if it doesn't exist
    sgnn_input_dir = Path(config.sgnn_gen_folder) / "input"
    sgnn_input_dir.mkdir(parents=True, exist_ok=True)
    
    # Save input sample for SGNN
    sgnn_input_file = sgnn_input_dir / "input_sample.csv"
    input_df.to_csv(sgnn_input_file, index=False)
    config.SGNN_csv_gen_smi = str(sgnn_input_file)

    # Generate SGNN data first
    combined_df, data_1H, data_13C, data_COSY, data_HSQC, csv_1H_path, csv_13C_path, csv_COSY_path, csv_HSQC_path = dg.main_run_data_generation(config)
    
    # Update config with generated paths
    config.csv_1H_path_SGNN = csv_1H_path
    config.csv_13C_path_SGNN = csv_13C_path
    config.csv_COSY_path_SGNN = csv_COSY_path
    config.csv_HSQC_path_SGNN = csv_HSQC_path
    
    # Set pickle file path to empty to force regeneration
    config.pickle_file_path = ""
        
    # Log paths for debugging
    logger.info(f"Generated paths:")
    logger.info(f"1H: {csv_1H_path}")
    logger.info(f"13C: {csv_13C_path}")
    logger.info(f"COSY: {csv_COSY_path}")
    logger.info(f"HSQC: {csv_HSQC_path}")
    
    # Set up validation data paths for SGNN
    config.csv_path_val = csv_1H_path  # Set the validation path to 1H data
    config.ref_data_type = "1H"        # Set reference data type to 1H
    config.dl_mode = "val"             # Set mode to validation
    config.training_mode = "1H_13C_HSQC_COSY_MF_MW"  # Set training modes
    config.data_type = "sgnn"          # Ensure data type is set to sgnn
    config.data_size = 1               # Set data size to 1 for single sample validation
    
    # Log config paths after setting
    logger.info(f"Config paths after setting:")
    logger.info(f"1H SGNN: {config.csv_1H_path_SGNN}")
    logger.info(f"13C SGNN: {config.csv_13C_path_SGNN}")
    logger.info(f"COSY SGNN: {config.csv_COSY_path_SGNN}")
    logger.info(f"HSQC SGNN: {config.csv_HSQC_path_SGNN}")
    logger.info(f"Val path: {config.csv_path_val}")
    logger.info(f"Ref data type: {config.ref_data_type}")
    logger.info(f"DL mode: {config.dl_mode}")
    logger.info(f"Training mode: {config.training_mode}")
    
    # Load models
    model_MMT = mrtf.load_MMT_model(config)
    model_CLIP = mrtf.load_CLIP_model(config)

    # Load data with the newly generated NMR  contain exactly one sample"
    val_dataloader = mrtf.load_data(config, stoi, stoi_MF, single=True, mode="val")
    val_dataloader_multi = mrtf.load_data(config, stoi, stoi_MF, single=False, mode="val")
    
    # Run tests
    results_dict = mrtf.run_test_mns_performance_CLIP_3(
        config, model_MMT, model_CLIP, val_dataloader, stoi, itos, True)
    
    results_dict, counter = mrtf.filter_invalid_inputs(results_dict)
    
    # Run greedy sampling
    config, results_dict_greedy = mrtf.run_greedy_sampling(
        config, model_MMT, val_dataloader_multi, itos, stoi)
    
    # Calculate performance metrics
    total_results = mrtf.run_test_performance_CLIP_3(
        config, model_MMT, val_dataloader, stoi)
    
    performance = total_results["statistics_multiplication_avg"][0]
    
    logger.info(f"Model performance: {performance}")
    return performance, results_dict, results_dict_greedy


def run_experimental_data(config: Dict, stoi: Dict, itos: Dict, stoi_MF: Dict) -> Tuple[float, Dict, Dict]:
    """Run the model on experimental data and generate molecules.
    
    Args:
        config: Configuration dictionary
        stoi: String to index mapping
        itos: Index to string mapping
        stoi_MF: String to index mapping for molecular fingerprints
        
    Returns:
        Tuple containing:
        - Performance score
        - Results dictionary
        - Greedy results dictionary
    """
    logger.info(f"Running model on experimental data with {config.multinom_runs} multinomial sampling runs...")
    
    # Log experimental data paths
    logger.info(f"Experimental data paths:")
    logger.info(f"1H: {config.csv_1H_path_SGNN}")
    logger.info(f"13C: {config.csv_13C_path_SGNN}")
    logger.info(f"COSY: {config.csv_COSY_path_SGNN}")
    logger.info(f"HSQC: {config.csv_HSQC_path_SGNN}")
    
    # Configure for experimental data testing
    config.dl_mode = "val"
    config.training_mode = "1H_13C_HSQC_COSY_MF_MW"
    config.data_type = "sgnn"
    config.data_size = 1  # Note: data_size remains at 1 for single sample validation
    
    # Load models
    model_MMT = mrtf.load_MMT_model(config)
    model_CLIP = mrtf.load_CLIP_model(config)
    
    # Load experimental data
    val_dataloader = mrtf.load_data(config, stoi, stoi_MF, single=True, mode="val")
    val_dataloader_multi = mrtf.load_data(config, stoi, stoi_MF, single=False, mode="val")
    
    # Run tests and generate molecules
    results_dict = mrtf.run_test_mns_performance_CLIP_3(
        config, model_MMT, model_CLIP, val_dataloader, stoi, itos, True)
    
    results_dict, counter = mrtf.filter_invalid_inputs(results_dict)
    
    # Run greedy sampling to generate molecules
    config, results_dict_greedy = mrtf.run_greedy_sampling(
        config, model_MMT, val_dataloader_multi, itos, stoi)
    
    # Calculate performance metrics
    total_results = mrtf.run_test_performance_CLIP_3(
        config, model_MMT, val_dataloader, stoi)
    
    performance = total_results["statistics_multiplication_avg"][0]
    
    logger.info(f"Experimental data performance: {performance}")
    return performance, results_dict, results_dict_greedy


def run_improvement_cycle(config: Dict, stoi: Dict, itos: Dict, stoi_MF: Dict, itos_MF: Dict, IR_config: Dict) -> float:
    """Run the improvement cycle workflow.
    
    Args:
        config: Configuration dictionary
        stoi: String to integer mapping
        itos: Integer to string mapping
        stoi_MF: String to integer mapping for molecular formulas
        itos_MF: Integer to string mapping for molecular formulas
        IR_config: IR configuration dictionary
        
    Returns:
        Final performance score
    """
    logger.info("Starting improvement cycle...")
    
    performance = 0
    iteration = 1  # Initialize iteration counter
    ic_results = []  # Initialize list to store iteration results
    while True:
        logger.info(f"Starting improvement cycle iteration {iteration}")
        # Step 1: Generate molecules using MF

        config, results_dict_MF = ex.SMI_generation_MF(config, stoi, stoi_MF, itos, itos_MF)
        
        # Clean up results
        results_dict_MF = {key: value for key, value in results_dict_MF.items() 
                          if not hf.contains_only_nan(value)}
        for key, value in results_dict_MF.items():
            results_dict_MF[key] = hf.remove_nan_from_list(value)
        
        # Transform results
        transformed_list_MF = [[key, value] for key, value in results_dict_MF.items()]
        src_smi_MF = list(results_dict_MF.keys())
        combined_list_MF = [item for sublist in transformed_list_MF for item in sublist[1][:]]
        
        # Step 2: Combine results
        all_gen_smis = combined_list_MF
        all_gen_smis = [smiles for smiles in all_gen_smis if smiles != 'NAN']

        # Filter potential hits
        val_data = pd.read_csv(config.csv_path_val)
        all_gen_smis = mrtf.filter_smiles(val_data, all_gen_smis)
        
        # Create DataFrame
        length_of_list = len(all_gen_smis)   
        random_number_strings = [f"GT_{str(i).zfill(7)}" for i in range(1, length_of_list + 1)]
        aug_mol_df = pd.DataFrame({'SMILES': all_gen_smis, 'sample-id': random_number_strings})


        # Step 3: Blend with training data
        config.train_data_blend = 0
        config, final_df = ex.blend_aug_with_train_data(config, aug_mol_df)
        
        # Step 4: Generate data
        # print(f"config: {config}")
        config = ex.gen_sim_aug_data(config, IR_config)
        # print(f"config after gen_sim_aug_data: {config}")

        # Step 5: Train transformer
        config.training_setup = "pretraining"
        mtf.run_MMT(config, stoi, stoi_MF)
        
        # Update model path and test configuration
        config = ex.update_model_path(config)
        # No need to reassign paths as they are already correctly set in config.json
        
        # Test current performance
        performance, results_dict, results_dict_greedy = test_pretrained_model(config, stoi, itos, stoi_MF)
        # performance = float(0.8)
        # results_dict = {}
        # results_dict_greedy = {}
        logger.info(f"Current cycle performance: {performance}")
        
        # Save iteration results
        iteration_results = {
            'iteration': iteration,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'performance': performance,
            'results': results_dict,
            'greedy_results': results_dict_greedy,
            'config': vars(config)
        }
        ic_results.append(iteration_results)
        
        if performance > config.IC_threshold:
            break
            
        iteration += 1
    
    return performance, ic_results

def prepare_experimental_data(nmr_type, json_path, output_csv, input_csv):
    """Prepare experimental NMR data from JSON for model input.
    
    Args:
        nmr_type (str): Type of NMR data ('1H', '13C', 'HSQC', 'COSY')
        json_path (str): Path to JSON file containing molecular data
        output_csv (str): Path to save the output CSV
        input_csv (str): Path to input CSV containing sample ID
    """
    # Read sample ID from input CSV
    input_df = pd.read_csv(input_csv)
    if 'sample-id' not in input_df.columns:
        raise ValueError("Input CSV must contain a 'sample-id' column with the sample ID")
    if len(input_df) != 1:
        raise ValueError("Input CSV must contain exactly one row")
    
    target_sample_id = input_df['sample-id'].iloc[0]
    logger.info(f"Processing sample ID: {target_sample_id}")
    
    # Load experimental data
    with open(json_path, 'r') as f:
        molecular_data = json.load(f)
    
    # Extract data for the specified sample
    if target_sample_id not in molecular_data:
        raise ValueError(f"Sample ID {target_sample_id} not found in molecular data")
    
    data = molecular_data[target_sample_id]
    logger.info(f"Data structure for sample {target_sample_id}: {data}")
    logger.info(f"Keys in data: {list(data.keys())}")
    
    # Access nmr_data through molecule_data
    if 'molecule_data' not in data or 'nmr_data' not in data['molecule_data']:
        raise ValueError(f"No NMR data found for sample {target_sample_id}")
    
    nmr_data = data['molecule_data']['nmr_data']
    logger.info(f"Keys in nmr_data: {list(nmr_data.keys())}")
    if f'{nmr_type}_exp' not in nmr_data:
        raise ValueError(f"No {nmr_type} experimental data found for sample {target_sample_id}")
    
    # Create output data
    exp_data = [{
        'sample-id': target_sample_id,
        'SMILES': data['smiles'],
        'NMR_Data': nmr_data[f'{nmr_type}_exp']
    }]
    
    # Save to CSV in required format
    df = pd.DataFrame(exp_data)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved experimental data for sample {target_sample_id} to {output_csv}")
    return output_csv

def get_sample_id_from_csv(csv_path: str) -> str:
    """Extract sample ID from input CSV file.
    
    Args:
        csv_path: Path to input CSV file
        
    Returns:
        Sample ID string
    """
    try:
        df = pd.read_csv(csv_path)
        if 'sample-id' in df.columns:
            return str(df['sample-id'].iloc[0])
        elif 'id' in df.columns:  # Fallback to 'id' column if exists
            return str(df['id'].iloc[0])
        return Path(csv_path).stem  # Fallback to filename without extension
    except Exception as e:
        logger.error(f"Error reading sample ID from CSV: {str(e)}")
        return "unknown_sample"

def get_sample_dirs(base_dir: Path, sample_id: str) -> dict:
    """Create and return dictionary of sample-specific directories.
    
    Args:
        base_dir: Base directory path
        sample_id: Sample ID string
        
    Returns:
        Dictionary containing paths for each subdirectory
    """
    sample_dir = base_dir / sample_id
    dirs = {
        'sample': sample_dir,
        'models': sample_dir / 'models',
        'sgnn_output': sample_dir / 'sgnn_output',
        'experimental_data': sample_dir / 'experimental_data',
        'test_results': sample_dir / 'test_results'
    }
    
    # Create all directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return dirs

# Add at the top with other imports
class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

def save_json_with_tensor_handling(data, output_file):
    """Save data to JSON file with proper tensor handling."""
    with open(output_file, 'w') as f:
        json.dump(data, f, cls=TensorEncoder, indent=2)


def parse_arguments(config, args):
    """Parse command line arguments into config object.
    
    Args:
        config: Existing config object loaded from config files
        args: Command line arguments parsed by argparse
        
    Returns:
        Updated config object with command line arguments
    """
    # Convert all path-like arguments to strings
    if args.input_csv:
        config.input_csv = str(args.input_csv)
    if args.output_dir:
        config.output_dir = str(args.output_dir)
    if args.model_save_dir:
        config.model_save_dir = str(args.model_save_dir)
    if args.config_dir:
        config.config_dir = str(args.config_dir)
    if args.mol2mol_model_path:
        config.mol2mol_model_path = str(args.mol2mol_model_path)
    if args.mol2mol_vocab_path:
        config.mol2mol_vocab_path = str(args.mol2mol_vocab_path)
    if args.sgnn_gen_folder:
        config.sgnn_gen_folder = str(args.sgnn_gen_folder)
    if args.exp_data_path:
        config.exp_data_path = str(args.exp_data_path)
    if args.run_mode:
        config.run_mode = args.run_mode
    
    # Mol2Mol parameters
    if args.MF_delta_weight:
        config.MF_delta_weight = args.MF_delta_weight
    if args.tanimoto_filter:
        config.tanimoto_filter = args.tanimoto_filter
    if args.MF_max_trails:
        config.MF_max_trails = args.MF_max_trails
    if args.max_scaffold_generations:
        config.max_scaffold_generations = args.max_scaffold_generations
    
    # SGNN parameters
    if args.sgnn_gen_folder:
        config.sgnn_gen_folder = args.sgnn_gen_folder
    
    # MMST parameters
    if args.MF_generations:
        config.MF_generations = args.MF_generations
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.prediction_batch_size:
        config.prediction_batch_size = args.prediction_batch_size
    if args.improvement_cycles:
        config.improvement_cycles = args.improvement_cycles
    if args.IC_threshold:
        config.IC_threshold = args.IC_threshold
    if args.learning_rate:
        config.lr_pretraining = [args.learning_rate]
        config.lr_finetuning = [args.learning_rate]
    
    # Experimental workflow parameters
    if args.nmr_types:
        config.nmr_types = args.nmr_types
    if args.multinom_runs:
        config.multinom_runs = args.multinom_runs
    return config

def main():
    parser = argparse.ArgumentParser(description='MMST Structure Prediction Workflow')
    # Common parameters
    parser.add_argument('--input_csv', required=True, help='Path to input CSV file containing a single SMILES and ID')
    parser.add_argument('--output_dir', required=True, help='Output directory for results and experiment data')
    parser.add_argument('--model_save_dir', required=True, help='Directory for saving fine-tuned models')
    parser.add_argument('--config_dir', required=True, help='Directory containing configuration files')
    parser.add_argument('--run_mode', choices=['test', 'improve', 'both'], default='both',
                        help='Run mode: test pre-trained model, run improvement cycle, or both')
    
    # Mol2Mol parameters
    parser.add_argument('--mol2mol_model_path', required=True, help='Path to Mol2Mol model')
    parser.add_argument('--mol2mol_vocab_path', required=True, help='Path to Mol2Mol vocabulary')
    parser.add_argument('--MF_delta_weight', type=int, default=100, help='Delta weight for Mol2Mol')
    parser.add_argument('--tanimoto_filter', type=float, default=0.2, help='Tanimoto filter threshold')
    parser.add_argument('--MF_max_trails', type=int, default=300, help='Maximum trails')
    parser.add_argument('--max_scaffold_generations', type=int, default=100, help='Maximum scaffold generations')
    
    # SGNN parameters
    parser.add_argument('--sgnn_gen_folder', required=True, help='SGNN generation folder')
    
    # MMST parameters
    parser.add_argument('--MF_generations', type=int, default=50, help='Number of analogues to generate')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of fine-tuning epochs')
    parser.add_argument('--prediction_batch_size', type=int, default=32, help='Batch size for prediction')
    parser.add_argument('--improvement_cycles', type=int, default=3, help='Number of improvement cycles to run')
    parser.add_argument('--IC_threshold', type=float, default=0.6, help='Performance threshold for improvement cycle')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate for finetuning during improvement cycles')
    

    # Experimental workflow parameters
    parser.add_argument('--nmr_types', nargs='+', type=str, choices=['1H', '13C', 'HSQC', 'COSY'], default=['1H'],
                        help='Types of NMR data to use')
    parser.add_argument('--exp_data_path', 
                        help='Path to experimental data JSON file')
    parser.add_argument('--multinom_runs', type=int, default=10,
                        help='Number of multinomial sampling runs to generate molecules')
    
    args = parser.parse_args()
    
    try:
        # Get sample ID and create sample-specific directories
        sample_id = get_sample_id_from_csv(args.input_csv)
        sample_dirs = get_sample_dirs(Path(args.output_dir), sample_id)
        
        # Update paths to use sample-specific directories
        args.model_save_dir = str(sample_dirs['models'])
        args.sgnn_gen_folder = str(sample_dirs['sgnn_output'])
        
        # Load configurations
        itos, stoi, stoi_MF, itos_MF = load_json_dics()
        IR_config, config = load_configs(args.config_dir)
        
        # Update config with command line arguments
        config = parse_arguments(config, args)
        
        # Store original model paths
        original_checkpoint_path = config.checkpoint_path
        original_mt_model_path = config.MT_model_path

        all_run_summaries = []  # List to collect all run summaries
        num_runs = config.improvement_cycles

        # Run the entire process N times
        for run_num in range(0, num_runs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting complete run {run_num}/{num_runs}")
            logger.info(f"{'='*50}\n")
            
            # Reset model paths to original pre-trained model for each run
            config.checkpoint_path = original_checkpoint_path
            config.MT_model_path = original_mt_model_path

            # Step 1: Test on simulated data
            logger.info(f"Run {run_num} - Step 1: Testing on simulated data...")
            # performance, results_dict, results_dict_greedy = test_pretrained_model(
            #     config, stoi, itos, stoi_MF) 
            performance = 0.1
            results_dict = {}
            results_dict_greedy = {}
            logger.info(f"Run {run_num} - Simulated data performance: {performance}")


            # Create run-specific directory for this iteration's results
            run_dir = sample_dirs['test_results'] / f'run_{run_num}'
            run_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"run_dir {run_dir}")

            # Save test results with run-specific path
            test_results = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'run_number': run_num,
                'performance': performance,
                'results': results_dict,
                'greedy_results': results_dict_greedy,
                'config': vars(config)
            }

            test_results_file = run_dir / 'simulated_test_results.json'
            save_json_with_tensor_handling(test_results, test_results_file)
            logger.info(f"test_results_file {test_results_file}")
            
                
            # Step 2: Check if performance meets threshold
            if performance >= config.IC_threshold:
                logger.info(f"Performance {performance} meets threshold {config.IC_threshold}")
                if config.exp_data_path:
                    # Define all NMR types
                    nmr_types = config.nmr_types
                    
                    # Prepare experimental data for each NMR type
                    exp_csv_paths = {}
                    for nmr_type in nmr_types:
                        exp_csv = sample_dirs['experimental_data'] / f"nmr_exp_{nmr_type}.csv"
                        prepare_experimental_data(
                            nmr_type, 
                            config.exp_data_path, 
                            exp_csv,
                            config.input_csv
                        )
                        exp_csv_paths[nmr_type] = str(exp_csv)
                    
                    # Save original input_csv
                    original_input = config.input_csv
                    
                    # Update config paths for each NMR type
                    config.csv_1H_path_SGNN = exp_csv_paths['1H']
                    config.csv_13C_path_SGNN = exp_csv_paths['13C']
                    config.csv_HSQC_path_SGNN = exp_csv_paths['HSQC']
                    config.csv_COSY_path_SGNN = exp_csv_paths['COSY']
                    config.csv_path_val = exp_csv_paths['HSQC']  # Set validation path to HSQC
                    config.pickle_file_path = ""  # Reset pickle file path
                    
                    # # Use HSQC data as main input for testing
                    # args.input_csv = exp_csv_paths['HSQC']
                    # config.input_csv = exp_csv_paths['HSQC']
                    
                    # Run on experimental data
                    logger.info("Running model on experimental data...")
                    exp_performance, exp_results, exp_results_greedy = run_experimental_data(
                        config, stoi, itos, stoi_MF)
                    logger.info(f"Experimental data performance: {exp_performance}")
                    
                    # Save experimental results in run-specific directory
                    exp_test_results = {
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'run_number': run_num,
                        'performance': exp_performance,
                        'results': exp_results,
                        'greedy_results': exp_results_greedy,
                        'model_save_path': str(sample_dirs['models']),
                        'improvement_cycle': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                        'config': vars(config)
                    }

                    exp_results_file = run_dir / 'experimental_results_wo_IC.json'
                    save_json_with_tensor_handling(exp_test_results, exp_results_file)

                
                # # Restore original input
                # args.input_csv = original_input
                # config.input_csv = original_input
            
            else:
                logger.info(f"Performance {performance} below threshold {config.IC_threshold}")
                logger.info("Starting improvement cycle...")
                
                # Create directories for improvement cycle within the run directory
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Create improvement cycle directory within run directory
                ic_dir = run_dir / f"improvement_cycle_{current_time}"
                ic_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"ic_dir {ic_dir}")

                # Create model directory with run number and timestamp
                model_save_path = sample_dirs['models'] / f"run_{run_num}_model_{current_time}"
                model_save_path.mkdir(parents=True, exist_ok=True)
                config.model_save_dir = str(model_save_path)

                # Update MOL2MOL config parameters to use the input file
                config.MF_csv_source_folder_location = str(sample_dirs['sample'])
                config.MF_csv_source_file_name = 'mmst_input'

                # Update learning rate for improvement cycle
                # config.lr_pretraining = 3e-4

                # Run improvement cycle
                final_performance, ic_results = run_improvement_cycle(config, stoi, itos, stoi_MF, itos_MF, IR_config)
                logger.info(f"Final improvement cycle performance: {final_performance}")

                # Save improvement cycle results within run directory
                ic_result_paths = []
                for i, result in enumerate(ic_results, 1):
                    ic_results_file = ic_dir / f'cycle_{i}_results.json'
                    save_json_with_tensor_handling(result, ic_results_file)
                    ic_result_paths.append(str(ic_results_file))

                # Save summary of improvement cycle
                ic_summary = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'run_number': run_num,
                    'initial_performance': performance,
                    'final_performance': final_performance,
                    'model_save_path': str(model_save_path),
                    'cycle_results': ic_result_paths
                }
                save_json_with_tensor_handling(ic_summary, ic_dir / 'improvement_cycle_summary.json')

                if final_performance >= config.IC_threshold and config.exp_data_path:
                    # If improved performance meets threshold, run on experimental data
                    # Define all NMR types
                    nmr_types = config.nmr_types

                    # Prepare experimental data for each NMR type
                    exp_csv_paths = {}
                    for nmr_type in nmr_types:
                        exp_csv = sample_dirs['experimental_data'] / f"nmr_exp_{nmr_type}.csv"
                        prepare_experimental_data(
                            nmr_type, 
                            config.exp_data_path, 
                            exp_csv,
                            config.input_csv
                        )
                        exp_csv_paths[nmr_type] = str(exp_csv)

                    # # Save original input
                    # original_input = args.input_csv

                    # Update config paths for each NMR type
                    config.csv_1H_path_SGNN = exp_csv_paths['1H']
                    config.csv_13C_path_SGNN = exp_csv_paths['13C']
                    config.csv_HSQC_path_SGNN = exp_csv_paths['HSQC']
                    config.csv_COSY_path_SGNN = exp_csv_paths['COSY']
                    config.csv_path_val = exp_csv_paths['HSQC']  # Set validation path to HSQC
                    config.pickle_file_path = ""  # Reset pickle file path

                    # # Use HSQC data as main input for testing
                    # args.input_csv = exp_csv_paths['HSQC']
                    # config.input_csv = exp_csv_paths['HSQC']

                    # Run on experimental data with improved model
                    logger.info("Running improved model on experimental data...")
                    exp_performance, exp_results, exp_results_greedy = run_experimental_data(
                        config, stoi, itos, stoi_MF)
                    logger.info(f"Experimental data performance with improved model: {exp_performance}")

                    # Save experimental results
                    exp_test_results = {
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'run_number': run_num,
                        'performance': exp_performance,
                        'results': exp_results,
                        'greedy_results': exp_results_greedy,
                        'model_save_path': str(model_save_path),
                        'improvement_cycle_dir': str(ic_dir),
                        'config': vars(config)
                    }

                    exp_results_file = ic_dir / 'experimental_results_after_IC.json'
                    save_json_with_tensor_handling(exp_test_results, exp_results_file)
                    logger.info(f"Saved experimental results to {exp_results_file}")

                    # Save individual run results
                    run_results = {
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'run_number': run_num,
                        'final_performance': final_performance,
                        'exp_results_file': str(exp_results_file),
                        'test_results_file': str(test_results_file),
                        'model_save_path': str(model_save_path),
                        'improvement_cycle_dir': str(ic_dir),
                        'improvement_cycle_results': ic_result_paths,
                    }

                    run_results_file = run_dir / f'run_{run_num}_results.json'
                    with open(run_results_file, 'w') as f:
                        json.dump(run_results, f, indent=4)
                    logger.info(f"Saved run {run_num} results to {run_results_file}")

                    all_run_summaries.append(run_results)

        # After all runs are complete, create final results file
        final_results_file = sample_dirs['test_results'] / 'mmst_final_results.json'
        final_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_runs': num_runs,
            'runs': all_run_summaries,
            'test_results_dir': str(sample_dirs['test_results'])
        }
        
        with open(final_results_file, 'w') as f:
            json.dump(final_results, f, indent=4)
        logger.info(f"Created final results file at {final_results_file}")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    sys.exit(main())


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/scripts/peak_matching_script.py ---
import sys
import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, List
import asyncio

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
TOOLS_DIR = PROJECT_ROOT / "LLM_Structure_Elucidator" / "agents" / "tools"
TEMP_DIR = PROJECT_ROOT / "LLM_Structure_Elucidator" / "_temp_folder" / "peak_matching"
CURRENT_RUN_DIR = TEMP_DIR / "current_run"
print("PROJECT_ROOT")
print(PROJECT_ROOT)
# Constants for peak matching
SUPPORTED_MATCHING_MODES = ['hung_dist_nn', 'euc_dist_all']
SUPPORTED_ERROR_TYPES = ['sum', 'avg']

# Configure logging
log_file = CURRENT_RUN_DIR / 'peak_matching.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging initialized. Log file: {log_file}")

# Add project root and tools directory to path for direct import
project_root_path = str(PROJECT_ROOT)
tools_path = str(TOOLS_DIR)
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)
    logger.info(f"Added to Python path: {project_root_path}")
if tools_path not in sys.path:
    sys.path.insert(0, tools_path)
    logger.info(f"Added to Python path: {tools_path}")

# Import required utilities
from utils_MMT.agents_code_v15_4_3 import generate_shifts_batch, add_atom_index_column
from utils_MMT import similarity_functions_exp_v15_4 as sfe

def extract_values(data_dict: Dict, key: str) -> List:
    """Extract values from potentially nested dictionary structures."""
    if not isinstance(data_dict, dict):
        return []
    if not data_dict.get(key):
        return []
    if isinstance(data_dict[key], dict):
        return list(data_dict[key].values())
    return data_dict[key]

async def compare_peaks(
    data1: Union[str, Dict],
    data2: Union[str, Dict],
    data_type: str,
    spectrum_type: str,
    config: Any,
    matching_mode: str = 'hung_dist_nn',
    error_type: str = 'sum'
) -> Dict:
    """Compare peaks between two inputs."""
    try:
        logger.info(f"Comparing peaks for {data_type} with spectrum type {spectrum_type}")
        
        # Validate input parameters
        if matching_mode not in SUPPORTED_MATCHING_MODES:
            raise ValueError(f"Unsupported matching mode: {matching_mode}. Must be one of {SUPPORTED_MATCHING_MODES}")
        if error_type not in SUPPORTED_ERROR_TYPES:
            raise ValueError(f"Unsupported error type: {error_type}. Must be one of {SUPPORTED_ERROR_TYPES}")
        
        logger.info(f"Using matching mode: {matching_mode}, error type: {error_type}")
        
        # Process data based on type
        if data_type == 'smiles_vs_smiles':
            # Initialize config for NMR generation with all required SGNN parameters
            config_dict = {
                'log_file': str(log_file),
                'output_directory': str(TEMP_DIR),
                'spectrum_type': spectrum_type.upper(),
                # SGNN required parameters
                'SGNN_gen_folder_path': str(TEMP_DIR / 'sgnn_gen'),
                'SGNN_size_filter': 550,  # Maximum molecular weight filter
                'SGNN_csv_save_folder': str(TEMP_DIR / 'sgnn_save'),
                'ML_dump_folder': str(TEMP_DIR / 'ml_dump'),
                'data_type': 'sgnn'
            }
            
            # Create necessary directories
            os.makedirs(config_dict['SGNN_gen_folder_path'], exist_ok=True)
            os.makedirs(config_dict['SGNN_csv_save_folder'], exist_ok=True)
            os.makedirs(config_dict['ML_dump_folder'], exist_ok=True)
            
            # Generate NMR data for both SMILES
            logger.info("Generating NMR data for SMILES comparison")
            nmr_data1, _, _, _ = generate_shifts_batch(config_dict, [data1])
            nmr_data2, _, _, _ = generate_shifts_batch(config_dict, [data2])
            peaks1 = nmr_data1[0][spectrum_type.upper()]
            peaks2 = nmr_data2[0][spectrum_type.upper()]
        
        elif data_type == 'smiles_vs_peaks':
            # Initialize config for NMR generation with all required SGNN parameters
            config_dict = {
                'log_file': str(log_file),
                'output_directory': str(TEMP_DIR),
                'spectrum_type': spectrum_type.upper(),
                # SGNN required parameters
                'SGNN_gen_folder_path': str(TEMP_DIR / 'sgnn_gen'),
                'SGNN_size_filter': 550,  # Maximum molecular weight filter
                'SGNN_csv_save_folder': str(TEMP_DIR / 'sgnn_save'),
                'ML_dump_folder': str(TEMP_DIR / 'ml_dump'),
                'data_type': 'sgnn'
            }
            
            # Create necessary directories
            os.makedirs(config_dict['SGNN_gen_folder_path'], exist_ok=True)
            os.makedirs(config_dict['SGNN_csv_save_folder'], exist_ok=True)
            os.makedirs(config_dict['ML_dump_folder'], exist_ok=True)
            
            # Generate NMR data for SMILES and use provided peaks
            logger.info("Generating NMR data for SMILES vs peaks comparison")
            nmr_data1, _, _, _ = generate_shifts_batch(config_dict, [data1])
            peaks1 = nmr_data1[0][spectrum_type.upper()]
            peaks2 = data2.get(spectrum_type.upper(), {})
        
        else:  # peaks_vs_peaks
            # Use provided peaks directly
            logger.info("Using provided peaks for comparison")
            peaks1 = data1.get(spectrum_type.upper(), {})
            peaks2 = data2.get(spectrum_type.upper(), {})

        logger.info("Converting data to DataFrames")
        
        # Handle potentially nested HSQC data
        if spectrum_type.upper() == 'HSQC':
            logger.info("Processing HSQC data")
            # Handle peaks1
            if isinstance(peaks1, dict):
                peaks1 = peaks1.get('HSQC', peaks1)
            if not peaks1:
                raise ValueError("No HSQC data found in peaks1")
                
            # Handle peaks2
            if isinstance(peaks2, dict):
                peaks2 = peaks2.get('HSQC', peaks2)
            if not peaks2:
                raise ValueError("No HSQC data found in peaks2")

        try:
            if spectrum_type.upper() in ['1H', '13C']:
                # Extract shifts for both peaks
                shifts1 = extract_values(peaks1, 'shifts')
                shifts2 = extract_values(peaks2, 'shifts')
                
                if not shifts1 or not shifts2:
                    logger.error(f"No shifts found in peaks for {spectrum_type}")
                    return {
                        'status': 'error',
                        'error': f'No shifts found for {spectrum_type}',
                        'type': 'data_error'
                    }
                
                # Convert to numpy arrays
                f1_ppm1 = np.array(shifts1, dtype=float)
                f1_ppm2 = np.array(shifts2, dtype=float)
                
                # Handle intensities based on spectrum type
                if spectrum_type.upper() == '1H':
                    # For 1H NMR, get intensities from data or use default 1.0
                    intensity1_raw = extract_values(peaks1, 'Intensity') or [1.0] * len(f1_ppm1)
                    intensity2_raw = extract_values(peaks2, 'Intensity') or [1.0] * len(f1_ppm2)
                else:  # 13C
                    # For 13C NMR, always use 1.0 for intensities
                    intensity1_raw = [1.0] * len(f1_ppm1)
                    intensity2_raw = [1.0] * len(f1_ppm2)
                
                # Convert intensities to numpy arrays
                intensity1_raw = np.array(intensity1_raw, dtype=float)
                intensity2_raw = np.array(intensity2_raw, dtype=float)
                
                # Store normalized intensity (all 1.0) for matching
                intensity1 = np.ones_like(intensity1_raw, dtype=float)
                intensity2 = np.ones_like(intensity2_raw, dtype=float)
                
                # Handle atom indices
                atom_idx1 = extract_values(peaks1, 'atom_index') or list(range(len(f1_ppm1)))
                atom_idx2 = extract_values(peaks2, 'atom_index') or list(range(len(f1_ppm2)))
                atom_idx1 = np.array(atom_idx1, dtype=int)
                atom_idx2 = np.array(atom_idx2, dtype=int)
                
                # Create DataFrames
                df1 = pd.DataFrame({
                    'shifts': f1_ppm1,
                    'Intensity': intensity1,  # Normalized intensity for matching
                    'actual_intensity': intensity1_raw,  # Actual intensity for reference
                    'atom_index': atom_idx1
                })
                
                df2 = pd.DataFrame({
                    'shifts': f1_ppm2,
                    'Intensity': intensity2,  # Normalized intensity for matching
                    'actual_intensity': intensity2_raw,  # Actual intensity for reference
                    'atom_index': atom_idx2
                })
            else:
                # For 2D spectra (HSQC, COSY)
                logger.info(f"Processing 2D spectrum type: {spectrum_type}")
                
                # Extract and validate F1 dimension
                f1_ppm1 = np.array(extract_values(peaks1, 'F1 (ppm)'), dtype=float)
                f1_ppm2 = np.array(extract_values(peaks2, 'F1 (ppm)'), dtype=float)
                if len(f1_ppm1) == 0 or len(f1_ppm2) == 0:
                    raise ValueError(f"Missing F1 dimension data for {spectrum_type}")
                
                # Extract and validate F2 dimension
                f2_ppm1 = np.array(extract_values(peaks1, 'F2 (ppm)'), dtype=float)
                f2_ppm2 = np.array(extract_values(peaks2, 'F2 (ppm)'), dtype=float)
                if len(f2_ppm1) == 0 or len(f2_ppm2) == 0:
                    raise ValueError(f"Missing F2 dimension data for {spectrum_type}")
                
                # Handle intensities with proper validation
                raw_intensity1 = extract_values(peaks1, 'Intensity')
                raw_intensity2 = extract_values(peaks2, 'Intensity')
                
                # Normalize intensities if present, otherwise use uniform weights
                if raw_intensity1:
                    intensity1 = np.array(raw_intensity1, dtype=float)
                    intensity1 = intensity1 / np.max(intensity1)  # Normalize to [0,1]
                else:
                    intensity1 = np.ones(len(f1_ppm1), dtype=float)
                
                if raw_intensity2:
                    intensity2 = np.array(raw_intensity2, dtype=float)
                    intensity2 = intensity2 / np.max(intensity2)  # Normalize to [0,1]
                else:
                    intensity2 = np.ones(len(f1_ppm2), dtype=float)
                
                # Handle atom indices
                atom_idx1 = np.array(extract_values(peaks1, 'atom_index') or list(range(len(f1_ppm1))), dtype=int)
                atom_idx2 = np.array(extract_values(peaks2, 'atom_index') or list(range(len(f1_ppm2))), dtype=int)
                
                # Create DataFrames with validated data
                df1 = pd.DataFrame({
                    'F1 (ppm)': f1_ppm1,
                    'F2 (ppm)': f2_ppm1,
                    'Intensity': intensity1,
                    'atom_index': atom_idx1
                })
                
                df2 = pd.DataFrame({
                    'F1 (ppm)': f1_ppm2,
                    'F2 (ppm)': f2_ppm2,
                    'Intensity': intensity2,
                    'atom_index': atom_idx2
                })
                
                logger.info(f"Created DataFrames for 2D spectra comparison: df1 shape {df1.shape}, df2 shape {df2.shape}")

        except (TypeError, ValueError) as e:
            logger.error(f"Error converting peak data: {str(e)}")
            logger.error(f"Peaks1 data: {peaks1}")
            logger.error(f"Peaks2 data: {peaks2}")
            raise

        logger.info("Calculating_similarity")
        # Calculate similarity using the unified calculation
        overall_error, df1_processed, df2_processed = sfe.unified_similarity_calculation(
            df1, df2, 
            spectrum_type.upper(),
            method=matching_mode,
            error_type=error_type
        )
        return {
            'status': 'success',
            'overall_error': float(overall_error),
            'spectrum_type': spectrum_type,
            'data_type': data_type,
            'error_type': error_type,
            'matching_mode': matching_mode,
            'matched_peaks': {
                'spectrum1': df1_processed.to_dict('records'),
                'spectrum2': df2_processed.to_dict('records')
            },
            'original_data': {
                'spectrum1': df1.to_dict('records'),
                'spectrum2': df2.to_dict('records')
            }
        }
            
    except Exception as e:
        logger.error(f"Error in peak comparison: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'status': 'error',
            'error': str(e),
            'type': 'comparison_error'
        }

async def process_peak_matching(input_path: str) -> Dict[str, Any]:
    """Process peak matching based on input data."""
    logger.info(f"Starting process_peak_matching with input: {input_path}")
    
    try:
        logger.info("Reading input data")
        with open(input_path) as f:
            input_data = json.load(f)
        logger.info(f"Input data loaded: {json.dumps(input_data, indent=2)}")
        
        # Extract parameters
        logger.info("Extracting parameters")
        data_type = input_data['type']
        spectra = input_data['spectra']
        matching_mode = input_data.get('matching_mode', 'hung_dist_nn')
        error_type = input_data.get('error_type', 'sum')
        
        # Process based on data type
        logger.info(f"Processing {data_type} comparison")
        
        results = {}
        for spectrum_type in spectra:
            if data_type == 'smiles_vs_smiles':
                # Initialize config for NMR generation with all required SGNN parameters
                config_dict = {
                    'log_file': str(log_file),
                    'output_directory': str(TEMP_DIR),
                    'spectrum_type': spectrum_type.upper(),
                    # SGNN required parameters
                    'SGNN_gen_folder_path': str(TEMP_DIR / 'sgnn_gen'),
                    'SGNN_size_filter': 550,  # Maximum molecular weight filter
                    'SGNN_csv_save_folder': str(TEMP_DIR / 'sgnn_save'),
                    'ML_dump_folder': str(TEMP_DIR / 'ml_dump'),
                    'data_type': 'sgnn'
                }
                
                # Create necessary directories
                os.makedirs(config_dict['SGNN_gen_folder_path'], exist_ok=True)
                os.makedirs(config_dict['SGNN_csv_save_folder'], exist_ok=True)
                os.makedirs(config_dict['ML_dump_folder'], exist_ok=True)
                
                result = await compare_peaks(
                    input_data['smiles1'],
                    input_data['smiles2'],
                    data_type,
                    spectrum_type,
                    config_dict,
                    matching_mode,
                    error_type
                )
            elif data_type == 'smiles_vs_peaks':
                # Initialize config for NMR generation with all required SGNN parameters
                config_dict = {
                    'log_file': str(log_file),
                    'output_directory': str(TEMP_DIR),
                    'spectrum_type': spectrum_type.upper(),
                    # SGNN required parameters
                    'SGNN_gen_folder_path': str(TEMP_DIR / 'sgnn_gen'),
                    'SGNN_size_filter': 550,  # Maximum molecular weight filter
                    'SGNN_csv_save_folder': str(TEMP_DIR / 'sgnn_save'),
                    'ML_dump_folder': str(TEMP_DIR / 'ml_dump'),
                    'data_type': 'sgnn'
                }
                
                # Create necessary directories
                os.makedirs(config_dict['SGNN_gen_folder_path'], exist_ok=True)
                os.makedirs(config_dict['SGNN_csv_save_folder'], exist_ok=True)
                os.makedirs(config_dict['ML_dump_folder'], exist_ok=True)
                
                result = await compare_peaks(
                    input_data['smiles'],
                    input_data['peaks'],
                    data_type,
                    spectrum_type,
                    config_dict,
                    matching_mode,
                    error_type
                )
            elif data_type == 'peaks_vs_peaks':
                result = await compare_peaks(
                    input_data['peaks1'],
                    input_data['peaks2'],
                    data_type,
                    spectrum_type,
                    None,  # TODO: Add config if needed
                    matching_mode,
                    error_type
                )
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
                
            results[spectrum_type] = result
        
        # Prepare final result
        final_result = {
            'status': 'success',
            'data': {
                'type': data_type,
                'spectra': spectra,
                'matching_mode': matching_mode,
                'error_type': error_type,
                'results': results
            }
        }
        
        # Save results
        logger.info("Saving results")
        result_path = Path(input_path).parent / 'results.json'
        with open(result_path, 'w') as f:
            json.dump(final_result, f, indent=2)
            
        logger.info(f"Results saved to {result_path}")
        return final_result
        
    except Exception as e:
        logger.error(f"Error in peak matching process: {str(e)}", exc_info=True)
        error_result = {
            'status': 'error',
            'error': str(e),
            'type': 'peak_matching_error'
        }
        
        # Save error result
        result_path = Path(input_path).parent / 'results.json'
        with open(result_path, 'w') as f:
            json.dump(error_result, f, indent=2)
            
        return error_result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python peak_matching_script.py <input_json_path>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        sys.exit(1)
        
    # Run the async function
    result = asyncio.run(process_peak_matching(input_path))


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/scripts/sgnn_script.py ---
import sys
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, Tuple, List, Optional
import shutil
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

# Import data generation module
try:
    import utils_MMT.data_generation_v15_4 as dg
    logger.info("Successfully imported data generation module")
except ImportError as e:
    logger.error(f"Failed to import data generation module: {e}")
    raise


class Config:
    """Configuration class that allows dot notation access to parameters."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getattr__(self, name):
        """Called when an attribute lookup has not found the attribute in the usual places."""
        raise AttributeError(f"Configuration has no attribute '{name}'. Available attributes: {', '.join(self.__dict__.keys())}")
    
    def to_dict(self):
        """Convert config back to dictionary if needed."""
        return self.__dict__


def simulate_nmr_data(config: Config) -> Config:
    """
    Simulate NMR data for molecules.
    
    Args:
        config: Configuration object containing simulation parameters
        
    Returns:
        Updated config
    """
    logger.info("Starting NMR data simulation")
    
    # Create simulation output directory if it doesn't exist
    output_dir = Path(config.SGNN_gen_folder_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    # Generate unique run ID if not provided
    if not hasattr(config, 'ran_num'):
        config.ran_num = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Run ID: {config.ran_num}")
    
    # Create run directory
    run_dir = output_dir / f"syn_{config.ran_num}"
    run_dir.mkdir(exist_ok=True)
    
    # Set paths in config
    config.SGNN_gen_folder_path = str(run_dir)
    
    # Run NMR data generation
    combined_df, data_1H, data_13C, data_COSY, data_HSQC, csv_1H_path, csv_13C_path, csv_COSY_path, csv_HSQC_path = dg.main_run_data_generation(config)
    
    # After data generation, copy files to temp folder with expected naming
    temp_folder = Path(config.csv_SMI_targets).parent  # This will be the _temp_folder
    
    # Copy files with expected naming convention
    file_mapping = {
        '1H': csv_1H_path,
        '13C': csv_13C_path,
        'COSY': csv_COSY_path,
        'HSQC': csv_HSQC_path
    }
    
    for nmr_type, source_path in file_mapping.items():
        target_path = temp_folder / f"nmr_prediction_{nmr_type}.csv"
        shutil.copy2(source_path, target_path)
        logger.info(f"Copied {nmr_type} NMR predictions to {target_path}")
        # Clean up source file after copying
        try:
            Path(source_path).unlink()
            logger.info(f"Cleaned up source file: {source_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up source file {source_path}: {str(e)}")
    
    # Clean up the temporary syn_ directory after copying results
    try:
        shutil.rmtree(run_dir)
        logger.info(f"Cleaned up temporary directory: {run_dir}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary directory {run_dir}: {str(e)}")
        

    return config

def read_input_csv(csv_path: str) -> pd.DataFrame:
    """
    Reads and validates input CSV file.
    
    Args:
        csv_path (str): Path to input CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing SMILES and sample-id columns
        
    Raises:
        ValueError: If required columns are missing
    """
    df = pd.read_csv(csv_path)
    required_cols = ['SMILES', 'sample-id']
    
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input CSV must contain columns: {required_cols}")
        
    return df

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run NMR simulation')
    parser.add_argument('--input_file', required=True, help='Path to input CSV file')
    args = parser.parse_args()
    
    # Get the directory containing the input file
    input_dir = os.path.dirname(os.path.abspath(args.input_file))
    
    # Configuration using input file location
    random_number = int(datetime.now().timestamp())
    config_dict = {
        'SGNN_gen_folder_path': input_dir,
        "SGNN_csv_save_folder": input_dir,
        'ran_num': str(random_number),
        "SGNN_size_filter": 550,
        'csv_SMI_targets': args.input_file,
        'SGNN_csv_gen_smi': args.input_file
    }
    
    # Convert dictionary to Config object
    config = Config(**config_dict)
    
    try:
        # Read input data
        logger.info(f"Reading input file: {args.input_file}")
        input_df = read_input_csv(config.csv_SMI_targets)
        logger.info(f"Input data: {input_df}")
        
        # Run simulation
        logger.info("Starting NMR simulation...")
        config = simulate_nmr_data(config)
        logger.info("NMR simulation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during simulation: {str(e)}", exc_info=True)
        sys.exit(1)

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/scripts/mol2mol_script.py ---
import sys
import os
import json
import random
from pathlib import Path
import pandas as pd
import argparse
from types import SimpleNamespace
from typing import Dict, Any, Optional
from rdkit import Chem

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

# Import Mol2Mol utilities
import utils_MMT.execution_function_v15_4 as ef

def load_json_dics(config_dir):
    """Load JSON dictionaries for tokenization"""
    with open(os.path.join(config_dir, 'itos.json'), 'r') as f:
        itos = json.load(f)
    with open(os.path.join(config_dir, 'stoi.json'), 'r') as f:
        stoi = json.load(f)
    with open(os.path.join(config_dir, 'stoi_MF.json'), 'r') as f:
        stoi_MF = json.load(f)
    with open(os.path.join(config_dir, 'itos_MF.json'), 'r') as f:
        itos_MF = json.load(f)    
    return itos, stoi, stoi_MF, itos_MF

def setup_molformer_config(params):
    """Create config namespace for Molformer"""
    config = {
        "MF_max_trails": params.max_trials,
        "MF_tanimoto_filter": params.tanimoto_filter,
        "MF_filter_higher": 1,  # True = generate more similar molecules
        "MF_delta_weight": params.delta_weight,
        "MF_generations": params.num_generations,
        "MF_model_path": params.model_path,
        "MF_vocab": params.vocab_path,
        "MF_csv_source_folder_location": os.path.dirname(params.input_csv),
        "MF_csv_source_file_name": Path(params.input_csv).stem,
        "MF_methods": ["MMP"], #scaffold , MMP
        "max_scaffold_generations": params.max_scaffold_generations,
    }
    
    return SimpleNamespace(**config)

def run_molformer(params):
    """Main function to run Molformer"""
    # Create output directory
    os.makedirs(params.output_dir, exist_ok=True)
    
    # Setup lock files
    running_lock = os.path.join(params.output_dir, "mol2mol_running.lock")
    complete_lock = os.path.join(params.output_dir, "mol2mol_complete.lock")
    
    try:
        # Load dictionaries
        itos, stoi, stoi_MF, itos_MF = load_json_dics(params.config_dir)
        
        # Setup configuration
        config = setup_molformer_config(params)
        
        # Run Molformer generation
        config, results_dict = ef.SMI_generation_MF(config, stoi, stoi_MF, itos, itos_MF)
        
        # Convert results to DataFrame and save
        df_results = pd.DataFrame.from_dict(results_dict, orient='index').transpose()
        output_file = os.path.join(params.output_dir, "generated_molecules.csv")
        df_results.to_csv(output_file, index=False)
        
        print(f"Successfully generated molecules. Results saved to: {output_file}")
        
        # Signal completion by creating complete lock and removing running lock
        if os.path.exists(running_lock):
            os.remove(running_lock)
        with open(complete_lock, 'w') as f:
            f.write('done')
        
        return df_results, output_file
        
    except Exception as e:
        print(f"Error occurred during molecule generation: {str(e)}")
        # Clean up lock files in case of error
        if os.path.exists(running_lock):
            os.remove(running_lock)
        if os.path.exists(complete_lock):
            os.remove(complete_lock)
        raise

def validate_smiles(smiles: str) -> bool:
    """Validate if a SMILES string can be parsed by RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def main(args):
    """
    Main function for Mol2Mol generation
    """
    # Input file validation
    input_file = Path(args.input_csv)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    # Read and validate standardized input
    df = pd.read_csv(input_file)
    if 'SMILES' not in df.columns or 'sample-id' not in df.columns:
        raise ValueError("Input file must contain SMILES and sample-id columns")
    
    # Validate SMILES strings
    invalid_smiles = []
    for idx, row in df.iterrows():
        if not validate_smiles(row['SMILES']):
            invalid_smiles.append((idx, row['SMILES'], row['sample-id']))
    
    if invalid_smiles:
        error_msg = "Invalid SMILES strings found:\n"
        for idx, smiles, sample_id in invalid_smiles:
            error_msg += f"Row {idx}: SMILES='{smiles}', sample-id='{sample_id}'\n"
        raise ValueError(error_msg)
    
    # Run Molformer generation
    df_results, output_file = run_molformer(args)
    print(f"Results saved to: {output_file}")
    print(f"Generated {len(df_results)} molecule analogues")
    print(f"Results saved to: {output_file}")    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Mol2Mol generation')
    parser.add_argument('--input_csv', required=True, help='Path to input CSV file')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--config_dir', required=True, help='Directory containing configuration files')
    parser.add_argument('--model_path', required=True, help='Path to the model')
    parser.add_argument('--vocab_path', required=True, help='Path to vocabulary file')
    parser.add_argument('--delta_weight', type=int, default=30, help='Delta weight parameter')
    parser.add_argument('--tanimoto_filter', type=float, default=0.2, help='Tanimoto filter threshold')
    parser.add_argument('--num_generations', type=int, default=10, help='Number of generations')
    parser.add_argument('--max_trials', type=int, default=100, help='Maximum number of trials')
    parser.add_argument('--max_scaffold_generations', type=int, default=10, help='Maximum scaffold generations')
    
    args = parser.parse_args()
    main(args)


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/scripts/stout_script.py ---
#!/usr/bin/env python3
"""
STOUT script for SMILES/IUPAC name conversion.
Handles both single and batch conversions in both directions (SMILES ↔ IUPAC).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Union
from STOUT import translate_forward, translate_reverse

def process_single_conversion(input_str: str, mode: str) -> Dict[str, Any]:
    """Process a single conversion in either direction."""
    try:
        if mode == "forward":
            result = translate_forward(input_str)
        else:
            result = translate_reverse(input_str)
            
        if not result:
            return {
                "status": "error",
                "error": f"Empty result from {'SMILES to IUPAC' if mode == 'forward' else 'IUPAC to SMILES'} conversion"
            }
            
        return {
            "status": "success",
            "input": input_str,
            "result": result,
            "mode": mode
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "input": input_str,
            "mode": mode
        }

def process_batch_conversion(input_data: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    """Process batch conversion for multiple molecules."""
    results = []
    for item in input_data:
        input_str = item.get('smiles' if mode == 'forward' else 'iupac')
        if not input_str:
            results.append({
                "status": "error",
                "error": f"Missing {'SMILES' if mode == 'forward' else 'IUPAC'} in input",
                "input_data": item
            })
            continue

        result = process_single_conversion(input_str, mode)
        if result["status"] == "success":
            # Merge original data with conversion result
            merged_result = {**item, **result}
            results.append(merged_result)
        else:
            results.append({**item, **result})

    return results

def main():
    parser = argparse.ArgumentParser(description='Convert between SMILES and IUPAC names using STOUT')
    parser.add_argument('--input', required=True, help='Input file path')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--mode', choices=['forward', 'reverse'], required=True,
                      help='Conversion direction: forward (SMILES→IUPAC) or reverse (IUPAC→SMILES)')
    parser.add_argument('--batch', action='store_true',
                      help='Process batch conversion from JSON list of molecules')
    
    args = parser.parse_args()
    
    try:
        # Read input
        with open(args.input, 'r') as f:
            if args.batch:
                input_data = json.load(f)
                if not isinstance(input_data, list):
                    raise ValueError("Batch input must be a JSON list of molecules")
                result = process_batch_conversion(input_data, args.mode)
            else:
                input_str = f.read().strip()
                result = process_single_conversion(input_str, args.mode)
        
        # Ensure output directory exists
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        
        # Write output
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
            
    except Exception as e:
        error_result = {
            "status": "error",
            "error": f"Processing failed: {str(e)}",
            "mode": args.mode
        }
        with open(args.output, 'w') as f:
            json.dump(error_result, f, indent=2)
        sys.exit(1)

if __name__ == "__main__":
    main()

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/scripts/imports_MMST.py ---
# Standard library imports
import argparse
import json
import os
import random
import sys
import time
from argparse import Namespace
from collections import defaultdict

# Third-party imports
## Data processing and scientific computing
import numpy as np
import pandas as pd
from tqdm import tqdm

## Machine learning and data visualization
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

## RDKit for cheminformatics
from rdkit import Chem
from rdkit.Chem import Draw, MolFromSmiles

# Local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Add chemprop to path
chemprop_ir_path = os.path.join(project_root, 'chemprop_IR')
if chemprop_ir_path not in sys.path:
    sys.path.append(chemprop_ir_path)

# Import utility functions
import utils_MMT.MT_functions_v15_4 as mtf
import utils_MMT.execution_function_v15_4 as ex
import utils_MMT.mmt_result_test_functions_15_4 as mrtf
import utils_MMT.helper_functions_pl_v15_4 as hf

from chemprop.train import make_predictions
from chemprop.parsing import modify_predict_args

# Helper functions
def load_json_dics():
    """Load JSON dictionaries for model vocabulary and mappings."""
    script_dir = os.path.dirname(__file__)
    mmt_dir = os.path.abspath(os.path.join(script_dir, '../../..'))  # Go up to MMT_explainability
    
    # Print paths for debugging
    print(f"Looking for JSON files in: {mmt_dir}")
    
    itos_path = os.path.join(mmt_dir, 'itos.json')
    stoi_path = os.path.join(mmt_dir, 'stoi.json')
    stoi_MF_path = os.path.join(mmt_dir, 'stoi_MF.json')
    itos_MF_path = os.path.join(mmt_dir, 'itos_MF.json')
    
    # Check if files exist
    for path in [itos_path, stoi_path, stoi_MF_path, itos_MF_path]:
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
    
    with open(itos_path, 'r') as f:
        itos = json.load(f)
    with open(stoi_path, 'r') as f:
        stoi = json.load(f)
    with open(stoi_MF_path, 'r') as f:
        stoi_MF = json.load(f)
    with open(itos_MF_path, 'r') as f:
        itos_MF = json.load(f)
    
    return itos, stoi, stoi_MF, itos_MF

def parse_arguments(hyperparameters):
    """Parse hyperparameters into a Namespace object."""
    # If already a Namespace, convert to dict
    if hasattr(hyperparameters, '__dict__'):
        hyperparameters = vars(hyperparameters)
    
    # Process the dictionary
    parsed_args = {
        key: val[0] if isinstance(val, (list, tuple)) else val 
        for key, val in hyperparameters.items()
    }
    return Namespace(**parsed_args)

def load_config(path):
    """Load configuration from a JSON file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def save_updated_config(config, path):
    """Save updated configuration to a JSON file."""
    config_dict = vars(config)
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=4)

def load_configs(config_dir=None):
    """Load both IR and main configurations."""
    if config_dir is None:
        script_dir = os.path.dirname(__file__)
        mmt_dir = os.path.abspath(os.path.join(script_dir, '../../..'))  # Go up to MMT_explainability
        config_dir = os.path.join(mmt_dir, 'utils_MMT')
    
    # Print paths for debugging
    print(f"Looking for config files in: {config_dir}")
    
    # Both config files are in config directory
    ir_config_path = os.path.join(config_dir, 'ir_config_V8.json')
    config_path = os.path.join(config_dir, 'config_V8.json')
    
    # Check if files exist
    if not os.path.exists(ir_config_path):
        print(f"Warning: IR config not found at: {ir_config_path}")
        raise FileNotFoundError(f"IR config file not found at {ir_config_path}")
        
    if not os.path.exists(config_path):
        print(f"Warning: Main config not found at: {config_path}")
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    print(f"Loading IR config from: {ir_config_path}")
    print(f"Loading main config from: {config_path}")
    
    IR_config_dict = load_config(ir_config_path)
    if IR_config_dict is None:
        raise FileNotFoundError(f"Failed to load IR config from {ir_config_path}")
    
    config_dict = load_config(config_path)
    if config_dict is None:
        raise FileNotFoundError(f"Failed to load main config from {config_path}")
    
    # Parse configs
    IR_config = parse_arguments(IR_config_dict)
    modify_predict_args(IR_config)
    config = parse_arguments(config_dict)
    
    return IR_config, config


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/scripts/chemformer_retro_script.py ---
#!/usr/bin/env python
from molbart.models import Chemformer
import hydra
import omegaconf
import pandas as pd
import sys
import os
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Run retrosynthesis predictions")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input text file with target SMILES")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for predictions")
    parser.add_argument("--n_beams", type=int, default=50, help="Number of beams for beam search")
    parser.add_argument("--n_unique_beams", type=int, default=-1, help="Number of unique beams to return. Use -1 for no limit")
    return parser.parse_args()

def create_config(args):
    """Create Chemformer configuration."""
    config = {
        'data_path': args.input_file,
        'vocabulary_path': args.vocab_path,
        'model_path': args.model_path,
        'task': 'backward_prediction',
        'output_sampled_smiles': args.output_file,
        'batch_size': args.batch_size,
        'n_beams': args.n_beams,
        "n_unique_beams": None if args.n_unique_beams == -1 else args.n_unique_beams,
        'n_gpus': 1 if torch.cuda.is_available() else 0,
        'train_mode': 'eval',
        'model_type': 'bart',
        'datamodule': ['SynthesisDataModule'],
        "device": "cuda" if torch.cuda.is_available() else "cpu",        
    }
    return OmegaConf.create(config)

def main():
    """Main function to run retrosynthesis predictions."""
    try:
        # Parse arguments
        args = parse_arguments()
        # logger.info("Creating Chemformer configuration")
        
        # Create config
        config = create_config(args)
        logger.info(f"Configuration created: {config}")
        
        # Initialize model
        logger.info("Initializing Chemformer model")
        chemformer = Chemformer(config)
        
        # Run prediction
        logger.info("Running predictions")
        smiles, log_lhs, target_smiles = chemformer.predict(dataset='full')
        
        logger.info(f"smiles {smiles}")
        logger.info(f"target_smiles {target_smiles}")

        # Save predictions to CSV
        logger.info("Saving predictions to CSV")
        predictions_df = pd.DataFrame({
            'target_smiles': target_smiles,
            'predicted_smiles': [s[0].item() if isinstance(s, (list, np.ndarray)) and len(s) > 0 else '' for s in smiles],
            'log_likelihood': [float(l[0]) if isinstance(l, (list, np.ndarray)) and len(l) > 0 else 0.0 for l in log_lhs],
            'all_predictions': [';'.join(map(str, s)) if isinstance(s, (list, np.ndarray)) else '' for s in smiles],
            'all_log_likelihoods': [';'.join(map(str, l)) if isinstance(l, (list, np.ndarray)) else '' for l in log_lhs]
        })
        
        # # Debug logging
        # logger.info(f"DataFrame shape: {predictions_df.shape}")
        # logger.info(f"DataFrame columns: {predictions_df.columns}")
        # if len(predictions_df) > 0:
        #     logger.info("First row of predictions:")
        #     logger.info(predictions_df.iloc[0].to_dict())
        
        predictions_df.to_csv(args.output_file, index=False)
        # logger.info(f"Predictions completed. Results saved to {args.output_file}")
        
    except Exception as e:
        logger.error(f"Error in retrosynthesis pipeline: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/scripts/_working_scripts_backup/chemformer_forward_script.py ---
#!/usr/bin/env python
from molbart.models import Chemformer
import hydra
import omegaconf
import pandas as pd
import sys
import os
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Run forward synthesis predictions")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input text file with reactant SMILES")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for predictions")
    parser.add_argument("--n_beams", type=int, default=50, help="Number of beams for beam search")
    return parser.parse_args()

def create_config(args):
    """Create Chemformer configuration."""
    config = {
        'data_path': args.input_file,
        'vocabulary_path': args.vocab_path,
        'model_path': args.model_path,
        "n_unique_beams":  args.n_unique_beams,
        'task': 'forward_prediction',
        'output_sampled_smiles': args.output_file,
        'batch_size': args.batch_size,
        'n_beams': args.n_beams,
        'n_gpus': 1 if torch.cuda.is_available() else 0,
        'train_mode': 'eval',
        'model_type': 'bart',
        'datamodule': ['SynthesisDataModule'],
        "device": "cuda" if torch.cuda.is_available() else "cpu",        

    }
    return OmegaConf.create(config)

def write_predictions(smiles, log_lhs, target_smiles, output_file):
    """Write predictions to CSV file."""
    try:
        # Debug logging
        logger.info(f"Number of predictions: {len(smiles)}")
        logger.info(f"Shape of first prediction: {np.array(smiles[0]).shape if smiles else 'empty'}")
        logger.info(f"Shape of first log_lhs: {np.array(log_lhs[0]).shape if log_lhs else 'empty'}")
        logger.info(f"Number of target smiles: {len(target_smiles)}")
        
        # Create DataFrame
        predictions_df = pd.DataFrame({
            'target_smiles': target_smiles,
            'predicted_smiles': [s[0].item() if isinstance(s, (list, np.ndarray)) and len(s) > 0 else '' for s in smiles],
            'log_likelihood': [float(l[0]) if isinstance(l, (list, np.ndarray)) and len(l) > 0 else 0.0 for l in log_lhs],
            'all_predictions': [';'.join(map(str, s)) if isinstance(s, (list, np.ndarray)) else '' for s in smiles],
            'all_log_likelihoods': [';'.join(map(str, l)) if isinstance(l, (list, np.ndarray)) else '' for l in log_lhs]
        })
        
        # Debug logging
        logger.info(f"DataFrame shape: {predictions_df.shape}")
        logger.info(f"DataFrame columns: {predictions_df.columns}")
        if len(predictions_df) > 0:
            logger.info("First row of predictions:")
            logger.info(predictions_df.iloc[0].to_dict())
        
        predictions_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error writing predictions: {str(e)}")
        raise

def main():
    """Main function to run forward synthesis predictions."""
    try:
        # Parse arguments
        args = parse_arguments()
        logger.info("Creating Chemformer configuration")
        
        # Create config
        config = create_config(args)
        logger.info(f"Configuration created: {config}")
        
        # Initialize model
        logger.info("Initializing Chemformer model")
        chemformer = Chemformer(config)
        
        # Run prediction
        logger.info("Running predictions")
        smiles, log_lhs, target_smiles = chemformer.predict(dataset='full')
        
        # Save predictions
        write_predictions(smiles, log_lhs, target_smiles, args.output_file)
        logger.info(f"Predictions completed. Results saved to {args.output_file}")
        
    except Exception as e:
        logger.error(f"Error in forward synthesis pipeline: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/scripts/_working_scripts_backup/chemformer_retro_script.py ---
#!/usr/bin/env python
from molbart.models import Chemformer
import hydra
import omegaconf
import pandas as pd
import sys
import os
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Run retrosynthesis predictions")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input text file with target SMILES")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for predictions")
    parser.add_argument("--n_beams", type=int, default=50, help="Number of beams for beam search")
    return parser.parse_args()

def create_config(args):
    """Create Chemformer configuration."""
    config = {
        'data_path': args.input_file,
        'vocabulary_path': args.vocab_path,
        'model_path': args.model_path,
        'task': 'backward_prediction',
        'output_sampled_smiles': args.output_file,
        'batch_size': args.batch_size,
        'n_beams': args.n_beams,
        "n_unique_beams":  args.n_unique_beams,
        'n_gpus': 1 if torch.cuda.is_available() else 0,
        'train_mode': 'eval',
        'model_type': 'bart',
        'datamodule': ['SynthesisDataModule'],
        "device": "cuda" if torch.cuda.is_available() else "cpu",        
    }
    return OmegaConf.create(config)

def main():
    """Main function to run retrosynthesis predictions."""
    try:
        # Parse arguments
        args = parse_arguments()
        logger.info("Creating Chemformer configuration")
        
        # Create config
        config = create_config(args)
        logger.info(f"Configuration created: {config}")
        
        # Initialize model
        logger.info("Initializing Chemformer model")
        chemformer = Chemformer(config)
        
        # Run prediction
        logger.info("Running predictions")
        smiles, log_lhs, target_smiles = chemformer.predict(dataset='full')
        
        # Debug logging
        logger.info(f"Number of predictions: {len(smiles)}")
        logger.info(f"Shape of first prediction: {np.array(smiles[0]).shape if smiles else 'empty'}")
        logger.info(f"Shape of first log_lhs: {np.array(log_lhs[0]).shape if log_lhs else 'empty'}")
        logger.info(f"Number of target smiles: {len(target_smiles)}")
        
        # Save predictions to CSV
        logger.info("Saving predictions to CSV")
        predictions_df = pd.DataFrame({
            'target_smiles': target_smiles,
            'predicted_smiles': [s[0].item() if isinstance(s, (list, np.ndarray)) and len(s) > 0 else '' for s in smiles],
            'log_likelihood': [float(l[0]) if isinstance(l, (list, np.ndarray)) and len(l) > 0 else 0.0 for l in log_lhs],
            'all_predictions': [';'.join(map(str, s)) if isinstance(s, (list, np.ndarray)) else '' for s in smiles],
            'all_log_likelihoods': [';'.join(map(str, l)) if isinstance(l, (list, np.ndarray)) else '' for l in log_lhs]
        })
        
        # Debug logging
        logger.info(f"DataFrame shape: {predictions_df.shape}")
        logger.info(f"DataFrame columns: {predictions_df.columns}")
        if len(predictions_df) > 0:
            logger.info("First row of predictions:")
            logger.info(predictions_df.iloc[0].to_dict())
        
        predictions_df.to_csv(args.output_file, index=False)
        logger.info(f"Predictions completed. Results saved to {args.output_file}")
        
    except Exception as e:
        logger.error(f"Error in retrosynthesis pipeline: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
 

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/scripts/_working_scripts_backup/Mol2Mol_script.py ---
import sys
import os
import json
import random
from pathlib import Path
import pandas as pd
import argparse
from types import SimpleNamespace
from typing import Dict, Any, Optional
from rdkit import Chem

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

# Import Mol2Mol utilities
import utils_MMT.execution_function_v15_4 as ef

def load_json_dics(config_dir):
    """Load JSON dictionaries for tokenization"""
    with open(os.path.join(config_dir, 'itos.json'), 'r') as f:
        itos = json.load(f)
    with open(os.path.join(config_dir, 'stoi.json'), 'r') as f:
        stoi = json.load(f)
    with open(os.path.join(config_dir, 'stoi_MF.json'), 'r') as f:
        stoi_MF = json.load(f)
    with open(os.path.join(config_dir, 'itos_MF.json'), 'r') as f:
        itos_MF = json.load(f)    
    return itos, stoi, stoi_MF, itos_MF

def setup_molformer_config(params):
    """Create config namespace for Molformer"""
    rand_num = str(random.randint(1, 10000000))
    
    config = {
        "MF_max_trails": params.max_trials,
        "MF_tanimoto_filter": params.tanimoto_filter,
        "MF_filter_higher": 1,  # True = generate more similar molecules
        "MF_delta_weight": params.delta_weight,
        "MF_generations": params.num_generations,
        "MF_model_path": params.model_path,
        "MF_vocab": params.vocab_path,
        "MF_csv_source_folder_location": os.path.dirname(params.input_csv),
        "MF_csv_source_file_name": Path(params.input_csv).stem,
        "MF_methods": ["MMP"], #scaffold , MMP
        "max_scaffold_generations": params.max_scaffold_generations,
        "ran_num": rand_num
    }
    
    return SimpleNamespace(**config)

def run_molformer(params):
    """Main function to run Molformer"""
    # Create output directory
    os.makedirs(params.output_dir, exist_ok=True)
    # Load dictionaries
    itos, stoi, stoi_MF, itos_MF = load_json_dics(params.config_dir)
    
    # Setup configuration
    config = setup_molformer_config(params)
    try:
        # Run Molformer generation
        config, results_dict = ef.SMI_generation_MF(config, stoi, stoi_MF, itos, itos_MF)
        
        # Convert results to DataFrame and save
        df_results = pd.DataFrame.from_dict(results_dict, orient='index').transpose()
        output_file = os.path.join(params.output_dir, f"generated_molecules_{config.ran_num}.csv")
        df_results.to_csv(output_file, index=False)
        
        print(f"Successfully generated molecules. Results saved to: {output_file}")
        return df_results, output_file
        
    except Exception as e:
        print(f"Error occurred during molecule generation: {str(e)}")
        raise

def validate_smiles(smiles: str) -> bool:
    """Validate if a SMILES string can be parsed by RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def main(args):
    """
    Main function for Mol2Mol generation
    """
    # Input file validation
    input_file = Path(args.input_csv)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    # Read and validate standardized input
    df = pd.read_csv(input_file)
    if 'SMILES' not in df.columns or 'sample-id' not in df.columns:
        raise ValueError("Input file must contain SMILES and sample-id columns")
    
    # Validate SMILES strings
    invalid_smiles = []
    for idx, row in df.iterrows():
        if not validate_smiles(row['SMILES']):
            invalid_smiles.append((idx, row['SMILES'], row['sample-id']))
    
    if invalid_smiles:
        error_msg = "Invalid SMILES strings found:\n"
        for idx, smiles, sample_id in invalid_smiles:
            error_msg += f"Row {idx}: SMILES='{smiles}', sample-id='{sample_id}'\n"
        raise ValueError(error_msg)
    
    # Run Molformer generation
    df_results, output_file = run_molformer(args)
    print(f"Results saved to: {output_file}")
    print(f"Generated {len(df_results)} molecule analogues")
    print(f"Results saved to: {output_file}")    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Mol2Mol generation')
    parser.add_argument('--input_csv', required=True, help='Path to input CSV file')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--config_dir', required=True, help='Directory containing configuration files')
    parser.add_argument('--model_path', required=True, help='Path to the model')
    parser.add_argument('--vocab_path', required=True, help='Path to vocabulary file')
    parser.add_argument('--delta_weight', type=int, default=30, help='Delta weight parameter')
    parser.add_argument('--tanimoto_filter', type=float, default=0.2, help='Tanimoto filter threshold')
    parser.add_argument('--num_generations', type=int, default=10, help='Number of generations')
    parser.add_argument('--max_trials', type=int, default=100, help='Maximum number of trials')
    parser.add_argument('--max_scaffold_generations', type=int, default=10, help='Maximum scaffold generations')
    
    args = parser.parse_args()
    main(args)


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/scripts/_working_scripts_backup/SGNN_script.py ---

import sys
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, Tuple, List, Optional
import shutil


# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

# Import SGNN utilities
import utils_MMT.data_generation_v15_4 as dg


class Config:
    """Configuration class that allows dot notation access to parameters."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getattr__(self, name):
        """Called when an attribute lookup has not found the attribute in the usual places."""
        raise AttributeError(f"Configuration has no attribute '{name}'. Available attributes: {', '.join(self.__dict__.keys())}")
    
    def to_dict(self):
        """Convert config back to dictionary if needed."""
        return self.__dict__


def simulate_nmr_data(config: Config) -> Config:
    """
    Simulate NMR data for molecules.
    
    Args:
        config: Configuration object containing simulation parameters
        
    Returns:
        Updated config
    """
    # Create simulation output directory if it doesn't exist
    output_dir = Path(config.SGNN_gen_folder_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique run ID if not provided
    if not hasattr(config, 'ran_num'):
        config.ran_num = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create run directory
    run_dir = output_dir / f"syn_{config.ran_num}"
    run_dir.mkdir(exist_ok=True)
    
    # Set paths in config
    config.SGNN_gen_folder_path = str(run_dir)
    
    # Run NMR data generation
    combined_df, data_1H, data_13C, data_COSY, data_HSQC, csv_1H_path, csv_13C_path, csv_COSY_path, csv_HSQC_path = dg.main_run_data_generation(config)
    
    # After data generation, copy files to temp folder with expected naming
    temp_folder = Path(config.csv_SMI_targets).parent  # This will be the _temp_folder
    
    # Copy files with expected naming convention
    file_mapping = {
        '1H': csv_1H_path,
        '13C': csv_13C_path,
        'COSY': csv_COSY_path,
        'HSQC': csv_HSQC_path
    }
    
    for nmr_type, source_path in file_mapping.items():
        target_path = temp_folder / f"nmr_prediction_{nmr_type}.csv"
        shutil.copy2(source_path, target_path)
        print(f"Copied {nmr_type} NMR predictions to {target_path}")

    return config

def read_input_csv(csv_path: str) -> pd.DataFrame:
    """
    Reads and validates input CSV file.
    
    Args:
        csv_path (str): Path to input CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing SMILES and sample-id columns
        
    Raises:
        ValueError: If required columns are missing
    """
    df = pd.read_csv(csv_path)
    required_cols = ['SMILES', 'sample-id']
    
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input CSV must contain columns: {required_cols}")
        
    return df

import random
def example_usage():
    """Example of how to use the NMR simulation function."""
    # Example configuration
    random_number = int(datetime.now().timestamp())
    config_dict  = {
        'SGNN_gen_folder_path': '/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/experiments',
        "SGNN_csv_save_folder":'/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/experiments',
        'ran_num': str(random_number),
        "SGNN_size_filter":550,
        'csv_SMI_targets': "/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/_temp_folder/current_molecule.csv",
        'SGNN_csv_gen_smi': "/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/_temp_folder/current_molecule.csv"  # Add this line
    }
    
    # Convert dictionary to Config object
    config = Config(**config_dict)

    # Read input data
    input_df = read_input_csv(config.csv_SMI_targets)  # Now we can use dot notation
    
    # Run simulation
    config = simulate_nmr_data(config)
    
    # Example of accessing results
    #print(f"Generated {len(combined_df)} simulated NMR spectra")
    
    # Paths to generated files are stored in config:
    print(f"Finished simulating NMR data")

if __name__ == "__main__":
    example_usage()

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/specialized/nmr_plot_agent.py ---
"""
NMR Plot Agent for handling spectral visualization requests.
"""
from typing import Dict, Any, Optional, Tuple, List
from ..base.base_agent import BaseAgent
from services.llm_service import LLMService
import json
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NMRPlotAgent(BaseAgent):
    """Agent specialized in handling NMR plot requests and visualization."""
    
    def __init__(self, llm_service: LLMService):
        super().__init__(
            name="NMR Plot",
            capabilities=["plot", "show", "display", "visualize", "spectrum", "nmr", 
                         "hsqc", "proton", "carbon", "cosy", "1h", "13c"]
        )
        self.llm_service = llm_service
        self.plot_types = {
            "hsqc": ["hsqc", "heteronuclear single quantum coherence"],
            "proton": ["proton", "1h", "1h-nmr", "hydrogen"],
            "carbon": ["carbon", "13c", "13c-nmr"],
            "cosy": ["cosy", "correlation spectroscopy"]
        }
        
        # Default plot parameters
        self.default_parameters = {
            "hsqc": {
                "title": "HSQC NMR Spectrum",
                "x_label": "F2 (ppm)",
                "y_label": "F1 (ppm)",
                "style": "default"
            },
            "proton": {
                "title": "1H NMR Spectrum",
                "x_label": "Chemical Shift (ppm)",
                "y_label": "Intensity",
                "style": "default"
            },
            "carbon": {
                "title": "13C NMR Spectrum",
                "x_label": "Chemical Shift (ppm)",
                "y_label": "Intensity",
                "style": "default"
            },
            "cosy": {
                "title": "COSY NMR Spectrum",
                "x_label": "F2 (ppm)",
                "y_label": "F1 (ppm)",
                "style": "default"
            }
        }

    async def process(self, message: str,context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a message to generate NMR visualizations.
        
        Args:
            message: The user message to process
            context: Additional context for processing (optional)
        """
        print("\n[NMR Plot Agent] ====== Starting Processing ======")
        print(f"[NMR Plot Agent] Message: {message}")
        model_choice = context.get('model_choice', 'gemini-flash')
        
        try:
            if not context or 'current_molecule' not in context:
                return {
                    "type": "error",
                    "content": "No molecule is currently selected. Please select a molecule first.",
                    "confidence": 0.0,
                    "reasoning": "Cannot generate NMR plot without a selected molecule"
                }
            
            logger.info(f"[NMR Plot Agent] Current molecule: {context['current_molecule']}")
            
            max_attempts = 3
            attempt = 1
            
            while attempt <= max_attempts:
                print(f"[NMR Plot Agent] Attempt {attempt} of {max_attempts}")
                
                # Use LLM for request analysis
                analysis_prompt = self._create_analysis_prompt(message, attempt > 1)
                analysis_response = await self.llm_service.get_completion(
                    message=analysis_prompt,
                    model=model_choice,
                    system="You are an NMR plot analysis assistant. Analyze plot requests and determine the appropriate visualization type and parameters. ONLY respond with the requested JSON format, no additional text."
                )
                
                plot_info = self._interpret_analysis(analysis_response)
                
                if plot_info.get("type") != "unknown":
                    if plot_info["confidence"] < 0.7:
                        return {
                            "type": "clarification",
                            "content": "I'm not quite sure which NMR plot you'd like to see. Could you specify if you want an HSQC, 1H (proton), 13C (carbon), or COSY spectrum?",
                            "confidence": plot_info["confidence"],
                            "reasoning": plot_info.get("reasoning", "Low confidence in plot type determination")
                        }
                    
                    # Create response
                    response = self._create_plot_response(plot_info["type"], plot_info["confidence"], plot_info.get("parameters"), context)
                    # Add reasoning to response
                    response["reasoning"] = plot_info.get("reasoning", "Successfully determined plot type and parameters")
                    print("[NMR Plot Agent] ====== Plot Request Processing Complete ======\n")
                    return response
                
                attempt += 1
            
            # If we've exhausted all attempts
            return {
                "type": "unknown",
                "confidence": 0.0,
                "content": "Unable to determine the appropriate NMR plot type after multiple attempts. Please rephrase your request.",
                "parameters": {},
                "reasoning": "Failed to determine plot type after multiple attempts"
            }
            
        except Exception as e:
            return {
                "type": "error",
                "content": f"Error processing plot request: {str(e)}",
                "confidence": 0.0,
                "reasoning": f"An error occurred while processing the plot request: {str(e)}"
            }

    def _create_plot_response(self, plot_type: str, confidence: float, custom_params: Optional[Dict] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a standardized plot response."""
        logger.info(f"Creating plot response for {plot_type} with confidence {confidence}")
        
        try:
            parameters = self.default_parameters.get(plot_type, {}).copy()

            # Get current SMILES from context if available
            current_smiles = None
            if context and 'current_molecule' in context:
                current_smiles = context['current_molecule'].get('SMILES')  # Note: Using uppercase SMILES
                print(f"[NMR Plot Agent] Current molecule SMILES from context: {current_smiles}")
            else:
                print("[NMR Plot Agent] No current molecule in context")
            
            # Generate NMR data using real data if available
            from utils.nmr_utils import generate_nmr_data
            nmr_data, is_random = generate_nmr_data(current_smiles, plot_type=plot_type, use_real_data=True)
            
            if not nmr_data:
                raise ValueError(f"No NMR data generated for {plot_type}")
            
            response = {
                "type": "plot",
                "plot_type": plot_type,
                "parameters": parameters,
                "confidence": confidence,
                "content": f"Displaying the {plot_type.upper()} spectrum for the current molecule."
            }
            
            if is_random:
                response["note"] = f"Note: Using simulated {plot_type.upper()} NMR spectrum as experimental data is not available."
            
            if plot_type in ['hsqc', 'cosy']:  # 2D NMR
                if len(nmr_data) == 3:
                    x_data, y_data, z_data = nmr_data
                    parameters['nmr_data'] = {
                        'x': x_data.tolist(),
                        'y': y_data.tolist(),
                        'z': z_data.tolist()
                    }
            else:  # 1D NMR
                x_data, y_data = nmr_data
                parameters['nmr_data'] = {
                    'x': x_data.tolist(),  # Convert numpy array to list
                    'y': y_data.tolist()   # Convert numpy array to list
                }

                response["content"] = f"Displaying the {plot_type.upper()} spectrum for the current molecule."
                # Only print non-data parameters to keep logs clean

            return response

        except Exception as e:
            logger.error(f"Error creating plot response: {str(e)}")
            return {
                "type": "error",
                "plot_type": plot_type,
                "parameters": self.default_parameters.get(plot_type, {}),
                "content": f"Error generating {plot_type.upper()} spectrum: {str(e)}",
                "confidence": 0.0,
                "reasoning": f"An error occurred while generating the plot response: {str(e)}"
            }
    
    def _create_analysis_prompt(self, message: str, is_retry: bool = False) -> str:
        """Create a prompt for analyzing the plot request."""
        retry_note = """
IMPORTANT: Previous attempt failed to generate valid JSON. 
Please ensure your response is ONLY valid JSON matching the required format below.
No additional text or explanations outside the JSON structure.""" if is_retry else ""

        return f"""Analyze this NMR plot request and determine the appropriate visualization.
Return ONLY a JSON response with NO additional text.{retry_note}

Rules for determining plot type:
1. If the request explicitly mentions '13C', 'carbon', or 'C13', use 'carbon' type
2. If the request explicitly mentions '1H', 'proton', or 'H1', use 'proton' type
3. If the request is ambiguous (just mentions 'NMR' or 'spectrum'), set confidence to 0.5
4. Default to 'carbon' type for requests about chemical shifts > 20 ppm
5. Default to 'proton' type for requests about chemical shifts < 10 ppm

Request: "{message}"

Available plot types:
- HSQC (Heteronuclear Single Quantum Coherence)
- 1H (Proton NMR)
- 13C (Carbon NMR)
- COSY (Correlation Spectroscopy)

Required JSON format:
{{
    "plot_request": {{
        "type": "plot_type",         # one of: hsqc, proton, carbon, cosy
        "confidence": 0.0,           # 0.0 to 1.0
        "reasoning": "explanation",   # Brief explanation of plot type choice. If confidence is low, explain why and suggest clearer prompts.
        "parameters": {{             # Optional parameters for the plot
            "title": "string",       # Custom title for the plot
            "x_label": "string",     # Custom x-axis label
            "y_label": "string",     # Custom y-axis label
            "style": "string"        # Plot style (default, publication, presentation)
        }}
    }}
}}"""
    
    def _interpret_analysis(self, analysis: str) -> Dict[str, Any]:
        """Interpret the LLM's analysis of the plot request."""
        try:
            # Clean and parse the JSON response
            content = analysis.get("content") if isinstance(analysis, dict) else analysis
            content = str(content).strip()
            
            # Clean markdown formatting if present
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json prefix
            content = content.replace("```", "").strip()
            
            # Parse JSON from content
            response = json.loads(content)

            # Extract plot request information
            plot_request = response.get("plot_request", {})
            plot_type = plot_request.get("type", "").lower()
            confidence = float(plot_request.get("confidence", 0.0))
            parameters = plot_request.get("parameters", {})
            reasoning = str(plot_request.get("reasoning", "No reasoning provided"))
            
            # Normalize plot type
            normalized_type = None
            for key, aliases in self.plot_types.items():
                if plot_type == key or plot_type in aliases:
                    normalized_type = key
                    break
            
            # Validate plot type
            if normalized_type is None:
                logger.info(f"[NMR Plot Agent] Unknown plot type: {plot_type}")
                return {
                    "type": "unknown",
                    "confidence": 0.0,
                    "parameters": {},
                    "reasoning": f"Could not determine NMR plot type from request: {plot_type}"
                }
            
            logger.info(f"[NMR Plot Agent] Normalized plot type '{plot_type}' to '{normalized_type}'")
            return {
                "type": normalized_type,
                "confidence": confidence,
                "parameters": parameters,
                "reasoning": reasoning
            }
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"[NMR Plot Agent] Error interpreting analysis: {str(e)}")
            return {
                "type": "unknown",
                "confidence": 0.0,
                "parameters": {},
                "reasoning": f"Error interpreting plot request: {str(e)}"
            }
    
    def get_available_plots(self) -> List[str]:
        """Return list of available plot types."""
        return list(self.plot_types.keys())
    
    def get_plot_description(self, plot_type: str) -> str:
        """Get a description of a specific plot type."""
        if plot_type not in self.plot_types:
            return "Unknown plot type"
            
        descriptions = {
            "hsqc": "2D NMR experiment showing correlations between directly bonded C-H pairs",
            "proton": "1D NMR spectrum showing hydrogen environments in the molecule",
            "carbon": "1D NMR spectrum showing carbon environments in the molecule",
            "cosy": "2D NMR experiment showing correlations between coupled protons"
        }
        return descriptions.get(plot_type, "No description available")


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/specialized/text_response_agent.py ---
from typing import Dict, Any, Optional
from ..base.base_agent import BaseAgent
from services.llm_service import LLMService

class TextResponseAgent(BaseAgent):
    """Agent for handling general text-based responses."""
    def __init__(self, llm_service: LLMService):
        capabilities = [
            "Natural language understanding",
            "General question answering",
            "Contextual responses",
            "Information retrieval"
        ]
        super().__init__(name="Text Response", capabilities=capabilities)
        self.llm_service = llm_service

    async def process(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a message and generate a text response.
        
        Args:
            message: The user message to process
            context: Additional context for processing (optional)
        """
        print("\n[Text Response Agent] ====== Starting Processing ======")
        print(f"[Text Response Agent] Message: {message}")
        print(f"[Text Response Agent] Context: {context}")
        model_choice = context.get('model_choice', 'gemini-flash')
        processing_mode = context.get('processing_mode', 'single')

        try:
            # Get response from LLM with proper system prompt
            print("[Text Response Agent] Getting response from LLM...")
            response = await self.llm_service.get_completion(
                message=message,
                model=model_choice,  
                system="You are an AI assistant specializing in chemical structure analysis and interpretation.",
                agent_name=self.name
            )
            
            # Handle error responses
            if response.startswith("Error in LLM completion:"):
                return {
                    "type": "error",
                    "content": {
                        "response": f"Error: {response}"
                    }
                }
            
            return response
            # return {
            #     "type": "text_response",
            #     "content": {
            #         "response": response
            #     }
            # }
            
        except Exception as e:
            error_msg = f"Error processing text response: {str(e)}"
            print(f"[Text Response Agent] {error_msg}")
            return {
                "type": "error",
                "content": {
                    "response": error_msg
                }
            }

    def can_handle(self, message):
        """Check if this agent should handle the message"""
        # This agent handles any text that doesn't match other specialized agents
        return True

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/specialized/analysis_agent.py ---
"""
Agent for analyzing molecular structures and spectral data.
"""
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
from enum import Enum
from pathlib import Path
from datetime import datetime

from ..base.base_agent import BaseAgent
from ..tools.analysis_enums import DataSource, RankingMetric
from ..tools.candidate_ranking_tool import CandidateRankingTool
from ..tools.structure_visualization_tool import StructureVisualizationTool
from ..tools.data_extraction_tool import DataExtractionTool
# from ..tools.molecular_visual_comparison_tool import MolecularVisualComparisonTool
from ..tools.spectral_comparison_tool import SpectralComparisonTool
from ..tools.final_analysis_tool import FinalAnalysisTool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Types of analysis that can be performed"""
    # Rank and select top candidate molecules
    TOP_CANDIDATES = "top_candidates"
    # Compare predicted vs experimental spectral data
    SPECTRAL_COMPARISON = "spectral_comparison"
    # LLM-based evaluation of spectral matches
    SPECTRAL_LLM_EVALUATION = "spectral_llm_evaluation"  
    # Compare structural features between molecules
    STRUCTURAL_COMPARISON = "structural_comparison"
    # Identify and analyze key functional groups
    FUNCTIONAL_GROUP = "functional_group"
    # Validate NMR coupling patterns
    COUPLING_PATTERN = "coupling_pattern"
    # Calculate overall confidence scores
    CONFIDENCE_SCORING = "confidence_scoring"
    # Check for contradictions in analysis results
    CONTRADICTION_CHECK = "contradiction_check"
    # Final comprehensive analysis
    FINAL_ANALYSIS = "final_analysis"

class AnalysisAgent(BaseAgent):
    """Agent responsible for analyzing molecular data using various analysis tools."""

    def __init__(self, llm_service: Optional[Any] = None):
        """Initialize the analysis agent."""
        capabilities = [
            "Top candidate selection and ranking",
            "Spectral analysis and comparison",
            "Structural comparison",
            "Functional group analysis",
            "Coupling pattern validation",
            "Confidence scoring",
            "Contradiction detection",
            "Final comprehensive analysis"
        ]
        super().__init__("Analysis Agent", capabilities)
        self.llm_service = llm_service
        self.data_tool = DataExtractionTool()
        self.ranking_tool = CandidateRankingTool(llm_service)
        self.structure_tool = StructureVisualizationTool()
        self.spectral_tool = SpectralComparisonTool(llm_service)
        self.final_tool = FinalAnalysisTool(llm_service)
        self.logger = logging.getLogger(__name__)

    def _create_temp_folder(self, sample_id: str) -> str:
        """Create a temporary folder for the current analysis run."""
        # Create base temp directory structure
        base_temp_dir = Path(__file__).resolve().parent.parent.parent / "_temp_folder"
        analysis_files_dir = base_temp_dir / "analysis_files"
        sample_temp_dir = analysis_files_dir / str(sample_id)
        
        # Create base temp folder and analysis_files if they don't exist
        base_temp_dir.mkdir(exist_ok=True)
        analysis_files_dir.mkdir(exist_ok=True)
        
        # Create a unique folder for this run using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_temp_dir = sample_temp_dir / timestamp
        run_temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created analysis directory at: {run_temp_dir}")
        return str(run_temp_dir)

    async def process(self, 
                    analysis_type: Union[AnalysisType, str],
                    workflow_data: Dict[str, Any],
                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a single analysis task based on the specified analysis type."""
        context = context or {}
        
        try:
            # Convert string to enum if needed
            if isinstance(analysis_type, str):
                analysis_type = AnalysisType(analysis_type)
                
            if analysis_type == AnalysisType.TOP_CANDIDATES:
                return await self._analyze_top_candidates(workflow_data, context)
            elif analysis_type == AnalysisType.SPECTRAL_COMPARISON:
                return await self._analyze_spectral_comparison(workflow_data, context)
            elif analysis_type == AnalysisType.SPECTRAL_LLM_EVALUATION:
                spectral_result = await self._analyze_spectral_llm_evaluation(workflow_data, context)
                # After spectral LLM evaluation, perform final analysis
            elif analysis_type == AnalysisType.FINAL_ANALYSIS:
                return await self._analyze_final_results(workflow_data, context)
               
            # elif analysis_type == AnalysisType.STRUCTURAL_COMPARISON:
            #     return await self._analyze_structural_comparison(workflow_data, context)
            # elif analysis_type == AnalysisType.FUNCTIONAL_GROUP:
            #     return await self._analyze_functional_groups(workflow_data, context)
            # elif analysis_type == AnalysisType.COUPLING_PATTERN:
            #     return await self._analyze_coupling_patterns(workflow_data, context)
            # elif analysis_type == AnalysisType.CONFIDENCE_SCORING:
            #     return await self._calculate_confidence_scores(workflow_data, context)
            # elif analysis_type == AnalysisType.CONTRADICTION_CHECK:
            #     return await self._check_contradictions(workflow_data, context)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
                
        except Exception as e:
            logger.error(f"Error in analysis process: {str(e)}")
            return {
                'type': 'error',
                'content': str(e),
                'metadata': {
                    'agent': 'ANALYSIS',
                    'confidence': 0.0,
                    'reasoning': 'Analysis process failed'
                }
            }

    async def process_all(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all analysis steps sequentially.
        
        Args:
            task_data: Dictionary containing task_input (with molecule_data and step_outputs) and context
        """
        try:
            # Add validation
            if not task_data.get('task_input'):
                raise KeyError("task_input missing from task_data")
            if not task_data['task_input'].get('workflow_data'):
                raise KeyError("workflow_data missing from task_input")
                
            workflow_data = task_data['task_input']["workflow_data"]
            if not workflow_data.get('molecule_data'):
                raise KeyError("molecule_data missing from workflow_data")        

            # Extract data from task input
            workflow_data = task_data['task_input']["workflow_data"]
            molecule_data = task_data['task_input']["workflow_data"]['molecule_data']
            context = task_data.get('context', {})
            # Log the result of each analysis step
            # logger.info(f"molecule_data {molecule_data}")
            # logger.info(f"context {context} ")
          
            # # Add step outputs to context if available
            # if 'step_outputs' in task_data['task_input']:
            #     context['step_outputs'] = task_data['task_input']['step_outputs']
            # logger.info(f"context {context} ")

            all_results = {}
            
            # Create temp folder for this analysis run
            sample_id = molecule_data.get('sample_id', 'unknown_sample')
            analysis_run_folder = self._create_temp_folder(sample_id)
            context['analysis_run_folder'] = analysis_run_folder
            context['from_orchestrator'] = True
            # logger.info(f"___context {context} ")

            # Sequential analysis pipeline
            analysis_steps = [
                (AnalysisType.TOP_CANDIDATES, self._analyze_top_candidates),
                (AnalysisType.SPECTRAL_COMPARISON, self._analyze_spectral_comparison),
                (AnalysisType.SPECTRAL_LLM_EVALUATION, self._analyze_spectral_llm_evaluation),
                (AnalysisType.FINAL_ANALYSIS, self._analyze_final_results)
                # (AnalysisType.STRUCTURAL_COMPARISON, self._analyze_structural_comparison),
                # (AnalysisType.FUNCTIONAL_GROUP, self._analyze_functional_groups),
                # (AnalysisType.COUPLING_PATTERN, self._analyze_coupling_patterns),
                # (AnalysisType.CONFIDENCE_SCORING, self._calculate_confidence_scores),
                # (AnalysisType.CONTRADICTION_CHECK, self._check_contradictions)
            ]

            for analysis_type, analysis_func in analysis_steps:
                try:
                    logger.info(f"Starting {analysis_type.value} analysis")
                    result = await analysis_func(workflow_data, context)
                    all_results[analysis_type.value] = result
                    
                    # Update context with results for next analysis
                    context['previous_analysis'] = context.get('previous_analysis', {})
                    context['previous_analysis'][analysis_type.value] = result
                    
                except Exception as e:
                    logger.error(f"Error in {analysis_type.value} analysis: {str(e)}")
                    all_results[analysis_type.value] = {
                        'status': 'error',
                        'error': str(e)
                    }

            return {
                'type': 'success',
                'content': all_results,
                'metadata': {
                    'agent': 'ANALYSIS',
                    'confidence': 1.0,
                    'reasoning': 'Completed all analysis steps'
                }
            }

        except Exception as e:
            logger.error(f"Error in process_all: {str(e)}")
            return {
                'type': 'error',
                'content': str(e),
                'metadata': {
                    'agent': 'ANALYSIS',
                    'confidence': 0.0,
                    'reasoning': 'Failed to complete analysis pipeline'
                }
            }


    async def _analyze_top_candidates(self, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze top candidate structures and prepare them for further analysis.
        """
        return await self.ranking_tool.analyze_top_candidates(
            workflow_data=workflow_data,
            context=context,
            data_tool=self.data_tool,
            ranking_tool=self.ranking_tool,
        )


    async def _analyze_spectral_comparison(self, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Compare spectral data across different types."""
        # Set default number of candidates if not in context
        if 'num_candidates' not in context:
            context['num_candidates'] = 3  # Default to 2 candidates
            logger.info(f"Setting default number of candidates to {context['num_candidates']}")
            
        return await self.spectral_tool.analyze_spectral_comparison(
            workflow_data=workflow_data,
            context=context,
            data_tool=self.data_tool,
            spectral_tool=self.spectral_tool,
            llm_service=self.llm_service
        )

    async def _analyze_spectral_llm_evaluation(self, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM-based evaluation of how well candidate structures match experimental NMR spectra.
        Uses vision capabilities to analyze structural features against spectral patterns.
        """
        try:
            self.logger.info("Starting spectral_llm_evaluation analysis")
            
            return await self.structure_tool.analyze_spectral_llm_evaluation(
                workflow_data=workflow_data,
                context=context,
                data_tool=self.data_tool,
                ranking_tool=self.ranking_tool,
                llm_service=self.llm_service
            )
        except Exception as e:
            self.logger.error(f"Error in spectral LLM evaluation: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    # async def _analyze_structural_comparison(self, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    #     """Compare structural features between molecules."""
    #     try:
    #         # Initialize molecular comparison tool if not already present
    #         if not hasattr(self, 'molecular_comparison_tool'):
    #             self.molecular_comparison_tool = MolecularVisualComparisonTool()

    #         return await self.molecular_comparison_tool.analyze_structural_comparison(
    #             workflow_data=workflow_data,
    #             context=context
    #         )
    #     except Exception as e:
    #         logger.error(f"Error in structural comparison analysis: {str(e)}")
    #         return {
    #             'status': 'error',
    #             'error': str(e)
    #         }

    async def _analyze_final_results(self, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final analysis using the FinalAnalysisTool."""
        return await self.final_tool.analyze_final_results(
            workflow_data=workflow_data,
            context=context,
            data_tool=self.data_tool,
            llm_service=self.llm_service
        )
        
    # async def _analyze_functional_groups(self, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    #     """Analyze functional groups and their consistency."""
    #     # TODO: Implement functional group analysis
    #     pass

    # async def _analyze_coupling_patterns(self, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    #     """Analyze and validate coupling patterns."""
    #     # TODO: Implement coupling pattern analysis
    #     pass

    
    # async def _calculate_confidence_scores(self, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    #     """Calculate confidence scores for predictions."""
    #     # TODO: Implement confidence scoring
    #     pass

    # async def _check_contradictions(self, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    #     """Check for contradictions in spectral and structural data."""
    #     # TODO: Implement contradiction checking
    #     pass

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/specialized/tool_agent.py ---
"""
Agent for managing and coordinating various tools in the system.
"""
from typing import Dict, Any, Optional, List
import json
import os
from pathlib import Path
import uuid
from datetime import datetime
import pandas as pd
from ..base.base_agent import BaseAgent
from ..tools.nmr_simulation_tool import NMRSimulationTool
from ..tools.mol2mol_tool import Mol2MolTool
from ..tools.retro_synthesis_tool import RetrosynthesisTool
from ..tools.forward_synthesis_tool import ForwardSynthesisTool
from ..tools.peak_matching_tool import EnhancedPeakMatchingTool
# from ..tools.molecular_visual_comparison_tool import MolecularVisualComparisonTool
from ..tools.threshold_calculation_tool import ThresholdCalculationTool
from ..tools.candidate_analyzer_tool import CandidateAnalyzerTool
from ..tools.mmst_tool import MMSTTool
from ..tools.stout_tool import STOUTTool
from .config.tool_descriptions import TOOL_DESCRIPTIONS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolAgent(BaseAgent):
    """Agent responsible for managing and coordinating tool operations."""

    def __init__(self, llm_service):
        """Initialize the tool agent with available tools."""
        capabilities = [
            "NMR spectrum simulation",
            "Molecular analogue generation",
            "Retrosynthesis prediction",
            "Forward synthesis prediction",
            "Peak matching and comparison",
            "Threshold calculation",
            "Tool coordination",
            "Tool execution management",
            "SMILES/IUPAC name conversion"
        ]
        super().__init__("Tool Agent", capabilities)
        self.llm_service = llm_service
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing ToolAgent")
        
        # Initialize tools
        self.tools = {}
        self._initialize_tools()
        self.logger.info(f"Available tools after initialization: {list(self.tools.keys())}")

    def _initialize_tools(self):
        """Initialize and register available tools."""
        try:
            # Register standard tools
            self.logger.info("Registering standard tools...")
            self.tools['nmr_simulation'] = NMRSimulationTool()
            self.tools['mol2mol'] = Mol2MolTool()
            self.tools['retro_synthesis'] = RetrosynthesisTool()
            self.tools['forward_synthesis'] = ForwardSynthesisTool()
            self.tools['peak_matching'] = EnhancedPeakMatchingTool()
            # self.tools['molecular_visual_comparison'] = MolecularVisualComparisonTool()
            self.tools['threshold_calculation'] = ThresholdCalculationTool()
            self.tools['forward_candidate_analysis'] = CandidateAnalyzerTool(analysis_type='forward')
            self.tools['mol2mol_candidate_analysis'] = CandidateAnalyzerTool(analysis_type='mol2mol')
            self.tools['mmst_candidate_analysis'] = CandidateAnalyzerTool(analysis_type='mmst')
            self.tools['mmst'] = MMSTTool()
            self.tools['stout'] = STOUTTool()
            
            self.logger.info(f"Successfully registered all tools: {list(self.tools.keys())}")
        except Exception as e:
            self.logger.error(f"Error during tool initialization: {str(e)}", exc_info=True)
            raise

    async def process(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a message and route it to the appropriate tool.
        
        Args:
            message: The user message to process
            context: Additional context for processing (optional)
        """
        try:
            self.logger.info(f"Processing message: {message}")
            self.logger.debug(f"Available tools: {list(self.tools.keys())}")
            
            # Update context with processing mode
            context = context or {}
            
            # Get model choice from context, default to gemini-flash
            model_choice = context.get('model_choice', 'gemini-flash')
            tool_name = await self._determine_tool_llm(message, model_choice)
            self.logger.info(f"Selected tool: {tool_name}")
            # self.logger.info(f"context: {context}")

            if tool_name == 'nmr_simulation':
                # Get current molecule data and ensure sample_id
                molecule_data = context.get('current_molecule', {})
                sample_id = molecule_data.get('sample_id')
                context['use_slurm'] = False

                if not sample_id:
                    self.logger.error("No sample_id found in current molecule data")
                    return {'status': 'error', 'message': 'No sample_id found in current molecule data'}

                # Create context with just the necessary data
                nmr_context = {
                    'smiles': molecule_data.get('smiles'),
                    'sample_id': sample_id
                }
                
                self.logger.info(f"Running NMR simulation for sample {sample_id}")
                result = await self.tools['nmr_simulation'].simulate_nmr(sample_id, nmr_context)    
                
                return self._format_tool_response(result, "NMR simulation completed")

            elif tool_name == 'mol2mol':
                # Force local execution by setting use_slurm to False in context
                mol2mol_context = context.copy() if context else {}
                mol2mol_context['use_slurm'] = False

                if self._current_processing_type == 'batch':
                    self.logger.info("Processing all molecules for mol2mol generation")
                    result = await self.tools['mol2mol'].process_all_molecules()
                else:
                    # Single molecule mode requires current molecule
                    if not context or 'current_molecule' not in context:
                        return {
                            "type": "error",
                            "content": "No molecule data available. Please load or select a molecule first.",
                            "metadata": {
                                "agent": "TOOL_AGENT",
                                "confidence": 0.0,
                                "reasoning": "Missing required molecule data"
                            }
                        }
                    
                    self.logger.info("Processing single molecule for mol2mol generation")
                    molecule_data = context['current_molecule']
                    # self.logger.info(f"Molecule data received: {molecule_data}")
                    self.logger.info(f"Molecule data type: {type(molecule_data)}")
                    if isinstance(molecule_data, dict):
                        self.logger.info(f"Molecule data keys: {molecule_data.keys()}")
                        self.logger.info(f"SMILES present (uppercase): {'SMILES' in molecule_data}")
                        self.logger.info(f"SMILES present (lowercase): {'smiles' in molecule_data}")
                        smiles = molecule_data.get('SMILES') or molecule_data.get('smiles')
                        if smiles:
                            self.logger.info(f"SMILES value: {smiles}")
                            self.logger.info(f"Sample ID: {molecule_data.get('sample_id')}")
                    
                    # Check for either uppercase or lowercase SMILES
                    smiles = molecule_data.get('SMILES') or molecule_data.get('smiles') if isinstance(molecule_data, dict) else None
                    sample_id = molecule_data.get('sample_id') or molecule_data.get('sample-id') if isinstance(molecule_data, dict) else None
                    if smiles:
                        result = await self.tools['mol2mol'].generate_analogues(
                            smiles,
                            sample_id
                        )
                        self.logger.info(f"Mol2Mol result: {result}")
                        
                        # If successful, update master data and return simplified response
                        if result.get('status') == 'success':
                            # await self._update_master_data_with_mol2mol(result, molecule_data)
                            return {
                                'type': 'success',
                                'content': {'status': 'success'},  # Simplified response that satisfies orchestrator validation
                                'predictions': result.get('predictions', {})
                            }
                        else:
                            return {
                                'type': 'error',
                                'content': result.get('message', 'Unknown error in mol2mol generation'),
                                'metadata': {
                                    'agent': 'TOOL_AGENT',
                                    'confidence': 0.0,
                                    'reasoning': 'Mol2mol generation failed'
                                }
                            }
                    else:
                        return {
                            "type": "error",
                            "content": "Invalid molecule data format. Expected dictionary with SMILES key.",
                            "metadata": {
                                "agent": "TOOL_AGENT",
                                "confidence": 0.0,
                                "reasoning": "Invalid data format"
                            }
                        }
                
            elif tool_name == 'retro_synthesis':
                # Get current molecule data and ensure sample_id
                molecule_data = context.get('current_molecule') if context else None
                retro_context = context.copy() if context else {}
                retro_context['use_slurm'] = False
                
                if not molecule_data:
                    return {
                        "type": "error", 
                        "content": "No molecule data available. Please select a molecule first.",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Missing required molecule data"
                        }
                    }
                    
                if isinstance(molecule_data, dict) and 'sample_id' not in molecule_data:
                    return {
                        "type": "error",
                        "content": "No sample ID found. Please ensure molecule has a sample ID.",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Missing required sample ID"
                        }
                    }
                molecule_data
                result = await self.tools['retro_synthesis'].predict_retrosynthesis(molecule_data, retro_context)
                return self._format_tool_response(result, "Retrosynthesis prediction completed")
            
            elif tool_name == 'forward_synthesis':
                # Get current molecule data and ensure sample_id
                molecule_data = context.get('current_molecule') if context else None
                forward_context = context.copy() if context else {}
                forward_context['use_slurm'] = False
                
                if not molecule_data:
                    return {
                        "type": "error", 
                        "content": "No molecule data available. Please select a molecule first.",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Missing required molecule data"
                        }
                    }
                    
                if isinstance(molecule_data, dict) and 'sample_id' not in molecule_data:
                    return {
                        "type": "error",
                        "content": "No sample ID found. Please ensure molecule has a sample ID.",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Missing required sample ID"
                        }
                    }
                
                result = await self.tools['forward_synthesis'].predict_forward_synthesis(molecule_data, forward_context)
                return self._format_tool_response(result, "Forward synthesis prediction completed")
 
            elif tool_name == 'peak_matching':
                self.logger.info("=== Peak Matching Context ===")
                
                if not context or 'current_molecule' not in context:
                    return {
                        "type": "error",
                        "content": "No molecule data available for peak matching",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Missing required molecule data"
                        }
                    }
                
                molecule_data = context['current_molecule']
                sample_id = molecule_data.get('sample_id')
                if not sample_id:
                    return {
                        "type": "error",
                        "content": "No sample ID found in molecule data",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Missing sample ID"
                        }
                    }
                
                # Prepare context with comparison mode info
                peak_context = self._prepare_peak_matching_context(message, context)
                
                # Log the comparison mode
                comparison_mode = peak_context.get('comparison_mode', 'default')
                self.logger.info(f"Using comparison mode: {comparison_mode}")
                
                # Use new process_peaks method that handles intermediate files
                result = await self.tools['peak_matching'].process_peaks(sample_id, peak_context)
                
                # Format response based on comparison mode
                success_message = f"Peak matching completed using {comparison_mode} mode"
                return self._format_tool_response(result, success_message)
            
            # elif tool_name == 'molecular_visual_comparison':
            #     if not context:
            #         return {
            #             "type": "error",
            #             "content": "No context provided for molecular comparison.",
            #             "metadata": {
            #                 "agent": "TOOL_AGENT",
            #                 "confidence": 0.0,
            #                 "reasoning": "Missing required context"
            #             }
            #         }
                
            #     # Determine input type and prepare data based on LLM analysis
            #     input_data = {}
                
            #     # Batch processing with CSV
            #     if self._current_processing_type == 'batch':
            #         if 'guess_smiles_csv' not in context:
            #             return {
            #                 "type": "error",
            #                 "content": "CSV file with guess molecules not provided for batch processing.",
            #                 "metadata": {
            #                     "agent": "TOOL_AGENT",
            #                     "confidence": 0.0,
            #                     "reasoning": "Missing required CSV file"
            #                 }
            #             }
                    
            #         if self._current_comparison_type == 'target' and 'target_smiles' in context:
            #             input_data = {
            #                 'comparison_type': 'batch_vs_target',
            #                 'guess_smiles_csv': context['guess_smiles_csv'],
            #                 'target_smiles': context['target_smiles']
            #             }
            #         elif self._current_comparison_type == 'starting_materials' and 'starting_materials_smiles' in context:
            #             input_data = {
            #                 'comparison_type': 'batch_vs_starting',
            #                 'guess_smiles_csv': context['guess_smiles_csv'],
            #                 'starting_materials_smiles': context['starting_materials_smiles']
            #             }
            #         else:
            #             return {
            #                 "type": "error",
            #                 "content": "Missing target or starting materials SMILES for batch comparison.",
            #                 "metadata": {
            #                     "agent": "TOOL_AGENT",
            #                     "confidence": 0.0,
            #                     "reasoning": "Missing required SMILES"
            #                 }
            #             }
                
            #     # Single molecule comparison
            #     else:
            #         if 'guess_smiles' not in context:
            #             return {
            #                 "type": "error",
            #                 "content": "Guess molecule SMILES not provided.",
            #                 "metadata": {
            #                     "agent": "TOOL_AGENT",
            #                     "confidence": 0.0,
            #                     "reasoning": "Missing required SMILES"
            #                 }
            #             }
                    
            #         if self._current_comparison_type == 'target' and 'target_smiles' in context:
            #             input_data = {
            #                 'comparison_type': 'guess_vs_target',
            #                 'guess_smiles': context['guess_smiles'],
            #                 'target_smiles': context['target_smiles']
            #             }
            #         elif self._current_comparison_type == 'starting_materials' and 'starting_materials_smiles' in context:
            #             input_data = {
            #                 'comparison_type': 'guess_vs_starting',
            #                 'guess_smiles': context['guess_smiles'],
            #                 'starting_materials_smiles': context['starting_materials_smiles']
            #             }
            #         else:
            #             return {
            #                 "type": "error",
            #                 "content": "Missing target or starting materials SMILES for comparison.",
            #                 "metadata": {
            #                     "agent": "TOOL_AGENT",
            #                     "confidence": 0.0,
            #                     "reasoning": "Missing required SMILES"
            #                 }
            #             }
                
            #     # Create run directory with unique ID
            #     run_id = str(uuid.uuid4())
            #     run_dir = Path("_temp_folder") / "molecular_visual_comparison" / run_id
            #     run_dir.mkdir(parents=True, exist_ok=True)

            #     # Prepare context with comparison type
            #     comparison_context = {
            #         **context,
            #         'run_dir': str(run_dir),
            #         'run_id': run_id,
            #         'comparison_type': input_data['comparison_type']
            #     }

            #     self.logger.info(f"Running molecular visual comparison - {input_data['comparison_type']}")
            #     result = await self.tools['molecular_visual_comparison'].compare_structures(
            #         input_data=input_data,
            #         context=comparison_context
            #     )
            #     return self._format_tool_response(result, "Molecular visual comparison completed")

            elif tool_name == 'threshold_calculation':
                self.logger.info("[TOOL_AGENT] Initiating threshold calculation process")
                if not context or 'current_molecule' not in context:
                    return {
                        "type": "error",
                        "content": "Missing molecule data for threshold calculation",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Required molecule data context not provided"
                        }
                    }
                    
                try:
                    current_molecule = context['current_molecule']
                    threshold_tool = self.tools['threshold_calculation']
                    self.logger.info(f"[TOOL_AGENT] Calculating thresholds for molecule: {current_molecule.get('sample_id')}")
                    self.logger.info(f"[TOOL_AGENT] Calculating current_molecule: {current_molecule}")

                    sample_id = current_molecule.get('sample_id')

                    # Call calculate_threshold directly with lowercase smiles key
                    result = await threshold_tool.calculate_threshold(
                        sample_id=sample_id,  # Changed from SMILES to smiles
                        context=context
                    )
                    
                    # self.logger.info(f"[TOOL_AGENT] Threshold calculation completed: {result}")
                    return {
                        "type": "success",
                        "content": {'threshold_data': result},
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 1.0,
                            "reasoning": "Threshold calculation completed successfully"
                        }
                    }
                except Exception as e:
                    error_msg = f"[TOOL_AGENT] Threshold calculation failed: {str(e)}"
                    self.logger.error(error_msg)
                    return {
                        "type": "error",
                        "content": str(e),
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": error_msg
                        }
                    }
                
            elif 'candidate_analysis' in tool_name:  # Handle any type of candidate analysis
                self.logger.info(f"Starting {tool_name} processing")
                if not context or 'current_molecule' not in context:
                    self.logger.info(f"Context validation failed. Context exists: {bool(context)}, Keys in context: {context.keys() if context else 'None'}")
                    return {
                        "type": "error",
                        "content": "No molecule data available for candidate analysis",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Missing required molecule data"
                        }
                    }
                
                molecule_data = context['current_molecule']
                sample_id = molecule_data.get('sample_id')
                
                # Determine the analysis type from the tool name
                if tool_name == 'mol2mol_candidate_analysis':
                    analysis_type = 'mol2mol'
                elif tool_name == 'forward_candidate_analysis':
                    analysis_type = 'forward'
                elif tool_name == 'mmst_candidate_analysis':
                    analysis_type = 'mmst'
                else:
                    analysis_type = 'general'  # Fallback for generic candidate_analysis
                
                self.logger.info(f"Running candidate analysis with type: {analysis_type}")
                
                # Get the appropriate tool instance
                analyzer_tool = self.tools.get(tool_name)
                if not analyzer_tool:
                    self.logger.error(f"No tool found for {tool_name}")
                    return {
                        "type": "error",
                        "content": f"Tool not found: {tool_name}",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Tool initialization error"
                        }
                    }
                
                try:
                    result = await analyzer_tool.process(molecule_data, context)
                    return {
                        "type": "success",
                        "content": result,
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 1.0,
                            "reasoning": f"Successfully processed {analysis_type} candidate analysis"
                        }
                    }
                except Exception as e:
                    error_msg = f"Error in {analysis_type} candidate analysis: {str(e)}"
                    self.logger.error(error_msg)
                    return {
                        "type": "error",
                        "content": error_msg,
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Processing error"
                        }
                    }
                           
            elif tool_name == 'mmst':
                # Ensure we have the necessary context
                if not context or 'current_molecule' not in context:
                    return {
                        "type": "error",
                        "content": "No reference molecule data available. Please provide a reference molecule first.",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Missing required molecule data"
                        }
                    }

                # Force local execution by default
                mmst_context = context.copy() if context else {}
                mmst_context['use_slurm'] = False

                # Get the reference molecule data
                molecule_data = context['current_molecule']
                smiles = molecule_data.get('SMILES') or molecule_data.get('smiles')
                molecule_id = molecule_data.get('sample_id') or molecule_data.get('ID')or molecule_data.get('sample-id')
                
                if not smiles:
                    return {
                        "type": "error",
                        "content": "Invalid molecule data format. Expected dictionary with SMILES key.",
                        "metadata": {
                            "agent": "TOOL_AGENT",
                            "confidence": 0.0,
                            "reasoning": "Invalid data format"
                        }
                    }

                try:
                    self.logger.info(f"Running MMST prediction for molecule {molecule_id} with SMILES: {smiles}")
                    mmst_result = await self.tools['mmst'].predict_structure(
                        reference_smiles=smiles,
                        molecule_id=molecule_id,
                        context=mmst_context
                    )
                    
                    if mmst_result['status'] == 'success':
                        # Update master data with MMST results
                        # await self._update_master_data_with_mmst(mmst_result, molecule_data)
                        
                        self.logger.info("MMST prediction completed successfully")
                        return {
                            'type': 'success',
                            'content': mmst_result,
                            'metadata': {
                                'agent': 'TOOL_AGENT',
                                'confidence': 1.0,
                                'reasoning': 'MMST prediction completed successfully'
                            }
                        }
                    else:
                        error_msg = mmst_result.get('message', 'Unknown error in MMST prediction')
                        self.logger.error(f"MMST prediction failed with error: {error_msg}")
                        return {
                            'type': 'error',
                            'content': error_msg,
                            'metadata': {
                                'agent': 'TOOL_AGENT',
                                'confidence': 0.0,
                                'reasoning': 'MMST prediction failed'
                            }
                        }
                except Exception as e:
                    self.logger.error(f"Exception during MMST prediction: {str(e)}", exc_info=True)
                    return {
                        'type': 'error',
                        'content': str(e),
                        'metadata': {
                            'agent': 'TOOL_AGENT',
                            'confidence': 0.0,
                            'reasoning': 'Error during MMST prediction'
                        }
                    }

            # elif tool_name == 'stout':
            #     if not context:
            #         return {
            #             "type": "error",
            #             "content": "No context provided for STOUT conversion",
            #             "metadata": {
            #                 "agent": "TOOL_AGENT",
            #                 "confidence": 0.0,
            #                 "reasoning": "Missing required context"
            #             }
            #         }

            #     # Handle batch processing of molecules
            #     if 'molecules' in context:
            #         try:
            #             result = await self.tools['stout'].process_molecule_batch(context['molecules'])
            #             if result['status'] == 'success':
            #                 return {
            #                     'type': 'success',
            #                     'content': result,
            #                     'metadata': {
            #                         'agent': 'TOOL_AGENT',
            #                         'confidence': 1.0,
            #                         'reasoning': "Successfully processed molecule batch"
            #                     }
            #                 }
            #         except Exception as e:
            #             return {
            #                 'type': 'error',
            #                 'content': str(e),
            #                 'metadata': {
            #                     'agent': 'TOOL_AGENT',
            #                     'confidence': 0.0,
            #                     'reasoning': 'Error during batch processing'
            #                 }
            #             }

            #     # Handle single molecule conversion
            #     if 'input_str' not in context:
            #         return {
            #             "type": "error",
            #             "content": "No input string provided for conversion",
            #             "metadata": {
            #                 "agent": "TOOL_AGENT",
            #                 "confidence": 0.0,
            #                 "reasoning": "Missing required input"
            #             }
            #         }

            #     input_str = context['input_str']
            #     mode = context.get('conversion_mode', 'forward')  # Default to SMILES→IUPAC
                
            #     try:
            #         if mode == 'forward':
            #             result = await self.tools['stout'].convert_smiles_to_iupac(input_str)
            #         else:
            #             result = await self.tools['stout'].convert_iupac_to_smiles(input_str)
                        
            #         if result['status'] == 'success':
            #             return {
            #                 'type': 'success',
            #                 'content': result,
            #                 'metadata': {
            #                     'agent': 'TOOL_AGENT',
            #                     'confidence': 1.0,
            #                     'reasoning': f"Successfully converted {'SMILES to IUPAC' if mode == 'forward' else 'IUPAC to SMILES'}"
            #                 }
            #             }
            #         else:
            #             return {
            #                 'type': 'error',
            #                 'content': result['error'],
            #                 'metadata': {
            #                     'agent': 'TOOL_AGENT',
            #                     'confidence': 0.0,
            #                     'reasoning': 'Conversion failed'
            #                 }
            #             }
                        
            #     except Exception as e:
            #         return {
            #             'type': 'error',
            #             'content': str(e),
            #             'metadata': {
            #                 'agent': 'TOOL_AGENT',
            #                 'confidence': 0.0,
            #                 'reasoning': 'Error during conversion'
            #             }
            #         }

            return {
                "type": "error",
                "content": f"Unknown tool: {tool_name}",
                "metadata": {
                    "agent": "TOOL_AGENT",
                    "confidence": 0.0,
                    "reasoning": f"Tool '{tool_name}' not found in available tools"
                }
            }
                
        except Exception as e:
            return {
                "type": "error",
                "content": str(e),
                "metadata": {
                    "agent": "TOOL_AGENT",
                    "confidence": 0.0,
                    "reasoning": f"Tool execution failed: {str(e)}"
                }
            }
            
    def _format_tool_response(self, result: Dict, success_message: str) -> Dict:
        """Format tool response in a consistent way."""
        if isinstance(result, dict) and result.get('status') == 'error':
            return {
                "type": "error",
                "content": result.get('error', 'Unknown error'),
                "metadata": {
                    "agent": "TOOL_AGENT",
                    "confidence": 0.0,
                    "reasoning": result.get('error', 'Tool execution failed')
                }
            }
        
        # Create content dictionary with status field
        content = result if isinstance(result, dict) else {'data': result}
        if 'status' not in content:
            content['status'] = 'success'
        
        return {
            "type": "success",
            "content": content,
            "metadata": {
                "agent": "TOOL_AGENT",
                "confidence": 1.0,
                "reasoning": success_message
            }
        }

    # async def _update_master_data_with_mol2mol(self, mol2mol_result: Dict, molecule_data: Dict) -> None:
    #     """Update master data JSON with mol2mol results."""
    #     if mol2mol_result['status'] == 'success':
    #         # Get path to master data
    #         master_data_path = Path(__file__).parent.parent.parent / 'data' / 'molecular_data' / 'molecular_data.json'
            
    #         # Read the output file from mol2mol
    #         output_file = Path(mol2mol_result['output_file'])
    #         if output_file.exists():
    #             try:
    #                 # Read mol2mol results
    #                 mol2mol_df = pd.read_csv(output_file)
    #                 mol2mol_data = mol2mol_df.to_dict('records')
                    
    #                 # Extract target SMILES and suggestions
    #                 target_smiles = molecule_data.get('SMILES') or molecule_data.get('smiles')
    #                 suggestions = []
    #                 for entry in mol2mol_data:
    #                     # Each entry is a dictionary with one key-value pair
    #                     # The key is the target SMILES and value is the suggested SMILES
    #                     for _, suggestion in entry.items():
    #                         suggestions.append(suggestion)
                    
    #                 # Read existing master data
    #                 with open(master_data_path, 'r') as f:
    #                     master_data = json.load(f)
                    
    #                 # Get the sample_id
    #                 sample_id = molecule_data.get('sample_id') or molecule_data.get('sample-id')
    #                 if sample_id and sample_id in master_data:
    #                     # Add mol2mol results with new structure where target SMILES is the key
    #                     master_data[sample_id]['mol2mol_results'] = {
    #                         'generated_analogues_target': {
    #                             target_smiles: suggestions
    #                         },
    #                         'timestamp': datetime.now().isoformat(),
    #                         'status': 'success'
    #                     }
                        
    #                     # Write updated data back
    #                     with open(master_data_path, 'w') as f:
    #                         json.dump(master_data, f, indent=2)
                            
    #                     self.logger.info(f"Updated master data with mol2mol results for sample {sample_id}")
    #                 else:
    #                     self.logger.error(f"Sample ID {sample_id} not found in master data")
                        
    #             except Exception as e:
    #                 self.logger.error(f"Failed to update master data with mol2mol results: {str(e)}")

    
    def _analyze_peak_matching_request(self, message: str) -> Dict:
        """Analyze the peak matching request to determine comparison mode."""
        # Default to exp vs sim comparison
        comparison_mode = {
            'type': 'default',
            'input_data': {}
        }
        
        # Look for SMILES vs SMILES comparison
        if 'compare smiles' in message.lower():
            comparison_mode['type'] = 'smiles_vs_smiles'
            # Note: actual SMILES should be provided in context
            
        # Look for SMILES vs peaks comparison
        elif 'compare smiles with peaks' in message.lower():
            comparison_mode['type'] = 'smiles_vs_peaks'
            
        # Look for peaks vs CSV comparison
        elif 'compare peaks with csv' in message.lower():
            comparison_mode['type'] = 'peaks_vs_smiles_csv'
            
        return comparison_mode

    def _prepare_peak_matching_context(self, message: str, context: Dict) -> Dict:
        """Prepare context for peak matching based on request type."""
        # Analyze the request
        comparison_info = self._analyze_peak_matching_request(message)
        
        # Start with existing context or empty dict
        peak_context = context.copy() if context else {}
        
        # Add comparison mode info
        if 'input_data' not in peak_context:
            peak_context['input_data'] = {}
        peak_context['input_data'].update(comparison_info['input_data'])
        
        # Add comparison type
        peak_context['comparison_mode'] = comparison_info['type']
        
        return peak_context



    async def _update_master_data_with_mmst(self, mmst_result: Dict, molecule_data: Dict) -> None:
        """Update master data JSON with MMST results.
        Args:
            mmst_result: Results from MMST prediction containing processed predictions
            molecule_data: Original molecule data dictionary containing sample_id
        """
        if mmst_result['status'] == 'success':
            # Get path to master data
            master_data_path = Path(__file__).parent.parent.parent / 'data' / 'molecular_data' / 'molecular_data.json'
            
            try:
                # Extract predictions from MMST results
                predictions = mmst_result.get('predictions', {})
                
                # Read existing master data
                with open(master_data_path, 'r') as f:
                    master_data = json.load(f)
                
                # Get the sample_id
                sample_id = molecule_data.get('sample_id') or molecule_data.get('sample-id')
                if sample_id and sample_id in master_data:
                    # Add MMST results with structured data
                    master_data[sample_id]['mmst_results'] = {
                        'generated_molecules': predictions['generated_molecules'],
                        'model_info': predictions['model_info'],
                        'performance': predictions['performance'],
                        'timestamp': predictions['timestamp'],
                        'status': 'success'
                    }
                    
                    # Write updated data back
                    with open(master_data_path, 'w') as f:
                        json.dump(master_data, f, indent=2)
                        
                    self.logger.info(f"Updated master data with MMST results for sample {sample_id}")
                else:
                    self.logger.error(f"Sample ID {sample_id} not found in master data")
                    
            except Exception as e:
                self.logger.error(f"Failed to update master data with MMST results: {str(e)}")

#     def _prepare_peak_matching_input(self, context: Dict) -> Dict:
#         """Prepare input data for peak matching based on context.
        
#         Supports multiple comparison modes based on input_data:
#         1. smiles_vs_smiles: Compare two SMILES structures
#    2. peaks_vs_peaks: Compare two peak lists
#    3. smiles_vs_peaks: Compare SMILES against peak list
#    4. peaks_vs_csv: Compare peaks against SMILES CSV file
#    5. smiles_vs_csv: Compare reference SMILES against CSV file
#    6. exp_vs_sim: (Default) Compare experimental vs simulated peaks from master.json
#         """
#         # Check for explicit comparison modes in input_data
#         input_data = context.get('input_data', {})
#         # Log the full context
#         self.logger.debug(f"Full context: {context}")
        
#         # Extract input_data from context and log it
#         input_data = context.get('input_data', {})
#         self.logger.debug(f"Extracted input_data: {input_data}")
        
#         # SMILES vs SMILES comparison
#         if 'smiles1' in input_data and 'smiles2' in input_data:
#             self.logger.info("Using SMILES vs SMILES comparison mode")
#             return input_data
            
#         # Peaks vs Peaks comparison
#         if 'peaks1' in input_data and 'peaks2' in input_data:
#             self.logger.info("Using Peaks vs Peaks comparison mode")
#             return input_data
            
#         # SMILES vs Peaks comparison
#         if 'smiles' in input_data and 'peaks' in input_data:
#             self.logger.info("Using SMILES vs Peaks comparison mode")
#             return input_data
            
#         # Peaks vs SMILES CSV comparison
#         if 'peaks' in input_data and 'smiles_csv' in input_data:
#             self.logger.info("Using Peaks vs SMILES CSV comparison mode")
#             return input_data
            
#         # Reference SMILES vs CSV comparison
#         if 'reference_smiles' in input_data and 'smiles_csv' in input_data:
#             self.logger.info("Using Reference SMILES vs CSV comparison mode")
#             return input_data
            
#         # Default: Experimental vs Simulated peaks comparison
#         self.logger.info("Using default Experimental vs Simulated peaks comparison mode")
#         try:
#             master_path = Path('data/molecular_data/molecular_data.json')
#             if not master_path.exists():
#                 raise FileNotFoundError("master.json not found")
                
#             with open(master_path, 'r') as f:
#                 master_data = json.load(f)
                
#             sample_id = context['current_molecule']['sample_id']
#             if sample_id not in master_data:
#                 raise KeyError(f"Sample {sample_id} not found in master.json")
                
#             # Update molecule data from master.json
#             context['current_molecule'].update(master_data[sample_id])
#             nmr_data = context['current_molecule']['nmr_data']
            
#             # Verify we have both experimental and simulated data
#             required_exp = ['1H_exp', '13C_exp', 'HSQC_exp', 'COSY_exp']
#             required_sim = ['1H_sim', '13C_sim', 'HSQC_sim', 'COSY_sim']
            
#             if not all(key in nmr_data for key in required_exp + required_sim):
#                 missing = [key for key in required_exp + required_sim if key not in nmr_data]
#                 raise ValueError(f"Missing required NMR data: {missing}")
            
#             # Format peaks for 1D NMR (1H)
#             def format_1d_peaks(peaks):
#                 """Format 1D NMR peaks as parallel lists of shifts and intensities."""
#                 if not isinstance(peaks, list):
#                     return {'shifts': [], 'intensity': []}
#                 shifts = [shift for shift, _ in peaks]
#                 intensities = [1.0] * len(shifts)  # Constant intensity of 1.0
#                 return {
#                     'shifts': shifts,
#                     'intensity': intensities
#                 }

#             # Format peaks for 2D NMR (HSQC, COSY)
#             def format_2d_peaks(peaks):
#                 """Format 2D NMR peaks as parallel lists of F1 and F2 ppm values."""
#                 if not isinstance(peaks, list):
#                     return {'F2 (ppm)': [], 'F1 (ppm)': []}
#                 f2_values = [f2 for f2, _ in peaks]
#                 f1_values = [f1 for _, f1 in peaks]
#                 return {
#                     'F2 (ppm)': f2_values,
#                     'F1 (ppm)': f1_values
#                 }

#             # Format 13C peaks (only shifts, constant intensity)
#             def format_13c_peaks(peaks):
#                 """Format 13C NMR peaks with shifts and constant intensity."""
#                 if not isinstance(peaks, list):
#                     return {'shifts': [], 'intensity': []}
#                 return {
#                     'shifts': peaks,
#                     'intensity': [1.0] * len(peaks)
#                 }

#             return {
#                 'peaks1': {
#                     '1H': format_1d_peaks(nmr_data['1H_exp']),
#                     '13C': format_13c_peaks(nmr_data['13C_exp']),
#                     'HSQC': format_2d_peaks(nmr_data['HSQC_exp']),
#                     'COSY': format_2d_peaks(nmr_data['COSY_exp'])
#                 },
#                 'peaks2': {
#                     '1H': format_1d_peaks(nmr_data['1H_sim']),
#                     '13C': format_13c_peaks(nmr_data['13C_sim']),
#                     'HSQC': format_2d_peaks(nmr_data['HSQC_sim']),
#                     'COSY': format_2d_peaks(nmr_data['COSY_sim'])
#                 },
#                 'matching_mode': 'hung_dist_nn',
#                 'error_type': 'sum',
#                 'spectra': ['1H', '13C', 'HSQC', 'COSY']
#             }
            
#         except Exception as e:
#             self.logger.error(f"Error preparing experimental vs simulated comparison: {str(e)}")
#             return None

    async def _determine_tool_llm(self, message: str, model_choice: str) -> str:
        """Use LLM to determine which tool to use based on the message content."""
        self.logger.info(f"Determining tool for message: {message}")
        self.logger.debug(f"Available tools: {list(TOOL_DESCRIPTIONS.keys())}")
        
        system_prompt = f"""You are a tool selection agent for a molecular analysis system. Your task is to analyze the user's message and select the most appropriate tool based on their needs.

Available tools and their capabilities:

{json.dumps(TOOL_DESCRIPTIONS, indent=2)}

Pay special attention to:
1. Processing scope:
       - Single molecule: When the user wants to process a specific molecule
   - Batch processing: When the task requires processing multiple molecules, such as:
     * Retrosynthesis predictions (always batch)
     * Forward synthesis predictions (always batch)
     * When explicitly requested ("all molecules", "every compound")
     * When comparing against a database or set of molecules

2. Type of processing:
   - Direct operations (e.g., NMR simulation on one molecule)
   - Comparative operations (e.g., peak matching between molecules)
   - Predictive operations (e.g., retrosynthesis, which needs context from all molecules)

3. Input sources:
   - Individual SMILES strings
   - CSV files
   - Molecular databases
   - Experimental data

Analyze the user's message and select the most appropriate tool. Return your response in JSON format with the following structure:
{{
    "selected_tool": "tool_name",
    "confidence": 0.0,  # 0.0 to 1.0
    "reasoning": "explanation of why this tool was selected",
    "processing_type": "single or batch",  # Indicate if batch processing is needed
    "comparison_type": "target or starting_materials",  # Type of comparison needed (if applicable)
    "batch_reason": "explanation of why batch processing was selected" # Only if processing_type is "batch"
}}

Please respond with the selected tool name."""

        try:
            self.logger.info("Sending request to LLM service")
            response = await self.llm_service.get_completion(
                message=message,
                system=system_prompt,
                require_json=True,
                model=model_choice
            )
            
            self.logger.debug(f"LLM response: {response}")
            result = json.loads(response)
            self.logger.info(f"Selected tool: {result['selected_tool']} with confidence {result['confidence']}")
            self.logger.debug(f"Selection reasoning: {result['reasoning']}")
            
            if result['confidence'] < 0.7:  # Confidence threshold
                self.logger.warning(f"Low confidence in tool selection: {result['confidence']}")
                raise ValueError(f"Low confidence in tool selection: {result['confidence']}")
            
            # Store additional context for tool use
            self._current_processing_type = result.get('processing_type', 'single')
            self._current_comparison_type = result.get('comparison_type', None)
                
            return result['selected_tool']
            
        except Exception as e:
            self.logger.error(f"Error determining tool: {str(e)}", exc_info=True)
            raise ValueError(f"Error determining tool: {str(e)}")

    def supports_message(self, message: str) -> bool:
        """Check if this agent can handle the given message."""
        # Always return True since we let the LLM decide
        return True


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/specialized/__init__.py ---
"""
Specialized agents for different tasks.
"""
from .molecule_plot_agent import MoleculePlotAgent
from .nmr_plot_agent import NMRPlotAgent
from .text_response_agent import TextResponseAgent
from .tool_agent import ToolAgent

__all__ = ['MoleculePlotAgent', 'NMRPlotAgent', 'TextResponseAgent', 'ToolAgent']

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/specialized/script_modifier_agent.py ---
"""
Agent for modifying script parameters in bash files.
"""
from typing import Dict, Any, Optional, List, Union
import re
from pathlib import Path
import logging
from ..base.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ScriptModifierAgent(BaseAgent):
    """Agent for modifying parameters in bash scripts while preserving structure."""
    
    def __init__(self):
        capabilities = [
            "Modify bash script parameters",
            "Preserve script structure",
            "Handle multiple parameter updates",
            "Support local and SLURM scripts"
        ]
        super().__init__("Script Modifier", capabilities)
        
        # Define script-specific parameter configurations
        self.script_configs = {
            "chemformer_forward": {
                "pattern": r".*chemformer_forward.*\.sh$",
                "parameters": {
                    "BATCH_SIZE": {"type": int, "min": 1, "description": "Batch size for predictions"},
                    "N_BEAMS": {"type": int, "min": 1, "description": "Number of beams for beam search"},
                    "N_UNIQUE_BEAMS": {"type": (int, str), "allowed_str": ["None"], "min": 1, "description": "Number of unique beams"}
                }
            },
            "chemformer_retro": {
                "pattern": r".*chemformer_retro.*\.sh$",
                "parameters": {
                    "BATCH_SIZE": {"type": int, "min": 1, "description": "Batch size for predictions"},
                    "N_BEAMS": {"type": int, "min": 1, "description": "Number of beams for beam search"},
                    "N_UNIQUE_BEAMS": {"type": (int, str), "allowed_str": ["None"], "min": 1, "description": "Number of unique beams"}
                }
            },
            "mol2mol": {
                "pattern": r".*mol2mol.*\.sh$",
                "parameters": {
                    "DELTA_WEIGHT": {"type": int, "min": 1, "description": "Delta weight parameter"},
                    "TANIMOTO_FILTER": {"type": float, "min": 0.0, "max": 1.0, "description": "Tanimoto filter threshold"},
                    "NUM_GENERATIONS": {"type": int, "min": 1, "description": "Number of generations"},
                    "MAX_TRIALS": {"type": int, "min": 1, "description": "Maximum number of trials"},
                    "MAX_SCAFFOLD_GENERATIONS": {"type": int, "min": 1, "description": "Maximum scaffold generations"}
                }
            },
            "sgnn": {
                "pattern": r".*sgnn.*\.sh$",
                "parameters": {}  # SGNN script uses command line arguments directly
            }
        }

    async def process(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a message to modify scripts.
        
        Args:
            message: The user message to process
            context: Additional context for processing (optional)
        """
        print("\n[Script Modifier Agent] ====== Starting Processing ======")
        print(f"[Script Modifier Agent] Message: {message}")
        model_choice = context.get('model_choice', 'gemini-flash')
        processing_mode = context.get('processing_mode', 'single')
        try:
            if isinstance(message, str):
                message = [message]
                
            results = {}
            for script_path in message:
                path = Path(script_path)
                if not path.exists():
                    results[script_path] = {
                        'status': 'error',
                        'message': f'Script not found: {script_path}'
                    }
                    continue
                
                try:
                    # Identify script type and get its config
                    script_config = self._get_script_config(path.name)
                    if not script_config:
                        results[script_path] = {
                            'status': 'error',
                            'message': f'Unsupported script type: {path.name}'
                        }
                        continue
                    
                    # Validate parameters against script-specific rules
                    invalid_params = self._validate_parameters({}, script_config["parameters"])
                    if invalid_params:
                        results[script_path] = {
                            'status': 'error',
                            'message': f'Invalid parameters: {", ".join(invalid_params)}'
                        }
                        continue
                    
                    # Read current script content
                    with open(path, 'r') as f:
                        content = f.read()
                    
                    # Modify parameters
                    modified_content = self._modify_parameters(content, {})
                    
                    # Write modified content back
                    with open(path, 'w') as f:
                        f.write(modified_content)
                    
                    results[script_path] = {
                        'status': 'success',
                        'message': f'Successfully modified parameters in {path.name}',
                        'modified_parameters': []
                    }
                    
                except Exception as e:
                    results[script_path] = {
                        'status': 'error',
                        'message': f'Error modifying {path.name}: {str(e)}'
                    }
            
            return {
                'status': 'success' if all(r['status'] == 'success' for r in results.values()) else 'partial_success',
                'results': results
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error in script modification: {str(e)}'
            }

    def _get_script_config(self, script_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific script type."""
        for config in self.script_configs.values():
            if re.match(config["pattern"], script_name):
                return config
        return None

    def _validate_parameters(self, parameters: Dict[str, Any], script_params: Dict[str, Dict[str, Any]]) -> List[str]:
        """Validate parameters against script-specific rules."""
        invalid_params = []
        for param_name, value in parameters.items():
            if param_name not in script_params:
                invalid_params.append(f'{param_name} (unknown parameter)')
                continue
                
            param_config = script_params[param_name]
            
            # Type validation
            if not isinstance(value, param_config["type"]):
                if isinstance(param_config["type"], tuple):
                    # Handle special cases like N_UNIQUE_BEAMS that can be int or "None"
                    if isinstance(value, str) and value in param_config.get("allowed_str", []):
                        continue
                invalid_params.append(f'{param_name} (invalid type: expected {param_config["type"]}, got {type(value)})')
                continue
            
            # Range validation for numeric types
            if isinstance(value, (int, float)):
                if "min" in param_config and value < param_config["min"]:
                    invalid_params.append(f'{param_name} (value below minimum: {param_config["min"]})')
                if "max" in param_config and value > param_config["max"]:
                    invalid_params.append(f'{param_name} (value above maximum: {param_config["max"]})')
                    
        return invalid_params

    def _modify_parameters(self, content: str, parameters: Dict[str, Any]) -> str:
        """Modify parameters in script content while preserving structure."""
        lines = content.split('\n')
        modified_lines = []
        
        for line in lines:
            # Check if line defines a parameter we want to modify
            for param_name, new_value in parameters.items():
                # Match parameter definition (handles different formats)
                pattern = rf'^{param_name}=.*$'
                if re.match(pattern, line.strip()):
                    # Preserve any comments that might be on the same line
                    comment = ''
                    if '#' in line:
                        comment = line[line.index('#'):]
                    
                    # Format the new value appropriately
                    if isinstance(new_value, str) and not new_value.startswith('"'):
                        formatted_value = f'"{new_value}"'
                    else:
                        formatted_value = str(new_value)
                    
                    # Create new line with preserved spacing
                    leading_space = len(line) - len(line.lstrip())
                    line = ' ' * leading_space + f'{param_name}={formatted_value}{comment}'
                    break
            
            modified_lines.append(line)
        
        return '\n'.join(modified_lines)

    def supports_message(self, message: str) -> bool:
        """Check if this agent can handle the given message."""
        return any(keyword in message.lower() for keyword in [
            'modify script', 'update parameter', 'change parameter',
            'script parameter', 'bash script', 'shell script'
        ])


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/specialized/molecule_plot_agent.py ---
"""
Specialized agent for handling molecule visualization requests.
"""
from typing import Dict, Any, Optional
from ..base.base_agent import BaseAgent
from services.llm_service import LLMService
from utils.visualization import create_molecule_response
from handlers.molecule_handler import get_molecular_data, get_nmr_data_from_json, set_current_molecule
import pandas as pd
import os
import random
import json

# Path to molecular data JSON file
TEST_SMILES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                               'data', 'molecular_data', 'molecular_data.json')

class MoleculePlotAgent(BaseAgent):
    """Agent for handling molecule visualization requests."""
    
    def __init__(self, llm_service: LLMService):
        """Initialize the molecule plot agent."""
        super().__init__(
            name="Molecule Plot",
            capabilities=[
                "molecule visualization",
                "structure interpretation",
                "2D/3D rendering",
                "molecular property calculation"
            ]
        )
        self.llm_service = llm_service
        
    # def get_random_smiles(self) -> Optional[str]:
    #     """Get a random SMILES string from the molecular data JSON file."""
    #     try:
    #         # Get molecular data
    #         data = get_molecular_data()
    #         if not data:
    #             print("[MoleculePlotAgent] No molecular data found")
    #             return None
                
    #         # Get a random molecule
    #         sample_id = random.choice(list(data.keys()))
    #         molecule = data[sample_id]
    #         smiles = molecule.get('smiles')
            
    #         if not smiles:
    #             print(f"[MoleculePlotAgent] No SMILES found for sample {sample_id}")
    #             return None
            
    #         print(f"[MoleculePlotAgent] Selected random SMILES: {smiles}")
    #         return smiles
            
    #     except Exception as e:
    #         print(f"[MoleculePlotAgent] Error getting random SMILES: {str(e)}")
    #         return None

    async def process(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a message to generate molecule visualizations.
        
        Args:
            message: The user message to process
            model_choice: The LLM model to use (default: 'gemini-flash')
            context: Additional context for processing (optional)
        """
        print("\n[Molecule Plot Agent] ====== Starting Processing ======")
        print(f"[Molecule Plot Agent] Message: {message}")
        #print(f"[Molecule Plot Agent] Context: {json.dumps(context, indent=2) if context else None}")
        model_choice = context.get('model_choice', 'gemini-flash')
        
        try:
            # First, analyze the request using LLM
            print("[Molecule Plot Agent] Creating analysis prompt...")
            analysis_prompt = self._create_analysis_prompt(message)
            #print(f"[Molecule Plot Agent] Analysis Prompt:\n{analysis_prompt}")
            
            print("[Molecule Plot Agent] Getting analysis from LLM...")
            analysis_response = await self.llm_service.get_completion(
                message=analysis_prompt,
                model=model_choice,  # Using Gemini Flash for quick analysis
                system="You are a molecule visualization assistant. Extract SMILES strings and molecule indices from requests. ONLY respond with the requested JSON format, no additional text."
            )
            
            print("[Molecule Plot Agent] Raw LLM Response:")
            print(json.dumps(analysis_response, indent=2))
            
            # Interpret the analysis
            print("[Molecule Plot Agent] Interpreting analysis...")
            molecule_info = self._interpret_analysis(analysis_response)
            print(f"[Molecule Plot Agent] Interpreted info: {json.dumps(molecule_info, indent=2)}")
            
            if not molecule_info or not molecule_info.get("smiles"):
                print("[Molecule Plot Agent] No valid SMILES found")
                return {
                    "type": "tool_error",
                    "content": "Unable to determine molecule to visualize. Please provide a SMILES string or molecule index.",
                    "metadata": {
                        "reasoning": "Unable to determine molecule to visualize. Please provide a SMILES string or molecule index.",
                        "confidence": molecule_info.get("confidence", 0.0)
                    }
                }
            
            # Generate both 2D and 3D visualizations
            smiles = molecule_info["smiles"]
            print(f"[Molecule Plot Agent] Using SMILES: {smiles}")
            
            # Get NMR data for the molecule
            nmr_data = get_nmr_data_from_json(smiles)
            
            # Set as current molecule with NMR data
            sample_id = nmr_data.get('sample_id', 'unknown')
            set_current_molecule(
                smiles=smiles,
                nmr_data={
                    '1h': nmr_data.get('1h_exp'),
                    '13c': nmr_data.get('13c_exp'),
                    'hsqc': nmr_data.get('hsqc_exp'),
                    'cosy': nmr_data.get('cosy_exp')
                },
                sample_id=sample_id
            )
            
            response_2d = create_molecule_response(smiles, is_3d=False)
            response_3d = create_molecule_response(smiles, is_3d=True)
            
            if not response_2d or not response_3d:
                raise ValueError(f"Failed to create visualizations for SMILES: {smiles}")
            
            # Add NMR data to response
            response_2d['nmr_data'] = nmr_data
            response_3d['nmr_data'] = nmr_data
            
            # Format response
            response = {
                "type": "molecule_plot",
                "data": {
                    "2d": {
                        **response_2d,
                        "container": "vis-content-nmr-1"
                    },
                    "3d": {
                        **response_3d,
                        "container": "vis-content-nmr-3"
                    }
                },
                "molecule_index": molecule_info.get("molecule_index")
            }
            
            print("[Molecule Plot Agent] Response data:")
            # print(json.dumps(response, indent=2))
            print("[Molecule Plot Agent] Successfully generated visualization")
            print("[Molecule Plot Agent] ====== Processing Complete ======\n")
            return response
            
        except Exception as e:
            error_msg = f"Failed to process molecule visualization: {str(e)}"
            print(f"[Molecule Plot Agent] ERROR: {error_msg}")
            print(f"[Molecule Plot Agent] Error type: {type(e)}")
            import traceback
            print(f"[Molecule Plot Agent] Traceback:\n{traceback.format_exc()}")
            return {
                "type": "error",
                "content": error_msg
            }

    def _create_analysis_prompt(self, message: str) -> str:
        """Create the analysis prompt for the LLM."""
        return f"""Analyze this molecule visualization request and extract SMILES string and molecule index if present.
Return ONLY a JSON response with NO additional text.

Request: "{message}"

Required JSON format:
{{
    "molecule_request": {{
        "smiles": "string or null",      # SMILES string if explicitly mentioned in the request
        "molecule_index": "number or null", # Index of molecule if specified (e.g., "show molecule 2" -> 2)
        "confidence": 0.0,               # 0.0 to 1.0
        "reasoning": "explanation"        # Brief explanation of what was found in the request. If confidence is low, explain why and suggest alternative phrasing for clarity.
    }}
}}

Example responses:

1. "Show me the molecule with SMILES: CC(=O)O"
{{
    "molecule_request": {{
        "smiles": "CC(=O)O",
        "molecule_index": null,
        "confidence": 0.95,
        "reasoning": "Request contains explicit SMILES string CC(=O)O"
    }}
}}

2. "Display molecule 3"
{{
    "molecule_request": {{
        "smiles": null,
        "molecule_index": 3,
        "confidence": 0.9,
        "reasoning": "Request specifies molecule index 3"
    }}
}}

3. "Show me"
{{
    "molecule_request": {{
        "smiles": null,
        "molecule_index": null,
        "confidence": 0.2,
        "reasoning": "Request is too vague. No specific molecule or index specified. Suggest rephrasing to 'Show me molecule X' or 'Display SMILES: [specific SMILES string]' for clarity."
    }}
}}"""

    def _interpret_analysis(self, analysis: Any) -> Dict[str, Any]:
        """Interpret the LLM's analysis of the visualization request."""
        try:
            print(f"\n[Molecule Plot Agent] Starting analysis interpretation")
            print(f"[Molecule Plot Agent] Raw analysis input type: {type(analysis)}")
            # print(f"[Molecule Plot Agent] Raw analysis content: {analysis}")
            
            # Extract content from dict response
            content = analysis.get("content") if isinstance(analysis, dict) else analysis
            print(f"[Molecule Plot Agent] Extracted content: {content}")
            
            # Parse JSON from content
            response = json.loads(str(content).strip())
            print(f"[Molecule Plot Agent] Parsed JSON response: {json.dumps(response, indent=2)}")
            
            # Extract molecule request information
            molecule_request = response.get("molecule_request", {})
            smiles = molecule_request.get("smiles")
            molecule_index = molecule_request.get("molecule_index")
            confidence = float(molecule_request.get("confidence", 0.0))
            
            print(f"[Molecule Plot Agent] Extracted values:")
            print(f"  - SMILES: {smiles}")
            print(f"  - Molecule Index: {molecule_index}")
            print(f"  - Confidence: {confidence}")
            
            # If index is provided but no SMILES, get SMILES from index
            if molecule_index is not None and smiles is None:
                try:
                    # Get molecular data from JSON
                    data = get_molecular_data()
                    if not data:
                        print("[MoleculePlotAgent] No molecular data found")
                        return None

                    # Get sorted list of sample IDs to maintain consistent order
                    sample_ids = sorted(data.keys())
                    # Convert to 0-based index
                    index = molecule_index - 1 if molecule_index > 0 else 0
                    if 0 <= index < len(sample_ids):
                        sample_id = sample_ids[index]
                        molecule = data[sample_id]
                        smiles = molecule.get('smiles')
                        print(f"[MoleculePlotAgent] Using SMILES for sample {sample_id} at index {molecule_index}: {smiles}")
                        
                        # Get NMR data for indexed molecule
                        nmr_data = get_nmr_data_from_json(smiles)
                        if nmr_data:
                            # Set as current molecule with NMR data
                            set_current_molecule(
                                smiles=smiles,
                                nmr_data={
                                    'proton': nmr_data.get('proton'),
                                    'carbon': nmr_data.get('carbon'),
                                    'hsqc': nmr_data.get('hsqc'),
                                    'cosy': nmr_data.get('cosy')
                                },
                                sample_id=sample_id  # Use the actual sample ID from the data
                            )
                except Exception as e:
                    print(f"[MoleculePlotAgent] Error getting SMILES from index: {str(e)}")
            
            # If SMILES is directly provided (pasted into chat)
            elif smiles is not None:
                print(f"[MoleculePlotAgent] Using provided SMILES: {smiles}")
                # For pasted SMILES, set as current molecule but with no NMR data
                set_current_molecule(
                    smiles=smiles,
                    nmr_data=None,  # No NMR data for pasted SMILES
                    sample_id='unknown'
                )
            
            # If no index provided and no SMILES, use index 0
            else:
                try:
                    df = pd.read_csv(TEST_SMILES_PATH)
                    if len(df) > 0:
                        smiles = df.iloc[0]['SMILES']
                        molecule_index = 1  # 1-based index for user display
                        print(f"[MoleculePlotAgent] Using default SMILES at index 1: {smiles}")
                        
                        # Get NMR data for default molecule from JSON
                        nmr_data = get_nmr_data_from_json(smiles)
                        if nmr_data:
                            set_current_molecule(
                                smiles=smiles,
                                nmr_data={
                                    'proton': nmr_data.get('proton'),
                                    'carbon': nmr_data.get('carbon'),
                                    'hsqc': nmr_data.get('hsqc'),
                                    'cosy': nmr_data.get('cosy')
                                },
                                sample_id=nmr_data.get('sample_id', 'unknown')
                            )
                except Exception as e:
                    print(f"[MoleculePlotAgent] Error getting default SMILES: {str(e)}")
            
            result = {
                "smiles": smiles,
                "molecule_index": molecule_index,
                "confidence": confidence
            }
            print(f"[MoleculePlotAgent] Final interpreted result: {json.dumps(result, indent=2)}")

            return result
            
        except Exception as e:
            print(f"[MoleculePlotAgent] Error interpreting analysis: {str(e)}")
            print(f"[MoleculePlotAgent] Raw analysis input: {analysis}")
            return None

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/specialized/config/tool_descriptions.py ---
"""
Tool descriptions configuration for specialized agents.
"""

TOOL_DESCRIPTIONS = {
    'nmr_simulation': {
        'name': 'NMR Spectrum Simulation',
        'description': 'Simulates NMR spectra for molecules using SGNN. Can predict 1H, 13C, COSY, and HSQC spectra.',
        'capabilities': [
            'Predict NMR spectra for input molecules',
            'Generate chemical shift predictions',
            'Create correlation spectra (COSY, HSQC)',
            'Handle multiple input molecules'
        ],
        'when_to_use': [
            'User wants to predict NMR spectra',
            'User needs chemical shift information',
            'User requests spectral analysis',
            'Questions about molecular structure characterization'
        ]
    },
    'mol2mol': {
        'name': 'Molecular Analogue Generation',
        'description': 'Generates molecular analogues using the Mol2Mol network.',
        'capabilities': [
            'Generate similar molecules',
            'Explore chemical space',
            'Create molecular variations'
        ],
        'when_to_use': [
            'User wants to find similar molecules',
            'User needs molecular analogues',
            'User requests structure modifications',
            'Questions about molecular diversity'
        ]
    },
    'retro_synthesis': {
        'name': 'Retro Synthesis Prediction',
        'description': 'Predicts retrosynthetic routes using the Chemformer model. Can handle single or batch predictions with configurable beam search parameters.',
        'capabilities': [
            'Generate retrosynthetic predictions',
            'Predict synthetic precursors',
            'Support batch processing',
            'Configurable beam search',
            'Handle complex molecules',
            'Support both local and cluster execution'
        ],
        'when_to_use': [
            'User wants to plan a synthesis',
            'User needs to identify precursor molecules',
            'Questions about synthetic routes',
            'Analysis of synthetic accessibility',
            'Batch processing of multiple targets',
            'Complex molecule synthesis planning'
        ]
    },
    'forward_synthesis': {
        'name': 'Forward Synthesis Prediction',
        'description': 'Predicts forward synthesis products using the Chemformer model. Can handle single or batch predictions with configurable beam search parameters.',
        'capabilities': [
            'Generate forward synthesis predictions',
            'Predict reaction products',
            'Support batch processing',
            'Configurable beam search',
            'Handle complex molecules',
            'Support both local and cluster execution'
        ],
        'when_to_use': [
            'User wants to predict reaction products',
            'User needs to identify possible products',
            'Questions about reaction outcomes',
            'Analysis of reaction feasibility',
            'Batch processing of multiple reactants',
            'Complex reaction prediction'
        ]
    },
    'peak_matching': {
        'name': 'Enhanced Peak Matching',
        'description': 'Advanced tool for matching and comparing NMR peaks between spectra, supporting 1H, 13C, HSQC, and COSY data.',
        'capabilities': [
            'Compare peaks between SMILES structures',
            'Match experimental vs predicted peaks',
            'Support multiple spectrum types',
            'Handle batch processing',
            'Normalize peak intensities',
            'Calculate matching scores'
        ],
        'when_to_use': [
            'User wants to compare NMR spectra',
            'User needs to validate peak assignments',
            'Questions about spectral similarity',
            'Analysis of experimental vs predicted data',
            'Batch processing of multiple spectra',
            'Peak pattern matching'
        ]
    },
    'threshold_calculation': {
        'name': 'NMR Threshold Calculation',
        'description': 'Calculates error thresholds for NMR spectral analysis by integrating retrosynthesis, NMR simulation, and peak matching.',
        'capabilities': [
            'Calculate thresholds for multiple NMR spectrum types',
            'Process 1H, 13C, HSQC, and COSY spectra',
            'Integrate with retrosynthesis prediction',
            'Handle peak matching and comparison',
            'Provide weighted threshold calculations',
            'Support local execution environment'
        ],
        'when_to_use': [
            'User needs error thresholds for NMR analysis',
            'User wants to validate spectral matching',
            'User requests threshold calculation',
            'Questions about spectral comparison accuracy'
        ]
    },
    'molecular_visual_comparison': {
        'name': 'Molecular Visual Comparison',
        'description': 'AI-powered tool for visually comparing molecular structures using Claude 3.5 Sonnet. Supports single and batch comparisons.',
        'capabilities': [
            'Compare guess molecule with target molecule',
            'Compare guess molecule with starting materials',
            'Batch processing of multiple guess molecules via CSV',
            'Detailed structural similarity analysis',
            'Visual validation of molecular structures',
            'Confidence scoring and pass/fail evaluation'
        ],
        'when_to_use': [
            'User wants to compare molecular structures visually',
            'User needs to validate a synthesized molecule against a target',
            'User wants to check if a molecule could be derived from starting materials',
            'User has multiple molecules to compare (batch processing)',
            'Questions about structural similarity or differences',
            'Validation of synthetic results'
        ]
    },
    # 'candidate_analysis': {
    #     'name': 'Candidate Analysis',
    #     'description': 'Analyzes and scores candidate molecules from various prediction sources using NMR data matching.',
    #     'capabilities': [
    #         'Analyze forward synthesis predictions',
    #         'Analyze mol2mol predictions',
    #         'Score candidates using NMR data',
    #         'Generate summary statistics',
    #         'Handle multiple prediction sources',
    #         'Batch process NMR predictions'
    #     ],
    #     'when_to_use': [
    #         'After forward synthesis predictions',
    #         'After mol2mol predictions',
    #         'When comparing multiple candidate molecules',
    #         'When scoring prediction results',
    #         'When analyzing structure matches'
    #     ]
    # },
    'forward_candidate_analysis': {
        'name': 'Forward Prediction Candidate Analysis',
        'description': 'Analyzes and scores candidate molecules specifically from forward synthesis predictions using NMR data matching.',
        'capabilities': [
            'Analyze forward synthesis predictions only',
            'Score candidates using NMR data',
            'Generate summary statistics',
            'Batch process NMR predictions'
        ],
        'when_to_use': [
            'After forward synthesis predictions',
            'When scoring forward prediction results',
            'When analyzing forward prediction matches',
            'When the command mentions analyzing forward synthesis candidates'
        ]
    },
    'mol2mol_candidate_analysis': {
        'name': 'Mol2Mol Candidate Analysis',
        'description': 'Analyzes and scores candidate molecules specifically from mol2mol analogues using NMR data matching.',
        'capabilities': [
            'Analyze mol2mol predictions only',
            'Score candidates using NMR data',
            'Generate summary statistics',
            'Batch process NMR predictions'
        ],
        'when_to_use': [
            'After mol2mol predictions',
            'When scoring mol2mol results',
            'When analyzing mol2mol matches',
            'When the command mentions analyzing mol2mol candidates'
        ]
    },
    'mmst_candidate_analysis': {
        'name': 'MMST Candidate Analysis',
        'description': 'Analyzes and scores candidate molecules specifically from MMST predictions using NMR data matching.',
        'capabilities': [
            'Analyze MMST predictions only',
            'Score candidates using NMR data',
            'Generate summary statistics',
            'Batch process NMR predictions'
        ],
        'when_to_use': [
            'After MMST predictions',
            'When scoring MMST results',
            'When analyzing MMST matches',
            'When the command mentions analyzing MMST candidates'
        ]
    },
    'mmst': {
        'name': 'Multi-Modal Spectral Transformer',
        'description': 'Advanced deep learning model that performs fine-tuning on molecular analogues to predict target structures from experimental NMR data.',
        'capabilities': [
            'Fine-tune model on molecular analogues',
            'Generate and simulate NMR data for training',
            'Predict target structures from experimental data',
            'Handle multiple NMR modalities (1H, 13C, COSY, HSQC)',
            'Iterative refinement through improvement cycles',
            'Support both local and cluster execution'
        ],
        'when_to_use': [
            'User needs to predict molecular structure from experimental NMR data',
            'User has reference molecules for fine-tuning',
            'Complex structure elucidation tasks',
            'When other tools provide insufficient results',
            'Need for iterative refinement of predictions',
            'Handling multiple NMR modalities simultaneously'
        ]
    }
}


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/specialized/config/__init__.py ---
"""
Configuration module for specialized agents.
"""
from .tool_descriptions import TOOL_DESCRIPTIONS

__all__ = ['TOOL_DESCRIPTIONS']


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/base/base_agent.py ---
"""
Base agent class and common utilities for the multi-agent system.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, name: str, capabilities: List[str]):
        self.name = name
        self.capabilities = capabilities
        self.memory: Dict[str, Any] = {}
    
    @abstractmethod
    async def process(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a message and return a response.
        
        Args:
            message: The input message to process
            context: Optional context information
            
        Returns:
            Dict containing the response with at least 'type' and 'content' keys
        """
        pass
    
    def can_handle(self, task: str) -> bool:
        """Check if the agent can handle a specific task."""
        return any(capability.lower() in task.lower() for capability in self.capabilities)
    
    def update_memory(self, key: str, value: Any) -> None:
        """Update agent's memory with new information."""
        self.memory[key] = value
    
    def get_from_memory(self, key: str) -> Optional[Any]:
        """Retrieve information from agent's memory."""
        return self.memory.get(key)
    
    def __str__(self) -> str:
        return f"{self.name} Agent (Capabilities: {', '.join(self.capabilities)})"


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/agents/base/__init__.py ---
"""Base agent package."""
from .base_agent import BaseAgent

__all__ = ['BaseAgent']


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/core/agents.py ---
"""
Agent initialization and setup.
"""
from services.llm_service import LLMService
from agents.coordinator.coordinator import CoordinatorAgent
from agents import (
    MoleculePlotAgent, NMRPlotAgent, TextResponseAgent,
    ToolAgent, OrchestrationAgent, AgentType
)

# Initialize LLM service
llm_service = LLMService()

if not llm_service:
    raise RuntimeError("No LLM service available. Please check your API keys in config/settings.py")

# Initialize coordinator first
agent_coordinator = CoordinatorAgent(llm_service)

# Initialize all other agents
molecule_plot_agent = MoleculePlotAgent(llm_service)
nmr_plot_agent = NMRPlotAgent(llm_service)
text_response_agent = TextResponseAgent(llm_service)
tool_agent = ToolAgent(llm_service)

# Initialize orchestration agent with its coordinator
orchestration_agent = OrchestrationAgent(llm_service, coordinator=agent_coordinator)

# Register all agents with coordinator
agent_coordinator.add_agent(AgentType.MOLECULE_PLOT, molecule_plot_agent)
agent_coordinator.add_agent(AgentType.NMR_PLOT, nmr_plot_agent)
agent_coordinator.add_agent(AgentType.TEXT_RESPONSE, text_response_agent)
agent_coordinator.add_agent(AgentType.TOOL_USE, tool_agent)
agent_coordinator.set_orchestration_agent(orchestration_agent)


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/core/app.py ---
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


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/core/__init__.py ---
"""
Core functionality for the LLM Structure Elucidator.
"""

from .app import app
from .socket import socketio
from .agents import agent_coordinator

__all__ = ['app', 'socketio', 'agent_coordinator']


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/core/socket.py ---
"""
Socket.IO setup and configuration.
"""
from flask_socketio import SocketIO
from .app import app

# Initialize Socket.IO
socketio = SocketIO(app, cors_allowed_origins="*")


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/services/__init__.py ---



--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/services/llm_service.py ---
"""
LLM API integration service.
"""
from typing import Dict, Any, Optional, List
import anthropic
from anthropic import AsyncAnthropic
import openai
import google.generativeai as genai
from google import genai as genai_think
import json
import base64
from pathlib import Path
from config.settings import ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY, DEEPSEEK_AZURE_ENDPOINT, DEEPSEEK_AZURE_API_KEY
import logging
import requests
import asyncio
import ast
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage

# Initialize logger
logger = logging.getLogger(__name__)

class LLMService:
    """Handles interactions with LLM APIs through the agent system."""

    _instance = None

    # Model mapping dictionary
    MODEL_MAPPING = {
        # Claude models
        "claude-3-5-haiku": "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet": "claude-3-5-sonnet-latest",
        # GPT models
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
        "gpt-4o": "gpt-4o",
        # O3 models
        "o3-mini": "o3-mini",
        # Gemini models
        "gemini-flash": "models/gemini-1.5-flash",
        "gemini-pro": "models/gemini-pro",
        "gemini-thinking": "gemini-2.0-flash-thinking-exp-01-21",
        # DeepSeek models
        "deepseek-reasoner": "deepseek-reasoner"
    }

    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
            openai.api_key = OPENAI_API_KEY
            # Initialize Gemini models
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_pro = genai.GenerativeModel(self.MODEL_MAPPING["gemini-pro"])
            self.gemini_flash = genai.GenerativeModel(self.MODEL_MAPPING["gemini-flash"])
            # Thinking model uses Client
            self.gemini_thinking = genai_think.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})
            self._initialized = True
            self.conversation_history: List[Dict[str, Any]] = []

    async def get_completion(self, 
                           message: str, 
                           model: str = "claude-3-5-haiku",
                           context: Optional[Dict[str, Any]] = None,
                           agent_name: Optional[str] = None,
                           max_tokens: int = 2000,
                           system: Optional[str] = None,
                           require_json: bool = False,
                           max_retries: int = 3) -> str:
        """Get completion from the specified LLM model through an agent.
        
        Args:
            message: The message to send to the LLM
            model: The model to use (e.g., "claude-3-5-haiku", "gemini-flash")
            context: Optional context dictionary
            agent_name: Optional agent name for conversation history
            max_tokens: Maximum tokens in response
            system: Optional system prompt
            require_json: If True, validates response is valid JSON and retries if not
            max_retries: Maximum number of retries for JSON validation
        """
        try:
            # Validate and normalize model name
            if model == "claude-3-haiku":  # Handle legacy model name
                model = "claude-3-5-haiku"

            if model not in self.MODEL_MAPPING:
                print(f"[LLM Service] Warning: Model {model} not found in mapping, falling back to claude-3-5-haiku")
                model = "claude-3-5-haiku"

            # Add message to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": message,
                "agent": agent_name
            })

            # Prepare context for the API call
            if context is None:
                context = {}
            if system:
                context["system"] = system

            # Add JSON formatting requirement if needed
            if require_json:
                print("\n[LLM Service] JSON response required")
                if system:
                    system = system + "\nIMPORTANT: You must respond with valid JSON only. No other text or explanation."
                else:
                    system = "IMPORTANT: You must respond with valid JSON only. No other text or explanation."
                context["system"] = system
                # print(f"[LLM Service] System prompt: {system}")

            # Prepare context for the API call
            full_context = self._prepare_context(context)
            print(f"\n[LLM Service] Sending request to model: {model}")

            # Try multiple times if JSON is required
            attempts = 0
            while attempts < max_retries:
                try:
                    # Get response based on model type
                    try:
                        if model.startswith("claude"):
                            # Map the model name to its full identifier
                            full_model_name = self.MODEL_MAPPING.get(model, model)
                            response = await self._get_anthropic_completion(message, full_model_name, full_context, max_tokens)
                        elif model.startswith("gpt") or model.startswith("o3"):
                            response = await self._get_openai_completion(message, model, full_context, max_tokens)
                        elif model.startswith("gemini"):
                            response = await self._get_gemini_completion(message, model, full_context, max_tokens)
                        # elif model == "deepseek-reasoner":
                        #     response = await self._query_deepseek(message, system)
                        else:
                            raise ValueError(f"Unsupported model: {model}")
                    except Exception as e:
                        print(f"[LLM Service] Error during attempt {attempts + 1}: {str(e)}")
                        if attempts < max_retries - 1:
                            attempts += 1
                            continue
                        else:
                            raise

                    print(f"\n[LLM Service] Raw response from {model}:")
                    print("----------------------------------------")

                    # Validate JSON if required
                    if require_json:
                        response = response.strip()
                        try:
                            # Test if response is valid JSON
                            _ = json.loads(response)
                            print("[LLM Service] Successfully validated JSON response")
                            # If we get here, JSON is valid, break the loop
                            break
                        except json.JSONDecodeError as e:
                            print(f"[LLM Service] JSON validation failed: {str(e)}")
                            if attempts < max_retries - 1:
                                print(f"[LLM Service] Attempt {attempts + 1}: Invalid JSON response, retrying...")
                                message = message + "\nYour previous response was not valid JSON. Please provide ONLY a valid JSON object with no additional text."
                                attempts += 1
                                continue
                            else:
                                raise ValueError("Failed to get valid JSON response after all retries")
                    else:
                        # If JSON not required, just return the response
                        break

                except Exception as e:
                    print(f"[LLM Service] Error during attempt {attempts + 1}: {str(e)}")
                    if attempts < max_retries - 1:
                        attempts += 1
                        continue
                    else:
                        raise

            # Add response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "agent": agent_name,
                "model": model
            })

            # Ensure we return a string
            return response if isinstance(response, str) else str(response)

        except Exception as e:
            error_msg = f"Error in LLM completion: {str(e)}"
            print(f"[LLM Service] {error_msg}")
            # Return error message as string instead of dict
            return error_msg

    def _prepare_context(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare context for API calls."""
        base_context = {
            "conversation_history": self.conversation_history[-5:] if self.conversation_history else []
        }
        if context:
            base_context.update(context)
        return base_context

    async def _get_anthropic_completion(self, message: str, model: str, context: Dict[str, Any], max_tokens: int) -> str:
        """Get completion from Anthropic's Claude."""
        system_prompt = self._create_system_prompt(context)

        # print(f"\n[LLM Service] Sending request to Anthropic's Claude model: {model}")
        # print(f"[LLM Service] System prompt: {system_prompt}")
        # print(f"[LLM Service] User message: {message}")

        response = await self.anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": message}]
        )
        # print(f"\n[LLM Service] Raw response from Claude: {response.content[0].text}")
        # print("----------------------------------------")
        return response.content[0].text

    async def _get_openai_completion(self, message: str, model: str, context: Dict[str, Any], max_tokens: int) -> str:
        """Get completion from OpenAI's GPT or O3."""
        system_prompt = self._create_system_prompt(context)

        # Base parameters for all OpenAI models
        params = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
        }
        
        # Add O3-specific parameters
        if model.startswith("o3"):
            params = {
            "model": model,
            "max_completion_tokens": 20000,
            "reasoning_effort": "high",  ###low or medium or high
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
        }

        response = openai.ChatCompletion.create(**params)
        return response.choices[0].message.content

    async def _get_gemini_completion(self, message: str, model: str, context: Dict[str, Any], max_tokens: int) -> str:
        """Get completion from Google's Gemini."""
        import ast
        
        system_prompt = self._create_system_prompt(context)
        
        # Combine system prompt and message
        full_prompt = f"{system_prompt}\n\nUser: {message}"

        try:
            if model == "gemini-thinking":
                # Use the specific client for the thinking model
                response = self.gemini_thinking.models.generate_content(
                    model=self.MODEL_MAPPING["gemini-thinking"],
                    contents=full_prompt
                )
                
                # For gemini-thinking, extract and parse JSON content
                raw_text = response.text
                return raw_text
                  
            else:
                # Get the appropriate model instance
                if model == "gemini-pro":
                    model_instance = self.gemini_pro
                elif model == "gemini-flash":
                    model_instance = self.gemini_flash
                else:
                    raise ValueError(f"Unknown Gemini model: {model}")

                # Generate content with the model
                response = model_instance.generate_content(
                    full_prompt,
                    generation_config={"max_output_tokens": 20000}
                )

                # Get response text
                text = response.text
                if text.startswith("```json"):
                    text = text[7:]  # Remove ```json prefix
                text = text.replace("```", "").strip()
                
                return text

        except Exception as e:
            logger.error(f"Error in Gemini completion: {str(e)}")
            raise

    # async def _query_deepseek(self, message: str, system: str = None) -> Dict[str, str]:
    #     """Query the DeepSeek API using the OpenAI library format.
        
    #     Returns:
    #         Dict containing 'content' and 'reasoning_content' from the DeepSeek response
    #     """
    #     try:
    #         # Configure OpenAI client for DeepSeek
    #         openai.api_base = "https://api.deepseek.com/v1"
    #         openai.api_key = DEEPSEEK_API_KEY

    #         # Prepare messages
    #         messages = []
    #         if system:
    #             messages.append({"role": "system", "content": system})
    #         messages.append({"role": "user", "content": message})

    #         # Create chat completion request using non-async method
    #         response = openai.ChatCompletion.create(
    #             model="deepseek-reasoner",
    #             messages=messages,
    #             stream=False,
    #             temperature=0.3
    #         )

    #         # Extract response content and reasoning
    #         message = response.choices[0].message
    #         return {
    #             'content': message['content'],
    #             'reasoning_content': message.get('reasoning_content', '')  # Get reasoning_content if available
    #         }

    #     except openai.error.AuthenticationError:
    #         logger.error("Authentication failed. Please check your DeepSeek API key.")
    #         raise Exception("Authentication failed")
    #     except openai.error.RateLimitError:
    #         logger.error("Rate limit exceeded. Please try again later.")
    #         raise Exception("Rate limit exceeded")
    #     except Exception as e:
    #         logger.error(f"Error querying DeepSeek API: {str(e)}")
    #         raise

    def _extract_parts(self, content: str) -> tuple:
        """Extract thinking and content parts from a string response.
        
        Args:
            content: String containing the full response text
            
        Returns:
            tuple: (thinking, content) where thinking is the text between <think> tags
                  and content is everything after </think> or the full text if no tags
        """
        # Extract thinking part (between <think> and </think>)
        think_start = content.find('<think>') + len('<think>')
        think_end = content.find('</think>')
        thinking = content[think_start:think_end].strip() if '<think>' in content else ''
        
        # Extract content part (everything after </think>)
        content = content[think_end + len('</think>'):].strip() if '</think>' in content else content
        
        return thinking, content


    async def query_deepseek_azure(self, message: str, system: str = None, max_tokens: int = 500) -> Dict[str, str]:
        """Query the DeepSeek API using Azure AI inference with streaming and exception handling."""
        try:
            client = ChatCompletionsClient(
                endpoint=DEEPSEEK_AZURE_ENDPOINT,
                credential=AzureKeyCredential(DEEPSEEK_AZURE_API_KEY),
            )

            messages = []
            if system:
                messages.append(SystemMessage(content=system))
            messages.append(UserMessage(content=message))

            # Initialize an empty string to accumulate the response
            full_response = ""

            # Stream response
            response = client.complete(
                model="DeepSeek-R1",
                messages=messages,
                stream=True,  # Enable streaming
                max_tokens=20000,  # Use the provided max_tokens parameter
            )

            # Iterate over the streamed response and accumulate the content
            for update in response:
                try:
                    content = update.choices[0].delta.content or ""
                    full_response += content
                    print(content, end="", flush=True)
                except Exception as e:
                    logger.error(f"Error processing streamed content: {str(e)}")
                    break  # Exit the loop on error

            # After the loop, full_response contains the complete response
            thinking, content = self._extract_parts(full_response)
            return thinking, content 


        except Exception as e:
            logger.error(f"Error querying DeepSeek Azure API: {str(e)}")
            return {"error": str(e)}


    def _create_system_prompt(self, context: Dict[str, Any]) -> str:
        """Create system prompt based on context."""
        if context.get("system"):
            base_prompt = context["system"]
        else:
            base_prompt = "You are an AI assistant specializing in chemical structure analysis and interpretation."

        if context.get("conversation_history"):
            base_prompt += "\nPrevious conversation context is provided for reference."

        return base_prompt

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    async def analyze_with_vision(self, 
                                prompt: str,
                                image_path: str,
                                model: str = "claude-3-5-sonnet",
                                max_tokens: int = 2048,
                                system: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze an image using LLM vision capabilities.
        
        Args:
            prompt: The prompt describing what to analyze in the image
            image_path: Path to the image file
            model: The model to use (default: claude-3-5-sonnet for best vision capabilities)
            max_tokens: Maximum tokens in response
            system: Optional system prompt
            
        Returns:
            The LLM's analysis of the image
        """
        try:
            # Validate and normalize model name
            if model == "claude-3-sonnet":  # Handle legacy model name
                model = "claude-3-5-sonnet"
            
            # For vision tasks, we only support Claude models currently
            if not model.startswith("claude"):
                logger.warning(f"Model {model} not supported for vision tasks, falling back to claude-3-5-sonnet")
                model = "claude-3-5-sonnet"
            
            # Get the full model name from mapping
            full_model_name = self.MODEL_MAPPING.get(model, model)
            
            # Read image file
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            with open(image_path, "rb") as f:
                image_data = f.read()

            # Create content array with image and text
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(image_data).decode()
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]

            # Set up system prompt
            if not system:
                system = ("You are an expert in chemical structure analysis and NMR spectroscopy. "
                         "Analyze the provided molecular structure image and provide detailed insights ")

            # Prepare the request body
            request_body = {
                "model": full_model_name,  # Use mapped model name if available, otherwise use as-is
                "max_tokens": max_tokens,
                "system": [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            }

            # Set up headers with API key, API version and beta features
            headers = {
                "x-api-key": self.anthropic_client.api_key,  # Use API key from initialized client
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "prompt-caching-2024-07-31",
                "content-type": "application/json"
            }

            # Make the API call with custom headers
            response = await self.anthropic_client._client.post(
                "https://api.anthropic.com/v1/messages",
                json=request_body,
                headers=headers
            )

            # Log the response status
            logger.info(f"Vision API response status: {response.status_code}")

            # Handle non-200 responses
            if response.status_code != 200:
                error_msg = f"Vision API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    'analysis_text': error_msg,
                    'confidence': 0.0,
                    'structural_matches': [],
                    'mismatches': []
                }

            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                error_msg = f"Failed to decode JSON response: {str(e)}"
                logger.error(error_msg)
                return {
                    'analysis_text': error_msg,
                    'confidence': 0.0,
                    'structural_matches': [],
                    'mismatches': []
                }

            # Log usage statistics if available
            try:
                if 'usage' in response_data:
                    print("\nUsage statistics for Vision Analysis:")
                    print(f"Input tokens: {response_data['usage'].get('input_tokens', 'N/A')}")
                    print(f"Output tokens: {response_data['usage'].get('output_tokens', 'N/A')}")
                    print(f"Cache creation input tokens: {response_data['usage'].get('cache_creation_input_tokens', 0)}")
                    print(f"Cache read input tokens: {response_data['usage'].get('cache_read_input_tokens', 0)}")
                    print("\n" + "-"*50 + "\n")
                else: 
                    logger.warning("No usage statistics in response")
            except Exception as e:
                logger.warning(f"Error logging usage statistics: {str(e)}")

            # Extract the analysis text with better error handling
            try:
                if 'content' in response_data and len(response_data['content']) > 0:
                    content = response_data['content'][0]
                    analysis_text = content['text'] if content['type'] == 'text' else str(content)
                    
                    # Return structured response
                    structured_response = {
                        'analysis_text': analysis_text,
                    }
                    
                    return structured_response
                else:
                    error_msg = "No content in response"
                    logger.error(error_msg)
                    return {
                        'analysis_text': error_msg,
                    }
            except Exception as e:
                error_msg = f"Error extracting content: {str(e)}"
                logger.error(error_msg)
                return {
                    'analysis_text': error_msg,
                }

        except Exception as e:
            error_msg = f"Error in vision analysis: {str(e)}"
            logger.error(error_msg)

--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/services/ai_handler.py ---
"""
AI Handler for managing LLM API calls.
"""
from typing import Dict, Any, Optional
from services.llm_service import LLMService

class AIHandler:
    def __init__(self):
        self.llm_service = LLMService()

    def make_api_call(self, model: str, message: str, system: Optional[str] = None) -> str:
        """Make an API call to the LLM service."""
        try:
            print(f"[AI Handler] Making API call with model: {model}")
            print(f"[AI Handler] Message: {message}")
            if system:
                print(f"[AI Handler] System prompt: {system}")

            # Create context with system prompt if provided
            context = {"system_prompt": system} if system else None

            response = self.llm_service.get_completion(
                message=message,
                model=model,
                context=context
            )
            
            if isinstance(response, dict):
                return response.get("content", "")
            return response

        except Exception as e:
            print(f"[AI Handler] Error in API call: {str(e)}")
            raise

# Global instance
ai_handler = AIHandler()


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/services/storage/vector_store.py ---
"""
Vector store service for embedding and retrieving molecular data.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    """Service for storing and retrieving vector embeddings of molecular data."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the vector store."""
        self.model = SentenceTransformer(model_name)
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
    
    def add_item(
        self,
        key: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an item to the vector store."""
        # Generate embedding
        embedding = self.model.encode([text])[0]
        
        # Store embedding and metadata
        self.embeddings[key] = embedding
        if metadata:
            self.metadata[key] = metadata
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search for similar items in the vector store."""
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Calculate similarities
        results = []
        for key, embedding in self.embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                embedding.reshape(1, -1)
            )[0][0]
            
            if similarity >= threshold:
                result = {
                    'key': key,
                    'similarity': float(similarity),
                    'metadata': self.metadata.get(key, {})
                }
                results.append(result)
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def get_item(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve an item by key."""
        if key in self.embeddings:
            return {
                'embedding': self.embeddings[key],
                'metadata': self.metadata.get(key, {})
            }
        return None
    
    def remove_item(self, key: str) -> bool:
        """Remove an item from the vector store."""
        if key in self.embeddings:
            del self.embeddings[key]
            if key in self.metadata:
                del self.metadata[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all items from the vector store."""
        self.embeddings.clear()
        self.metadata.clear()


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/services/audio/speech_service.py ---
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


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/utils/sbatch_utils.py ---
"""Utility functions for handling SBATCH job submissions and monitoring."""
import logging
import subprocess
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

# Constants for SBATCH operations
SBATCH_JOB_CHECK_INTERVAL = 10  # seconds
SBATCH_TIMEOUT = 3600  # 1 hour timeout

logger = logging.getLogger(__name__)

async def execute_sbatch(script_path: Path, *args: str) -> str:
    """Execute sbatch script and return job ID.
    
    Args:
        script_path: Path to the sbatch script to execute
        *args: Additional arguments to pass to the script
        
    Returns:
        str: Job ID of the submitted batch job
        
    Raises:
        RuntimeError: If sbatch execution fails
    """
    try:
        cmd = ['sbatch', str(script_path)]
        if args:
            cmd.extend(args)
            
        logger.info(f"Executing sbatch command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        # Extract job ID from sbatch output (typical format: "Submitted batch job 123456")
        job_id = result.stdout.strip().split()[-1]
        logger.info(f"Submitted batch job {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to execute sbatch: {e.stderr}")
        raise RuntimeError(f"Sbatch execution failed: {e.stderr}")

async def check_job_status(job_id: str) -> bool:
    """Check if sbatch job is complete.
    
    Args:
        job_id: ID of the job to check
        
    Returns:
        bool: True if job is complete, False if still running
    """
    try:
        result = subprocess.run(
            ['squeue', '-j', job_id],
            capture_output=True,
            text=True
        )
        # If job not found in queue, it's complete
        return "Invalid job id specified" in result.stderr or job_id not in result.stdout
    except subprocess.CalledProcessError:
        # If squeue fails, assume job is complete
        return True

async def wait_for_job_completion(job_id: str, timeout: Optional[int] = None) -> bool:
    """Wait for sbatch job to complete with timeout.
    
    Args:
        job_id: ID of the job to wait for
        timeout: Optional timeout in seconds (defaults to SBATCH_TIMEOUT)
        
    Returns:
        bool: True if job completed successfully, False if failed
        
    Raises:
        TimeoutError: If job doesn't complete within timeout period
    """
    timeout = timeout or SBATCH_TIMEOUT
    start_time = datetime.now()
    
    try:
        while True:
            try:
                if await check_job_status(job_id):
                    logger.info(f"Job {job_id} completed")
                    return True
            except RuntimeError as e:
                logger.error(f"Job failed: {str(e)}")
                return False
            
            if (datetime.now() - start_time).total_seconds() > timeout:
                raise TimeoutError(f"Job {job_id} timed out after {timeout} seconds")
            
            await asyncio.sleep(SBATCH_JOB_CHECK_INTERVAL)
    except Exception as e:
        logger.error(f"Error waiting for job completion: {str(e)}")
        return False



--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/utils/nmr_utils.py ---
"""
Utility functions for generating and loading NMR spectral data.
"""
import numpy as np
import pandas as pd
import json
import os

def generate_random_2d_correlation_points(x_range=(0, 10), y_range=(0, 200), num_points=20, diagonal=False):
    """Generate random correlation points for 2D NMR spectra."""
    if diagonal:
        # For COSY: Generate points near the diagonal
        x = np.random.uniform(x_range[0], x_range[1], num_points)
        y = x + np.random.normal(0, 0.3, num_points)  # Points close to diagonal with more spread
        # Add some off-diagonal correlations
        off_diag = int(num_points * 0.4)  # 40% off-diagonal peaks
        off_x = np.random.uniform(x_range[0], x_range[1], off_diag)
        off_y = np.random.uniform(y_range[0], y_range[1], off_diag)
        # Filter out points too close to diagonal
        mask = np.abs(off_y - off_x) > 0.5
        off_x = off_x[mask]
        off_y = off_y[mask]
        x = np.concatenate([x, off_x])
        y = np.concatenate([y, off_y])
    else:
        # For HSQC: Generate random points with clustering
        num_clusters = 5
        points_per_cluster = num_points // num_clusters
        x = []
        y = []
        for _ in range(num_clusters):
            center_x = np.random.uniform(x_range[0], x_range[1])
            center_y = np.random.uniform(y_range[0], y_range[1])
            cluster_x = center_x + np.random.normal(0, (x_range[1] - x_range[0])/20, points_per_cluster)
            cluster_y = center_y + np.random.normal(0, (y_range[1] - y_range[0])/20, points_per_cluster)
            x.extend(cluster_x)
            y.extend(cluster_y)
        x = np.array(x)
        y = np.array(y)
    
    # Generate varying intensities with some correlation to position
    z = 0.3 + 0.7 * np.random.beta(2, 2, len(x))  # Beta distribution for more realistic intensities
    
    return x, y, z

def generate_nmr_peaks(x=None, peak_positions=None, intensities=None):
    """Generate NMR peaks with Lorentzian line shape and multiplicity."""
    # Generate default values if not provided
    if x is None:
        x = np.linspace(0, 10, 1000)
    if peak_positions is None:
        peak_positions = np.random.uniform(1, 9, 8)  # 8 random peaks
    if intensities is None:
        intensities = np.random.uniform(0.3, 1.0, len(peak_positions))
        
    y = np.zeros_like(x)
    for pos, intensity in zip(peak_positions, intensities):
        # Lorentzian peak shape
        gamma = 0.02  # Peak width
        # Add main peak
        y += intensity * gamma**2 / ((x - pos)**2 + gamma**2)
        
        # Randomly add multiplicity (doublets, triplets)
        multiplicity = np.random.choice([1, 2, 3])  # singlet, doublet, triplet
        if multiplicity > 1:
            j_coupling = 0.1  # Typical J-coupling constant
            for i in range(1, multiplicity):
                # Add satellite peaks
                y += (intensity * 0.9) * gamma**2 / ((x - (pos + i*j_coupling))**2 + gamma**2)
                y += (intensity * 0.9) * gamma**2 / ((x - (pos - i*j_coupling))**2 + gamma**2)
    
    return x, y

def generate_default_nmr_data(plot_type='proton'):
    """Generate default NMR data based on plot type."""
    if plot_type in ['hsqc', 'cosy']:
        x_range = (0, 10) if plot_type == 'hsqc' else (0, 10)
        y_range = (0, 200) if plot_type == 'hsqc' else (0, 10)
        return generate_random_2d_correlation_points(x_range, y_range, 20, diagonal=(plot_type == 'cosy'))
    else:
        return generate_nmr_peaks()

def load_nmr_data_from_csv(smiles, csv_path=None):
    """Load NMR data for a specific molecule from the CSV file."""
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              'test_data', 'test_smiles_with_nmr.csv')
    
    try:
        if not os.path.exists(csv_path):
            print(f"[NMR Utils] CSV file not found: {csv_path}")
            return None
            
        df = pd.read_csv(csv_path)
        molecule_data = df[df['SMILES'] == smiles]
        
        if len(molecule_data) == 0:
            print(f"[NMR Utils] No data found for SMILES: {smiles}")
            return None
            
        # Parse the NMR data from the first matching row
        row = molecule_data.iloc[0]
        try:
            return {
                'proton': json.loads(row['1H_NMR']),
                'carbon': json.loads(row['13C_NMR']),
                'hsqc': json.loads(row['HSQC']),
                'cosy': json.loads(row['COSY'])
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[NMR Utils] Error parsing NMR data: {str(e)}")
            return None
    except Exception as e:
        print(f"[NMR Utils] Error loading NMR data: {str(e)}")
        return None

def generate_nmr_data(smiles, plot_type='proton', use_real_data=True):
    """Generate NMR data, using real data if available, falling back to random if not."""
    try:
        print(f"\n[NMR Utils] Generating NMR data for SMILES: {smiles}")

        using_random_data = False
        if use_real_data and smiles:
            print("[NMR Utils] Attempting to load real NMR data...")
            real_data = load_nmr_data_from_csv(smiles)
            if real_data and plot_type in real_data:
                print(f"[NMR Utils] Found real NMR data for {plot_type}")
                if plot_type in ['hsqc', 'cosy']:
                    return process_2d_nmr_data(real_data[plot_type], plot_type), False
                else:
                    return process_1d_nmr_data(real_data[plot_type], plot_type), False
            else:
                print("[NMR Utils] No real NMR data found, falling back to default data")
                using_random_data = True
                
        print("[NMR Utils] Generating random NMR data")
        return generate_default_nmr_data(plot_type), True
    except Exception as e:
        print(f"[NMR Utils] Error generating NMR data: {str(e)}")
        raise RuntimeError(f"Failed to generate {plot_type} NMR data: {str(e)}")

def process_1d_nmr_data(peaks_data, plot_type):
    """Process 1D NMR data (proton or carbon) into plottable format."""
    x = np.linspace(0, 10 if plot_type == 'proton' else 200, 1000)
    y = np.zeros_like(x)
    
    # Handle different data formats based on NMR type
    if plot_type == 'proton':
        # Proton NMR should always have [position, intensity] pairs
        if not isinstance(peaks_data[0], (list, tuple)):
            raise ValueError("Proton NMR data must be in [position, intensity] pairs format")
        for position, intensity in peaks_data:
            gamma = 0.02  # Narrow peaks for proton NMR
            y += intensity * gamma**2 / ((x - position)**2 + gamma**2)
    elif plot_type == 'carbon':
        # Carbon NMR can be either just positions or [position, intensity] pairs
        if isinstance(peaks_data[0], (list, tuple)):
            for position, intensity in peaks_data:
                gamma = 0.5  # Broader peaks for carbon NMR
                y += intensity * gamma**2 / ((x - position)**2 + gamma**2)
        else:
            for position in peaks_data:
                gamma = 0.5  # Broader peaks for carbon NMR
                y += 1.0 * gamma**2 / ((x - position)**2 + gamma**2)
    else:
        raise ValueError(f"Unsupported NMR type: {plot_type}")
    
    return x, y

def process_2d_nmr_data(correlation_data, plot_type):
    """Process 2D NMR data (HSQC or COSY) into plottable format."""
    x = []
    y = []
    z = []
    
    for correlation in correlation_data:
        x_pos, y_pos = correlation[:2]
        intensity = 1.0 if len(correlation) < 3 else correlation[2]
        x.append(x_pos)
        y.append(y_pos)
        z.append(intensity)
    
    return np.array(x), np.array(y), np.array(z)


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/utils/visualization.py ---
"""
Visualization utilities for molecule and plot generation.
"""
import io
import base64
import numpy as np
import plotly.graph_objects as go
from models.molecule import MoleculeHandler
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from PIL import Image

def create_molecule_image(mol, size=(400, 400)):
    """Create a 2D PIL Image of the molecule."""
    print(f"\n[Visualization] Creating 2D molecule image:")
    print(f"  - Molecule type: {type(mol)}")
    print(f"  - Molecule SMILES: {Chem.MolToSmiles(mol) if mol else None}")
    print(f"  - Requested size: {size}")
    
    try:
        # Generate 2D depiction
        img = Draw.MolToImage(mol, size=size)
        print("[Visualization] Successfully generated 2D molecule image:")
        print(f"  - Image type: {type(img)}")
        print(f"  - Image size: {img.size}")
        print(f"  - Image mode: {img.mode}")
        return img
    except Exception as e:
        print(f"[Visualization] ERROR creating molecule image:")
        print(f"  - Error type: {type(e)}")
        print(f"  - Error message: {str(e)}")
        print(f"  - Error args: {e.args}")
        import traceback
        print(f"  - Traceback:\n{traceback.format_exc()}")
        return None

def create_molecule_response(smiles, is_3d=False):
    """Create a response containing molecule visualization data."""
    print(f"\n[Visualization] Creating molecule response:")
    print(f"  - SMILES: {smiles}")
    print(f"  - 3D Mode: {is_3d}")
    
    try:
        # Convert SMILES to molecule
        print("[Visualization] Converting SMILES to RDKit molecule...")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("[Visualization] ERROR: Failed to create molecule from SMILES")
            print(f"  - Input SMILES: {smiles}")
            print("  - Possible issues:")
            print("    * Invalid SMILES syntax")
            print("    * Unsupported chemical features")
            print("    * Malformed input string")
            return None
            
        print("[Visualization] Successfully created RDKit molecule:")
        print(f"  - Canonical SMILES: {Chem.MolToSmiles(mol)}")
        print(f"  - Number of atoms: {mol.GetNumAtoms()}")
        print(f"  - Number of bonds: {mol.GetNumBonds()}")
        
        if is_3d:
            print("[Visualization] Preparing 3D response...")
            # For 3D, we'll send the SMILES string to be rendered by 3Dmol.js
            response = {
                            'smiles': smiles,
                            'is_3d': True,
                            'format': '3dmol',
                            'molecular_weight': "{:.2f}".format(MoleculeHandler().calculate_molecular_weight(mol))
            }
            print("[Visualization] Created 3D response object:")
            print(f"  - Response keys: {list(response.keys())}")
            
        else:
            print("[Visualization] Preparing 2D response...")
            # Generate 2D image
            img = create_molecule_image(mol)
            if img is None:
                print("[Visualization] ERROR: Failed to generate 2D molecule image")
                return None
                
            print("[Visualization] Converting image to base64...")
            # Convert image to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            print(f"  - Base64 string length: {len(img_str)}")
            
            # Create response data
            response = {
                            'smiles': smiles,
                            'image': img_str,
                            'format': 'png',
                            'encoding': 'base64',
                            'molecular_weight': "{:.2f}".format(MoleculeHandler().calculate_molecular_weight(mol)),
                            'is_3d': False,
                            'image_size': f"{img.size[0]}x{img.size[1]}"
            }
            print("[Visualization] Created 2D response object:")
            print(f"  - Response keys: {list(response.keys())}")
            print(f"  - Image size: {response['image_size']}")
        
        print("[Visualization] Successfully generated molecule response")
        return response
        
    except Exception as e:
        print(f"\n[Visualization] ERROR creating molecule response:")
        print(f"  - Error type: {type(e)}")
        print(f"  - Error message: {str(e)}")
        print(f"  - Generation mode: {'3D' if is_3d else '2D'}")
        import traceback
        print(f"  - Traceback:\n{traceback.format_exc()}")
        return None

def generate_random_molecule():
    """Generate a random molecule for testing."""
    print("[Visualization] Generating random molecule")
    try:
        # List of common SMILES for testing
        test_molecules = [
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',  # Testosterone
            'CC(C)(C)NCC(O)C1=CC(O)=CC(O)=C1',  # Salbutamol
            'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(N)(=O)=O)C(F)(F)F',  # Celecoxib
            'CC1=C(C=C(C=C1)O)C(=O)CC2=CC=C(C=C2)OC',  # Nabumetone
            'CC1=C(C(=O)N(N1C)C2=CC=CC=C2)C3=CC=CC=C3',  # Antipyrine
            'CC1=CC=C(C=C1)NC(=O)CN2CCN(CC2)CC3=CC=C(C=C3)OCC4=CC=CC=C4',  # Cinnarizine
        ]
        import random
        smiles = random.choice(test_molecules)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Failed to create molecule from SMILES: {smiles}")
            
        print(f"[Visualization] Generated random molecule: {smiles}")
        return mol, smiles
        
    except Exception as e:
        print(f"[Visualization] Error generating random molecule: {str(e)}")
        print(f"[Visualization] Error type: {type(e)}")
        print(f"[Visualization] Error args: {e.args}")
        return None, None

def create_plot_response(data):
    """Create a response object for plot visualization."""
    try:
        # Create figure
        fig = go.Figure()
        
        if data.get('type') == '2d':
            # Create scatter plot for HSQC or COSY
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='markers',
                marker=dict(
                    size=data.get('sizes', [10] * len(data['x'])),
                    color=data['z'],
                    colorscale=data.get('colorscale', 'Viridis'),
                    showscale=True,
                    opacity=0.7
                ),
                name='Correlations'
            ))
            
            # Reverse y-axis for NMR convention
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(autorange="reversed")
            
        else:  # 1D plot (1H or 13C NMR)
            # Main spectrum line
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='lines',
                line=dict(
                    color='rgb(0, 100, 200)',
                    width=1.5
                ),
                name='Spectrum'
            ))
            
            # Add vertical lines for peaks
            if 'peak_x' in data and 'peak_y' in data:
                fig.add_trace(go.Scatter(
                    x=data['peak_x'],
                    y=data['peak_y'],
                    mode='lines',
                    line=dict(
                        color='rgba(0, 100, 200, 0.5)',
                        width=1
                    ),
                    showlegend=False
                ))
            
            # Reverse x-axis for NMR convention
            fig.update_xaxes(autorange="reversed")
        
        # Update layout
        fig.update_layout(
            title=data.get('title', 'NMR Spectrum'),
            xaxis_title=data.get('x_label', 'Chemical Shift (ppm)'),
            yaxis_title=data.get('y_label', 'Intensity'),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=50, b=50),
            font=dict(family="Arial", size=12),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='lightgray',
                zerolinewidth=1
            )
        )
        
        # Convert to JSON
        plot_json = fig.to_json()
        
        return {
            'plot': plot_json,
            'title': data.get('title', 'NMR Spectrum')
        }
        
    except Exception as e:
        print(f"Error creating plot response: {str(e)}")
        return None


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/utils/__init__.py ---



--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/utils/results_manager.py ---
# """Unified manager for storing and retrieving all calculation and conversation results."""
# import pandas as pd
# from pathlib import Path
# import json
# from datetime import datetime
# import hashlib
# from typing import Dict, Any, Optional, List, Union, Tuple
# import logging
# from dataclasses import dataclass, asdict
# from enum import Enum

# class ResultType(Enum):
#     """Types of results that can be stored.
    
#     Types:
#         TARGET_MOLECULES: Input molecules (single or batch) with their properties
#         STARTING_MATERIALS: Starting materials (provided or from retrosynthesis)
#         FORWARD_PREDICTIONS: Forward synthesis predictions from starting materials
#         MORPHOMER_MOLECULES: Molecules generated by Morphomer
#         MMST_MOLECULES: Molecules generated by MMST
#         NMR_SIMULATION: Simulated NMR spectra for molecules
#         EXPERIMENTAL_DATA: Experimental NMR data
#         THRESHOLD_CALCULATION: Error thresholds for spectral matching
#         PEAK_MATCHING: Peak matching results per spectrum type
#         RETROSYNTHESIS: Retrosynthesis predictions and pathways
#         CONVERSATION_FULL: Complete conversation history with agents
#         CONVERSATION_SUMMARY: Condensed summary of key conversation points
#         WORKFLOW_STATE: Current state of the workflow
#         ERROR_LOG: Any errors or warnings during processing
#         PERFORMANCE_METRICS: Timing and resource usage metrics
#     """
#     # Molecule Types and Generation
#     TARGET_MOLECULES = "target_molecules"  # Input molecules with properties
#     STARTING_MATERIALS = "starting_materials"  # Either provided or from retrosynthesis
#     FORWARD_PREDICTIONS = "forward_predictions"  # Forward synthesis from starting materials
#     MORPHOMER_MOLECULES = "morphomer_molecules"  # Generated by Morphomer
#     MMST_MOLECULES = "mmst_molecules"  # Generated by MMST
    
#     # Spectral and Analysis Data
#     NMR_SIMULATION = "nmr_simulation"  # Per spectrum type (1H, 13C, HSQC, COSY)
#     EXPERIMENTAL_DATA = "experimental_data"  # Experimental NMR data
#     THRESHOLD_CALCULATION = "threshold_calculation"  # Error thresholds
#     PEAK_MATCHING = "peak_matching"  # Per spectrum type matching results
    
#     # Synthesis and Predictions
#     RETROSYNTHESIS = "retrosynthesis"  # Retrosynthesis paths and predictions
    
#     # Conversation and Workflow
#     CONVERSATION_FULL = "conversation_full"  # Complete conversation logs
#     CONVERSATION_SUMMARY = "conversation_summary"  # Condensed key points
#     WORKFLOW_STATE = "workflow_state"  # Current workflow state
    
#     # Monitoring and Performance
#     ERROR_LOG = "error_log"  # Error tracking
#     PERFORMANCE_METRICS = "performance_metrics"  # Timing and resources

# @dataclass
# class NMRMetadata:
#     """Common metadata for all NMR experiments."""
#     frequency: float  # Spectrometer frequency in MHz
#     solvent: str
#     temperature: float  # Temperature in K
#     experiment_type: str  # Pulse sequence details
#     acquisition_time: Optional[float] = None
#     number_of_scans: Optional[int] = None
#     relaxation_delay: Optional[float] = None
#     pulse_width: Optional[float] = None
#     digital_resolution: Optional[float] = None  # Hz/point

# @dataclass
# class ProtonNMRData:
#     """Structure for 1H NMR data."""
#     shifts: List[float]  # Chemical shifts in ppm
#     intensities: List[float]  # Peak intensities/integrals
#     multiplicities: Optional[List[str]] = None  # s, d, t, q, m, etc.
#     j_couplings: Optional[List[float]] = None  # J coupling constants in Hz
#     metadata: Optional[NMRMetadata] = None

# @dataclass
# class CarbonNMRData:
#     """Structure for 13C NMR data."""
#     shifts: List[float]  # Chemical shifts in ppm
#     peak_types: Optional[List[str]] = None  # CH3, CH2, CH, C
#     metadata: Optional[NMRMetadata] = None

# @dataclass
# class HSQCData:
#     """Structure for HSQC 2D NMR data."""
#     f1_shifts: List[float]  # 13C dimension shifts
#     f2_shifts: List[float]  # 1H dimension shifts
#     correlation_indices: List[Tuple[int, int]]  # Indices connecting F1 and F2 peaks
#     peak_types: Optional[List[str]] = None  # CH3, CH2, CH
#     metadata: Optional[NMRMetadata] = None

# @dataclass
# class COSYData:
#     """Structure for COSY 2D NMR data."""
#     f1_shifts: List[float]  # 1H dimension shifts
#     f2_shifts: List[float]  # 1H dimension shifts
#     correlation_indices: List[Tuple[int, int]]  # Indices connecting F1 and F2 peaks
#     correlation_types: Optional[List[str]] = None  # Strong, medium, weak
#     metadata: Optional[NMRMetadata] = None

# @dataclass
# class NMRData:
#     """Container for all types of NMR data."""
#     spectrum_type: str  # '1H', '13C', 'HSQC', 'COSY'
#     data: Union[ProtonNMRData, CarbonNMRData, HSQCData, COSYData]
    
#     @classmethod
#     def create_proton_nmr(cls, shifts: List[float], intensities: List[float], **kwargs) -> 'NMRData':
#         """Create a 1H NMR data instance."""
#         return cls(
#             spectrum_type='1H',
#             data=ProtonNMRData(shifts=shifts, intensities=intensities, **kwargs)
#         )
    
#     @classmethod
#     def create_carbon_nmr(cls, shifts: List[float], **kwargs) -> 'NMRData':
#         """Create a 13C NMR data instance."""
#         return cls(
#             spectrum_type='13C',
#             data=CarbonNMRData(shifts=shifts, **kwargs)
#         )
    
#     @classmethod
#     def create_hsqc(cls, f1_shifts: List[float], f2_shifts: List[float], 
#                    correlation_indices: List[Tuple[int, int]], **kwargs) -> 'NMRData':
#         """Create an HSQC data instance."""
#         return cls(
#             spectrum_type='HSQC',
#             data=HSQCData(
#                 f1_shifts=f1_shifts,
#                 f2_shifts=f2_shifts,
#                 correlation_indices=correlation_indices,
#                 **kwargs
#             )
#         )
    
#     @classmethod
#     def create_cosy(cls, f1_shifts: List[float], f2_shifts: List[float],
#                    correlation_indices: List[Tuple[int, int]], **kwargs) -> 'NMRData':
#         """Create a COSY data instance."""
#         return cls(
#             spectrum_type='COSY',
#             data=COSYData(
#                 f1_shifts=f1_shifts,
#                 f2_shifts=f2_shifts,
#                 correlation_indices=correlation_indices,
#                 **kwargs
#             )
#         )

# @dataclass
# class SpectrumThreshold:
#     """Threshold information for a specific spectrum type."""
#     value: float  # The calculated threshold value
#     confidence: Optional[float] = None  # Confidence score (0-1)
#     method: Optional[str] = None  # Method used to calculate threshold
#     parameters: Optional[Dict[str, Any]] = None  # Parameters used in calculation
#     statistics: Optional[Dict[str, float]] = None  # Statistical metrics (mean, std, etc.)

# @dataclass
# class ThresholdData:
#     """Complete threshold information for all spectrum types."""
#     proton: Optional[SpectrumThreshold] = None  # 1H NMR thresholds
#     carbon: Optional[SpectrumThreshold] = None  # 13C NMR thresholds
#     hsqc: Optional[SpectrumThreshold] = None    # HSQC thresholds
#     cosy: Optional[SpectrumThreshold] = None    # COSY thresholds
#     overall: Optional[float] = None             # Combined threshold
#     weights: Optional[Dict[str, float]] = None  # Weights used for overall calculation
    
#     def to_dict(self) -> Dict[str, float]:
#         """Convert to simple dictionary format for backward compatibility."""
#         return {
#             '1H': self.proton.value if self.proton else None,
#             '13C': self.carbon.value if self.carbon else None,
#             'HSQC': self.hsqc.value if self.hsqc else None,
#             'COSY': self.cosy.value if self.cosy else None
#         }
    
#     @classmethod
#     def from_dict(cls, data: Dict[str, float], **kwargs) -> 'ThresholdData':
#         """Create ThresholdData from simple dictionary format."""
#         return cls(
#             proton=SpectrumThreshold(value=data['1H']) if '1H' in data else None,
#             carbon=SpectrumThreshold(value=data['13C']) if '13C' in data else None,
#             hsqc=SpectrumThreshold(value=data['HSQC']) if 'HSQC' in data else None,
#             cosy=SpectrumThreshold(value=data['COSY']) if 'COSY' in data else None,
#             **kwargs
#         )

# @dataclass
# class MoleculeResult:
#     """Complete result set for a molecule."""
#     # Core Identifiers
#     molecule_id: str
#     smiles: str
#     sample_id: Optional[str] = None  # Sample identifier from experimental data
#     inchi: Optional[str] = None
#     inchi_key: Optional[str] = None
    
#     # Target Molecule Information
#     is_batch: bool = False  # Whether this is part of a batch
#     batch_id: Optional[str] = None  # ID of the batch this molecule belongs to
#     molecular_properties: Optional[Dict[str, Any]] = None  # Properties like MW, logP, etc.
    
#     # Starting Materials
#     starting_materials: Optional[List[str]] = None  # SMILES of starting materials
#     starting_materials_source: Optional[str] = None  # 'provided' or 'retrosynthesis'
    
#     # NMR Data (per spectrum type)
#     predicted_nmr: Optional[Dict[str, NMRData]] = None  # Simulated spectra
#     experimental_nmr: Optional[Dict[str, NMRData]] = None  # Experimental spectra
    
#     # Threshold Data
#     thresholds: Optional[ThresholdData] = None  # Structured threshold data
#     threshold_calculation_params: Optional[Dict[str, Any]] = None
    
#     # Peak Matching (per spectrum type)
#     peak_matches: Optional[Dict[str, List[Dict[str, float]]]] = None
#     match_scores: Optional[Dict[str, float]] = None
#     peak_matching_params: Optional[Dict[str, Any]] = None
    
#     # Retrosynthesis Data
#     retrosynthesis_predictions: Optional[List[Dict[str, Any]]] = None
#     retrosynthesis_params: Optional[Dict[str, Any]] = None
    
#     # Conversation History
#     conversation_history: Optional[List[Dict[str, Any]]] = None  # Full conversation
#     conversation_summary: Optional[str] = None  # Condensed summary
#     last_conversation_timestamp: Optional[str] = None
    
#     # Workflow State
#     workflow_state: Optional[str] = None  # Current state in the pipeline
#     completed_steps: Optional[List[str]] = None  # Steps completed so far
#     next_steps: Optional[List[str]] = None  # Upcoming steps
    
#     # Error Tracking
#     errors: Optional[List[Dict[str, Any]]] = None  # List of errors encountered
#     warnings: Optional[List[Dict[str, Any]]] = None  # List of warnings
    
#     # Performance Metrics
#     calculation_times: Optional[Dict[str, float]] = None  # Timing per operation
#     resource_usage: Optional[Dict[str, Any]] = None  # Memory, CPU usage etc.
    
#     # Metadata
#     timestamp: str = None  # Creation timestamp
#     last_modified: str = None  # Last modification
#     calculation_params: Optional[Dict[str, Any]] = None  # Global calculation parameters
#     status: str = "pending"  # Overall status
#     version: str = "1.0"  # Schema version
    
#     def __post_init__(self):
#         """Initialize timestamps if not provided."""
#         if not self.timestamp:
#             self.timestamp = datetime.now().isoformat()
#         if not self.last_modified:
#             self.last_modified = self.timestamp
            
#     def update_conversation(self, message: Dict[str, Any], generate_summary: bool = True):
#         """Add a conversation message and optionally update the summary."""
#         if not self.conversation_history:
#             self.conversation_history = []
        
#         self.conversation_history.append(message)
#         self.last_conversation_timestamp = datetime.now().isoformat()
        
#         if generate_summary:
#             # Here we would call an LLM to generate a summary
#             # For now, just take the last message
#             self.conversation_summary = f"Last action: {message.get('content', '')[:100]}..."
            
#     def add_error(self, error_type: str, message: str, details: Optional[Dict] = None):
#         """Add an error to the error log."""
#         if not self.errors:
#             self.errors = []
            
#         self.errors.append({
#             'type': error_type,
#             'message': message,
#             'details': details,
#             'timestamp': datetime.now().isoformat()
#         })
        
#     def update_workflow_state(self, new_state: str, completed_step: Optional[str] = None):
#         """Update the workflow state and completed steps."""
#         self.workflow_state = new_state
#         if completed_step:
#             if not self.completed_steps:
#                 self.completed_steps = []
#             self.completed_steps.append(completed_step)
#         self.last_modified = datetime.now().isoformat()

# class ResultsManager:
#     """Manages storage and retrieval of all calculation and conversation results."""
    
#     def __init__(self, base_dir: Optional[Path] = None):
#         """Initialize the results manager."""
#         self.base_dir = base_dir or Path(__file__).parent.parent / "data"
#         self.results_dir = self.base_dir / "results"
#         self.index_file = self.base_dir / "results_index.json"
        
#         # Create directories
#         self.results_dir.mkdir(parents=True, exist_ok=True)
        
#         # Initialize or load index
#         self.index = self._load_index()
#         self.logger = logging.getLogger(__name__)
    
#     def _serialize_dataclass(self, obj: Any) -> Dict[str, Any]:
#         """Serialize dataclass objects with proper type handling."""
#         if hasattr(obj, '__dataclass_fields__'):
#             result = {}
#             for field in obj.__dataclass_fields__:
#                 value = getattr(obj, field)
#                 if value is not None:
#                     if hasattr(value, '__dataclass_fields__'):
#                         result[field] = self._serialize_dataclass(value)
#                     elif isinstance(value, dict):
#                         result[field] = {k: self._serialize_dataclass(v) if hasattr(v, '__dataclass_fields__') else v
#                                        for k, v in value.items()}
#                     elif isinstance(value, (list, tuple)):
#                         result[field] = [self._serialize_dataclass(item) if hasattr(item, '__dataclass_fields__') else item
#                                        for item in value]
#                     else:
#                         result[field] = value
#             return result
#         return obj
    
#     def _deserialize_dataclass(self, data: Dict[str, Any], cls: Any) -> Any:
#         """Deserialize dictionary into appropriate dataclass."""
#         if not data:
#             return None
            
#         field_types = {field.name: field.type for field in cls.__dataclass_fields__.values()}
#         kwargs = {}
        
#         for key, value in data.items():
#             if key in field_types:
#                 field_type = field_types[key]
#                 if hasattr(field_type, '__dataclass_fields__'):
#                     kwargs[key] = self._deserialize_dataclass(value, field_type)
#                 elif hasattr(field_type, '__origin__') and field_type.__origin__ is dict:
#                     if value is None:
#                         kwargs[key] = None
#                     else:
#                         key_type, val_type = field_type.__args__
#                         if hasattr(val_type, '__dataclass_fields__'):
#                             kwargs[key] = {k: self._deserialize_dataclass(v, val_type) for k, v in value.items()}
#                         else:
#                             kwargs[key] = value
#                 else:
#                     kwargs[key] = value
                    
#         return cls(**kwargs)
    
#     def _load_index(self) -> Dict[str, Any]:
#         """Load or create the index file with proper deserialization."""
#         if self.index_file.exists():
#             with open(self.index_file, 'r') as f:
#                 raw_data = json.load(f)
#                 return {
#                     'molecules': {
#                         k: self._deserialize_dataclass(v, MoleculeResult)
#                         for k, v in raw_data.get('molecules', {}).items()
#                     },
#                     'calculations': raw_data.get('calculations', {}),
#                     'metadata': raw_data.get('metadata', {
#                         'last_updated': datetime.now().isoformat(),
#                         'version': '1.0'
#                     })
#                 }
#         return {
#             'molecules': {},
#             'calculations': {},
#             'metadata': {
#                 'last_updated': datetime.now().isoformat(),
#                 'version': '1.0'
#             }
#         }
    
#     def _save_index(self):
#         """Save the current index with proper serialization."""
#         save_data = {
#             'molecules': {
#                 k: self._serialize_dataclass(v)
#                 for k, v in self.index['molecules'].items()
#             },
#             'calculations': self.index['calculations'],
#             'metadata': {
#                 'last_updated': datetime.now().isoformat(),
#                 'version': '1.0'
#             }
#         }
#         with open(self.index_file, 'w') as f:
#             json.dump(save_data, f, indent=2)
    
#     def store_result(
#         self,
#         result_type: ResultType,
#         smiles: str,
#         data: Dict[str, Any],
#         context: Optional[Dict[str, Any]] = None
#     ) -> str:
#         """Store calculation or conversation results."""
#         try:
#             molecule_id = self._generate_molecule_id(smiles, context or {})
            
#             # Create or get existing molecule result
#             if molecule_id not in self.index['molecules']:
#                 molecule_result = MoleculeResult(
#                     molecule_id=molecule_id,
#                     smiles=smiles
#                 )
#             else:
#                 molecule_result = self.index['molecules'][molecule_id]
            
#             # Start performance tracking
#             start_time = datetime.now()
            
#             # Update based on result type
#             self._update_result(molecule_result, result_type, data, context)
            
#             # Track performance
#             end_time = datetime.now()
#             if not molecule_result.calculation_times:
#                 molecule_result.calculation_times = {}
#             molecule_result.calculation_times[result_type.value] = (end_time - start_time).total_seconds()
            
#             # Update metadata
#             molecule_result.last_modified = datetime.now().isoformat()
#             molecule_result.status = "completed"
            
#             # Save to index
#             self.index['molecules'][molecule_id] = molecule_result
#             self._save_index()
            
#             return molecule_id
            
#         except Exception as e:
#             self.logger.error(f"Error storing results: {str(e)}")
#             raise
    
#     def _update_result(
#         self,
#         molecule_result: MoleculeResult,
#         result_type: ResultType,
#         data: Dict[str, Any],
#         context: Optional[Dict[str, Any]]
#     ):
#         """Update molecule result based on result type."""
#         try:
#             if result_type == ResultType.TARGET_MOLECULES:
#                 molecule_result.molecular_properties = data.get('properties')
#                 molecule_result.is_batch = data.get('is_batch', False)
#                 molecule_result.batch_id = data.get('batch_id')
#                 if context and 'sample_id' in context:
#                     molecule_result.sample_id = context['sample_id']
                
#             elif result_type == ResultType.NMR_SIMULATION:
#                 if not molecule_result.predicted_nmr:
#                     molecule_result.predicted_nmr = {}
#                 spectrum_type = data['spectrum_type']
#                 nmr_data = self._create_nmr_data(spectrum_type, data['data'])
#                 molecule_result.predicted_nmr[spectrum_type] = nmr_data
                
#             elif result_type == ResultType.EXPERIMENTAL_DATA:
#                 if not molecule_result.experimental_nmr:
#                     molecule_result.experimental_nmr = {}
#                 spectrum_type = data['spectrum_type']
#                 nmr_data = self._create_nmr_data(spectrum_type, data['data'])
#                 molecule_result.experimental_nmr[spectrum_type] = nmr_data
                
#             elif result_type == ResultType.THRESHOLD_CALCULATION:
#                 molecule_result.thresholds = ThresholdData.from_dict(
#                     data.get('individual_thresholds'),
#                     overall=data.get('overall_threshold'),
#                     weights=data.get('weights')
#                 )
#                 molecule_result.threshold_calculation_params = context
                
#             elif result_type == ResultType.PEAK_MATCHING:
#                 molecule_result.peak_matches = data.get('matches')
#                 molecule_result.match_scores = data.get('scores')
#                 molecule_result.peak_matching_params = context
                
#             elif result_type == ResultType.RETROSYNTHESIS:
#                 molecule_result.starting_materials = data.get('starting_materials')
#                 molecule_result.starting_materials_source = 'retrosynthesis'
#                 molecule_result.retrosynthesis_predictions = data.get('predictions')
#                 molecule_result.retrosynthesis_params = context
                
#             elif result_type == ResultType.FORWARD_PREDICTIONS:
#                 if not hasattr(molecule_result, 'forward_predictions'):
#                     molecule_result.forward_predictions = []
#                 molecule_result.forward_predictions.append({
#                     'predictions': data.get('predictions'),
#                     'parameters': context,
#                     'timestamp': datetime.now().isoformat()
#                 })
                
#             elif result_type == ResultType.MORPHOMER_MOLECULES:
#                 if not hasattr(molecule_result, 'morphomer_results'):
#                     molecule_result.morphomer_results = []
#                 molecule_result.morphomer_results.append({
#                     'molecules': data.get('molecules'),
#                     'parameters': context,
#                     'timestamp': datetime.now().isoformat()
#                 })
                
#             elif result_type == ResultType.MMST_MOLECULES:
#                 if not hasattr(molecule_result, 'mmst_results'):
#                     molecule_result.mmst_results = []
#                 molecule_result.mmst_results.append({
#                     'molecules': data.get('molecules'),
#                     'parameters': context,
#                     'timestamp': datetime.now().isoformat()
#                 })
                
#             elif result_type == ResultType.CONVERSATION_FULL:
#                 molecule_result.update_conversation(data, generate_summary=True)
                
#             elif result_type == ResultType.ERROR_LOG:
#                 molecule_result.add_error(
#                     error_type=data.get('type'),
#                     message=data.get('message'),
#                     details=data.get('details')
#                 )
                
#             elif result_type == ResultType.WORKFLOW_STATE:
#                 molecule_result.update_workflow_state(
#                     new_state=data.get('state'),
#                     completed_step=data.get('completed_step')
#                 )
                
#         except Exception as e:
#             self.logger.error(f"Error updating result type {result_type}: {str(e)}")
#             raise
    
#     def _create_nmr_data(self, spectrum_type: str, data: Dict[str, Any]) -> NMRData:
#         """Create appropriate NMR data instance based on spectrum type."""
#         if spectrum_type == '1H':
#             return NMRData.create_proton_nmr(**data)
#         elif spectrum_type == '13C':
#             return NMRData.create_carbon_nmr(**data)
#         elif spectrum_type == 'HSQC':
#             return NMRData.create_hsqc(**data)
#         elif spectrum_type == 'COSY':
#             return NMRData.create_cosy(**data)
#         else:
#             raise ValueError(f"Unknown spectrum type: {spectrum_type}")
    
#     def delete_result(self, molecule_id: str):
#         """Delete a result from the database."""
#         if molecule_id in self.index['molecules']:
#             del self.index['molecules'][molecule_id]
#             self._save_index()
    
#     def export_database(self, output_file: Path):
#         """Export the entire database to a file."""
#         with open(output_file, 'w') as f:
#             json.dump(self._serialize_dataclass(self.index), f, indent=2)
    
#     def import_database(self, input_file: Path):
#         """Import database from a file."""
#         with open(input_file, 'r') as f:
#             raw_data = json.load(f)
#             self.index = self._deserialize_dataclass(raw_data, dict)
#             self._save_index()

#     def get_molecules_by_timestamp(self, start_time: datetime = None, end_time: datetime = None) -> List[MoleculeResult]:
#         """Get molecules filtered by timestamp range.
        
#         Args:
#             start_time: Optional start time to filter from
#             end_time: Optional end time to filter to
            
#         Returns:
#             List of MoleculeResult objects within the time range
#         """
#         results = []
#         for molecule in self.index['molecules'].values():
#             timestamp = datetime.fromisoformat(molecule.timestamp)
#             if start_time and timestamp < start_time:
#                 continue
#             if end_time and timestamp > end_time:
#                 continue
#             results.append(molecule)
#         return results
    
#     def get_molecules_by_source(self, source: str) -> List[MoleculeResult]:
#         """Get molecules filtered by source.
        
#         Args:
#             source: Source identifier (e.g., 'csv_upload')
            
#         Returns:
#             List of MoleculeResult objects from the specified source
#         """
#         results = []
#         for molecule in self.index['molecules'].values():
#             context = molecule.calculation_params or {}
#             if context.get('source') == source:
#                 results.append(molecule)
#         return results
    
#     def get_latest_upload_results(self) -> List[MoleculeResult]:
#         """Get molecules from the most recent CSV upload.
        
#         Returns:
#             List of MoleculeResult objects from the latest upload
#         """
#         # Find the latest upload timestamp
#         latest_time = None
#         latest_molecules = []
        
#         for molecule in self.index['molecules'].values():
#             context = molecule.calculation_params or {}
#             if context.get('source') == 'csv_upload':
#                 timestamp = datetime.fromisoformat(context.get('timestamp', '1970-01-01T00:00:00'))
#                 if not latest_time or timestamp > latest_time:
#                     latest_time = timestamp
#                     latest_molecules = [molecule]
#                 elif timestamp == latest_time:
#                     latest_molecules.append(molecule)
        
#         return latest_molecules


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/utils/file_utils.py ---
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


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/utils/file_handlers.py ---
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


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/routes/main.py ---
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


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/routes/audio.py ---
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


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/routes/file_upload.py ---
"""
File upload handling routes for the LLM Structure Elucidator.
"""
from flask import Blueprint, request, jsonify
import pandas as pd
from datetime import datetime
import json
import ast
import shutil
from rdkit import Chem
from rdkit.Chem import Descriptors
from models.molecule import MoleculeHandler
from utils.file_utils import save_uploaded_file, uploaded_smiles
from pathlib import Path
from typing import Dict, Any
from agents.orchestrator.workflow_definitions import determine_workflow_type, WorkflowType

file_upload = Blueprint('file_upload', __name__)

def parse_csv_data(row: pd.Series) -> Dict[str, Any]:
    """Parse all data from CSV row into a dictionary structure with molecule_data key.
    
    The structure will be:
    {
        'sample_id': str,
        'smiles': str,
        'molecule_data': {
            'smiles': str,
            'sample_id': str,
            'molecular_weight': float,
            'starting_materials': List[str],
            'timestamp': str,
            'nmr_data': {
                '1H_exp': {...},
                '13C_exp': {...},
                'HSQC_exp': {...},
                'COSY_exp': {...}
            }
        }
    }
    """
    # Get sample ID or generate one if not present
    sample_id = row.get('sample-id', f"SAMPLE_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    smiles = row.get('SMILES', '')
    
    molecule_data = {
        'smiles': smiles,
        'sample_id': sample_id,
        'starting_smiles': row.get('starting_smiles', '').split(';') if row.get('starting_smiles') else [],
        'timestamp': datetime.now().isoformat(),
        'nmr_data': {}
    }
    
    # Calculate molecular formula and weight using RDKit
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            molecule_data['molecular_weight'] = Descriptors.ExactMolWt(mol)
            # Calculate molecular formula
            molecule_data['molecular_formula'] = Chem.rdMolDescriptors.CalcMolFormula(mol)
        else:
            print(f"Warning: Could not parse SMILES string for molecular calculations: {smiles}")
    except Exception as e:
        print(f"Error calculating molecular properties: {str(e)}")

    if '1H_NMR' in row and row['1H_NMR']:
        try:
            peaks = ast.literal_eval(row['1H_NMR'])
            molecule_data['nmr_data']['1H_exp'] = peaks
        except Exception as e:
            print(f"Error parsing 1H NMR data: {str(e)}")

    if '13C_NMR' in row and row['13C_NMR']:
        try:
            shifts = ast.literal_eval(row['13C_NMR'])
            molecule_data['nmr_data']['13C_exp'] = shifts
        except Exception as e:
            print(f"Error parsing 13C NMR data: {str(e)}")

    if 'HSQC' in row and row['HSQC']:
        try:
            correlations = ast.literal_eval(row['HSQC'])
            molecule_data['nmr_data']['HSQC_exp'] = correlations
        except Exception as e:
            print(f"Error parsing HSQC data: {str(e)}")

    if 'COSY' in row and row['COSY']:
        try:
            correlations = ast.literal_eval(row['COSY'])
            molecule_data['nmr_data']['COSY_exp'] = correlations
        except Exception as e:
            print(f"Error parsing COSY data: {str(e)}")

    return {
        'sample_id': sample_id,
        'smiles': row.get('SMILES', ''),
        'molecule_data': molecule_data
    }

def save_molecular_data(all_data: Dict[str, Any], filepath: str):
    """Save complete molecular data dictionary to a JSON file.
    Archives existing data file if present.
    
    Args:
        all_data: Dictionary containing all molecular data with sample_ids as keys
        filepath: Path to the source CSV file
    """
    data_dir = Path(__file__).parent.parent / "data" / "molecular_data"
    archive_dir = data_dir / "archive"
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    json_file = data_dir / "molecular_data.json"
    
    # Archive existing file if it exists
    if json_file.exists():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_file = archive_dir / f"molecular_data_{timestamp}.json"
        shutil.move(json_file, archive_file)
        print(f"Archived existing molecular_data.json to {archive_file}")
    
    # Save new data to file
    with open(json_file, 'w') as f:
        json.dump(all_data, f, indent=2)

def archive_molecular_data():
    """Archive existing molecular_data.json file if it exists."""
    molecular_data_dir = Path(__file__).parent.parent / "data" / "molecular_data"
    target_file = molecular_data_dir / "molecular_data.json"
    
    if target_file.exists():
        archive_dir = molecular_data_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"molecular_data_{timestamp}.json"
        shutil.move(target_file, archive_path)
        print(f"Archived existing molecular_data.json to {archive_path}")
        return True
    return False

@file_upload.route('/upload_file', methods=['POST'])
def upload_file():
    """Handle file upload for both CSV and JSON files."""
    try:
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        
        # Handle JSON file upload
        if file.filename.lower() == 'molecular_data.json':
            molecular_data_dir = Path(__file__).parent.parent / "data" / "molecular_data"
            target_file = molecular_data_dir / "molecular_data.json"
            
            # Archive existing file if present
            archive_molecular_data()
            
            # Save new file
            file.save(target_file)
            print(f"Saved new molecular_data.json to {target_file}")
            return jsonify({'message': 'Successfully uploaded molecular_data.json'})
            
        # Handle CSV file upload
        elif file.filename.endswith('.csv'):
            # Save the file
            filepath = save_uploaded_file(file)
            print(f"Saved file to: {filepath}")

            # Read CSV file
            df = pd.read_csv(filepath)
            print(f"Loaded CSV with columns: {df.columns.tolist()}")
            
            # Process each row and build complete data dictionary
            molecule_handler = MoleculeHandler()
            all_molecular_data = {}
            processed_samples = []
            
            for idx, row in df.iterrows():
                smiles = row['SMILES']
                if molecule_handler.validate_smiles(smiles):
                    # Parse data for this sample
                    data = parse_csv_data(row)

                    sample_id = data['sample_id']
                    
                    # Add source file and row index
                    data["molecule_data"]['source_file'] = str(filepath)
                    data["molecule_data"]['row_index'] = idx
                    
                    # Add workflow type
                    workflow_type = determine_workflow_type(data)
                    data["molecule_data"]['workflow_type'] = workflow_type.value
                    
                    # Add to complete dictionary
                    all_molecular_data[sample_id] = data#["molecule_data"]

                    processed_samples.append(sample_id)
            
            if not processed_samples:
                return jsonify({'error': 'No valid molecules found in CSV'}), 400
            
            # Save complete data dictionary
            save_molecular_data(all_molecular_data, filepath)
            
            return jsonify({
                'message': f'Successfully processed {len(processed_samples)} molecules',
                'sample_ids': processed_samples
            })
            
        else:
            print(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Please upload a CSV file or molecular_data.json'}), 400
            
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@file_upload.route('/get_molecular_data', methods=['GET'])
def get_molecular_data():
    """Get all molecular data or filter by sample_id."""
    try:
        sample_id = request.args.get('sample_id')
        data_file = Path(__file__).parent.parent / "data" / "molecular_data" / "molecular_data.json"
        
        if not data_file.exists():
            return jsonify({'error': 'No molecular data found'}), 404
            
        with open(data_file, 'r') as f:
            all_data = json.load(f)
            
        if sample_id:
            if sample_id not in all_data:
                return jsonify({'error': f'Sample ID {sample_id} not found'}), 404
            return jsonify(all_data[sample_id])
        
        return jsonify(all_data)
            
    except Exception as e:
        print(f"Error getting molecular data: {str(e)}")
        return jsonify({'error': str(e)}), 500


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/routes/__init__.py ---



--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/routes/structure.py ---
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


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/routes/data_routes.py ---
from flask import jsonify
from pathlib import Path
import json

def init_data_routes(app):
    @app.route('/get_molecular_data', methods=['GET'])
    def get_molecular_data():
        try:
            json_path = Path(app.root_path) / "data" / "molecular_data" / "molecular_data.json"
            if not json_path.exists():
                return jsonify({"error": "No molecular data found"}), 404
                
            with open(json_path, 'r') as f:
                data = json.load(f)
            return jsonify({"status": "success", "data": data})
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/get_first_molecule_json', methods=['GET'])
    def get_first_molecule_json():
        try:
            json_path = Path(app.root_path) / "data" / "molecular_data" / "molecular_data.json"
            if not json_path.exists():
                return jsonify({"error": "No molecular data found"}), 404
                
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            if not data:
                return jsonify({"error": "No molecules found in database"}), 404
                
            # Get the first molecule's data
            first_molecule_id = next(iter(data))
            first_molecule = data[first_molecule_id]
            
            return jsonify({
                "status": "success",
                "sample_id": first_molecule_id,
                "smiles": first_molecule.get("smiles"),
                "inchi": first_molecule.get("inchi"),
                "inchi_key": first_molecule.get("inchi_key"),
                "nmr_data": first_molecule.get("nmr_data", {})
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/config/__init__.py ---



--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/config/settings.py ---
"""
Configuration settings for the LLM Structure Elucidator application.
"""
import os
from datetime import datetime
from .config import anthropic_api_key, openai_api_key, elevenlabs_key, gemini_api_key, deepseek_api_key

# API Keys
ANTHROPIC_API_KEY = anthropic_api_key
OPENAI_API_KEY = openai_api_key
ELEVENLABS_KEY = elevenlabs_key
GEMINI_API_KEY = gemini_api_key
DEEPSEEK_API_KEY = deepseek_api_key

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


--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/config/config.py ---
# config.py

# Anthropic API key
anthropic_api_key = "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"

# OpenAI API key - strip the 'export' part and just use the key value
openai_api_key = "sk-XgwEtky_vrBaMegIMfbl7Mnh6qRSoA7nkzL7-F5KtVT3BlbkFJab5vQBYLC0svpyPM7B410PCCBxZnmxMwodsvrgIMgA"

elevenlabs_key = "sk_7cae25cb57b6d2758af3df95a8e15ff60c9246fbffe42deb"

gemini_api_key = "AIzaSyDWOYTJyf3LqXDBOKENGZ6yWJRtt12ztRA"

deepseek_api_key = "sk-e98b388e947e4e53a15d9b73fa1fe654"

deepseek_azure_api_key = "A3KWGWtHtx6wkhQr9pqGohdcqN4N2nEXE6H7agwliDgIsDkEarZbJQQJ99BBACHYHv6XJ3w3AAAAACOGXUoW"



--- /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_explainability/LLM_Structure_Elucidator/config/config.template.py ---
# config.template.py
# Copy this file to config.py and replace with your actual API keys

# Anthropic API key
anthropic_api_key = "your-anthropic-api-key-here"

# OpenAI API key
openai_api_key = "your-openai-api-key-here"

# ElevenLabs API key
elevenlabs_key = "your-elevenlabs-api-key-here"

# Gemini API key
gemini_api_key = "your-google-api-key-here"

