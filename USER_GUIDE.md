# ChemStructLLM User Guide

## Overview

ChemStructLLM is an AI-powered structure elucidation tool that combines experimental NMR data with large language models to assist in molecular structure determination. This guide will walk you through all the features and how to use them effectively.

## Getting Started

### 1. Starting the Application

Run the application using:
```bash
python app.py
```

The web interface will be available at `http://localhost:5000`

**📸 Screenshot needed:** *Main application startup screen*

### 2. Remote Server Access (SSH Tunneling)

If you're running ChemStructLLM on a remote server or cluster node, you'll need to set up SSH tunneling to access the web interface from your local machine.

#### Setting Up SSH Tunnel

1. **Basic SSH Tunnel Setup**
   ```bash
   ssh -L 5000:localhost:5000 username@remote-server.domain.com
   ```

2. **Custom Port Tunneling** (if port 5000 is occupied)
   ```bash
   # Tunnel remote port 5000 to local port 5001
   ssh -L 5001:localhost:5000 username@remote-server.domain.com
   
   # Then access via: http://localhost:5001
   ```

3. **Example for Cluster Access**
   ```bash
   # Replace with your actual username and server
   ssh -L 5001:localhost:5000 your_username@cluster-node-name
   ```

#### After SSH Tunnel is Established

1. **On the remote server**, start the application:
   ```bash
   cd LLM_Structure_Elucidator
   python run.py
   # Application runs on remote localhost:5000
   ```

2. **On your local machine**, open your browser and navigate to:
   - `http://localhost:5000` (if using direct port mapping)
   - `http://localhost:5001` (if using custom port mapping)

#### Troubleshooting SSH Tunneling

- **Port already in use**: Try a different local port (e.g., 5001, 5002)
- **Connection refused**: Ensure the application is running on the remote server
- **Tunnel drops**: Add `-o ServerAliveInterval=60` to keep connection alive
  ```bash
  ssh -L 5001:localhost:5000 -o ServerAliveInterval=60 username@server
  ```



---

## Core Features

### 1. File Upload and Data Management

#### Uploading Experimental Data

1. **Choose Your Data File**
   - Click the file upload area or drag and drop your CSV file
   - Supported format: CSV files containing experimental NMR data
   - Required columns: `SMILES`, `Sample_ID`, `1H_NMR`, `13C_NMR`, `HSQC`, `COSY`

2. **File Processing**
   - The system automatically parses your CSV data
   - Molecular structures are generated from SMILES strings
   - NMR data is stored and linked to each molecule

**📸 Screenshots needed:**
- *File upload interface*
- *Example CSV file format*
- *Successful upload confirmation*

#### Supported Data Format

Your CSV file should contain:
```csv
SMILES,Sample_ID,1H_NMR,13C_NMR,HSQC,COSY
"Cc1ccc(NC(C(=O)O)c2ccc3c(c2)OCO3)cc1","SAMPLE002","[(2.2513137, 3), (5.049414, 1), (5.968915, 2), (6.631792, 0.5), ...]","[19.225267, 58.290585, 59.04912, 64.90056, 104.40221, ...]","[[2.2513137, 19.225267], [5.049414, 59.04912], [5.968915, 104.40221], ...]","[(2.2513137, 2.2513137), (5.049414, 5.049414), (6.651792, 6.6657114), ...]"
```

**Data Format Explanation:**
- **1H NMR**: `[(chemical_shift, intensity), (chemical_shift, intensity), ...]`
  - Example: `[(2.2513137, 3), (5.049414, 1)]` = peak at 2.25 ppm with intensity 3
- **13C NMR**: `[chemical_shift, chemical_shift, chemical_shift, ...]`
  - Example: `[19.225267, 58.290585, 59.04912]` = carbon peaks at these chemical shifts
- **HSQC**: `[[1H_shift, 13C_shift], [1H_shift, 13C_shift], ...]`
  - Example: `[[2.2513137, 19.225267], [5.049414, 59.04912]]` = correlations between 1H and 13C
- **COSY**: `[(1H_shift1, 1H_shift2), (1H_shift1, 1H_shift2), ...]`
  - Example: `[(2.2513137, 2.2513137), (6.651792, 6.6657114)]` = proton-proton correlations

---

### 2. Interactive Chat Interface

#### Text-Based Interaction

1. **Type Your Queries**
   - Use the text input at the bottom of the chat panel
   - Ask questions about molecules, request NMR plots, or seek analysis
   - Examples:
     - "Show me the 1H NMR spectrum"
     - "Display molecule 1"
     - "What is the molecular weight?"

2. **Send Messages**
   - Click the "Send" button or press Enter
   - The AI will process your request and provide responses

**📸 Screenshots needed:**
- *Chat interface with example queries*
- *AI responses with molecular information*

#### Voice Interaction

1. **Voice Input**
   - Click the microphone button 🎤
   - Speak your question clearly
   - The system will convert speech to text automatically

2. **Voice Output**
   - The AI can read responses aloud
   - Toggle audio output in the interface settings

**📸 Screenshots needed:**
- *Microphone button and voice input interface*
- *Voice-to-text conversion in action*

---

### 3. AI Model Selection

#### Choosing Your AI Model

1. **Model Options**
   - Multiple AI models available (GPT, Claude, Gemini, etc.)
   - Each model has different strengths for chemical analysis
   - Switch models based on your specific needs

2. **Model Selection**
   - Use the model dropdown in the chat interface
   - Changes take effect for subsequent queries
   - Previous conversation history is maintained

**📸 Screenshots needed:**
- *Model selection dropdown*
- *Different model responses comparison*

---

### 4. NMR Data Visualization

#### Available NMR Spectra

1. **1D NMR Spectra**
   - **1H NMR**: Proton NMR spectra
   - **13C NMR**: Carbon-13 NMR spectra
   - Request with: "show 1H" or "show 13C"

2. **2D NMR Spectra**
   - **HSQC**: Heteronuclear Single Quantum Coherence
   - **COSY**: Correlation Spectroscopy
   - Request with: "show HSQC" or "show COSY"

#### Viewing Spectra

1. **Request Spectra**
   - Type your request in the chat: "show HSQC"
   - The system uses your uploaded experimental data
   - Interactive plots appear in the visualization panel

2. **Plot Features**
   - Zoom and pan capabilities
   - Peak identification
   - Integration values (where applicable)

**📸 Screenshots needed:**
- *1H NMR spectrum display*
- *HSQC 2D plot*
- *Interactive plot controls*

---

### 5. Molecular Visualization

#### 2D Molecular Structures

1. **Viewing Molecules**
   - Request specific molecules: "show mol 1"
   - 2D structures are automatically generated
   - High-quality chemical structure drawings

2. **Molecule Navigation**
   - Browse through uploaded molecules
   - Compare different structures side-by-side

#### 3D Molecular Models

1. **3D Visualization**
   - Interactive 3D molecular models
   - Rotate, zoom, and explore molecular geometry
   - Useful for understanding stereochemistry

**📸 Screenshots needed:**
- *2D molecular structure display*
- *3D molecular model interface*
- *Multiple molecule comparison view*

---

### 6. Structure Elucidation Workflow

#### Running the Elucidation Process

1. **The Green Button** 🟢
   - Located in the chat interface: "Run Structure Elucidation"
   - Initiates the complete AI-powered analysis workflow
   - Combines all available NMR data with AI reasoning

2. **Workflow Process**
   - **Data Integration**: Combines 1H, 13C, HSQC, and COSY data
   - **AI Analysis**: Multiple AI agents analyze different aspects
   - **Structure Prediction**: Generates possible molecular structures
   - **Confidence Scoring**: Provides reliability metrics

3. **Output Generation**
   - Creates a comprehensive JSON file with results
   - Contains all analysis steps and reasoning
   - Includes structure predictions and confidence scores

**📸 Screenshots needed:**
- *Structure elucidation button*
- *Workflow progress indicators*
- *Generated JSON output preview*

#### Understanding the Results

The generated JSON file contains:
```json
{
  "analysis_results": {
    "molecular_formula": "C16H15NO4",
    "predicted_structures": [...],
    "confidence_scores": {...},
    "nmr_analysis": {...},
    "ai_reasoning": [...]
  }
}
```

---

### 7. Advanced Analysis with Jupyter Notebooks

#### Post-Processing Analysis

1. **Generated Data Files**
   - JSON files contain complete analysis results
   - Located in the `results/` directory
   - Timestamped for easy identification

2. **Jupyter Notebook Integration**
   - Use provided analysis notebooks
   - Load JSON results for detailed examination
   - Create custom visualizations and reports

3. **Available Notebooks**
   - `structure_analysis.ipynb`: Detailed structure comparison
   - `nmr_interpretation.ipynb`: Advanced NMR analysis
   - `confidence_evaluation.ipynb`: Result validation

**📸 Screenshots needed:**
- *Generated JSON file structure*
- *Jupyter notebook with loaded results*
- *Custom analysis plots*

---

### 8. Customizing the Workflow

#### Workflow Configuration Options

1. **Agent Configuration**
   - Located in `config/agent_config.json`
   - Modify AI model assignments for different tasks
   - Adjust analysis parameters

2. **NMR Processing Settings**
   - Configure peak picking algorithms
   - Adjust integration parameters
   - Set noise filtering options

3. **Structure Generation Options**
   - Modify structure prediction algorithms
   - Set confidence thresholds
   - Configure output formats

#### Customization Locations

```
ChemStructLLM/
├── config/
│   ├── agent_config.json      # AI agent settings
│   ├── nmr_config.json        # NMR processing parameters
│   └── workflow_config.json   # Overall workflow settings
├── agents/
│   └── specialized/           # Individual agent implementations
└── workflows/
    └── elucidation/          # Structure elucidation workflows
```

**📸 Screenshots needed:**
- *Configuration file examples*
- *Workflow customization interface*

---

## Interface Features

### Chat Management

1. **Clear Chat History**
   - Reset conversation for new analysis
   - Maintains uploaded data and current molecule selection

2. **Export Conversations**
   - Save chat history for documentation
   - Include AI reasoning and analysis steps

### Data Selection

1. **Molecule Selection**
   - Choose which uploaded molecule to analyze
   - Switch between different samples easily
   - Maintain context across different queries

2. **Spectrum Selection**
   - Choose which NMR data to display
   - Compare different spectral types
   - Overlay multiple spectra when relevant

**📸 Screenshots needed:**
- *Chat management controls*
- *Data selection interface*
- *Molecule switching demonstration*

---

## Tips for Best Results

### Data Preparation

1. **CSV File Quality**
   - Ensure SMILES strings are valid
   - Include all available NMR data types
   - Use consistent data formatting

2. **NMR Data Format**
   - Provide peak lists or full spectral data
   - Include integration values when available
   - Specify measurement conditions if relevant

### Effective Querying

1. **Specific Requests**
   - Be specific about what you want to see
   - Use standard NMR terminology
   - Ask follow-up questions for clarification

2. **Sequential Analysis**
   - Start with basic molecular information
   - Progress to specific spectral analysis
   - Use structure elucidation as final step

---

## Troubleshooting

### Common Issues

1. **File Upload Problems**
   - Check CSV format matches requirements
   - Ensure SMILES strings are valid
   - Verify file size is within limits

2. **NMR Display Issues**
   - Confirm experimental data was uploaded correctly
   - Check that molecule selection is active
   - Verify NMR data format in CSV

3. **AI Response Issues**
   - Try different AI models for varied perspectives
   - Rephrase questions if responses are unclear
   - Check internet connection for model access

### Getting Help

- Check the console logs for detailed error messages
- Review the generated JSON files for analysis details
- Consult the technical documentation for advanced configuration

---

## Conclusion

ChemStructLLM provides a comprehensive platform for AI-assisted structure elucidation. By combining experimental NMR data with advanced AI reasoning, it offers powerful tools for molecular analysis and structure determination.

For technical support or advanced customization, refer to the developer documentation or contact the development team.

---

*This guide covers the main features of ChemStructLLM. For the most current information and updates, please refer to the project repository.*
