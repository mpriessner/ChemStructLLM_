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
        # WORKFLOW_STEPS['nmr_simulation'],
        # WORKFLOW_STEPS['peak_matching'],
        WORKFLOW_STEPS['retrosynthesis'],
        WORKFLOW_STEPS['forward_prediction'],
        WORKFLOW_STEPS['forward_candidate_analysis'],
        WORKFLOW_STEPS['mol2mol'], 
        WORKFLOW_STEPS['mol2mol_candidate_analysis'],
        WORKFLOW_STEPS['mmst'],
        WORKFLOW_STEPS['mmst_candidate_analysis'],
        WORKFLOW_STEPS['analysis'] 
    ],
    WorkflowType.STARTING_MATERIAL: [
        # WORKFLOW_STEPS['threshold_calculation'],
        # # WORKFLOW_STEPS['nmr_simulation'],
        # WORKFLOW_STEPS['forward_prediction'], # need to add different logic for experimental data comparison?
        # WORKFLOW_STEPS['forward_candidate_analysis'],
        # WORKFLOW_STEPS['mol2mol'],
        # WORKFLOW_STEPS['mol2mol_candidate_analysis'],
        # WORKFLOW_STEPS['mmst'],
        WORKFLOW_STEPS['mmst_candidate_analysis'],
        # WORKFLOW_STEPS['candidate_analysis'],
        WORKFLOW_STEPS['analysis']
    ],
    WorkflowType.MULTIPLE_TARGETS: [
        # WORKFLOW_STEPS['threshold_calculation'],
        # WORKFLOW_STEPS['retrosynthesis'],
        # WORKFLOW_STEPS['nmr_simulation'],
        # WORKFLOW_STEPS['peak_matching'],
        # WORKFLOW_STEPS['forward_prediction'],
        # WORKFLOW_STEPS['forward_candidate_analysis'],
        # WORKFLOW_STEPS['mol2mol'],
        # WORKFLOW_STEPS['mol2mol_candidate_analysis'],
        # WORKFLOW_STEPS['mmst'],
        # WORKFLOW_STEPS['mmst_candidate_analysis'],
        # WORKFLOW_STEPS['candidate_analysis'],
        # WORKFLOW_STEPS['analysis']
    ],
    WorkflowType.SPECTRAL_ONLY: [
        # WORKFLOW_STEPS['threshold_calculation'],
        # WORKFLOW_STEPS['peak_matching'],
        # WORKFLOW_STEPS['analysis']
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
