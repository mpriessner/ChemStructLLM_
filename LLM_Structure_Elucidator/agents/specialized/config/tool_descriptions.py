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
