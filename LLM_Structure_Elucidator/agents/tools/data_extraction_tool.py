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
