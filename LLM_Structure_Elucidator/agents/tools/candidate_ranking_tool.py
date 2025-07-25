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
            analysis_types = ['forward_synthesis', 'mol2mol', 'mmst']
            
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
            # Get IUPAC names for top results only
            for result in top_results:
                iupac_result = await self.stout_ops.get_iupac_name(result['smiles'])
                result['iupac_name'] = iupac_result.get('iupac_name', 'Not available')

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
                    prompt = ""
                    # prompt = f"""
                    # Analyze why this molecule ranked #{rank}:
                    # SMILES: {result['smiles']}
                    # IUPAC: {result.get('iupac_name', 'Not available')}
                    
                    # HSQC NMR Error Score: {result['scores']['HSQC']:.6f}
                    
                    # Explain the ranking focusing on the HSQC score, which has been shown to be the most reliable indicator 
                    # for structural matching in this analysis.
                    # """
                    # # NMR Error Scores:
                    # # Overall: {result['scores']['overall']:.6f}
                    # # 1H NMR: {result['scores']['1H']:.6f}
                    # # 13C NMR: {result['scores']['13C']:.6f}
                    # # HSQC: {result['scores']['HSQC']:.6f}
                    # # COSY: {result['scores']['COSY']:.6f}
                    # reasoning = await self.llm_service.get_completion(
                    #     message=prompt,
                    #     model="claude-3-5-sonnet",   #### maybe make it flexible to adapt to the user input
                    #     system="You are an expert in NMR spectroscopy analysis. Analyze the molecule's ranking based on its NMR matching scores."
                    # )
                    reasoning = f"""Ranking Analysis for Candidate #{rank}:
                    - Structure: {result['smiles']}
                    - IUPAC Name: {result.get('iupac_name', 'Not available')}
                    - HSQC Error Score: {result['scores']['HSQC']:.6f}

                    This molecule ranked #{rank} among the top {top_n} candidates based on its HSQC NMR spectral matching. 
                    The HSQC error score of {result['scores']['HSQC']:.6f} represents the average deviation between 
                    predicted and experimental HSQC correlations across the entire molecule."""
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
                        top_n=5,    ### top 5 candidates
                        mw_tolerance=0.5,     ### mw tolerance 
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
