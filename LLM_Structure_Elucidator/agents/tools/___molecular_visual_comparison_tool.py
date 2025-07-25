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
