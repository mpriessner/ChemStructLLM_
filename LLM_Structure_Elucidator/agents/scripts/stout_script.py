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