#!/usr/bin/env python3
"""
Simple test script for chemformer_retro_script.py
This helps debug issues without running through the web interface.
"""

import os
import tempfile
from pathlib import Path

def create_test_input():
    """Create a simple test input file with a SMILES string."""
    # Create a temporary input file with a simple SMILES
    test_smiles = "CCO"  # Ethanol - simple molecule for testing
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_smiles + '\n')
        return f.name

def main():
    print("=== Chemformer Retro Script Test ===")
    
    # Get the script directory
    script_dir = Path(__file__).parent / "LLM_Structure_Elucidator/agents/scripts"
    chemformer_script = script_dir / "chemformer_retro_script.py"
    
    print(f"Looking for script at: {chemformer_script}")
    
    if not chemformer_script.exists():
        print(f"Error: Script not found at {chemformer_script}")
        print("Let me check what's actually in the directory...")
        
        # Debug: show what's actually there
        if script_dir.exists():
            print(f"Contents of {script_dir}:")
            for item in script_dir.iterdir():
                print(f"  - {item.name}")
        else:
            print(f"Directory {script_dir} doesn't exist")
            # Try to find the actual path
            base_dir = Path(__file__).parent
            print(f"Base directory: {base_dir}")
            print("Contents of base directory:")
            for item in base_dir.iterdir():
                if item.is_dir():
                    print(f"  - {item.name}/")
        return
    
    # Create test input file
    input_file = create_test_input()
    output_file = "/tmp/chemformer_test_output.csv"
    
    print(f"Created test input file: {input_file}")
    print(f"Output will be saved to: {output_file}")
    
    # You'll need to update these paths to your actual model and vocab files
    # These are the paths from the shell script but need to be adjusted for your system
    vocab_path = "UPDATE_THIS_PATH/vocab.pkl"  # Update this
    model_path = "UPDATE_THIS_PATH/model.ckpt"  # Update this
    
    print("\n=== Test Configuration ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Vocab path: {vocab_path}")
    print(f"Model path: {model_path}")
    print(f"Script path: {chemformer_script}")
    
    # Test command (you can run this manually)
    test_command = f"""
python "{chemformer_script}" \\
    --input_file="{input_file}" \\
    --output_file="{output_file}" \\
    --vocab_path="{vocab_path}" \\
    --model_path="{model_path}" \\
    --batch_size=1 \\
    --n_beams=5 \\
    --n_unique_beams=3
"""
    
    print("\n=== Manual Test Command ===")
    print("Run this command to test the script:")
    print(test_command)
    
    print("\n=== Next Steps ===")
    print("1. Update the vocab_path and model_path above with your actual file paths")
    print("2. Run the command above to test the script")
    print("3. Check for any import errors or path issues")
    print(f"4. Clean up test file when done: rm {input_file}")

if __name__ == "__main__":
    main()
