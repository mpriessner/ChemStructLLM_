#!/usr/bin/env python3
"""
Test script to check if MMST imports work correctly
"""

import sys
import os
from pathlib import Path

# Get the directory where this script is located
test_script_dir = Path(__file__).resolve().parent

# Find the ChemStructLLM_ root directory (should be the same as test_script_dir)
project_root = test_script_dir

# Change to project root like the bash script does
os.chdir(project_root)

# Add project root to path for imports
sys.path.insert(0, str(project_root))

# Add scripts directory to path for imports_MMST
scripts_dir = project_root / "LLM_Structure_Elucidator" / "agents" / "scripts"
sys.path.insert(0, str(scripts_dir))

print("Testing MMST imports...")
print(f"Current working directory: {os.getcwd()}")
print(f"Project root: {project_root}")
print(f"Scripts dir: {scripts_dir}")

try:
    print("\n1. Testing utils_MMT imports...")
    from utils_MMT.execution_function_v15_4 import *
    print("✅ utils_MMT.execution_function_v15_4 imported successfully")
    
    from utils_MMT.mmt_result_test_functions_15_4 import *
    print("✅ utils_MMT.mmt_result_test_functions_15_4 imported successfully")
    
    import utils_MMT.data_generation_v15_4 as dg
    print("✅ utils_MMT.data_generation_v15_4 imported successfully")
    
    print("\n2. Testing imports_MMST...")
    from imports_MMST import (
        mtf, ex, mrtf, hf,
        parse_arguments, load_config, save_updated_config,
        load_configs, load_json_dics
    )
    print("✅ imports_MMST imported successfully")
    
    print("\n🎉 All imports successful! MMST script should work.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:5]}")
    
    # Check if directories exist
    print(f"\nDebugging info:")
    print(f"utils_MMT exists: {(project_root / 'utils_MMT').exists()}")
    print(f"scripts dir exists: {scripts_dir.exists()}")
    if scripts_dir.exists():
        print(f"imports_MMST.py exists: {(scripts_dir / 'imports_MMST.py').exists()}")
        
except Exception as e:
    print(f"❌ Other error: {e}")
