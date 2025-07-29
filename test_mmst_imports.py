#!/usr/bin/env python3
"""
Test script to check if MMST imports work correctly
"""

import sys
import os
from pathlib import Path

# Change to CONFIG_BASE_DIR like the bash script does
os.chdir('/Users/mpriessner/windsurf_repos/ChemStructLLM_')

# Add project root to path for imports
project_root = os.getcwd()
sys.path.insert(0, project_root)

# Add script directory to path
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

print("Testing MMST imports...")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path includes: {sys.path[:3]}")

try:
    print("\n1. Testing utils_MMT imports...")
    from utils_MMT.execution_function_v15_4 import *
    print("✅ utils_MMT.execution_function_v15_4 imported successfully")
    
    from utils_MMT.mmt_result_test_functions_15_4 import *
    print("✅ utils_MMT.mmt_result_test_functions_15_4 imported successfully")
    
    import utils_MMT.data_generation_v15_4 as dg
    print("✅ utils_MMT.data_generation_v15_4 imported successfully")
    
    print("\n2. Testing imports_MMST...")
    # Add the scripts directory to path for imports_MMST
    scripts_dir = Path('/Users/mpriessner/windsurf_repos/ChemStructLLM_/LLM_Structure_Elucidator/agents/scripts')
    sys.path.insert(0, str(scripts_dir))
    
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
except Exception as e:
    print(f"❌ Other error: {e}")
