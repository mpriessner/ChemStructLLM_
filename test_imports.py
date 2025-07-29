#!/usr/bin/env python3
"""
Test script to check if the imports and paths work correctly
for chemformer_retro_script.py
"""

import sys
import os
from pathlib import Path

def test_path_resolution():
    """Test the path resolution logic from chemformer_retro_script.py"""
    print("=== Testing Path Resolution ===")
    
    # Simulate the path logic from chemformer_retro_script.py
    script_dir = Path(__file__).parent / "ChemStructLLM_/LLM_Structure_Elucidator/agents/scripts"
    parent_dir = script_dir.parent.parent.parent  # Go up 3 levels
    
    print(f"Script directory: {script_dir}")
    print(f"Parent directory: {parent_dir}")
    print(f"Expected chemformer_public path: {parent_dir / 'chemformer_public'}")
    
    # Check if paths exist
    chemformer_public_path = parent_dir / "chemformer_public"
    if chemformer_public_path.exists():
        print("✅ chemformer_public directory found!")
        molbart_path = chemformer_public_path / "molbart"
        if molbart_path.exists():
            print("✅ molbart subdirectory found!")
        else:
            print("❌ molbart subdirectory not found")
    else:
        print("❌ chemformer_public directory not found")
        print("Available directories in parent:")
        if parent_dir.exists():
            for item in parent_dir.iterdir():
                if item.is_dir():
                    print(f"  - {item.name}")

def test_imports():
    """Test the imports that chemformer_retro_script.py needs"""
    print("\n=== Testing Imports ===")
    
    # Test basic imports first
    try:
        import torch
        print("✅ torch imported successfully")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
    
    # Test the path resolution and chemformer import
    try:
        # Add the path like the script does
        script_dir = Path(__file__).parent / "ChemStructLLM_/LLM_Structure_Elucidator/agents/scripts"
        parent_dir = script_dir.parent.parent.parent
        sys.path.insert(0, str(parent_dir))
        
        from chemformer_public.molbart.models import Chemformer
        print("✅ Chemformer imported successfully!")
    except ImportError as e:
        print(f"❌ Chemformer import failed: {e}")
        print("   This might be due to missing chemformer_public directory or dependencies")
    except Exception as e:
        print(f"❌ Unexpected error importing Chemformer: {e}")

def main():
    print("=== Chemformer Import Diagnostic ===")
    test_path_resolution()
    test_imports()
    
    print("\n=== Summary ===")
    print("If you see ❌ errors above:")
    print("1. Check that chemformer_public directory exists in the right location")
    print("2. Install missing Python packages (torch, pandas, numpy)")
    print("3. Check that the chemformer_public code is properly set up")
    print("\nIf all ✅ green, the imports should work in the actual script!")

if __name__ == "__main__":
    main()
