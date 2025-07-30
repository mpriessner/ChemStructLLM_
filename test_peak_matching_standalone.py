#!/usr/bin/env python3
"""
Standalone test for peak matching script to bypass the main process and get direct error feedback.
This test creates dummy input data and calls the bash script directly.
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

def create_dummy_input_data():
    """Create minimal dummy input data for peak matching test."""
    dummy_data = {
        "type": "peaks_vs_peaks",
        "spectra": ["1H"],
        "matching_mode": "hung_dist_nn",
        "error_type": "sum",
        "peaks1": {
            "1H": {
                "Chemical Shift (ppm)": [1.0, 2.0, 3.0],
                "Intensity": [100, 200, 150]
            }
        },
        "peaks2": {
            "1H": {
                "Chemical Shift (ppm)": [1.1, 2.1, 3.1],
                "Intensity": [110, 190, 140]
            }
        }
    }
    return dummy_data

def test_peak_matching_direct():
    """Test the peak matching bash script directly."""
    print("🧪 Testing Peak Matching Script Directly")
    print("=" * 50)
    
    # Get project root
    project_root = Path(__file__).parent
    print(f"📁 Project root: {project_root}")
    
    # Create test directory
    test_dir = project_root / "test_peak_matching_temp"
    test_dir.mkdir(exist_ok=True)
    print(f"📁 Test directory: {test_dir}")
    
    # Create dummy input file
    input_file = test_dir / "test_input.json"
    dummy_data = create_dummy_input_data()
    
    with open(input_file, 'w') as f:
        json.dump(dummy_data, f, indent=2)
    print(f"✅ Created dummy input file: {input_file}")
    print(f"📄 Input data:\n{json.dumps(dummy_data, indent=2)}")
    
    # Path to bash script
    bash_script = project_root / "LLM_Structure_Elucidator" / "agents" / "scripts" / "peak_matching_local.sh"
    print(f"🔧 Bash script: {bash_script}")
    
    if not bash_script.exists():
        print(f"❌ ERROR: Bash script not found at {bash_script}")
        return False
    
    # Make bash script executable
    os.chmod(bash_script, 0o755)
    
    print("\n🚀 Running bash script...")
    print(f"Command: {bash_script} {input_file}")
    print("-" * 50)
    
    try:
        # Run the bash script with timeout
        start_time = time.time()
        
        result = subprocess.run(
            [str(bash_script), str(input_file)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"⏱️  Script completed in {duration:.2f} seconds")
        print(f"🔢 Return code: {result.returncode}")
        
        if result.stdout:
            print(f"\n📤 STDOUT:\n{result.stdout}")
        
        if result.stderr:
            print(f"\n📥 STDERR:\n{result.stderr}")
        
        # Check for results file
        results_file = test_dir / "results.json"
        if results_file.exists():
            print(f"\n✅ Results file created: {results_file}")
            with open(results_file) as f:
                results = json.load(f)
            print(f"📄 Results:\n{json.dumps(results, indent=2)}")
            return True
        else:
            print(f"\n❌ No results file found at {results_file}")
            
            # Check for error file
            error_file = test_dir / "error.log"
            if error_file.exists():
                print(f"📄 Error file found: {error_file}")
                with open(error_file) as f:
                    error_content = f.read()
                print(f"❌ Error content:\n{error_content}")
            else:
                print("📄 No error file found either")
            
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n⏰ TIMEOUT: Script took longer than 120 seconds")
        return False
        
    except Exception as e:
        print(f"\n💥 EXCEPTION: {str(e)}")
        return False

def test_python_script_direct():
    """Test just the Python script directly (bypass bash)."""
    print("\n🐍 Testing Python Script Directly")
    print("=" * 50)
    
    # Get project root
    project_root = Path(__file__).parent
    
    # Create test directory
    test_dir = project_root / "test_peak_matching_temp"
    test_dir.mkdir(exist_ok=True)
    
    # Create dummy input file
    input_file = test_dir / "test_input_python.json"
    dummy_data = create_dummy_input_data()
    
    with open(input_file, 'w') as f:
        json.dump(dummy_data, f, indent=2)
    
    # Path to Python script
    python_script = project_root / "LLM_Structure_Elucidator" / "agents" / "scripts" / "peak_matching_script.py"
    print(f"🐍 Python script: {python_script}")
    
    if not python_script.exists():
        print(f"❌ ERROR: Python script not found at {python_script}")
        return False
    
    print(f"\n🚀 Running Python script directly...")
    print(f"Command: python {python_script} {input_file}")
    print("-" * 50)
    
    try:
        # Change to project root (like bash script does)
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        start_time = time.time()
        
        result = subprocess.run(
            ["python", str(python_script), str(input_file)],
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Restore original directory
        os.chdir(original_cwd)
        
        print(f"⏱️  Script completed in {duration:.2f} seconds")
        print(f"🔢 Return code: {result.returncode}")
        
        if result.stdout:
            print(f"\n📤 STDOUT:\n{result.stdout}")
        
        if result.stderr:
            print(f"\n📥 STDERR:\n{result.stderr}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"\n⏰ TIMEOUT: Python script took longer than 60 seconds")
        os.chdir(original_cwd)
        return False
        
    except Exception as e:
        print(f"\n💥 EXCEPTION: {str(e)}")
        os.chdir(original_cwd)
        return False

if __name__ == "__main__":
    print("🔬 Peak Matching Standalone Test")
    print("=" * 50)
    print("This test bypasses the main process and tests the scripts directly.")
    print("It will help identify where exactly the hanging/timeout occurs.\n")
    
    # Test 1: Python script only (fastest, most direct)
    print("TEST 1: Python script only")
    python_success = test_python_script_direct()
    
    # Test 2: Full bash script (includes environment setup)
    print("\nTEST 2: Full bash script")
    bash_success = test_peak_matching_direct()
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print(f"🐍 Python script direct: {'✅ SUCCESS' if python_success else '❌ FAILED'}")
    print(f"🔧 Bash script full: {'✅ SUCCESS' if bash_success else '❌ FAILED'}")
    
    if not python_success:
        print("\n💡 RECOMMENDATION: Python script is failing - check imports and path setup")
    elif not bash_success:
        print("\n💡 RECOMMENDATION: Python works but bash fails - check environment setup (conda/CUDA)")
    else:
        print("\n🎉 BOTH TESTS PASSED: The issue is likely in the main process communication")
