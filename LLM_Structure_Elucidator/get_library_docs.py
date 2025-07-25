import importlib
import sys
from io import StringIO

def generate_library_docs(library_name):
    """Generate documentation for a library and save it to a file."""
    try:
        # Import the library
        library = importlib.import_module(library_name)
        
        # Redirect stdout to capture help output
        old_stdout = sys.stdout
        result = StringIO()
        sys.stdout = result
        
        # Get help information
        help(library)
        
        # Restore stdout
        sys.stdout = old_stdout
        
        # Save to file
        with open(f'{library_name}_documentation.txt', 'w', encoding='utf-8') as f:
            f.write(result.getvalue())
            
        print(f"Documentation for {library_name} has been saved to {library_name}_documentation.txt")
        
    except ImportError:
        print(f"Could not import {library_name}. Make sure it's installed.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        library_name = sys.argv[1]
        generate_library_docs(library_name)
    else:
        print("Please provide a library name as argument")
