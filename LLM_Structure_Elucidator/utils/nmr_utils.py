"""
Utility functions for generating and loading NMR spectral data.
"""
import numpy as np
import pandas as pd
import json
import os

def generate_random_2d_correlation_points(x_range=(0, 10), y_range=(0, 200), num_points=20, diagonal=False):
    """Generate random correlation points for 2D NMR spectra."""
    if diagonal:
        # For COSY: Generate points near the diagonal
        x = np.random.uniform(x_range[0], x_range[1], num_points)
        y = x + np.random.normal(0, 0.3, num_points)  # Points close to diagonal with more spread
        # Add some off-diagonal correlations
        off_diag = int(num_points * 0.4)  # 40% off-diagonal peaks
        off_x = np.random.uniform(x_range[0], x_range[1], off_diag)
        off_y = np.random.uniform(y_range[0], y_range[1], off_diag)
        # Filter out points too close to diagonal
        mask = np.abs(off_y - off_x) > 0.5
        off_x = off_x[mask]
        off_y = off_y[mask]
        x = np.concatenate([x, off_x])
        y = np.concatenate([y, off_y])
    else:
        # For HSQC: Generate random points with clustering
        num_clusters = 5
        points_per_cluster = num_points // num_clusters
        x = []
        y = []
        for _ in range(num_clusters):
            center_x = np.random.uniform(x_range[0], x_range[1])
            center_y = np.random.uniform(y_range[0], y_range[1])
            cluster_x = center_x + np.random.normal(0, (x_range[1] - x_range[0])/20, points_per_cluster)
            cluster_y = center_y + np.random.normal(0, (y_range[1] - y_range[0])/20, points_per_cluster)
            x.extend(cluster_x)
            y.extend(cluster_y)
        x = np.array(x)
        y = np.array(y)
    
    # Generate varying intensities with some correlation to position
    z = 0.3 + 0.7 * np.random.beta(2, 2, len(x))  # Beta distribution for more realistic intensities
    
    return x, y, z

def generate_nmr_peaks(x=None, peak_positions=None, intensities=None):
    """Generate NMR peaks with Lorentzian line shape and multiplicity."""
    # Generate default values if not provided
    if x is None:
        x = np.linspace(0, 10, 1000)
    if peak_positions is None:
        peak_positions = np.random.uniform(1, 9, 8)  # 8 random peaks
    if intensities is None:
        intensities = np.random.uniform(0.3, 1.0, len(peak_positions))
        
    y = np.zeros_like(x)
    for pos, intensity in zip(peak_positions, intensities):
        # Lorentzian peak shape
        gamma = 0.02  # Peak width
        # Add main peak
        y += intensity * gamma**2 / ((x - pos)**2 + gamma**2)
        
        # Randomly add multiplicity (doublets, triplets)
        multiplicity = np.random.choice([1, 2, 3])  # singlet, doublet, triplet
        if multiplicity > 1:
            j_coupling = 0.1  # Typical J-coupling constant
            for i in range(1, multiplicity):
                # Add satellite peaks
                y += (intensity * 0.9) * gamma**2 / ((x - (pos + i*j_coupling))**2 + gamma**2)
                y += (intensity * 0.9) * gamma**2 / ((x - (pos - i*j_coupling))**2 + gamma**2)
    
    return x, y

def generate_default_nmr_data(plot_type='proton'):
    """Generate default NMR data based on plot type."""
    if plot_type in ['hsqc', 'cosy']:
        x_range = (0, 10) if plot_type == 'hsqc' else (0, 10)
        y_range = (0, 200) if plot_type == 'hsqc' else (0, 10)
        return generate_random_2d_correlation_points(x_range, y_range, 20, diagonal=(plot_type == 'cosy'))
    else:
        return generate_nmr_peaks()

def load_nmr_data_from_csv(smiles, csv_path=None):
    """Load NMR data for a specific molecule from the CSV file."""
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              'test_data', 'test_smiles_with_nmr.csv')
    
    try:
        if not os.path.exists(csv_path):
            print(f"[NMR Utils] CSV file not found: {csv_path}")
            return None
            
        df = pd.read_csv(csv_path)
        molecule_data = df[df['SMILES'] == smiles]
        
        if len(molecule_data) == 0:
            print(f"[NMR Utils] No data found for SMILES: {smiles}")
            return None
            
        # Parse the NMR data from the first matching row
        row = molecule_data.iloc[0]
        try:
            return {
                'proton': json.loads(row['1H_NMR']),
                'carbon': json.loads(row['13C_NMR']),
                'hsqc': json.loads(row['HSQC']),
                'cosy': json.loads(row['COSY'])
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[NMR Utils] Error parsing NMR data: {str(e)}")
            return None
    except Exception as e:
        print(f"[NMR Utils] Error loading NMR data: {str(e)}")
        return None

def generate_nmr_data(smiles, plot_type='proton', use_real_data=True):
    """Generate NMR data for a given SMILES string and plot type.
    
    Args:
        smiles: SMILES string of the molecule
        plot_type: Type of NMR plot ('proton', 'carbon', 'hsqc', 'cosy')
        use_real_data: Whether to try loading real experimental data first
        
    Returns:
        tuple: (nmr_data, is_random) where is_random indicates if data was simulated
    """
    print(f"\n[NMR Utils] Generating {plot_type} NMR data for SMILES: {smiles}")
    print(f"[NMR Utils] Use real data: {use_real_data}")
    
    if use_real_data and smiles:
        # First try to get data from uploaded molecular_data.json
        try:
            from handlers.molecule_handler import get_nmr_data_from_json
            json_data = get_nmr_data_from_json(smiles)
            print(f"[NMR Utils] JSON data retrieved: {json_data is not None}")
            
            if json_data:
                # Map plot types to data keys
                key_mapping = {
                    'proton': '1H_exp',
                    'carbon': '13C_exp', 
                    'hsqc': 'HSQC_exp',
                    'cosy': 'COSY_exp'
                }
                
                data_key = key_mapping.get(plot_type)
                if data_key and json_data.get(data_key):
                    real_data = json_data[data_key]
                    print(f"[NMR Utils] Found real {plot_type} data in uploaded files")
                    
                    # Process the real data based on plot type
                    if plot_type in ['proton', 'carbon']:
                        processed_data = process_1d_nmr_data(real_data, plot_type)
                    else:  # hsqc, cosy
                        processed_data = process_2d_nmr_data(real_data, plot_type)
                    
                    return processed_data, False  # False = not random, real data
                else:
                    print(f"[NMR Utils] No {plot_type} data found in uploaded files")
        except Exception as e:
            print(f"[NMR Utils] Error loading from JSON: {str(e)}")
        
        # Fallback to CSV loading (original behavior)
        csv_data = load_nmr_data_from_csv(smiles)
        if csv_data:
            # Map plot types to CSV data keys
            csv_key_mapping = {
                'proton': 'proton',
                'carbon': 'carbon',
                'hsqc': 'hsqc',
                'cosy': 'cosy'
            }
            
            csv_key = csv_key_mapping.get(plot_type)
            if csv_key and csv_data.get(csv_key):
                print(f"[NMR Utils] Found real {plot_type} data in CSV")
                
                # Process the real data based on plot type
                if plot_type in ['proton', 'carbon']:
                    processed_data = process_1d_nmr_data(csv_data[csv_key], plot_type)
                else:  # hsqc, cosy
                    processed_data = process_2d_nmr_data(csv_data[csv_key], plot_type)
                
                return processed_data, False  # False = not random, real data
    
    # Generate default (simulated) data if no real data found
    print(f"[NMR Utils] No real data found, generating simulated {plot_type} data")
    default_data = generate_default_nmr_data(plot_type)
    return default_data, True  # True = random/simulated data

def process_1d_nmr_data(peaks_data, plot_type):
    """Process 1D NMR data (proton or carbon) into plottable format."""
    x = np.linspace(0, 10 if plot_type == 'proton' else 200, 1000)
    y = np.zeros_like(x)
    
    # Handle different data formats based on NMR type
    if plot_type == 'proton':
        # Proton NMR should always have [position, intensity] pairs
        if not isinstance(peaks_data[0], (list, tuple)):
            raise ValueError("Proton NMR data must be in [position, intensity] pairs format")
        for position, intensity in peaks_data:
            gamma = 0.02  # Narrow peaks for proton NMR
            y += intensity * gamma**2 / ((x - position)**2 + gamma**2)
    elif plot_type == 'carbon':
        # Carbon NMR can be either just positions or [position, intensity] pairs
        if isinstance(peaks_data[0], (list, tuple)):
            for position, intensity in peaks_data:
                gamma = 0.5  # Broader peaks for carbon NMR
                y += intensity * gamma**2 / ((x - position)**2 + gamma**2)
        else:
            for position in peaks_data:
                gamma = 0.5  # Broader peaks for carbon NMR
                y += 1.0 * gamma**2 / ((x - position)**2 + gamma**2)
    else:
        raise ValueError(f"Unsupported NMR type: {plot_type}")
    
    return x, y

def process_2d_nmr_data(correlation_data, plot_type):
    """Process 2D NMR data (HSQC or COSY) into plottable format."""
    x = []
    y = []
    z = []
    
    for correlation in correlation_data:
        x_pos, y_pos = correlation[:2]
        intensity = 1.0 if len(correlation) < 3 else correlation[2]
        x.append(x_pos)
        y.append(y_pos)
        z.append(intensity)
    
    return np.array(x), np.array(y), np.array(z)
