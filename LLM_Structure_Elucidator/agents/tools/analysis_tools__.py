"""
Tools for analysis-related operations.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
from utils.nmr_utils import generate_nmr_peaks, generate_random_2d_correlation_points

class AnalysisTools:
    """Collection of tools for data analysis and interpretation."""
    
    @staticmethod
    def analyze_nmr_spectrum(peaks: List[Dict[str, float]]) -> Dict[str, Any]:
        """Analyze NMR spectrum peaks."""
        try:
            # Convert peaks to numpy array for analysis
            peak_values = np.array([peak["ppm"] for peak in peaks])
            intensities = np.array([peak["intensity"] for peak in peaks])
            
            return {
                "num_peaks": len(peaks),
                "max_intensity": float(np.max(intensities)),
                "min_intensity": float(np.min(intensities)),
                "mean_intensity": float(np.mean(intensities)),
                "chemical_shift_range": (float(np.min(peak_values)), float(np.max(peak_values)))
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def find_correlations(points: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Analyze 2D correlation points."""
        try:
            # Convert points to numpy arrays
            x_values = np.array([p[0] for p in points])
            y_values = np.array([p[1] for p in points])
            
            # Calculate correlation coefficient
            correlation = float(np.corrcoef(x_values, y_values)[0, 1])
            
            return {
                "num_points": len(points),
                "correlation": correlation,
                "x_range": (float(np.min(x_values)), float(np.max(x_values))),
                "y_range": (float(np.min(y_values)), float(np.max(y_values)))
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def generate_test_data() -> Dict[str, Any]:
        """Generate test data for analysis."""
        try:
            peaks = generate_nmr_peaks()
            correlations = generate_random_2d_correlation_points()
            
            return {
                "peaks": peaks,
                "correlations": correlations,
                "peak_analysis": AnalysisTools.analyze_nmr_spectrum(peaks),
                "correlation_analysis": AnalysisTools.find_correlations(correlations)
            }
        except Exception as e:
            return {"error": str(e)}
