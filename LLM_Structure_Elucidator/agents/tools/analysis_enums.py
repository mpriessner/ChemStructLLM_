"""
Shared enums for analysis tools.
"""
from enum import Enum

class DataSource(Enum):
    MASTER_FILE = "master_file"
    INTERMEDIATE = "intermediate"

class RankingMetric(Enum):
    """Available metrics for ranking candidates"""
    OVERALL = "overall"
    PROTON = "1H"
    CARBON = "13C"
    HSQC = "HSQC"
    COSY = "COSY"
