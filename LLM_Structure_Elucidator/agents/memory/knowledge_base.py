"""
Knowledge base for storing and retrieving analysis results and workflow data.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import json
import sqlite3
import logging


class KnowledgeBase:
    """
    Persistent storage for analysis results and workflow data.
    Implements a SQLite-based storage with JSON serialization for complex data.
    """
    
    def __init__(self):
        """Initialize knowledge base with SQLite storage."""
        # Set up storage directory in temp/memory
        self.base_dir = Path(__file__).parent.parent.parent / 'temp' / 'memory' / 'knowledge'
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = self.base_dir / 'knowledge.db'
        self._initialize_db()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self._initialize_logging()
    
    def _initialize_logging(self):
        """Initialize logging system."""
        self.logger.setLevel(logging.INFO)
        
        # Create log directory
        log_dir = self.base_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'knowledge_base_{timestamp}.log'
        
        # Add file handler
        file_handler = logging.FileHandler(str(log_file))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def _initialize_db(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Create results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    workflow_type TEXT NOT NULL,
                    step_type TEXT NOT NULL,
                    data JSON NOT NULL,
                    confidence REAL,
                    metadata JSON
                )
            """)
            
            # Create workflow_context table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_context (
                    workflow_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    workflow_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    context JSON NOT NULL,
                    metadata JSON
                )
            """)
            
            # Create index on workflow_id and timestamp
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_results_workflow 
                ON results(workflow_id, timestamp)
            """)
            
            conn.commit()
    
    def add_results(self, workflow_id: str, workflow_type: str, 
                   step_type: str, data: Dict, confidence: float = None,
                   metadata: Dict = None) -> int:
        """
        Add analysis results to the knowledge base.
        
        Args:
            workflow_id: ID of the workflow that generated the results
            workflow_type: Type of workflow (e.g., STARTING_MATERIAL)
            step_type: Type of step that generated the results
            data: Results data (will be JSON serialized)
            confidence: Optional confidence score for the results
            metadata: Optional metadata about the results
            
        Returns:
            ID of the inserted result record
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO results 
                    (timestamp, workflow_id, workflow_type, step_type, data, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    workflow_id,
                    workflow_type,
                    step_type,
                    json.dumps(data),
                    confidence,
                    json.dumps(metadata) if metadata else None
                ))
                
                result_id = cursor.lastrowid
                conn.commit()
                
                self.logger.info(f"Added results for workflow {workflow_id}, step {step_type}")
                return result_id
                
        except Exception as e:
            self.logger.error(f"Failed to add results: {str(e)}")
            raise
    
    def update_workflow_context(self, workflow_id: str, workflow_type: str,
                              status: str, context: Dict, metadata: Dict = None):
        """
        Update workflow context in the knowledge base.
        
        Args:
            workflow_id: ID of the workflow
            workflow_type: Type of workflow
            status: Current workflow status
            context: Current workflow context
            metadata: Optional workflow metadata
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO workflow_context
                    (workflow_id, timestamp, workflow_type, status, context, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    workflow_id,
                    datetime.now().isoformat(),
                    workflow_type,
                    status,
                    json.dumps(context),
                    json.dumps(metadata) if metadata else None
                ))
                
                conn.commit()
                
                self.logger.info(f"Updated context for workflow {workflow_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to update workflow context: {str(e)}")
            raise
    
    def get_workflow_results(self, workflow_id: str) -> List[Dict]:
        """Get all results for a specific workflow."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM results 
                    WHERE workflow_id = ?
                    ORDER BY timestamp ASC
                """, (workflow_id,))
                
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    result['data'] = json.loads(result['data'])
                    if result['metadata']:
                        result['metadata'] = json.loads(result['metadata'])
                    results.append(result)
                
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to get workflow results: {str(e)}")
            raise
    
    def get_workflow_context(self, workflow_id: str) -> Optional[Dict]:
        """Get current context for a specific workflow."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM workflow_context 
                    WHERE workflow_id = ?
                """, (workflow_id,))
                
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    result['context'] = json.loads(result['context'])
                    if result['metadata']:
                        result['metadata'] = json.loads(result['metadata'])
                    return result
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get workflow context: {str(e)}")
            raise
    
    def get_similar_results(self, workflow_type: str, data_pattern: Dict,
                          min_confidence: float = 0.0) -> List[Dict]:
        """
        Find similar results based on workflow type and data pattern.
        
        Args:
            workflow_type: Type of workflow to search
            data_pattern: Dictionary of key-value pairs to match in results
            min_confidence: Minimum confidence score for results
            
        Returns:
            List of matching result records
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Convert pattern to JSON string fragments for LIKE matching
                patterns = []
                for key, value in data_pattern.items():
                    patterns.append(f'"%{key}": "{value}%"')
                
                # Build query with pattern matching
                query = """
                    SELECT * FROM results 
                    WHERE workflow_type = ?
                    AND confidence >= ?
                """
                params = [workflow_type, min_confidence]
                
                for pattern in patterns:
                    query += f" AND data LIKE ?"
                    params.append(pattern)
                
                cursor.execute(query, params)
                
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    result['data'] = json.loads(result['data'])
                    if result['metadata']:
                        result['metadata'] = json.loads(result['metadata'])
                    results.append(result)
                
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to get similar results: {str(e)}")
            raise
    
    def size(self) -> Dict[str, int]:
        """Get current size of the knowledge base."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Get counts from both tables
                cursor.execute("SELECT COUNT(*) FROM results")
                results_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM workflow_context")
                context_count = cursor.fetchone()[0]
                
                return {
                    'results': results_count,
                    'contexts': context_count
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get knowledge base size: {str(e)}")
            raise
