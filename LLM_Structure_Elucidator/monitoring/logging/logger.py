"""
Centralized logging configuration and setup.
"""
import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional
from pathlib import Path

class LLMLogger:
    """Centralized logging for the LLM Structure Elucidator."""
    
    def __init__(self, log_dir: Optional[str] = None):
        """Initialize logger with optional custom log directory."""
        self.log_dir = log_dir or os.path.join(os.path.dirname(__file__), '../../logs')
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Create loggers for different components
        self.app_logger = self._setup_logger('app', 'app.log')
        self.api_logger = self._setup_logger('api', 'api.log')
        self.llm_logger = self._setup_logger('llm', 'llm.log')
        self.error_logger = self._setup_logger('error', 'error.log', level=logging.ERROR)
    
    def _setup_logger(self, name: str, filename: str, level: int = logging.INFO) -> logging.Logger:
        """Set up a logger with file and console handlers."""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        if not logger.handlers:
            # File handler
            file_handler = logging.handlers.RotatingFileHandler(
                os.path.join(self.log_dir, filename),
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def log_request(self, method: str, path: str, params: dict, response_time: float) -> None:
        """Log API request details."""
        self.api_logger.info(
            f"Request: {method} {path} - Params: {params} - Response Time: {response_time:.2f}s"
        )
    
    def log_llm_interaction(self, model: str, prompt_tokens: int, completion_tokens: int, 
                          duration: float, success: bool) -> None:
        """Log LLM interaction details."""
        self.llm_logger.info(
            f"Model: {model} - Prompt Tokens: {prompt_tokens} - "
            f"Completion Tokens: {completion_tokens} - Duration: {duration:.2f}s - "
            f"Success: {success}"
        )
    
    def log_error(self, error: Exception, context: dict = None) -> None:
        """Log error with context."""
        context = context or {}
        self.error_logger.error(
            f"Error: {str(error)} - Type: {type(error).__name__} - Context: {context}",
            exc_info=True
        )
    
    def log_app_event(self, event_type: str, details: dict) -> None:
        """Log application events."""
        self.app_logger.info(f"Event: {event_type} - Details: {details}")
    
    def get_recent_errors(self, hours: int = 24) -> list:
        """Get list of errors from the last N hours."""
        errors = []
        error_log_path = os.path.join(self.log_dir, 'error.log')
        
        if not os.path.exists(error_log_path):
            return errors
        
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        with open(error_log_path, 'r') as f:
            for line in f:
                try:
                    # Parse timestamp from log line
                    timestamp_str = line.split(' - ')[0]
                    timestamp = datetime.strptime(
                        timestamp_str, '%Y-%m-%d %H:%M:%S,%f'
                    ).timestamp()
                    
                    if timestamp >= cutoff_time:
                        errors.append(line.strip())
                except (ValueError, IndexError):
                    continue
        
        return errors
