"""Utility functions for handling SBATCH job submissions and monitoring."""
import logging
import subprocess
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

# Constants for SBATCH operations
SBATCH_JOB_CHECK_INTERVAL = 10  # seconds
SBATCH_TIMEOUT = 3600  # 1 hour timeout

logger = logging.getLogger(__name__)

async def execute_sbatch(script_path: Path, *args: str) -> str:
    """Execute sbatch script and return job ID.
    
    Args:
        script_path: Path to the sbatch script to execute
        *args: Additional arguments to pass to the script
        
    Returns:
        str: Job ID of the submitted batch job
        
    Raises:
        RuntimeError: If sbatch execution fails
    """
    try:
        cmd = ['sbatch', str(script_path)]
        if args:
            cmd.extend(args)
            
        logger.info(f"Executing sbatch command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        # Extract job ID from sbatch output (typical format: "Submitted batch job 123456")
        job_id = result.stdout.strip().split()[-1]
        logger.info(f"Submitted batch job {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to execute sbatch: {e.stderr}")
        raise RuntimeError(f"Sbatch execution failed: {e.stderr}")

async def check_job_status(job_id: str) -> bool:
    """Check if sbatch job is complete.
    
    Args:
        job_id: ID of the job to check
        
    Returns:
        bool: True if job is complete, False if still running
    """
    try:
        result = subprocess.run(
            ['squeue', '-j', job_id],
            capture_output=True,
            text=True
        )
        # If job not found in queue, it's complete
        return "Invalid job id specified" in result.stderr or job_id not in result.stdout
    except subprocess.CalledProcessError:
        # If squeue fails, assume job is complete
        return True

async def wait_for_job_completion(job_id: str, timeout: Optional[int] = None) -> bool:
    """Wait for sbatch job to complete with timeout.
    
    Args:
        job_id: ID of the job to wait for
        timeout: Optional timeout in seconds (defaults to SBATCH_TIMEOUT)
        
    Returns:
        bool: True if job completed successfully, False if failed
        
    Raises:
        TimeoutError: If job doesn't complete within timeout period
    """
    timeout = timeout or SBATCH_TIMEOUT
    start_time = datetime.now()
    
    try:
        while True:
            try:
                if await check_job_status(job_id):
                    logger.info(f"Job {job_id} completed")
                    return True
            except RuntimeError as e:
                logger.error(f"Job failed: {str(e)}")
                return False
            
            if (datetime.now() - start_time).total_seconds() > timeout:
                raise TimeoutError(f"Job {job_id} timed out after {timeout} seconds")
            
            await asyncio.sleep(SBATCH_JOB_CHECK_INTERVAL)
    except Exception as e:
        logger.error(f"Error waiting for job completion: {str(e)}")
        return False

