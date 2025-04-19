import os
import sys
import subprocess
from typing import Tuple
from flask import current_app
import logging

logger = logging.getLogger(__name__)

def convert_stl_to_dxf(stl_path: str) -> Tuple[bool, str]:
    """
    Convert STL file to DXF using the external converter script.
    
    Args:
        stl_path: Path to the STL file
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Get the directory containing the STL file
        stl_dir = os.path.dirname(stl_path)
        stl_filename = os.path.basename(stl_path)
        dxf_filename = os.path.splitext(stl_filename)[0] + '.dxf'
        dxf_path = os.path.join(stl_dir.replace('stl', 'dxf'), dxf_filename)
        
        # Ensure the DXF directory exists
        os.makedirs(os.path.dirname(dxf_path), exist_ok=True)
        
        # Get the path to the converter script
        converter_script = os.path.join(current_app.config['PROJECT_ROOT'], 'stl_to_dxf_works.py')
        
        # Run the conversion command
        cmd = [
            'python',
            converter_script,
            stl_path,
            '-t', '2000',  # Target faces for simplification
            '-v',  # Verbose output
            '--no-sections',  # Disable sections
            '-o', dxf_path  # Output path
        ]
        
        logger.info(f"Running conversion command: {' '.join(cmd)}")
        
        # Run the command and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the process to complete
        stdout, stderr = process.communicate()
        
        # Log the output
        if stdout:
            logger.info(f"Converter output:\n{stdout}")
        if stderr:
            logger.warning(f"Converter warnings/errors:\n{stderr}")
        
        # Check if the conversion was successful
        if process.returncode == 0 and os.path.exists(dxf_path):
            return True, dxf_path
        else:
            error_msg = f"Conversion failed with return code {process.returncode}"
            if stderr:
                error_msg += f"\nError: {stderr}"
            return False, error_msg
            
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        return False, str(e) 