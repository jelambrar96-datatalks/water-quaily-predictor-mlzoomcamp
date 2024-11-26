import pickle
from typing import Any, Optional
from pathlib import Path
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_to_pickle(
    obj: Any,
    file_path: str,
    create_dir: bool = True,
    compression: Optional[str] = None
) -> bool:
    """
    Save an object to a pickle file with error handling.
    
    Parameters:
    -----------
    obj : Any
        The Python object to save
    file_path : str
        Path where the pickle file will be saved
    create_dir : bool, optional (default=True)
        If True, creates the directory if it doesn't exist
    compression : str, optional (default=None)
        Compression protocol to use ('gzip', 'bz2', 'lzma' or None)
        
    Returns:
    --------
    bool
        True if save was successful, False otherwise
        
    Examples:
    --------
    >>> data = {'key': 'value'}
    >>> save_to_pickle(data, 'data/my_dict.pkl')
    >>> save_to_pickle(data, 'data/my_dict.pkl.gz', compression='gzip')
    """
    try:
        # Convert to Path object for better path handling
        path = Path(file_path)
        
        # Create directory if it doesn't exist and create_dir is True
        if create_dir:
            path.parent.mkdir(parents=True, exist_ok=True)
            
        # Determine the appropriate open function and mode
        if compression:
            if compression == 'gzip':
                import gzip
                open_func = gzip.open
            elif compression == 'bz2':
                import bz2
                open_func = bz2.open
            elif compression == 'lzma':
                import lzma
                open_func = lzma.open
            else:
                raise ValueError(f"Unsupported compression format: {compression}")
            mode = 'wb'
        else:
            open_func = open
            mode = 'wb'
        
        # Save the object
        with open_func(path, mode) as f:
            pickle.dump(obj, f)
            
        logger.info(f"Successfully saved object to {path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving object to {file_path}: {str(e)}")
        return False

def load_from_pickle(
    file_path: str,
    default_value: Any = None,
    compression: Optional[str] = None
) -> Any:
    """
    Load an object from a pickle file with error handling.
    
    Parameters:
    -----------
    file_path : str
        Path to the pickle file
    default_value : Any, optional (default=None)
        Value to return if loading fails
    compression : str, optional (default=None)
        Compression protocol used ('gzip', 'bz2', 'lzma' or None)
        
    Returns:
    --------
    Any
        The loaded object if successful, default_value if failed
        
    Examples:
    --------
    >>> data = load_from_pickle('data/my_dict.pkl')
    >>> data = load_from_pickle('data/my_dict.pkl.gz', compression='gzip')
    >>> data = load_from_pickle('data/my_dict.pkl', default_value={})
    """
    try:
        # Convert to Path object
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            logger.error(f"File not found: {path}")
            return default_value
            
        # Determine the appropriate open function and mode
        if compression:
            if compression == 'gzip':
                import gzip
                open_func = gzip.open
            elif compression == 'bz2':
                import bz2
                open_func = bz2.open
            elif compression == 'lzma':
                import lzma
                open_func = lzma.open
            else:
                raise ValueError(f"Unsupported compression format: {compression}")
            mode = 'rb'
        else:
            open_func = open
            mode = 'rb'
        
        # Load the object
        with open_func(path, mode) as f:
            obj = pickle.load(f)
            
        logger.info(f"Successfully loaded object from {path}")
        return obj
        
    except Exception as e:
        logger.error(f"Error loading object from {file_path}: {str(e)}")
        return default_value

# Example usage
if __name__ == "__main__":
    # Example data
    example_data = {
        'name': 'John Doe',
        'age': 30,
        'scores': [85, 90, 95]
    }
    
    # Example 1: Basic usage
    save_to_pickle(example_data, 'data/user_data.pkl')
    loaded_data = load_from_pickle('data/user_data.pkl')
    
    # Example 2: Using compression
    save_to_pickle(example_data, 'data/user_data.pkl.gz', compression='gzip')
    loaded_compressed = load_from_pickle('data/user_data.pkl.gz', compression='gzip')
    
    # Example 3: Using default value
    missing_data = load_from_pickle('nonexistent.pkl', default_value={})
