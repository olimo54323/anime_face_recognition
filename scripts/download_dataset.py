import os
import sys
import shutil
import zipfile
import argparse
import glob

# Define default paths
DEFAULT_DATASET_PATH = 'dataset'
DEFAULT_KAGGLE_DATASET = 'thedevastator/anime-face-dataset-by-character-name'

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Download anime face dataset')
    
    parser.add_argument(
        '--dataset_path', 
        type=str, 
        default=DEFAULT_DATASET_PATH,
        help=f'Path to store the dataset (default: {DEFAULT_DATASET_PATH})'
    )
    
    parser.add_argument(
        '--kaggle_dataset', 
        type=str, 
        default=DEFAULT_KAGGLE_DATASET,
        help=f'Kaggle dataset identifier (default: {DEFAULT_KAGGLE_DATASET})'
    )
    
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Force download even if dataset already exists'
    )
    
    parser.add_argument(
        '--min_images', 
        type=int, 
        default=0,
        help='Filter out character folders with fewer than this many images (default: 0, no filtering)'
    )
    
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='/root/.cache/kagglehub/datasets',
        help='Path to the cache directory where kagglehub stores datasets'
    )
    
    return parser.parse_args()

def find_downloaded_dataset(kaggle_dataset, cache_dir):
    """
    Find dataset in cache directory if it exists.
    
    Args:
        kaggle_dataset: Kaggle dataset identifier (e.g., 'thedevastator/anime-face-dataset-by-character-name')
        cache_dir: Path to kagglehub cache directory
        
    Returns:
        Path to the downloaded dataset or None if not found
    """
    # Split dataset identifier into owner and name
    if '/' in kaggle_dataset:
        owner, dataset_name = kaggle_dataset.split('/')
    else:
        owner = kaggle_dataset
        dataset_name = ''
    
    # Check if dataset exists in cache
    owner_dir = os.path.join(cache_dir, owner)
    if os.path.exists(owner_dir) and os.path.isdir(owner_dir):
        # Check for dataset directory
        dataset_dir = os.path.join(owner_dir, dataset_name)
        if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
            # Check for versions subdirectory
            versions_dir = os.path.join(dataset_dir, 'versions')
            if os.path.exists(versions_dir) and os.path.isdir(versions_dir):
                # Get the latest version (assuming highest number is latest)
                versions = [d for d in os.listdir(versions_dir) 
                          if os.path.isdir(os.path.join(versions_dir, d))]
                if versions:
                    try:
                        # Try to sort numerically
                        versions = sorted(versions, key=int)
                    except ValueError:
                        # If not all are numbers, sort alphabetically
                        versions = sorted(versions)
                    
                    latest_version = versions[-1]
                    dataset_path = os.path.join(versions_dir, latest_version)
                    print(f"Found dataset in cache: {dataset_path}")
                    return dataset_path
    
    return None

def download_dataset(dataset_path, kaggle_dataset, force=False, cache_dir=None):
    """
    Download the dataset from Kaggle.
    
    Args:
        dataset_path: Path to store the dataset
        kaggle_dataset: Kaggle dataset identifier
        force: Force download even if dataset already exists
        cache_dir: Path to kagglehub cache directory
        
    Returns:
        Path to the dataset
    """
    # Check if dataset already exists in target directory
    if os.path.exists(dataset_path) and len(os.listdir(dataset_path)) > 0 and not force:
        print(f"Dataset already exists at {dataset_path}. Use --force to redownload.")
        return dataset_path
    
    # Create dataset directory if it doesn't exist
    os.makedirs(dataset_path, exist_ok=True)
    
    # If force is True and the directory exists, clear it
    if force and os.path.exists(dataset_path):
        print(f"Clearing existing dataset at {dataset_path}")
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    
    # Try to find dataset in cache first
    cache_dataset_path = None
    if cache_dir:
        cache_dataset_path = find_downloaded_dataset(kaggle_dataset, cache_dir)
    
    # If found in cache, copy to target directory
    if cache_dataset_path:
        print(f"Using dataset from cache: {cache_dataset_path}")
        
        # Check if it's a directory with contents
        if os.path.isdir(cache_dataset_path) and os.listdir(cache_dataset_path):
            # Check for nested directory structure
            for item in os.listdir(cache_dataset_path):
                item_path = os.path.join(cache_dataset_path, item)
                
                # If it's a file, copy it
                if os.path.isfile(item_path):
                    print(f"Copying file: {item}")
                    shutil.copy2(item_path, os.path.join(dataset_path, item))
                
                # If it's a directory, copy it recursively
                elif os.path.isdir(item_path):
                    print(f"Copying directory: {item}")
                    dest_path = os.path.join(dataset_path, item)
                    if not os.path.exists(dest_path):
                        shutil.copytree(item_path, dest_path)
            
            print(f"Dataset copied to {dataset_path}")
            return dataset_path
        
        # If it's a single file (possibly a zip)
        elif os.path.isfile(cache_dataset_path):
            print(f"Found file in cache: {cache_dataset_path}")
            
            # If it's a zip file, extract it
            if cache_dataset_path.endswith('.zip'):
                print("Extracting ZIP file...")
                with zipfile.ZipFile(cache_dataset_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_path)
                print(f"Dataset extracted to {dataset_path}")
                return dataset_path
            
            # Otherwise, just copy the file
            else:
                print(f"Copying file to {dataset_path}")
                shutil.copy2(cache_dataset_path, dataset_path)
                return dataset_path
    
    # If not found in cache or force is True, download using kagglehub
    print(f"Downloading dataset '{kaggle_dataset}' to {dataset_path}")
    
    try:
        # Try with kagglehub first
        try:
            import kagglehub
            dataset_file = kagglehub.dataset_download(kaggle_dataset)
            
            print(f"Dataset downloaded to: {dataset_file}")
            
            # Check if it's a file or directory
            if os.path.isfile(dataset_file):
                # If it's a ZIP file, extract it
                if dataset_file.endswith('.zip'):
                    print("Extracting ZIP file...")
                    with zipfile.ZipFile(dataset_file, 'r') as zip_ref:
                        zip_ref.extractall(dataset_path)
                    print(f"Dataset extracted to {dataset_path}")
                else:
                    print(f"Copying file to {dataset_path}")
                    shutil.copy2(dataset_file, dataset_path)
            elif os.path.isdir(dataset_file):
                print(f"Dataset already extracted at: {dataset_file}")
                
                # Copy contents to target directory
                for item in os.listdir(dataset_file):
                    src_path = os.path.join(dataset_file, item)
                    dst_path = os.path.join(dataset_path, item)
                    
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, dst_path)
                    elif os.path.isdir(src_path):
                        if not os.path.exists(dst_path):
                            shutil.copytree(src_path, dst_path)
                
                print(f"Dataset contents copied to {dataset_path}")
            
        except ImportError:
            print("kagglehub not found, trying kaggle API...")
            # If kagglehub fails, try kaggle API
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate()
                
                # Download dataset
                api.dataset_download_files(
                    kaggle_dataset,
                    path=dataset_path,
                    unzip=True,
                    quiet=False
                )
                
                print(f"Dataset downloaded and extracted to {dataset_path}")
            except ImportError:
                print("Kaggle API not found. Installing...")
                os.system("pip install kaggle")
                print("Please retry after installation.")
                sys.exit(1)
            except Exception as e:
                print(f"Error with Kaggle API: {e}")
                print("\nAlternative method: Download manually from https://www.kaggle.com/datasets/thedevastator/anime-face-dataset-by-character-name")
                print("and extract to 'dataset' folder.")
                sys.exit(1)
        
        return dataset_path
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)

def find_character_folders(dataset_path):
    """
    Find character folders in the dataset, handling different directory structures.
    
    Args:
        dataset_path: Path to the dataset
        
    Returns:
        List of character folders
    """
    print("Looking for character folders...")
    
    # Direct approach: look for directories in the dataset path
    character_folders = [f for f in os.listdir(dataset_path) 
                       if os.path.isdir(os.path.join(dataset_path, f))]
    
    if character_folders:
        # Check if these folders contain images (then they are character folders)
        for folder in character_folders[:5]:  # Check a few to get an idea
            folder_path = os.path.join(dataset_path, folder)
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(glob.glob(os.path.join(folder_path, ext)))
                image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
            if image_files:
                print(f"Found {len(character_folders)} character folders in root directory")
                return character_folders
        
        # If checked folders don't contain images, they might not be character folders
        print(f"Found {len(character_folders)} folders in root directory, but they might not be character folders")
    
    # Check for a specific 'dataset' subdirectory that contains character folders
    dataset_subdir = os.path.join(dataset_path, 'dataset')
    if os.path.exists(dataset_subdir) and os.path.isdir(dataset_subdir):
        dataset_folders = [f for f in os.listdir(dataset_subdir) 
                          if os.path.isdir(os.path.join(dataset_subdir, f))]
        
        if dataset_folders:
            print(f"Found {len(dataset_folders)} potential character folders in {dataset_subdir}")
            
            # Verify a few folders to see if they contain images
            for folder in dataset_folders[:5]:  # Check a few to get an idea
                folder_path = os.path.join(dataset_subdir, folder)
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    image_files.extend(glob.glob(os.path.join(folder_path, ext)))
                    image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
                
                if image_files:
                    print(f"Found character folders in 'dataset' subdirectory")
                    
                    # Return full paths instead of just names
                    return [os.path.join('dataset', f) for f in dataset_folders]
    
    # If no character folders found, check for nested structure
    print("No character folders found in expected locations, checking for nested structure...")
    
    # First check for /dataset/1/dataset/ structure (common in Kaggle datasets)
    version_dirs = [f for f in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, f)) and f.isdigit()]
    
    for version in version_dirs:
        version_path = os.path.join(dataset_path, version)
        
        # Check for 'dataset' subdirectory
        dataset_subdir = os.path.join(version_path, 'dataset')
        if os.path.exists(dataset_subdir) and os.path.isdir(dataset_subdir):
            dataset_folders = [f for f in os.listdir(dataset_subdir) 
                             if os.path.isdir(os.path.join(dataset_subdir, f))]
            
            if dataset_folders:
                print(f"Found {len(dataset_folders)} character folders in {dataset_subdir}")
                
                # Move these folders to the root dataset directory
                print(f"Moving character folders from {dataset_subdir} to {dataset_path}")
                for folder in dataset_folders:
                    src = os.path.join(dataset_subdir, folder)
                    dst = os.path.join(dataset_path, folder)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                    else:
                        print(f"Folder {folder} already exists, merging...")
                        # Clone files from src to dst
                        for file in os.listdir(src):
                            src_file = os.path.join(src, file)
                            dst_file = os.path.join(dst, file)
                            if os.path.isfile(src_file) and not os.path.exists(dst_file):
                                shutil.copy2(src_file, dst_file)
                
                # Recheck character folders
                character_folders = [f for f in os.listdir(dataset_path) 
                                   if os.path.isdir(os.path.join(dataset_path, f))]
                
                if character_folders:
                    print(f"Now have {len(character_folders)} character folders in root directory")
                    return character_folders
    
    # Check for any subdirectories that might contain character folders
    subdirs = [f for f in os.listdir(dataset_path) 
             if os.path.isdir(os.path.join(dataset_path, f))]
    
    for subdir in subdirs:
        subdir_path = os.path.join(dataset_path, subdir)
        
        # Check if this subdir contains image files directly (could be a character folder itself)
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(subdir_path, ext)))
            image_files.extend(glob.glob(os.path.join(subdir_path, ext.upper())))
        
        if image_files:
            # This is likely a character folder itself
            print(f"Found a potential character folder: {subdir} with {len(image_files)} images")
            return subdirs
        
        # Check if the subdirectory contains other directories (potential character folders)
        sub_character_folders = [f for f in os.listdir(subdir_path) 
                              if os.path.isdir(os.path.join(subdir_path, f))]
        
        if sub_character_folders:
            print(f"Found {len(sub_character_folders)} potential character folders in {subdir}")
            
            # Check if these folders contain images
            for folder in sub_character_folders[:5]:  # Check a few to get an idea
                folder_path = os.path.join(subdir_path, folder)
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    image_files.extend(glob.glob(os.path.join(folder_path, ext)))
                    image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
                
                if image_files:
                    print(f"Found character folders in {subdir_path}")
                    
                    # Return full paths instead of just names
                    return [os.path.join(subdir, f) for f in sub_character_folders]
    
    print("No character folders found in any subdirectory")
    return []

def check_dataset_structure(dataset_path, min_images=0):
    """
    Check the structure of the dataset and provide summary statistics.
    
    Args:
        dataset_path: Path to the dataset
        min_images: Minimum number of images per character folder
    """
    print("\nAnalyzing dataset structure...")
    
    # Find character folders
    character_folders = find_character_folders(dataset_path)
    
    # Empty list check
    if not character_folders:
        print("No character folders found.")
        print("Please check that the dataset was downloaded correctly.")
        print(f"Contents of {dataset_path}:")
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                print(f"  Directory: {item} ({len(os.listdir(item_path))} items)")
            else:
                print(f"  File: {item} ({os.path.getsize(item_path)} bytes)")
        return
    
    print(f"Found {len(character_folders)} character folders")
    
    # Count images per character
    character_counts = {}
    total_images = 0
    
    for character in character_folders:
        # Handle full paths from find_character_folders
        char_name = os.path.basename(character)
        char_path = os.path.join(dataset_path, character)
        
        # Check if the path exists as is or needs to be joined with dataset_path
        if not os.path.exists(char_path):
            char_path = character  # Use as is if it was returned as a full path
        
        image_files = []
        
        # Get all image files with different extensions
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(char_path, ext)))
            image_files.extend(glob.glob(os.path.join(char_path, ext.upper())))
        
        character_counts[char_name] = len(image_files)
        total_images += len(image_files)
    
    # Handle empty dataset case
    if len(character_counts) == 0:
        print("No character folders with images found.")
        return
    
    # Filter by minimum images if requested
    if min_images > 0:
        filtered_characters = [char for char, count in character_counts.items() 
                             if count >= min_images]
        
        filtered_total = sum(character_counts[char] for char in filtered_characters)
        
        print(f"\nAfter filtering (min {min_images} images per character):")
        print(f"  Characters: {len(filtered_characters)} (reduced from {len(character_counts)})")
        print(f"  Total images: {filtered_total} (reduced from {total_images})")
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Total characters: {len(character_counts)}")
    print(f"  Total images: {total_images}")
    
    # Avoid division by zero
    if len(character_counts) > 0:
        print(f"  Average images per character: {total_images/len(character_counts):.2f}")
        print(f"  Min images per character: {min(character_counts.values())}")
        print(f"  Max images per character: {max(character_counts.values())}")
    
        # Print top characters
        sorted_chars = sorted(character_counts.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 characters by number of images:")
        for i, (char, count) in enumerate(sorted_chars[:10]):
            print(f"  {i+1}. {char}: {count} images")
        
        # Print characters with fewest images
        print("\nCharacters with fewest images:")
        for i, (char, count) in enumerate(sorted_chars[-10:]):
            print(f"  {i+1}. {char}: {count} images")
    else:
        print("  No character folders with images found.")


if __name__ == "__main__":
    args = parse_arguments()
    
    # Download dataset
    dataset_path = download_dataset(args.dataset_path, args.kaggle_dataset, args.force, args.cache_dir)
    
    # Check dataset structure
    check_dataset_structure(dataset_path, args.min_images)
    
    print("\nDataset download and validation complete!")