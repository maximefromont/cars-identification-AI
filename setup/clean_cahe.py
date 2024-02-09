import os
import glob
import shutil

def delete_files(paths):
    for path in paths:
        try:
            # If path is a directory, delete all .pkl files in it
            if os.path.isdir(path):
                pkl_files = glob.glob(os.path.join(path, '*.pkl'))
                for file in pkl_files:
                    print('here')
                    print(f"Deleting file: {file}")
                    os.remove(file)
                print(path)   
                if path != './.':
                    print(f"Deleting directory: {path}")
                    shutil.rmtree(path)                    
            # If path is a file, delete it
            elif os.path.isfile(path):
                print(f"Deleting file: {path}")
                os.remove(path)
            else:
                print(f"File or directory not found: {path}")
        except Exception as e:
            print(f"Error deleting {path}: {e}")

# List of file paths to delete
files_to_delete = [  
    '.',  
    'brand-dataset',
    'car-dataset',
    'car-model/plot',
    'car-model/model',
    'car-model/model-data',
    'car-model',
    'brand-model/plot',
    'brand-model/model',
    'brand-model/model-data',
    'brand-model',
    'the-car-connection-picture-dataset',
    'sorted-dataset',
    'complete-brand-dataset',
    'complete-car-dataset',]

# Specify the root directory
root_directory = "."

# Construct full paths
full_paths = [os.path.join(root_directory, path) for path in files_to_delete]

# Delete files
delete_files(full_paths)