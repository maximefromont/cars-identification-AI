import os
import shutil

def delete_files(file_paths):
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"File deleted: {file_path}")
                else:
                    shutil.rmtree(file_path)
                    print(f"Directory deleted: {file_path}")
            else:
                print(f"File or directory not found: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# List of file paths to delete
files_to_delete = [
    'brand-dataset',
    'car-dataset',
    'car-model/plot',
    'car-model/model',
    'car-model/model-data',
    'brand-model/plot',
    'brand-model/model',
    'brand-model/model-data',
    'the-car-connection-picture-dataset'
]

# Specify the root directory
root_directory = "."

# Construct full paths
full_paths = [os.path.join(root_directory, path) for path in files_to_delete]

# Delete files
delete_files(full_paths)





