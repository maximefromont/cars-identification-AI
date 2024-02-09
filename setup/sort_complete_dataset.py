import os
import shutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Define the paths
source_folder = "the-car-connection-picture-dataset"
destination_folder = "sorted-dataset"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

def process_file(filename):
    # Split the filename into groups separated by underscores
    groups = filename.split("_")
    
    # Create the subfolder path based on the first two groups
    subfolder = os.path.join(destination_folder, f"{groups[0]} - {groups[1]}")
    
    # Check if the subfolder already exists
    if not os.path.exists(subfolder):
        os.makedirs(subfolder, exist_ok=True)
        
    # Check if the file already exists in the subfolder
    if not os.path.exists(os.path.join(subfolder, filename)):
        # Move the picture to the subfolder
        shutil.copy(os.path.join(source_folder, filename), os.path.join(subfolder, filename))
    
    # Move the picture to the subfolder
    shutil.copy(os.path.join(source_folder, filename), os.path.join(subfolder, filename))

def sort_complete_dataset():
    # Get the number of available cores
    num_cores = cpu_count()

    # Create a pool of worker processes
    pool = Pool(processes=num_cores)

    # Iterate through the pictures in the source folder and process them in parallel
    with tqdm(total=len(os.listdir(source_folder))) as pbar:
        for _ in pool.imap_unordered(process_file, os.listdir(source_folder)):
            pbar.update(1)


if __name__ == "__main__":
    sort_complete_dataset()
    print("The dataset has been sorted successfully!")

