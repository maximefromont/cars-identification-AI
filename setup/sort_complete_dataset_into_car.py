import os
import shutil
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count

# Define the paths
source_folder = "the-car-connection-picture-dataset"
destination_folder = "complete-car-dataset"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

def process_file(filename):
    # Split the filename into groups separated by underscores
    groups = filename.split("_")

    # found = False
    # for i in range(len(brands)):
    #     for car_model in car[i]:
    #         if groups[1] in car_model:
    #             found = True
    #             break

    # if not found:
    #     return   
    
    # Create the subfolder path based on the f
    subfolder = os.path.join(destination_folder,groups[0],f"{groups[0]}_{groups[1]}")
  
    
    # Check if the subfolder already exists
    if not os.path.exists(subfolder):
        os.makedirs(subfolder, exist_ok=True)
        
    # Check if the file already exists in the subfolder
    if not os.path.exists(os.path.join(subfolder, filename)):
        # Move the picture to the subfolder
        shutil.copy(os.path.join(source_folder, filename), os.path.join(subfolder, filename))
    
    # Move the picture to the subfolder
    shutil.copy(os.path.join(source_folder, filename), os.path.join(subfolder, filename))

def sort_complete_dataset_into_car():
    # creat subfolder of car-dataset with brand name
    # for i in range(len(brands)):
    #     subfolder = os.path.join(destination_folder, f"{brands[i]}")
    #     if not os.path.exists(subfolder):
    #         os.makedirs(subfolder, exist_ok=True)


    num_cores = cpu_count()
    
    # Create a pool of worker processes
    pool = Pool(processes=num_cores)
    process_file_with_brands = partial(process_file)
    
    # Iterate through the pictures in the source folder and process them in parallel
    with tqdm(total=len(os.listdir(source_folder))) as pbar:
        for _ in pool.imap_unordered(process_file_with_brands, os.listdir(source_folder)):
            pbar.update(1)

