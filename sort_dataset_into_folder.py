import os
import shutil

# Define the paths
source_folder = "the-car-connection-picture-dataset"
destination_folder = "sorted-dataset"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Iterate through the pictures in the source folder
for filename in os.listdir(source_folder):
    # Split the filename into groups separated by underscores
    groups = filename.split("_")
    
    # Create the subfolder path based on the first two groups
    subfolder = os.path.join(destination_folder, f"{groups[0]} - {groups[1]}")
    
    # Create the subfolder if it doesn't exist
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    
    # Move the picture to the subfolder
    shutil.copy(os.path.join(source_folder, filename), os.path.join(subfolder, filename))