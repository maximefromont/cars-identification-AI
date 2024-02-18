import os
import shutil
import csv

#get the number of brand by counting the number of folder in brand-sorted-dataset and the number of for each brand
def get_number_of_brand(root_folder="brand-sorted-dataset"):
    # Initialize an empty dictionary to store counts per brand
    brand_counts = {}

    # Iterate through subfolders
    for brand_folder in os.listdir(root_folder):
        brand_path = os.path.join(root_folder, brand_folder)

        # Check if the item in the root folder is a directory
        if os.path.isdir(brand_path):
            # Count the number of image files in the brand subfolder
            num_images = len([f for f in os.listdir(brand_path) if f.endswith(('.jpg', '.jpeg', '.png'))])

            # Store the count in the dictionary
            brand_counts[brand_folder] = num_images

    # Print the counts per brand
    for brand, count in brand_counts.items():
        print(f"{brand}: {count} images")

    #sort the dictionary by value
    brand_counts = dict(sorted(brand_counts.items(), key=lambda item: item[1], reverse=True))
    #save this information to csv file   
    with open('stat-'+root_folder+'.csv', 'w') as f:
        for key in brand_counts.keys():
            f.write("%s,%s\n"%(key,brand_counts[key]))

    return brand_counts

#get the number of car model by counting the number of folder in sorted-dataset and the number picture of for each car model
def get_number_of_car_model(root_folder="sorted-dataset"):    
    # Initialize an empty dictionary to store counts per brand
    car_model_counts = {}

    # Iterate through subfolders
    for car_model_folder in os.listdir(root_folder):
        car_model_path = os.path.join(root_folder, car_model_folder)

        # Check if the item in the root folder is a directory
        if os.path.isdir(car_model_path):
            # Count the number of image files in the brand subfolder
            num_images = len([f for f in os.listdir(car_model_path) if f.endswith(('.jpg', '.jpeg', '.png'))])

            # Store the count in the dictionary
            car_model_counts[car_model_folder] = num_images

    # Print the counts per brand
    for car_model, count in car_model_counts.items():
        print(f"{car_model}: {count} images")
    #sort the dictionary by value
    car_model_counts = dict(sorted(car_model_counts.items(), key=lambda item: item[1], reverse=True))
    #save this information to csv file
    with open('stat-'+root_folder+'.csv', 'w') as f:
        for key in car_model_counts.keys():
            f.write("%s,%s\n"%(key,car_model_counts[key]))
    return car_model_counts

#get the number of brand by counting the number of folder in brand-sorted-dataset and the number of for each brand
def get_number_of_model_per_brand(root_folder="complete-car-dataset"):
    # Initialize an empty dictionary to store counts per brand
    brand_counts = {}

    # Iterate through subfolders
    for brand_folder in os.listdir(root_folder):
        brand_path = os.path.join(root_folder, brand_folder)

        # Check if the item in the root folder is a directory
        if os.path.isdir(brand_path):
            # Count the number of subfolder in the brand subfolder
            num_images = len([f for f in os.listdir(brand_path) if os.path.isdir(os.path.join(brand_path, f))])
            

            # Store the count in the dictionary
            brand_counts[brand_folder] = num_images

    # Print the counts per brand
    for brand, count in brand_counts.items():
        print(f"{brand}: {count} images")

    #sort the dictionary by value
    brand_counts = dict(sorted(brand_counts.items(), key=lambda item: item[1], reverse=True))
    #save this information to csv file   
    with open('stat-'+'model_per_brand'+'.csv', 'w') as f:
        for key in brand_counts.keys():
            f.write("%s,%s\n"%(key,brand_counts[key]))

    return brand_counts



# brand_date = get_number_of_brand()
# model_data=get_number_of_car_model()
get_number_of_model_per_brand()






