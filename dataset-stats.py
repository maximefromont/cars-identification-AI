import os
import shutil

#get the number of brand by counting the number of folder in brand-sorted-dataset and the number of for each brand
def get_number_of_brand():
    # Define the paths
    source_folder = "brand-sorted-dataset"
    #count the number of folder in brand-sorted-dataset
    number_of_brand = len([d for d in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, d))])
    #count the number of file in each folder
    number_of_file = []
    for i in range(number_of_brand):
        number_of_file.append(len([d for d in os.listdir(source_folder + "/" + str(i)) if os.path.isfile(os.path.join(source_folder + "/" + str(i), d))]))
    return number_of_brand, number_of_file

#get the number of car model by counting the number of folder in sorted-dataset and the number picture of for each car model
def get_number_of_car_model():
    # Define the paths
    source_folder = "sorted-dataset"
    #count the number of folder in sorted-dataset
    number_of_car_model = len([d for d in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, d))])
    #count the number of file in each folder
    number_of_file = []
    for i in range(number_of_car_model):
        number_of_file.append(len([d for d in os.listdir(source_folder + "/" + str(i)) if os.path.isfile(os.path.join(source_folder + "/" + str(i), d))]))
    return number_of_car_model, number_of_file

print(get_number_of_brand())
print(get_number_of_car_model())





