import pandas as pd
import sort_dataset_into_brand
import sort_dataset_into_car
import brand_model
import car_model
import pickle


data_brand = pd.read_csv('data-stats/stat-brand-sorted-dataset.csv')
data_car = pd.read_csv('data-stats/stat-sorted-dataset.csv')
model_car_brand = ['Nissan','Ford','BMW','Audi']
epochs_brand = 15
epochs_car = 20
resolution = 480
with open('resolution.pkl', 'wb') as f:
    pickle.dump(resolution, f)





###################################GET THE DATASET#############################################
import get_dataset_from_kaggle
###################################FILTER THE DATASET BY BRAND#############################################

choice = True
brand_nbr= 0
print("You can choice four brand in the dataset to build your model use")
while choice:
    print("1. help")
    print("2. brand list")
    print("3. use de default brand (Nissan, Ford, BMW, Mercedes-Benz)")
    print("or enter a car brand name")
    choice = input(":")
    if choice == "1":
        print("You need to select a brands to build your model")
        print("Try to use brand with a similar number of picture to have a better model")
    elif choice == "2":       
        print(data_brand.iloc[:,0])
      
    elif choice == "3":
        print("default brand selected")
        model_car_brand = ['Nissan','Ford','BMW','Mercedes-Benz']
        choice = False
    else:
        if choice in data_brand.iloc[:,0].values:
            print("brand selected")
            model_car_brand[brand_nbr] = choice    
            brand_nbr += 1        
        else:
            print("brand not found")
            choice = True
    if brand_nbr == 4:
        choice = False

#create the folder with the brand dataset
sort_dataset_into_brand.sort_dataset_into_brand(model_car_brand)

###################################BUILD THE BRAND MODEL#############################################
brand_model.build_brand_model(epochs=epochs_brand,resolution=resolution)

###################################FILTER THE DATASET BY CAR#############################################
#loop in the data_car and print the first col
brand_car = [[],[],[],[]]
for i in range(len(data_car)):
    print(data_car.iloc[i,0])
    if model_car_brand[0] in data_car.iloc[i,0]:
        brand_car[0].append(data_car.iloc[i,0])
    elif model_car_brand[1] in data_car.iloc[i,0]:
        brand_car[1].append(data_car.iloc[i,0])
    elif model_car_brand[2] in data_car.iloc[i,0]:
        brand_car[2].append(data_car.iloc[i,0])
    elif model_car_brand[3] in data_car.iloc[i,0]:
        brand_car[3].append(data_car.iloc[i,0])



#kept for each sub brand the 4 most represented car
for i in range(len(brand_car)):
    brand_car[i] = brand_car[i][:4]

#create the folder with the car dataset
sort_dataset_into_car.sort_dataset_into_car(model_car_brand,brand_car)   
###################################BUILD THE CATEGORY MODEL#############################################

for i in range(len(model_car_brand)):
    car_model.build_car_model(model_car_brand[i],epochs=epochs_car,resolution=resolution)

