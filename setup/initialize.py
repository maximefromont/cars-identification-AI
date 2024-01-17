import pandas as pd
import sort_dataset_into_brand
import brand_model
data_brand = pd.read_csv('data-stats/stat-brand-sorted-dataset.csv')
data_car = pd.read_csv('data-stats/stat-sorted-dataset.csv')



###################################GET THE DATASET#############################################
#import get_dataset_from_kaggle
###################################FILTER THE DATASET BY BRAND#############################################
model_car_brand = ['Nissan','Ford','BMW','Mercedes-Benz']
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
        #print only the the second col of the data_brand dataframe
        print(data_brand.iloc[:,0])
      
    elif choice == "3":
        print("default brand selected")
        model_car_brand = ['Nissan','Ford','BMW','Mercedes-Benz']
        choice = False
    else:
        #test if the input is in the dataframe
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
brand_model.build_brand_model(epochs=10)



        


###################################BUILD THE CATEGORY MODEL#############################################

###################################PRNT AI MODEL CARACTERISTIC#############################################

###################################TEST THE MODELS#############################################