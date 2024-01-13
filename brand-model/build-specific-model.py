from brand_model import basic_brand_model

#ask user to choice betwwen basic model and transfer learning model
print("Enter 1 for basic model")
choice = int(input("Enter your choice: "))

if choice == 1:
    #ask user for the nmber of epoch
    epoch = int(input("Enter the number of epoch: "))
    #run the basic brand model for epoch give by user 
    basic_brand_model(epoch)
