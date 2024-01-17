from keras.models import load_model
import tensorflow as tf
import os
import numpy as np
import pickle

img_height = 180
img_width = 180
#find the file the the folder brand-model/model
brand_model_path = os.listdir('brand-model/model')
print(brand_model_path)

car_model_paths = os.listdir('car-model/model')
print(car_model_paths)

with open('brands_names.pkl', 'rb') as f:
        brands_names = pickle.load(f) 
with open('resolution.pkl', 'rb') as f:
        resolution = pickle.load(f) 


def get_brand(picture_path):
    model = load_model('brand-model/model/'+str(brand_model_path[0]))       
    picture = test_image = tf.keras.preprocessing.image.load_img(
        picture_path, target_size=(resolution, resolution)
    )
    test_image_array = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image_array = tf.expand_dims(test_image_array, 0) # Create a batch
    predictions = model.predict(test_image_array)
    score = tf.nn.softmax(predictions[0])
    print(score)
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(brands_names[np.argmax(score)], 100 * np.max(score))
    )
    return brands_names[np.argmax(score)],score

    


def get_car(brand,picture):
    #regarder si on a une coorespondance entre un des paths de car_model_paths et le brand
    #si oui on charge le model et on fait la prédiction
    for path in car_model_paths:
        if brand in path:
            model = load_model('car-model/model/'+str(path))
            with open(str(brand)+'_car_names.pkl', 'rb') as f:
                car_names = pickle.load(f)    
            test_image = tf.keras.preprocessing.image.load_img(
                picture, target_size=(resolution, resolution)
            )
            test_image_array = tf.keras.preprocessing.image.img_to_array(test_image)
            test_image_array = tf.expand_dims(test_image_array, 0)
            predictions = model.predict(test_image_array)
            score = tf.nn.softmax(predictions[0])
            print(score)
            print(
                "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(car_names[np.argmax(score)], 100 * np.max(score))
            )
            return car_names[np.argmax(score)],score

def car_classifier(picture_path):
    brand,brand_score = get_brand(picture_path)    
    car,car_score = get_car(brand,picture_path)
    print('The car is a '+str(car)+' from '+str(brand))
    return brand,brand_score,car,car_score
    
    

print(car_classifier('/home/death-joke/Bureau/APP5/IA/cars-identification-AI/car-dataset/BMW/BMW_3-Series/BMW_3-Series_2011_46_17_230_30_6_70_54_181_18_RWD_4_2_Convertible_avK.jpg'))





    
    

