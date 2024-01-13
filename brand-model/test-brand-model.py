import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import PIL
import PIL.Image
from tensorflow.keras.models import load_model
import os


###################################SETUP###################################

# '0': Display all logs (default behavior).
# '1': Display only warning and error logs.
# '2': Display only error logs.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = load_model('brand_model_v1_car_classifier.h5')
batch_size = 32
img_height = 180
img_width = 180
class_directory = "brand-sorted-dataset"
class_names= sorted([d for d in os.listdir(class_directory) if os.path.isdir(os.path.join(class_directory, d))])
test_image_directory = "test-dataset"
test_image_name = sorted([d for d in os.listdir(test_image_directory) if os.path.isfile(os.path.join(test_image_directory, d))])




###################################test model with test images###################################
for i in range(len(test_image_name)):
    print("=========================================")
    print(test_image_name[i])
    test_image_path = test_image_directory + "/" + test_image_name[i]
    test_image = tf.keras.preprocessing.image.load_img(
        test_image_path, target_size=(img_height, img_width)
    )
    test_image_array = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image_array = tf.expand_dims(test_image_array, 0) # Create a batch
    predictions = model.predict(test_image_array)
    score = tf.nn.softmax(predictions[0])
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
#     # plt.imshow(test_image)
#     # plt.show()
#     # print(test_image_path)
#     # print(class_name[np.argmax(score)])
#     # print(100 * np.max(score))
#     # print("=========================================")
#     # print("=========================================")