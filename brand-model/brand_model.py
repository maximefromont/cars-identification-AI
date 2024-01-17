import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt
import os


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os
import os


def basic_brand_model(epochs ):
######################SETUP###############################


  # '0': Display all logs (default behavior).
  # '1': Display only warning and error logs.
  # '2': Display only error logs.
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  batch_size = 32
  img_height = 180
  img_width = 180

  data_dir="brand-sorted-dataset"

  ######################CREATE DATASET######################

  train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

  val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
  class_names = train_ds.class_names
  # print(class_names)


  ######################CREATE MODEL#########################

  num_classes = len(class_names)

  model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  model.summary()

  ######################TRAIN MODEL#########################  
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
  )


  ######################PLOT MODEL#########################

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)


  model_name = 'brand_model_car_classifier'
  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  # Create the "plot" directory if it doesn't exist
  os.makedirs("plot", exist_ok=True)

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  os.makedirs("brand-model/model-data", exist_ok=True)
  plt.savefig('brand-model/plot/basic'+model_name+'_'+str(epochs)+'.png')

  ######################SAVE MODEL#########################
  model.save('brand-model/model/basic'+model_name+'_'+str(epochs)+'.h5')
  #save the model data to a csv file with data model name and last value of  accuracy, loss, validation accuracy, validation loss
  # Create the "model-data" directory if it doesn't exist
  os.makedirs("brand-model/model-data", exist_ok=True)

  # Get the last values of accuracy, loss, validation accuracy, and validation loss
  last_acc = acc[-1]
  last_loss = loss[-1]
  last_val_acc = val_acc[-1]
  last_val_loss = val_loss[-1]

  # Save the model data to a csv file
  with open('brand-model/model-data/'+'basic'+model_name+'_'+str(epochs)+'.csv', 'w') as f:
    f.write('model_name, accuracy, loss, val_accuracy, val_loss\n')
    f.write('basic'+model_name+'_'+str(epochs)+', '+str(last_acc)+', '+str(last_loss)+', '+str(last_val_acc)+', '+str(last_val_loss))

def dropout_brand_model(epochs, dropout_rate_1=0.5, dropout_rate_2=0.5, dropout_rate_3=0.5 ):
######################SETUP###############################


  # '0': Display all logs (default behavior).
  # '1': Display only warning and error logs.
  # '2': Display only error logs.
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  batch_size = 32
  img_height = 480
  img_width = 480

  data_dir="short-brand-sorted-dataset"

  ######################CREATE DATASET######################

  train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

  val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
  class_names = train_ds.class_names
  # print(class_names)


  ######################CREATE MODEL#########################

  num_classes = len(class_names)

  model = Sequential([    


  layers.Rescaling(1./255),

  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(dropout_rate_1),

  layers.Conv2D(32, 3, padding='same', activation='relu'),  
  layers.MaxPooling2D(),
  layers.Dropout(dropout_rate_1),

  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(dropout_rate_1),
  
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  
  layers.Dense(num_classes, name="outputs"),
  
])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  
  model.build(input_shape=(None, img_height, img_width, 3))

  model.summary()

  ######################TRAIN MODEL#########################  
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
  )


  ######################PLOT MODEL#########################

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)


  model_name = 'short_480_dropdown_brand_model_car_classifier_'+str(dropout_rate_1)+'_'+str(dropout_rate_2)+'_'+str(dropout_rate_3)
  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  # Create the "plot" directory if it doesn't exist
  os.makedirs("plot", exist_ok=True)

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  os.makedirs("brand-model/model-data", exist_ok=True)
  plt.savefig('brand-model/plot/'+model_name+'_'+str(epochs)+'.png')

  ######################SAVE MODEL#########################
  model.save('brand-model/model/'+model_name+'_'+str(epochs)+'.h5')
  #save the model data to a csv file with data model name and last value of  accuracy, loss, validation accuracy, validation loss
  # Create the "model-data" directory if it doesn't exist
  os.makedirs("brand-model/model-data", exist_ok=True)

  # Get the last values of accuracy, loss, validation accuracy, and validation loss
  last_acc = acc[-1]
  last_loss = loss[-1]
  last_val_acc = val_acc[-1]
  last_val_loss = val_loss[-1]

  # Save the model data to a csv file
  with open('brand-model/model-data/'+model_name+'_'+str(epochs)+'.csv', 'w') as f:
    f.write('model_name, accuracy, loss, val_accuracy, val_loss\n')
    f.write(model_name+'_'+str(epochs)+', '+str(last_acc)+', '+str(last_loss)+', '+str(last_val_acc)+', '+str(last_val_loss))

def optimized_brand_model(epochs ):
######################SETUP###############################


  # '0': Display all logs (default behavior).
  # '1': Display only warning and error logs.
  # '2': Display only error logs.
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  batch_size = 32
  img_height = 180
  img_width = 180

  data_dir="brand-sorted-dataset"

  ######################CREATE DATASET######################

  train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

  val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
  class_names = train_ds.class_names
  # print(class_names)


  ######################CREATE MODEL#########################

  num_classes = len(class_names)

  model = Sequential([    


  layers.Rescaling(1./255),

  layers.Conv2D(16, 3, padding='same', activation='relu'),  
  layers.MaxPooling2D(),
 

  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),


  layers.Conv2D(64, 3, padding='same', activation='relu'),  
  layers.MaxPooling2D(),
 
  
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  
  layers.Dense(num_classes, name="outputs"),
  
])
  
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)


  model.compile(optimizer=optimizer, 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

  
  model.build(input_shape=(None, img_height, img_width, 3))

  model.summary()

  ######################TRAIN MODEL#########################  
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
  )


  ######################PLOT MODEL#########################

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)


  model_name = 'opti_brand_model_car_classifier_0.5_0.5_0.5'
  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  # Create the "plot" directory if it doesn't exist
  os.makedirs("plot", exist_ok=True)

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  os.makedirs("brand-model/model-data", exist_ok=True)
  plt.savefig('brand-model/plot/'+model_name+'_'+str(epochs)+'.png')

  ######################SAVE MODEL#########################
  model.save('brand-model/model/'+model_name+'_'+str(epochs)+'.h5')
  #save the model data to a csv file with data model name and last value of  accuracy, loss, validation accuracy, validation loss
  # Create the "model-data" directory if it doesn't exist
  os.makedirs("brand-model/model-data", exist_ok=True)

  # Get the last values of accuracy, loss, validation accuracy, and validation loss
  last_acc = acc[-1]
  last_loss = loss[-1]
  last_val_acc = val_acc[-1]
  last_val_loss = val_loss[-1]

  # Save the model data to a csv file
  with open('brand-model/model-data/'+model_name+'_'+str(epochs)+'.csv', 'w') as f:
    f.write('model_name, accuracy, loss, val_accuracy, val_loss\n')
    f.write(model_name+'_'+str(epochs)+', '+str(last_acc)+', '+str(last_loss)+', '+str(last_val_acc)+', '+str(last_val_loss))

def normalyze_brand_model(epochs ):
  ######################SETUP###############################


  # '0': Display all logs (default behavior).
  # '1': Display only warning and error logs.
  # '2': Display only error logs.
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  batch_size = 32
  img_height = 180
  img_width = 180

  data_dir="brand-sorted-dataset"

  ######################CREATE DATASET######################

  train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

  val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
  class_names = train_ds.class_names
  # print(class_names)


  ######################CREATE MODEL#########################

  num_classes = len(class_names)

  model = Sequential([    


  layers.Rescaling(1./255),

  layers.Conv2D(16, 3, padding='same', activation='relu'),  
  layers.BatchNormalization(),
  layers.MaxPooling2D(),
  

  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.BatchNormalization(),
  layers.MaxPooling2D(),
  

  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.BatchNormalization(),
  layers.MaxPooling2D(),
 
  
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  
  layers.Dense(num_classes, name="outputs"),
  
])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  
  model.build(input_shape=(None, img_height, img_width, 3))

  model.summary()

  ######################TRAIN MODEL#########################  
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
  )


  ######################PLOT MODEL#########################

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)


  model_name = 'normalyzed_brand_model_car_classifier_0.5_0.5_0.5'
  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  # Create the "plot" directory if it doesn't exist
  os.makedirs("plot", exist_ok=True)

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  os.makedirs("brand-model/model-data", exist_ok=True)
  plt.savefig('brand-model/plot/'+model_name+'_'+str(epochs)+'.png')

  ######################SAVE MODEL#########################
  model.save('brand-model/model/'+model_name+'_'+str(epochs)+'.h5')
  #save the model data to a csv file with data model name and last value of  accuracy, loss, validation accuracy, validation loss
  # Create the "model-data" directory if it doesn't exist
  os.makedirs("brand-model/model-data", exist_ok=True)

  # Get the last values of accuracy, loss, validation accuracy, and validation loss
  last_acc = acc[-1]
  last_loss = loss[-1]
  last_val_acc = val_acc[-1]
  last_val_loss = val_loss[-1]

  # Save the model data to a csv file
  with open('brand-model/model-data/'+model_name+'_'+str(epochs)+'.csv', 'w') as f:
    f.write('model_name, accuracy, loss, val_accuracy, val_loss\n')
    f.write(model_name+'_'+str(epochs)+', '+str(last_acc)+', '+str(last_loss)+', '+str(last_val_acc)+', '+str(last_val_loss))

def denser_brand_model(epochs ):
  ######################SETUP###############################


  # '0': Display all logs (default behavior).
  # '1': Display only warning and error logs.
  # '2': Display only error logs.
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  batch_size = 32
  img_height = 180
  img_width = 180

  data_dir="brand-sorted-dataset"

  ######################CREATE DATASET######################

  train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

  val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
  class_names = train_ds.class_names
  # print(class_names)


  ######################CREATE MODEL#########################

  num_classes = len(class_names)

  model = Sequential([    


  layers.Rescaling(1./255),

  layers.Conv2D(16, 3, padding='same', activation='relu'),  
  layers.MaxPooling2D(),
   layers.Dropout(0.5),
  

  layers.Conv2D(32, 3, padding='same', activation='relu'), 
  layers.MaxPooling2D(),
  layers.Dropout(0.5),
  

  layers.Conv2D(64, 3, padding='same', activation='relu'), 
  layers.MaxPooling2D(),
 

  layers.Conv2D(128, 3, padding='same', activation='relu'), 
  layers.MaxPooling2D(),
  layers.Dropout(0.5),

  layers.Conv2D(256, 3, padding='same', activation='relu'), 
  layers.MaxPooling2D(),
  layers.Dropout(0.5),


  
  layers.Flatten(),
  layers.Dense(128, activation='relu'),

 
  
  layers.Dense(num_classes, name="outputs"),
  
])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  
  model.build(input_shape=(None, img_height, img_width, 3))

  model.summary()

  ######################TRAIN MODEL#########################  
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
  )


  ######################PLOT MODEL#########################

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)


  model_name = 'denser_brand_model_car_classifier_0.5_0.5_0.5'
  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  # Create the "plot" directory if it doesn't exist
  os.makedirs("plot", exist_ok=True)

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  os.makedirs("brand-model/model-data", exist_ok=True)
  plt.savefig('brand-model/plot/'+model_name+'_'+str(epochs)+'.png')

  ######################SAVE MODEL#########################
  model.save('brand-model/model/'+model_name+'_'+str(epochs)+'.h5')
  #save the model data to a csv file with data model name and last value of  accuracy, loss, validation accuracy, validation loss
  # Create the "model-data" directory if it doesn't exist
  os.makedirs("brand-model/model-data", exist_ok=True)

  # Get the last values of accuracy, loss, validation accuracy, and validation loss
  last_acc = acc[-1]
  last_loss = loss[-1]
  last_val_acc = val_acc[-1]
  last_val_loss = val_loss[-1]

  # Save the model data to a csv file
  with open('brand-model/model-data/'+model_name+'_'+str(epochs)+'.csv', 'w') as f:
    f.write('model_name, accuracy, loss, val_accuracy, val_loss\n')
    f.write(model_name+'_'+str(epochs)+', '+str(last_acc)+', '+str(last_loss)+', '+str(last_val_acc)+', '+str(last_val_loss))

def CNN_brand_model(epochs ):
  ######################SETUP###############################


  # '0': Display all logs (default behavior).
  # '1': Display only warning and error logs.
  # '2': Display only error logs.
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  batch_size = 32
  img_height = 224
  img_width = 224

  data_dir="short-brand-sorted-dataset"

  ######################CREATE DATASET######################

  train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

  val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
  class_names = train_ds.class_names
  # print(class_names)


  ######################CREATE MODEL#########################

  num_classes = len(class_names)
  base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_height, 3))


  model = Sequential([
     layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
      base_model,
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))
,
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))
      
,
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
      layers.Dropout(0.2)
,
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
  ])


  
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


  model.compile(optimizer=optimizer, 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

  
  model.build(input_shape=(None, 180, 180, 3))

  model.summary()

  ######################TRAIN MODEL#########################  
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
  )


  ######################PLOT MODEL#########################

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)


  model_name = 'CNN_brand_model_car_classifier'
  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  # Create the "plot" directory if it doesn't exist
  os.makedirs("plot", exist_ok=True)

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  os.makedirs("brand-model/model-data", exist_ok=True)
  plt.savefig('brand-model/plot/'+model_name+'_'+str(epochs)+'.png')

  ######################SAVE MODEL#########################
  model.save('brand-model/model/'+model_name+'_'+str(epochs)+'.h5')
  #save the model data to a csv file with data model name and last value of  accuracy, loss, validation accuracy, validation loss
  # Create the "model-data" directory if it doesn't exist
  os.makedirs("brand-model/model-data", exist_ok=True)

  # Get the last values of accuracy, loss, validation accuracy, and validation loss
  last_acc = acc[-1]
  last_loss = loss[-1]
  last_val_acc = val_acc[-1]
  last_val_loss = val_loss[-1]

  # Save the model data to a csv file
  with open('brand-model/model-data/'+model_name+'_'+str(epochs)+'.csv', 'w') as f:
    f.write('model_name, accuracy, loss, val_accuracy, val_loss\n')
    f.write(model_name+'_'+str(epochs)+', '+str(last_acc)+', '+str(last_loss)+', '+str(last_val_acc)+', '+str(last_val_loss))



#only run if call this file directly
if __name__ == "__main__":
 #create a dropdown model , a optimized model, a normalyzed model, a denser model
  epochs = 30
  dropout_brand_model(epochs)
  # optimized_brand_model(epochs)
  # normalyze_brand_model(epochs)
  # denser_brand_model(epochs)
  #CNN_brand_model(10)

  epochs = 30
  #build different dropout rate model
  # dropout_brand_model(epochs, 0.2, 0.2, 0.2)
  # dropout_brand_model(epochs, 0.3, 0.3, 0.3)
  # dropout_brand_model(epochs, 0.4, 0.4, 0.4)
  # dropout_brand_model(epochs, 0.6, 0.6, 0.6)
  # dropout_brand_model(epochs, 0.7, 0.7, 0.7)
  # dropout_brand_model(epochs, 0.8, 0.8, 0.8)



