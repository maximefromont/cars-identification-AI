import pickle
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


def build_dropout_complete_brand_model(epochs,data_dir='complete-brand-dataset', dropout_rate_1=0.5, dropout_rate_2=0.5, dropout_rate_3=0.5,resolution=180,batch_size=32):
######################SETUP###############################

  # '0': Display all logs (default behavior).
  # '1': Display only warning and error logs.
  # '2': Display only error logs.
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  img_height = resolution
  img_width = resolution
  

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

  ######################CREATE MODEL#########################

  num_classes = len(class_names)

  model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(dropout_rate_2),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(dropout_rate_2),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(dropout_rate_2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
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


  model_name = 'initial_brand_classifier_'+str(dropout_rate_1)+'_'+str(dropout_rate_2)+'_'+str(dropout_rate_3)+'_epoch_'+str(epochs)
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
  os.makedirs("complete-brand-model/model-data", exist_ok=True)
  os.makedirs("complete-brand-model/plot", exist_ok=True)
  plt.savefig('complete-brand-model/plot/'+model_name+'.png')

  ######################SAVE MODEL#########################
  os.makedirs("complete-brand-model/model", exist_ok=True)
  model.save('complete-brand-model/model/'+model_name+'.h5')
  #save the model data to a csv file with data model name and last value of  accuracy, loss, validation accuracy, validation loss
  # Create the "model-data" directory if it doesn't exist
  os.makedirs("complete-brand-model/model-data", exist_ok=True)

  # Get the last values of accuracy, loss, validation accuracy, and validation loss
  last_acc = acc[-1]
  last_loss = loss[-1]
  last_val_acc = val_acc[-1]
  last_val_loss = val_loss[-1]

  # Save the model data to a csv file
  with open('complete-brand-model/model-data/'+model_name+'.csv', 'w') as f:
    f.write('model_name, accuracy, loss, val_accuracy, val_loss\n')
    f.write(model_name+'_'+str(epochs)+', '+str(last_acc)+', '+str(last_loss)+', '+str(last_val_acc)+', '+str(last_val_loss))


def build_dropout_complete_brand_model_with_resolution(epochs,data_dir='complete-brand-dataset', dropout_rate_1=0.5, dropout_rate_2=0.5, dropout_rate_3=0.5,resolution=180,batch_size=32):
######################SETUP###############################

  # '0': Display all logs (default behavior).
  # '1': Display only warning and error logs.
  # '2': Display only error logs.
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  img_height = resolution
  img_width = resolution
  

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

  ######################CREATE MODEL#########################

  num_classes = len(class_names)

  model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(dropout_rate_1),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(dropout_rate_2),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(dropout_rate_3),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
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


  model_name = 'initial_brand_classifier_with_resolution_'+str(resolution)+'_'+str(dropout_rate_1)+'_'+str(dropout_rate_2)+'_'+str(dropout_rate_3)+'_epoch_'+str(epochs)
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
  os.makedirs("complete-brand-model/model-data", exist_ok=True)
  os.makedirs("complete-brand-model/plot", exist_ok=True)
  plt.savefig('complete-brand-model/plot/'+model_name+'.png')

  ######################SAVE MODEL#########################
  os.makedirs("complete-brand-model/model", exist_ok=True)
  model.save('complete-brand-model/model/'+model_name+'.h5')
  #save the model data to a csv file with data model name and last value of  accuracy, loss, validation accuracy, validation loss
  # Create the "model-data" directory if it doesn't exist
  os.makedirs("complete-brand-model/model-data", exist_ok=True)

  # Get the last values of accuracy, loss, validation accuracy, and validation loss
  last_acc = acc[-1]
  last_loss = loss[-1]
  last_val_acc = val_acc[-1]
  last_val_loss = val_loss[-1]

  # Save the model data to a csv file
  with open('complete-brand-model/model-data/'+model_name+'.csv', 'w') as f:
    f.write('model_name, accuracy, loss, val_accuracy, val_loss\n')
    f.write(model_name+'_'+str(epochs)+', '+str(last_acc)+', '+str(last_loss)+', '+str(last_val_acc)+', '+str(last_val_loss))



if __name__ == "__main__":
    build_dropout_complete_brand_model(30)
    build_dropout_complete_brand_model_with_resolution(15, resolution=90)
    build_dropout_complete_brand_model_with_resolution(15, resolution=180)
    build_dropout_complete_brand_model_with_resolution(15, resolution=360)
    build_dropout_complete_brand_model_with_resolution(15, resolution=480)
    build_dropout_complete_brand_model_with_resolution(15, resolution=720)
    