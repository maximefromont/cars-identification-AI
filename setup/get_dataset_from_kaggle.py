import pandas as pd
import os
import opendatasets as od

# Assign the Kaggle data set URL into variable
dataset = 'https://www.kaggle.com/datasets/prondeau/the-car-connection-picture-dataset'
# Using opendatasets let's download the data sets
od.download(dataset)