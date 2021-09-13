import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from tensorflow.keras.preprocessing.image import ImageDataGenerator
tfds.disable_progress_bar()

# Make all other necessary imports.
import warnings
warnings.filterwarnings('ignore')

import time
import json
import matplotlib.pyplot as plt
import numpy as np

import os, random

import PIL
from PIL import Image

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import argparse
parser = argparse.ArgumentParser(description='Prediction script')

parser.add_argument('--image_path', default ='/home/workspace/test_images/cautleya_spicata.jpg', action='store', type=str)
parser.add_argument('--checkpoint', default ='checkpoint.pth', action='store', type=str )
parser.add_argument('--top_k', dest = 'top_k', default=5 ,action='store',  type=int)
parser.add_argument('--category_names', action='store', default = 'load_map.json')

args = parser.parse_args()


#Loading the data
def load_data():
    
    # Load the dataset with TensorFlow Datasets. Hint: use tfds.load()
    dataset, dataset_info = tfds.load('oxford_flowers102', as_supervised=True, with_info=True)

    # Create a training set, a validation set and a test set.
    train_set, valid_set, test_set = dataset['train'], dataset['validation'], dataset['test']

    # Get the number of examples in each set from the dataset info.
    n_train_ex = dataset_info.splits['train'].num_examples # number of train set example
    n_valid_ex = dataset_info.splits['validation'].num_examples # number of validation set example
    n_test_ex = dataset_info.splits['test'].num_examples # number of test set example

    # Get the number of classes in the dataset from the dataset info.
    n_class = dataset_info.features['label'].num_classes
    
    return train_set, valid_set, test_set, n_train_ex

# Image normalization
batch_size = 32
image_size  = 224
def image_fmt(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image

# split the data into training, validation and test batches
def data_batch(train_set, valid_set, test_set, n_train_ex):
    train_batches = train_set.shuffle(n_train_ex//4).map(image_fmt).batch(batch_size).prefetch(1)
    valid_batches = valid_set.map(image_fmt).batch(batch_size).prefetch(1)
    test_batches = test_set.map(image_fmt).batch(batch_size).prefetch(1)
    return train_batches, valid_batches, test_batches

#mapping integers to their corresponding flower names
def data_mapping(flower_names):
    with open(flower_names, 'r') as f:
        class_names = json.load(f)
    return class_names

#load the model
def model_load(model_path):
    load_model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer},compile=False)
    return load_model

# prediction
def prediction(image_path, model, top_k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = Image.open(image_path) # load the images
    image = np.asarray(image) # convert image object to a numpy array        
    image = image_fmt(image)
    image = np.expand_dims(image, axis = 0) # adding an extra dimension to the image
    
    pred = model.predict(image)
    
    top_p, top_class = tf.nn.top_k(pred, k=top_k)
    top_p = top_p.numpy()
    classes = top_class.numpy()
    
        
    return top_p[0],classes[0]

# check out predict function
model_load = model_load(args.checkpoint)
class_names = data_mapping(args.category_names)
image_path = args.image_path
top_p,classes  = prediction(image_path, model_load, args.top_k)
flower_names = [class_names[str(i+1)] for i in classes]
print(top_p)
print(classes)
print(flower_names)