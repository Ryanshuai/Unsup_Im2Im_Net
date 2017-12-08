import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from scipy.io import loadmat


current_dir = os.getcwd()
parent_path = os.path.dirname(current_dir)

svhn_dir = os.path.join(parent_path, 'SVHN-data')

svhn_train_data = loadmat(svhn_dir + '/train_32x32.mat')


print(type(svhn_train_data['X']))
print(type(svhn_train_data['y']))

#data processing func used in map
def preprocessing(filename):
    image_string = tf.read_file(filename)
    image = tf.image.decode_png(image_string)
    image = tf.image.resize_images(image, [128, 128])
    image.set_shape([128, 128, 3])
    # image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_brightness(image, max_delta=0.1)
    # image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image

#make dataset
dataset = tf.data.Dataset.from_tensor_slices(tensor_image_paths)
#dataset = dataset.repeat(32)
dataset = dataset.map(preprocessing)
# dataset = dataset.shuffle(3200)
dataset = dataset.batch(BS)
