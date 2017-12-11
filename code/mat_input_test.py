import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from scipy.io import loadmat
from PIL import Image
import cv2

current_dir = os.getcwd()
parent_path = os.path.dirname(current_dir)
mnist_dir = os.path.join(parent_path, 'MNIST-data')
mnist = input_data.read_data_sets(mnist_dir)

svhn_dir = os.path.join(parent_path, 'SVHN-data')
svhn_train_data = loadmat(svhn_dir + '/train_32x32.mat')
XXX = svhn_train_data['X'].transpose((3, 0, 1, 2))
yyy = svhn_train_data['y']

def preprocessing(image, label):  # data processing func used in map
    image.set_shape([32, 32, 3])
    # image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_brightness(image, max_delta=0.1)
    # image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image, label

dataset = tf.data.Dataset.from_tensor_slices((XXX, yyy))
# dataset = dataset.repeat(32)
dataset = dataset.map(preprocessing)
# dataset = dataset.shuffle(3200)
dataset = dataset.batch(32)

svhn_iterator = dataset.make_initializable_iterator()

with tf.Session() as sess:
    sess.run(svhn_iterator.initializer)

    A2B_path = parent_path + '/generated_image/epoch' + str(233) + '/A2B'
    B2A_path = parent_path + '/generated_image/epoch' + str(233) + '/B2A'
    if not os.path.exists(A2B_path):
        os.makedirs(A2B_path)
    if not os.path.exists(B2A_path):
        os.makedirs(B2A_path)


    for i in range(32):
        cv2.imwrite(A2B_path + '/A' + str(i) + '.jpg', XA_28_1[i] * 255)
        cv2.imwrite(B2A_path + '/B' + str(i) + '.jpg', XB[i] * 255)
