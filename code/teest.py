import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from scipy.io import loadmat
from PIL import Image


current_dir = os.getcwd()
parent_path = os.path.dirname(current_dir)
svhn_dir = os.path.join(parent_path, 'SVHN-data')
svhn_train_data = loadmat(svhn_dir + '/train_32x32.mat')

XXX = svhn_train_data['X'].transpose((3, 0, 1, 2))
yyy = svhn_train_data['y']

#data processing func used in map
def preprocessing(image, label):
    image = tf.image.resize_images(image, [28, 28])
    image.set_shape([28, 28, 3])
    # image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_brightness(image, max_delta=0.1)
    # image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.cast(image, tf.float32)
    #image = image / 255.0
    return image, label

dataset = tf.data.Dataset.from_tensor_slices((XXX, yyy))
#dataset = dataset.repeat(32)
dataset = dataset.map(preprocessing)
# dataset = dataset.shuffle(3200)
dataset = dataset.batch(32)


iterator = dataset.make_initializable_iterator()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    test_data = sess.run(iterator.get_next())
    print(test_data[0].shape)
    print(test_data[1].shape)
    for i in range(32):
        im = Image.fromarray(test_data[0][i].astype("uint8"))
        im.show()
        print(test_data[1][i])

