# DCGAN digit recognition and generation
# Tom Young - Fall 2018
# Adapted from Felix Mohr's tutorial
# https://github.com/FelixMohr/Deep-learning-with-Python/blob/master/DCGAN-MNIST.ipynb

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# hand drawn data set
mnist = input_data.read_data_sets('MNIST_data')

# start graph clean
tf.reset_default_graph()

batch_size = 64 # number of data points in a batch
n_noise = 64 # input noise

# TENSOR PLACEHOLDERS
noise = tf.placeholder(dtype=tf.float32, shape=[None, n_noise])
X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob') # used by drop-out layers
is_training = tf.placeholder(dtype=tf.bool, name='is_training') # used for batch normalization


# leaky rectified linear unit
# activation function
# https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))


# used for computing loss function
def binary_cross_entropy(x, z):
    eps = 1e-12
    return -(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps))


# the digit discriminator (recognizer)
def discriminator(img_in, reuse=None, keep_prob=keep_prob):
    activation = lrelu
    with tf.variable_scope("discriminator", reuse=reuse):

        # reshape tensor
        x = tf.reshape(img_in, shape=[-1, 28, 28, 1])

        # kernel convolutions
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob) # random dropout helps prevent over-fitting
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=128, activation=activation)

        # sigmoid
        x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
        return x


