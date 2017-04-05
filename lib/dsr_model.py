# coding=utf-8
#using dsr as a Q-value estimator
import tensorflow as tf
import numpy as np


class DSR(object):


    def __init__(self,args):
        self.kernel_sizes = args.kernel.size  # add to  define every filter's size
        self.edim = args.size  # define vector dimentions


    def build_model(self):
        # 输入的真实值Y和当前选取动作actions
        x_conv = tf.placehold(tf.float32,[None,size])


        # Filters
        F1 = tf.Variable(tf.random_normal([self.kernel_sizes[0], self.edim, 1, self.n_filters], stddev=self.std_dev),
                         dtype='float32')
        F2 = tf.Variable(tf.random_normal([self.kernel_sizes[1], self.edim, 1, self.n_filters], stddev=self.std_dev),
                         dtype='float32')
        F3 = tf.Variable(tf.random_normal([self.kernel_sizes[2], self.edim, 1, self.n_filters], stddev=self.std_dev),
                         dtype='float32')
        FB1 = tf.Variable(tf.constant(0.1, shape=[self.n_filters]))
        FB2 = tf.Variable(tf.constant(0.1, shape=[self.n_filters]))
        FB3 = tf.Variable(tf.constant(0.1, shape=[self.n_filters]))
        # Weight for final layer
        W = tf.Variable(tf.random_normal([3 * self.n_filters, 2], stddev=self.std_dev), dtype='float32')
        b = tf.Variable(tf.constant(0.1, shape=[1, 2]), dtype='float32')
        # Convolutions
        C1 = tf.add(tf.nn.conv2d(x_conv, F1, [1, 1, 1, 1], padding='VALID'), FB1)
        C2 = tf.add(tf.nn.conv2d(x_conv, F2, [1, 1, 1, 1], padding='VALID'), FB2)
        C3 = tf.add(tf.nn.conv2d(x_conv, F3, [1, 1, 1, 1], padding='VALID'), FB3)

        C1 = tf.nn.relu(C1)
        C2 = tf.nn.relu(C2)
        C3 = tf.nn.relu(C3)

        # Max pooling
        maxC1 = tf.nn.max_pool(C1, [1, C1.get_shape()[1], 1, 1], [1, 1, 1, 1], padding='VALID')
        maxC1 = tf.squeeze(maxC1, [1, 2])
        maxC2 = tf.nn.max_pool(C2, [1, C2.get_shape()[1], 1, 1], [1, 1, 1, 1], padding='VALID')
        maxC2 = tf.squeeze(maxC2, [1, 2])
        maxC3 = tf.nn.max_pool(C3, [1, C3.get_shape()[1], 1, 1], [1, 1, 1, 1], padding='VALID')
        maxC3 = tf.squeeze(maxC3, [1, 2])
        # Concatenating pooled features
        z = tf.concat(axis=1, values=[maxC1, maxC2, maxC3])
        zd = tf.nn.dropout(z, self.cur_drop_rate)
        # Fully connected layer
        self.y = tf.add(tf.matmul(zd, W), b)


    def train(self):
        pass