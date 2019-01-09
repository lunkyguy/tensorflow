#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#下载数据集
mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)
#x为训练图像的占位符，y_为训练图像标签的占位符
x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
#将单张图片从784维重新还原成28*28的矩阵图片
x_image = tf.reshape(x, [-1, 28,28,1])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides = [1, 2 ,2, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1, 2 ,2, 1], padding = 'SAME')

#第一层卷积层
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#全连接层
W_fc1 = weight_variable([7*7*64,1024])
