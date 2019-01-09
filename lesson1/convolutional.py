#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#下载数据集
mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)
#x为训练图像的占位符，y_为训练图像标签的占位符
x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
