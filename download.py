#coding:utf-8
from tensorflow.examples.tutorials.mnist import input_data
#下载数据集
mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)
