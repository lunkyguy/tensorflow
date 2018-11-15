#coding:utf-8
from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os

#读取MNIST数据集
mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)
#创建存储文件夹
save_dir = 'MNIST_data/raw/'
if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)

#保存前20张图片
for i in range(20):
    image_arrary = mnist.train.images[i,:]
    image_arrary = image_arrary.reshape(28,28)
    fliename = save_dir + 'mnist_train_%d.jpg'%i
    scipy.misc.toimage(image_arrary,cmin = 0.0,cmax =1.0).save(fliename)