#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#下载数据集
mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)
#创建占位符
x = tf.placeholder(tf.float32,[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#模型输出
y = tf.nn.softmax(tf.matmul(x,W) + b)
#实际标签
y_ = tf.placeholder(tf.float32,[None,10])
#计算交叉熵损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y) + (1-y_)* tf.log(1-y_)))
#梯度下降
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#创建Session
sess = tf.InteractiveSession()
#变量初始化，分配内存
tf.global_variables_initializer().run()

#进行1000梯度下降
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

#正确预测结果
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#最终准确率
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))