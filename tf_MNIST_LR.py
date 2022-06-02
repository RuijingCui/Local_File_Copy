import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


'''
可以从模块 input_data 给出的 TensorFlow 示例中获取 MNIST 的输入数据。该 one_hot 标志设置为真，以使用标签的 one_hot 编码。
这产生了两个张量，大小为 [55000，784] 的 mnist.train.images 和大小为 [55000，10] 的 mnist.train.labels。mnist.train.images 的每项都是一个范围介于 0 到 1 的像素强度
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

## 数据常量
n_input = 784
n_classes = 10
# htyperparameters
max_epochs = 10000
learning_rate = 0.5
batch_size = 10
seed = 0
n_hidden = 30 ## 隐藏层神经元数量

## 需要sigmoid函数的导数来进行权重更新，
def sigmaprime(x):
	return tf.multiply(tf.sigmoid(x),tf.subtract(tf.constant(1.0),tf.sigmoid(x)))

# 在 TensorFlow 图中为训练数据集的输入 x 和标签 y 创建占位符：
x_in = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.float32,[None,n_classes])

# 定义权重和偏置变量
# W = tf.Variable(tf.zeros([784,10],name='W'))
# b = tf.Variable(tf.zeros([10]),name='b')
weights = {
	'h1':tf.Variable(tf.random_normal([n_input,n_hidden],seed=seed)),
	'out':tf.Variable(tf.random_normal([n_hidden,n_classes],seed=seed)),
}
biases = {
	'h1':tf.Variable(tf.random_normal([1,n_hidden],seed=seed)),
	'out':tf.Variable(tf.random_normal([1,n_classes],seed=seed)),
}

# 创建模型
def multilayer_perceptron(x,weights,biases):
	## hidden layers with RELU
	h_layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['h1'])
	out_layer_1 = tf.sigmoid(h_layer_1)
	## output layer with linear activation
	h_out = tf.matmul(out_layer_1,weights['out'] + biases['out'])
	return tf.sigmoid(h_out),h_out,out_layer_1,h_layer_1
	
# 为正向传播、误差、梯度和更新计算创建计算图：
## forward pass
y_hat,h_2,o_1,h_1 = multilayer_perceptron(x_in,weights,biases)

## Errors
err = y_hat - y

# backward pass
delta_2 = tf.multiply(err,sigmaprime(h_2))
delta_w_2 = tf.matmul(tf.transpose(o_1),delta_2)

wtd_error = tf.matmul(delta_2,tf.transpose(weights['out']))
delta_1 = tf.multiply(wtd_error,sigmaprime(h_1))
delta_w_1 = tf.matmul(tf.transpose(x_in),delta_1)

eta = tf.constant(learning_rate)

## Update weights
step = [
	tf.assign(weights['h1'],tf.subtract(weights['h1'],tf.multiply(eta,delta_w_1))),
	tf.assign(biases['h1'],tf.subtract(biases['h1'],tf.multiply(eta,tf.reduce_mean(delta_1,axis=[0])))),
	tf.assign(weights['out'],tf.subtract(weights['out'],tf.multiply(eta,delta_w_2))),
	tf.assign(biases['out'],tf.subtract(biases['out'],tf.multiply(eta,tf.reduce_mean(delta_2,axis=[0])))),
]


# 定义计算精度 accuracy 的操作
acc_mat = tf.equal(tf.argmax(y_hat,1), tf.argmax(y,1))
accuracy = tf.reduce_sum(tf.cast(acc_mat,tf.float32))

# # 训练时添加 summary 操作来收集数据。使用直方图以便看到权重和偏置随时间相对于彼此值的变化关系。可以通过 TensorBoard Histogtam 选项卡看到：
# w_h = tf.summary.histogram("weights",W)
# b_h = tf.summary.histogram("biases",b)
#
# # 定义交叉熵（cross-entropy）和损失（loss）函数，并添加 name scope 和 summary 以实现更好的可视化。使用 scalar summary 来获得随时间变化的损失函数。scalar summary 在 Events 选项卡下可见：
# with tf.name_scope("wx_b") as scope:
# 	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_hat))
# 	tf.summary.scalar('cross-entropy',loss)
#
# # 采用 TensorFlow GradientDescentOptimizer，学习率为 0.01。为了更好地可视化，定义一个 name_scope：
# with tf.name_scope("Train") as scope:
# 	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
	
# 为变量进行初始化
init = tf.global_variables_initializer()

# 执行图
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(max_epochs):
		batch_xs, batch_ys, = mnist.train.next_batch(batch_size)
		sess.run(step, feed_dict = {x_in: batch_xs, y: batch_ys})
		if epoch % 1000 == 0:
			acc_test = sess.run(accuracy, feed_dict={x_in:mnist.test.images, y:mnist.test.labels})
			acc_train = sess.run(accuracy,feed_dict = {x_in:mnist.train.images, y:mnist.test.labels})
			print('Epoch:{0} Accuracy Train %:{1}  Accuracy test%:{2}'.format(epoch,acc_train/600,(acc_test/100)))



# # 组合所有的 summary 操作：
# merged_summary_op = tf.summary.merge_all()
#
# # 定义会话并将所有的 summary 存储在定义的文件夹中
# with tf.Session() as sess:
# 	## initialize the Variables
# 	sess.run(init)
# 	max_epochs = 50
# 	batch_size = 100
# 	accuracy = 0
# 	## create an event file
# 	summary_writer = tf.summary.FileWriter('graphs',sess.graph)
# 	## training
# 	for epoch in range(max_epochs):
# 		loss_avg = 0
# 		num_of_batch = int(mnist.train.num_examples/batch_size)
# 		for i in range(num_of_batch):
# 			## get the next batch of data
# 			batch_xs,batch_ys = mnist.train.next_batch(100)
# 			## run the optimizer
# 			_, l, summary_str = sess.run([optimizer,loss,merged_summary_op],feed_dict={x:batch_xs,y:batch_ys})
# 			loss_avg += l
# 			## add all summaries per batch
# 			summary_writer.add_summary(summary_str,epoch*num_of_batch + i)
# 		loss_avg = loss_avg/num_of_batch
# 		print('Epoch{0}:  Loss{1}'.format(epoch,loss_avg))
# 	print('done')
# 	print(sess.run(accuracy,feed_dict =  {x:mnist.test.images,y:mnist.test.labels}))
		



