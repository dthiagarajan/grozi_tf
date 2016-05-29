## WARNING: This file takes a long time to run
import numpy as np

num_points = 100
data = []
for i in xrange(num_points):
	x1= np.random.normal(0.0, 0.9)
	y1= x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.05)
	data.append([x1, y1])

x_data = [v[0] for v in data]
y_data = [v[1] for v in data]


import matplotlib.pyplot as plt
print 'here'
#Graphic display
'''
plt.plot(x_data, y_data, 'ro')
plt.legend()
plt.show()
'''
import tensorflow as tf


W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
import time
print 'Starting training'
plt.ion()
for step in range(101):
	plt.plot(x_data, y_data, 'ro')
	plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
	plt.legend()
	plt.draw()
	plt.pause(0.001)
	sess.run(train)
	print(step, sess.run(W), sess.run(b))
