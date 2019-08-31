from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
import math
from pprint import pprint
sess = tf.Session()

FIRST = ord('a')
LAST = ord('c')
N_CODES = (LAST - FIRST)+1

def onehot(str):
	chars = list(map(lambda c: max(0, ord(c) - FIRST), list(str)))
	twoDim = sess.run(tf.one_hot(chars, N_CODES))
	# flatten the 2D array:
	return [item for sublist in twoDim for item in sublist]
	
import pandas
df = pandas.read_csv(sys.argv[1])
train_inputs = df['input'].map(onehot).values.tolist()
train_labels = df['output'].map(onehot).values.tolist()
sess.close()
print('INPUTS:')
pprint(train_inputs)
print('OUTPUTS:')
pprint(train_labels)

SAMPLES = len(train_inputs)
INPUT_VAR_CODES = len(train_inputs[0])
INPUT_VARS = INPUT_VAR_CODES // N_CODES
OUTPUT_VAR_CODES = len(train_labels[0])
OUTPUT_VARS = OUTPUT_VAR_CODES // N_CODES
HIDDEN=100
num_epochs = 100000
learning_rate = 0.005

print("is=%d iv=%d os=%d oc=%d" % (SAMPLES, INPUT_VAR_CODES, SAMPLES, OUTPUT_VAR_CODES))

def neural_net(X):
	parameters = {
		'W1': tf.Variable(tf.random_normal([INPUT_VAR_CODES, HIDDEN])),
		'b1': tf.Variable(tf.random_normal([HIDDEN])),
		'W2': tf.Variable(tf.random_normal([HIDDEN, OUTPUT_VAR_CODES])),
		'b2': tf.Variable(tf.random_normal([OUTPUT_VAR_CODES]))
	}
	Z1 = tf.add(tf.matmul(X, parameters['W1']), parameters['b1'])
	A2 = tf.nn.relu(Z1)
	Z2 = tf.add(tf.matmul(A2, parameters['W2']), parameters['b2'])
	print("Returning NN for X=" + str(X))
	return Z2

def train(X,Y):
	Z = neural_net(X)
	costs = []
	for i in range(OUTPUT_VARS):
		c = Z[0][N_CODES*i:N_CODES*i+N_CODES]
		costs.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=c,  labels=Y[0][N_CODES*i:N_CODES*i+N_CODES])))
	tf.Print('costs:',costs)
	cost = tf.reduce_sum(costs)
	print(costs)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		epoch = 0
		while epoch<num_epochs:
			for i in range(SAMPLES):
				_ , c = sess.run([optimizer, cost], feed_dict={
					X: np.reshape(train_inputs[i],[1,INPUT_VAR_CODES]), 
					Y: np.reshape(train_labels[i],[1,OUTPUT_VAR_CODES])
				}) 
				if c <= 0.000001:
					epoch = num_epochs
					break
			if epoch % 200 == 0:
				print ("Cost after epoch %i: %f" % (epoch+200, c))
			epoch += 1

		# Test predictions by computing the output using training set as input

		for row in range(len(train_inputs)):

			g = train_inputs[row]
			g = np.reshape(g,[1,INPUT_VAR_CODES])
			output = neural_net(g)
			outputs = []
			for i in range(OUTPUT_VARS):
				kk = tf.nn.softmax(output[0][N_CODES*i : N_CODES*i+N_CODES])
				outputs.append(kk)
				
			init1 = tf.global_variables_initializer()
			sess.run(init1)
			
			out = sess.run(outputs)
			out = np.reshape(list(map(list,out)), [1, OUTPUT_VAR_CODES])[0].tolist()
			out = list(map(lambda x: "%.1f" % x, out))
			print("\nROW #" + str(row))
			print("Expected: " + str(train_labels[row]))
			print("Actual:")
			pprint(out)

		#import code
		#code.interact(local=locals())

# setup the Neural Network
X = tf.placeholder(tf.float32, shape=[1, INPUT_VAR_CODES])
Y = tf.placeholder(tf.float32, shape=[1, OUTPUT_VAR_CODES])
train(X,Y)
