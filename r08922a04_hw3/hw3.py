import numpy as np
import pandas as pd
from tqdm import tqdm 
import argparse
import matplotlib.pyplot as plt


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--training', help = 'trianing data', default = 'hw3_train.dat')
	parser.add_argument('--testing', help = 'testing data', default = 'hw3_test.dat')
	# parser.add_argument('--sgd', help = 'IS SGD?', type = int,default = 0)
	# parser.add_argument('--lr', help ='learning rate', type = float, default = 0.01)

	args = parser.parse_args()

	return args

def load(path):
	with open(path)as infile:
		data = infile.readlines()
	x = []
	# y =[]
	y = np.zeros((len(data),1))
	for i in range(len(data)):
		t = data[i].split()
		temp = []
		temp.append(1.0)
		temp += t
		#print(temp)
		x.append(temp[0:-1])
		y[i] = float(temp[-1])
		# y.append(temp[-1])
	x = np.array(x).astype(np.float)
	y = np.array(y).astype(np.float)

	return x,y


def theta(something):
	return np.exp(something)/(1+np.exp(something))

def gradient_descent(x, y, W):
	"""1/N sum(θ(-y_n w_t^T x_n))(-y_n x_n)"""
	gradient = np.zeros((len(x[0]),1))
	N = len(x)
	yWx = y * np.dot(x,W)
	grad = (1/N)*(sum(theta(-yWx)* -y*x))
	for i in range(len(grad)):
		gradient[i] = grad[i]
	return gradient



def testing(x, y, W):
	predict_y = np.sign(np.dot(x,W))
	# print(np.sign(predict_y))
	error_rate = sum(predict_y != y)/len(x)

	return error_rate

def testing_print(x, y, W):
	predict_y = np.sign(np.dot(x,W))
	# print(np.sign(predict_y))
	# for i in range(len(predict_y)):
	# 	if(predict_y[i]!=1):
	# 		print(i)
	error_rate = sum(predict_y != y)/len(x)

	return error_rate


def Logestic_Regression_GD(train_x,train_y, test_x, test_y, learning_rate = 0.01, train_iter = 2000):
	"""Logestic Regression with gradient descent"""
	W = np.zeros((len(train_x[0]), 1), dtype = float)
	lr = learning_rate
	E_in = []
	E_out = []
	for i in tqdm(range(train_iter)):
		W = W - lr * gradient_descent(train_x, train_y, W)
		e_in = testing(train_x, train_y, W)
		E_in.append(e_in)
		e_out = testing(test_x, test_y, W)
		E_out.append(e_out)
	return E_in, E_out


def stochastic_gradient_descent(x, y, W):
	gradient = np.zeros((len(x),1))
	# print('x,y,W')
	# print(x,y, W)
	N = len(x)
	yWx = y * np.dot(x,W)
	# print('ywx',yWx)
	# print(-y*x)
	grad = theta(-yWx) * -y * x
	for i in range(len(grad)):
		gradient[i] = grad[i]
	# print(grad)
	return gradient

def Logestic_Regression_SGD(train_x, train_y, test_x, test_y, learning_rate = 0.001, train_iter = 2000):
	""" Logestic Regression with stochastic gradient descent"""
	W = np.zeros((len(train_x[0]), 1), dtype = float)

	lr = learning_rate
	E_in = []
	E_out = []
	idx = 0
	for i in tqdm(range(train_iter)):
		# print(train_x[idx % 1000],train_y[idx % 1000], '\n')
		W = W - lr * stochastic_gradient_descent(train_x[idx % 1000], train_y[idx % 1000], W)
		idx += 1
		# if(idx>3):
		# 	exit()
		e_in = testing(train_x, train_y, W)
		E_in.append(e_in)
		e_out = testing(test_x, test_y, W)
		E_out.append(e_out)
	test = testing_print(test_x, test_y, W)
	# print(W)
	return E_in, E_out

def plot_figure(E_GD, E_SGD, i):
	plt.plot(E_GD, label = 'E_GD')
	plt.plot(E_SGD, label='E_SGD')
	plt.xlabel('iterations')

	plt.legend()
	if(i==0):
		plt.title('E_in')
		plt.ylabel('E_in')
	else:
		plt.title('E_out')
		plt.ylabel('E_out')
	plt.show()

def main():
	args = parse_args()
	"""data loading"""
	train_x, train_y =load(args.training)
	test_x, test_y = load(args.testing)

	# tmp = -train_y*np.dot(train_x,W)
	GD_E_in, GD_E_out = Logestic_Regression_GD(train_x, train_y, test_x, test_y)
	print(GD_E_in[-1], GD_E_out[-1])
	SGD_E_in, SGD_E_out = Logestic_Regression_SGD(train_x, train_y, test_x, test_y)
	print(SGD_E_in[-1], SGD_E_out[-1])
	# print(E_in,E_out)
	# gradient_descent(train_x, train_y, W)
	plot_figure(GD_E_in, SGD_E_in,0)
	plot_figure(GD_E_out, SGD_E_out,1)












if __name__ =='__main__':
	main()