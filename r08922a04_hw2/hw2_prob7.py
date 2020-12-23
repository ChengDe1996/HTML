import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
def parse_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--iter', type = int, help = 'how many times to run the algorithm', default = 1000)
	parser.add_argument('--noise_rate', type = int, help = 'noise_rate', default = 0.2)
	parser.add_argument('--num_data','--N', type = int, help = '# data', default = 20)
	parser.add_argument('--train', help = 'training data path', default = 'hw2_train.dat')
	parser.add_argument('--test', help = 'testing data path', default = 'hw2_test.dat')
	parser.add_argument('--dim', type = int, help = ' 1 if only 1 dim, 2 if more than 1', default = 1)
	args = parser.parse_args()

	return args


def load(path):
	with open(path)as infile:
		data = infile.readlines()
	x = []
	y = []
	for i in range(len(data)):
		temp = data[i].split()
		#print(temp)
		x.append(temp[0:-1])
		y.append(temp[-1])
	x = np.array(x)
	y = np.array(y)
	return x,y

def data_generation(N,error_rate):
	#generate data by the describtion 
	# a.Generate x by a uniform distribution in [−1, 1]
	# b.Generate y by f(x) = s ̃(x) + noise where s ̃(x) = sign(x) and the noise flips the result with 20% probability. 
	data = np.zeros((N,2))
	for i in range(N):
		data[i,0] = np.random.uniform(-1,1)
		if (np.random.uniform(0,1)<error_rate):
			point_y = -1*np.sign(data[i,0])
		else:
			point_y = np.sign(data[i,0])
		data[i,1] = point_y

	return data

def hypothesis(s, theta, x):
	#h_function
	# print(s, theta, x)
	# print(s*np.sign(float(x)-theta))
	# exit()
	return s*np.sign(float(x)-theta)

def thetaset_generation(data):
	theta_set = []
	sorted_data = np.sort(data[:,0])
	theta_set.append(sorted_data[0]-1)
	for i in range(len(sorted_data)-1):
		theta_set.append(np.round((sorted_data[i]+sorted_data[i+1])/2,3))
	theta_set.append(sorted_data[-1]+1)
	return theta_set

def nd_thetaset_generation(data, dim):
	theta_set = []
	sorted_data = np.sort(data[:,dim])
	theta_set.append(float(sorted_data[0])-1)
	for i in range(len(sorted_data)-1):
		theta_set.append(np.round((float(sorted_data[i])+float(sorted_data[i+1]))/2,3))
	theta_set.append(float(sorted_data[-1])+1)
	return theta_set


def error_estimation(data, s, theta):
	err_list = []
	for i in range(len(data)):
		if hypothesis(s, theta, data[i,0]) != data[i,1]:
			err_list.append(i)
	#print(err_list)
	err_rate = len(err_list)/len(data)
	#print(err_rate)
	return err_rate

def nd_error_estimation(xtrain, ytrain, s, theta, dim):
	err_list = []
	for i in range(len(xtrain)):
		if int(hypothesis(s, theta, xtrain[i,dim])) != int(ytrain[i]):
			# print(hypothesis(s, theta, xtrain[i,dim]), ytrain[i])
			err_list.append(i)
	return len(err_list)/len(xtrain)


def E_out_estimate(s, theta):
	#calculate E_out with the formula
	return 0.5 + 0.3 * s*(np.abs(theta)-1)

def multi_dimensional_DS(train_path, test_path):
	xtrain, ytrain =load(train_path)
	xtest, ytest = load(test_path)
	dim = len(xtrain[0])
	s_set = [1, -1]
	E_in_matrix = np.zeros((dim, len(s_set), len(xtrain)+1))
	for i in range(dim):
		theta_set = nd_thetaset_generation(xtrain, i)
		for j in range(len(s_set)):
			for k in range(len(theta_set)):
				E_in_matrix[i,j,k] = nd_error_estimation(xtrain, ytrain, s_set[j], theta_set[k], i)

	bs_idx = (E_in_matrix.argmin() % (len(s_set) * len(theta_set))) // len(theta_set)
	bt_idx = (E_in_matrix.argmin() % (len(s_set) * len(theta_set))) % len(theta_set)
	bd_idx = E_in_matrix.argmin() // (len(s_set) * len(theta_set))
	best_s = s_set[bs_idx]
	best_theta = theta_set[bt_idx]
	best_dim = bd_idx

	E_in = np.min(E_in_matrix)
	E_out = nd_error_estimation(xtest, ytest, best_s, best_theta, best_dim)

	return E_in, E_out




def decision_stump(noise_rate, num_data):
	data = data_generation(num_data, noise_rate)
	theta_set = thetaset_generation(data)
	s_set = [1,-1]
	E_in_matrix = np.zeros((len(s_set), len(theta_set)))
	#E[0,:] : s=1 
	#E[1,:] : s=-1
	for i in range(len(s_set)):
		for j in range(len(theta_set)):
			E_in_matrix[i,j] = error_estimation(data, s_set[i], theta_set[j])
	#print(E_in_matrix)
	bs_index = E_in_matrix.argmin()//len(theta_set)
	#print('bs_index: ', bs_index)
	best_s = s_set[bs_index]
	bt_index = E_in_matrix.argmin()%len(theta_set)
	#print('bt_index: ',bt_index)
	best_theta = theta_set[bt_index]
	E_in = np.min(E_in_matrix)
	#print(best_s, best_theta,E_in)
	#print(bs_index,bt_index)
	#print(E_in_matrix.argmin())


	E_out = E_out_estimate(best_s, best_theta)
	#print(E_out)
	#print(return_thetaset(data))
	#print(pd.DataFrame(sorted_data))

	return E_in, E_out
	
def plt_histogram(diff, N):
	print(max(diff), min(diff))
	x = [np.round(i/10-0.5,2) for i in range(8) ]
	print(x)
	y = np.zeros((8))
	for i in range(len(diff)):
		for j in range(len(x)-1):
			if(diff[i]>x[j] and diff[i]<x[j+1]):
				y[j]+=1
	print(y)
	plt.bar(x,y, width = 0.07)
	plt.xlabel('E_in - E_out')
	plt.ylabel('Numbers')
	plt.savefig('hw2_N={}'.format(N))




def hw2(args):
	# args = parse_args()
	E_in =[]
	E_out = []
	for i in tqdm(range(args.iter)):
		ein, eout = decision_stump(args.noise_rate, args.num_data)
		E_in.append(ein)
		E_out.append(eout)

	E_in_avg = sum(E_in)/len(E_in)
	E_out_avg = sum(E_out)/len(E_out)
	print(E_in_avg, E_out_avg)
	diff = [E_in[i]-E_out[i] for i in range(len(E_in))]
	x = [i for i in range(len(E_in))]
	plt_histogram(diff, args.num_data)
	print(np.average(np.array(diff)))
	np.array(E_in)
	np.array(E_out)
	np.save('E_in_N={}.npy'.format(args.num_data),E_in)
	np.save('E_out_N={}.npy'.format(args.num_data), E_out)


def test(args):
	# args = parse_args()
	E_in, E_out = multi_dimensional_DS(args.train, args.test)
	print(E_in, E_out)

def main():
	args = parse_args()
	if args.dim == 1:
		hw2(args)
	else:
		test(args)


if __name__ =='__main__':
	main()
	# test()









