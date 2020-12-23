import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import sys
import random
import matplotlib.pyplot as plt

def parse_args():
    """ Parse args. """
    parser = argparse.ArgumentParser()
    # initial
    parser.add_argument('-s', '--seed', type = int, default = 6174,
                        help='random seed')
    parser.add_argument('-i','--input', help = 'input file')
    parser.add_argument('-t','--test',help = 'test file')
    # number of repeat times
    parser.add_argument('-r', '--repeat', type = int, default = 1126, help = 'repeat')

    args = parser.parse_args()
    return args

def load_data(path):
	#input_file = pd.read_csv(path)
	input_file = np.genfromtxt(path)
	print(input_file.shape)
	return np.array(input_file)
def CheckError(data,w):
	error = 0
	for i in range(len(data)):
		x = [1.0]
		x.extend(data[i][0:4])
		target_y = data[i][4]
		if(np.sign(np.dot(w,x))!= np.sign(target_y)):
			error +=1
	return (error)
def PLA(data,test):
	#weight vec
	history_e = []
	history_w = []
	glo_err_best = 9999
	glo_w_best = np.zeros(5)
	for i in tqdm(range(1126)):
		#print(data)
		np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)
		#print(data)
		w = np.zeros(5)
		update = 0
		error_best = 99999
		w_best = np.zeros(5)
		while (True) :
			for j in range(len(data)):
				x = [1.0]
				x.extend(data[j][0:4])
				#print(x)
				target_y = data[j][4]
				if(np.sign(np.dot(w,x)) == np.sign(target_y) ):
					continue
				else:
					#print(np.sign(np.dot(w,x)),np.sign(target_y))
					w = w + target_y *np.array(x)
					error = CheckError(data,w)
					update += 1
					if(error<error_best):
						w_best = w
						error_best = error
					if(update == 100):
						break
			if(update == 100):
				break
		test_err = CheckError(test,w_best)
		history_e.append(test_err)
		#history_e.append(error_best)
		#history_w.append(w_best)
		'''
		if(glo_err_best>error_best):
			glo_err_best = error_best
			glo_w_best = w_best
		'''
	#return history_e, history_w,glo_err_best,glo_w_best
	return history_e
def Plot(history,l):
	m = np.min(history)
	M = np.max(history)
	counts = np.zeros(M-m+1)
	for i in range(len(history)):
		counts[history[i]-m] += 1
	x = []
	for i in range(M-m+1):
		x.append(i+m)
	x = np.asarray(x)
	x2 = [round(i/l,3) for i in x]
	plt.bar(x,counts, width = 0.5,tick_label = x2)
	plt.xlabel('number of updates')
	plt.ylabel('frequency')
	plt.title('hw1_6')
	plt.show()

	

	

def run():
	args = parse_args()
	data = load_data(args.input)
	test = load_data(args.test)
	random.seed(args.seed)
	history_e = PLA(data,test)
	print(np.mean(history_e))
	Plot(history_e,len(test))
	np.savetxt('hw7.txt',history_e)

if __name__ == '__main__':
	run()
