import numpy as np
import pandas as pd
# from tqdm import tqdm
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
    # number of repeat times
    parser.add_argument('-r', '--repeat', type = int, default = 1126, help = 'repeat')

    args = parser.parse_args()
    return args

def load_data(path):
	#input_file = pd.read_csv(path)
	input_file = np.genfromtxt(path)
	print(input_file.shape)
	return np.array(input_file)

def PLA(data):
	#weight vec
	history = []
	for i in tqdm(range(1126)):
		#print(data)
		np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)
		#print(data)
		w = np.zeros(5)
		update = 0
		Stop = False
		while (not Stop) :
			for j in range(len(data)):
				Stop = True
				x = [1.0]
				x.extend(data[j][0:4])
				#print(x)
				target_y = data[j][4]
				if(np.sign(np.dot(w,x)) == np.sign(target_y) ):
					continue
				else:
					#print(np.sign(np.dot(w,x)),np.sign(target_y))
					w = w + target_y *np.array(x)
					Stop = False
					update += 1
		history.append(update)
	return history

def Plot(history):
	m = np.min(history)
	M = np.max(history)
	counts = np.zeros(M-m+1)
	for i in range(len(history)):
		counts[history[i]-m] += 1
	y = []
	for i in range(M-m+1):
		y.append(i+m)
	y = np.asarray(y)
	plt.bar(y,counts, width = 0.8)
	plt.xlabel('number of updates')
	plt.ylabel('frequency')
	plt.title('hw1_6')
	plt.show()



def run():
	args = parse_args()
	data = load_data(args.input)
	random.seed(args.seed)
	history = PLA(data)
	print(np.mean(history))
	Plot(history)

if __name__ == '__main__':
	run()
