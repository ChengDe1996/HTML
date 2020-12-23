import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import sys
import random
import matplotlib.pyplot as plt

def Plot(history,l):
	#history is the array which
	#have all 1126 time's result
	m = np.min(history)
	M = np.max(history)
	print(M,m)
	counts = np.zeros(int(M-m+1))
	for i in range(len(history)):
		counts[int(history[i]-m)] += 1
	x = []
	for i in range(int(M-m+1)):
		x.append(i+m)
	x = np.asarray(x)
	x2 = [round(i/l,3) for i in x]
	plt.bar(x,counts, width = 0.5,tick_label = x2)
	plt.xlabel('error rate')
	plt.ylabel('frequency')
	plt.title('hw1_8')
	plt.show()
a = np.genfromtxt('hw7.txt')
b = np.genfromtxt('hw8.txt')

Plot(a,500)
Plot(b,500)
