import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import csv


blob = datasets.make_blobs(300,2,center_box=(-20.0,20.0),cluster_std = 1,centers = [[8,0],[12,6],[3,6]])
BLOB = [[blob[0][i],blob[1][i]] for i in range(len(blob[0]))]

blob1 = datasets.make_blobs(400,2,center_box=(-20.0,20.0),cluster_std = 2,centers = [[-5,-5],[-7,3],[16,-5],[5,-10]])
BLOB1 = [[blob1[0][i],blob1[1][i]+3] for i in range(len(blob1[0]))]

blob2 = datasets.make_moons(100)
BLOB2 = [[blob2[0][i],blob2[1][i]+7] for i in range(len(blob2[0]))]

DATA = BLOB + BLOB1 + BLOB2
DATA=np.array(DATA)
plt.figure(1)
X = list(map(lambda x : x[0][0],DATA))
Y = list(map(lambda x : x[0][1],DATA))
L = list(map(lambda x : x[1],DATA))
plt.scatter(X,Y,c=L)
plt.show()

with open('2D-Dataset-10.csv', 'w', newline='') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
	for i in range(len(DATA)):
		spamwriter.writerow([DATA[i][0][0],DATA[i][0][1],DATA[i][1]])