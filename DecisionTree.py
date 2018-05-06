from sklearn import *
import numpy as np
import matplotlib.pyplot as plt
import time
from functools import reduce
from operator import add
# TODO
# Membership at each leaf corresponding for each class
# Linear and Conic
# Draw Method

dataset_two_class_1 = []

data = datasets.make_blobs(400,2,4,center_box=(-40.0,40.0))
DATA = [[data[0][i],data[1][i]] for i in range(len(data[0]))]

circle = datasets.make_circles(100)
CIRCLE = [[circle[0][i],circle[1][i]] for i in range(len(circle[0]))]

moons = datasets.make_moons(100)
MOONS = [[moons[0][i],moons[1][i]] for i in range(len(moons[0]))]

no_of_trees = 2
no_of_geometric_primitive = 10
max_depth = 4
color = 0
box_data = []
max_labels = 4

def read_data(file):

	global dataset_two_class_1

	f = open(file)

	for l in f.readlines():
		x = l.strip()
		x = x.split(',')
		dataset_two_class_1.append([[float(x[1]),float(x[2])],x[0]])

	dataset_two_class_1 = np.array(dataset_two_class_1)


def shannon_entropy(labels):
	unique_labels = np.unique(labels,return_counts = True)
	fraction_labels = np.divide(unique_labels[1],np.sum(unique_labels[1]))
	entropy = -1*np.sum([x*np.log(x) for x in fraction_labels])
	return entropy

def information_gain(label_left,label_right,label_all):
	info_gain = shannon_entropy(label_all) - (np.size(label_left)*shannon_entropy(label_left) + np.size(label_right)*shannon_entropy(label_right))/np.size(label_all)
	return info_gain

class Node():

	def __init__(self):
		self.data = np.array([])
		self.ltree = None
		self.rtree = None
		self.is_leaf = False
		self.depth = 0
		self.threshold = 0
		self.phi = np.array([])
		self.psi = np.array([])
		self.membership = np.array([]) # TODO

	def generate_psi(self,learner_type,phi_length):
		if learner_type == 'axis-aligned':
			psi = np.zeros(phi_length)
			psi[np.random.randint(0,phi_length)] = 1
			return psi

		elif learner_type == 'linear':
			psi = np.random.randn(phi_length)
			norm = np.linalg.norm(psi)
			if norm != 0:
				return np.divide(psi,norm)
			else:
				return None

		elif learner_type == 'conic':
			psi = np.random.randn(phi_length,phi_length)
			norm = np.sqrt(np.sum([np.linalg.norm(row)**2 for row in psi]))
			if norm != 0:
				return np.divide(psi,norm)
			else:
				return None

	def choose_phi(self,dimension):
		i = np.ceil(np.random.rand()+2)
		i = min(i,dimension)
		#parameters = np.zeros(dimension)
		indices = np.unique(np.random.randint(0,dimension,i))
		#print(indices)
		#parameters = [1 if (x in indices) else 0 for x in range(dimension)]
		return indices

	def process_node(self,dimension,learner_type):
		
		#print(self.data)
		global color,leaf_data,box_data,no_of_geometric_primitive

		best_gain = 0 # TODO
		best_phi = []
		best_psi = []
		best_threshold = 0
		best_evaluation = []

		evaluation = []

		if self.depth < max_depth:	

			if len(self.data) == 0 :
				return

			print("Incoming Data Size : %s"%len(self.data))
			
			if learner_type == 'conic':
				no_of_geometric_primitive = 100

			for i in range(no_of_geometric_primitive):
				temp_phi = self.choose_phi(dimension)
				temp_psi = self.generate_psi(learner_type,len(temp_phi))

				#if i % 10 == 0:
				print (i)

				filtered_data = np.array(list(map(lambda x : [x[0][i] for i in temp_phi],self.data)))

				labels = self.data[:,1]
				
				if shannon_entropy(labels) < 0.00001 :
					self.is_leaf = True
					self.membership=[np.count_nonzero(self.data[:,1]==i)/len(self.data) for i in range(0,max_labels)]
					print(self.membership)
					return

				filtered_data_with_label = [[filtered_data[i],labels[i]] for i in range(len(self.data))]

				if learner_type == 'conic':
					evaluation = np.array([np.matmul(np.matmul(x,temp_psi),np.transpose(x)) for x in filtered_data])
				else :
					evaluation = np.sum(np.multiply(filtered_data,temp_psi),1)
				evaluation = np.insert(evaluation,len(evaluation),-1000) #TODO
				evaluation = np.insert(evaluation,len(evaluation),1000)

				thresholds = np.sort(np.unique(evaluation))

				for ind in range(len(thresholds)-1):
					th = (thresholds[ind] + thresholds[ind+1])/2.0
					
					left_set = [filtered_data_with_label[i][1] for i in range(len(evaluation)-2) if evaluation[i] < th]
					right_set = [filtered_data_with_label[i][1] for i in range(len(evaluation)-2) if evaluation[i] >= th]

					temp_information_gain = information_gain(left_set,right_set,labels)

					if temp_information_gain > best_gain :

						best_gain = temp_information_gain
						best_phi = temp_phi
						best_psi = temp_psi
						best_threshold = th	
						best_evaluation = evaluation	

			filtered_data = np.array(list(map(lambda x : [x[0][i] for i in best_phi],self.data)))
			labels = self.data[:,1]

			filtered_data = [[filtered_data[i],labels[i]] for i in range(len(self.data))]

			left_set = [self.data[i] for i in range(len(best_evaluation)-2) if best_evaluation[i] < best_threshold]
			right_set = [self.data[i] for i in range(len(best_evaluation)-2) if best_evaluation[i] >= best_threshold]

			if len(left_set) != 0 and len(right_set) != 0:				
	
				print("Best Gain : %s"%best_gain)
				print("Best Threshold : %s"%best_threshold)
				print("Best Phi : %s"%best_phi)
				print("Best Psi : %s"%best_psi)
				print ("Left Child : %s"%len(left_set))
				print ("Right Child : %s"%len(right_set))

				self.is_leaf = False

				self.psi = best_psi
				self.phi = best_phi
				self.threshold = best_threshold

				self.ltree = Node()
				self.rtree = Node()
				self.ltree.depth = self.depth+1
				self.rtree.depth = self.depth+1
				self.ltree.data = np.array(left_set)
				self.rtree.data = np.array(right_set)

				if learner_type == 'axis-aligned':
					
					if len(self.phi) == 1 :
						if self.phi[0] == 1:
							box_data.append([0,0,0,0,self.psi[0],-self.threshold])
						else:
							box_data.append([0,0,0,self.psi[0],0,-self.threshold])
					elif self.psi[0] == 1.0 :
						box_data.append([0,0,0,0,self.psi[0],-self.threshold])
					else :
						box_data.append([0,0,0,self.psi[0],0,-self.threshold])

				elif learner_type == 'linear':
					if len(best_psi) == 1 :
						if self.phi[0] == 1:
							box_data.append([0,0,0,0,self.psi[0],-self.threshold])
						else:
							box_data.append([0,0,0,self.psi[0],0,-self.threshold])
					elif abs(best_psi[1]) < 0.000001 :
						box_data.append([0,0,0,0,self.psi[0],-self.threshold])
					else :
						box_data.append([0,0,0,self.psi[0],self.psi[1],-self.threshold])

				else :
					if len(self.psi) == 1:
						if self.phi[0] == 1:
							print ("%f*y^2 = %f"%(self.psi,self.threshold))
							box_data.append([0,0,self.psi,-self.threshold])
						else:
							print ("%f*x^2 = %f"%(self.psi,self.threshold))
							box_data.append([self.psi,0,0,-self.threshold])
					else:
						print ("%f*x^2 + %f*xy + %f*y^2 = %f"%(self.psi[0][0],self.psi[0][1]+self.psi[1][0],self.psi[1][1],self.threshold))
						box_data.append([self.psi[0][0],self.psi[0][1]+self.psi[1][0],self.psi[1][1],-self.threshold])

				self.ltree.process_node(dimension,learner_type)
				self.rtree.process_node(dimension,learner_type)

			else :
				self.is_leaf = True
				self.membership=[np.count_nonzero(self.data[:,1]==i)/len(self.data) for i in range(0,max_labels)]
				print(self.membership)


		else :
			self.is_leaf = True
			self.membership=[np.count_nonzero(self.data[:,1]==i)/len(self.data) for i in range(0,max_labels)]
			print(self.membership)


	def evaluate_node(self,test_data_instance,learner_type): 

		#TODO For learner type

		if self.is_leaf :
			return self.membership
		else : 
			#relevant_params = [example[x] for x in range(len(example)) if x in best_phi]
			#print(test_data_instance)
			if learner_type == 'conic':
				evaluation = np.matmul(np.array([test_data_instance[i] for i in self.phi]),np.matmul(self.psi,np.transpose(np.array([test_data_instance[i] for i in self.phi]))))
			else:
				evaluation = np.sum(np.multiply(np.array([test_data_instance[i] for i in self.phi]),self.psi))

			if evaluation < self.threshold :
				return self.ltree.evaluate_node(test_data_instance,learner_type)
			else :
				return self.rtree.evaluate_node(test_data_instance,learner_type)


class DecisionTree():
	
	def __init__(self):
		self.root = Node()
		self.data = np.array([])

	def train_tree(self,train_dataset,dimension,learner_type):
		self.root.data = np.array(train_dataset)
		self.root.process_node(dimension,learner_type)

	def test_tree(self,test_dataset,learner_type):
		return [self.root.evaluate_node(x,learner_type) for x in test_dataset ]# TODO For entire test dataset

class RandomForest():
	
	def __init__(self):
		self.no_of_trees = no_of_trees
		self.trees = [DecisionTree() for _ in range(no_of_trees)]

	def train_forest(self,train_dataset,dimension,learner_type):

		for i in range(self.no_of_trees):
			self.trees[i].train_tree(train_dataset,dimension,learner_type)
			print("\n\n\n\n\nTree %d trained\n\n\n\n\n"%i)

	def test_forest(self,test_dataset,learner_type):

		membership = [0 for _ in range(no_of_trees)] 

		# TODO Membership
		memberships = [self.trees[i].test_tree(test_dataset,learner_type) for i in range(0,no_of_trees)]
		combined_memberships=reduce(lambda x,y: list(map(lambda u,v: list(map(add,u,v)) ,x,y)),memberships)
		print(combined_memberships)
		predictions = [[combined_memberships[i].index(max(combined_memberships[i])),max(combined_memberships[i])/self.no_of_trees] for i in range(0,len(test_dataset))]
		predictions = np.array(predictions)
		return predictions


def main():
	#root = Node()
	#root.data = np.array(DATA)
	#root.process_node(2,'linear')
	#plt.figure(2)
	

	xcord = np.linspace(-40, 40, 100)
	ycord = np.linspace(-40, 40, 100)
	x, y = np.meshgrid(xcord, ycord)

	test_dataset = []
	for i in xcord:
		for j in ycord:
			test_dataset.append([i,j])

	#print(test_dataset)
	test_dataset = np.array(test_dataset)
	#for [a,h,b,f,g,c] in box_data:
	#	plt.contour(x, y,(a*x**2 + h*x*y + b*y**2 + f*x + g*y + c), [0], colors='k')

	forest = RandomForest();
	forest.train_forest(DATA,2,'axis-aligned')

	

	
	#prediction[0] gives labels and predictions[1] gives its probability
	predictions = forest.test_forest(test_dataset,'axis-aligned')

	print(predictions)
	plt.scatter(test_dataset[:,0],test_dataset[:,1],c=predictions[:,0])
	plt.show()

	#plt.figure(1)
	#X = list(map(lambda x : x[0][0],DATA))
	#Y = list(map(lambda x : x[0][1],DATA))
	#L = list(map(lambda x : x[1],DATA))
	#plt.scatter(X,Y,c=L)
	#plt.show()

read_data('data1.csv')
main()
'''
if __name__ == '__main__':
	main()
'''
