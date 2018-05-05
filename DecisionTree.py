from sklearn import *
import numpy as np
import matplotlib.pyplot as plt
import time

# TODO
# Membership at each leaf corresponding for each class
# Linear and Conic
# Draw Method


data = datasets.make_blobs(1200,2,12,center_box=(-40.0,40.0))
DATA = [[data[0][i],data[1][i]] for i in range(len(data[0]))]

circle = datasets.make_circles(100)
CIRCLE = [[circle[0][i],circle[1][i]] for i in range(len(circle[0]))]

moons = datasets.make_moons(100)
MOONS = [[moons[0][i],moons[1][i]] for i in range(len(moons[0]))]

no_of_trees = 200
no_of_geometric_primitive = 10
max_depth = 4
color = 0
box_data = []

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

				#if not isinstance(temp_psi,list):
				#	continue

				filtered_data = np.array(list(map(lambda x : [x[0][i] for i in temp_phi],self.data)))

				#print(temp_psi)

				labels = self.data[:,1]
				
				if shannon_entropy(labels) < 0.00001 :
					self.is_leaf = True
					return

				filtered_data_with_label = [[filtered_data[i],labels[i]] for i in range(len(self.data))]

				# print (filtered_data)

				#print("TESTS %s"%np.multiply(np.multiply(filtered_data[0],temp_psi),np.transpose(filtered_data[0])))

				if learner_type == 'conic':
					evaluation = np.array([np.matmul(np.matmul(x,temp_psi),np.transpose(x)) for x in filtered_data])
				else :
					evaluation = np.sum(np.multiply(filtered_data,temp_psi),1)
				evaluation = np.insert(evaluation,len(evaluation),-1000) #TODO
				evaluation = np.insert(evaluation,len(evaluation),1000)
				
				#print(evaluation)

				thresholds = np.sort(np.unique(evaluation))

				#print(len(filtered_data_with_label))

				#print(len(evaluation))

				for ind in range(len(thresholds)-1):
					th = (thresholds[ind] + thresholds[ind+1])/2.0
					
					left_set = [filtered_data_with_label[i][1] for i in range(len(evaluation)-2) if evaluation[i] < th]
					right_set = [filtered_data_with_label[i][1] for i in range(len(evaluation)-2) if evaluation[i] >= th]

					temp_information_gain = information_gain(left_set,right_set,labels)

					#print ("Info : %s"%temp_information_gain)

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
							box_data.append([range(-80,80,1),[best_threshold]*160])
						else:
							box_data.append([[best_threshold]*160,range(-80,80,1)])
					elif self.psi[0] == 1.0 :
						box_data.append([range(-80,80,1),[best_threshold]*160])
					else :
						box_data.append([range(-80,80,1),[best_threshold]*160])

				elif learner_type == 'linear':
					if len(best_psi) == 1 :
						if self.phi[0] == 1:
							box_data.append([range(-80,80,1),[best_threshold]*160])
						else:
							box_data.append([[best_threshold]*160,range(-80,80,1)])
					elif abs(best_psi[1]) < 0.000001 :
						box_data.append([range(-80,80,1),[best_threshold]*160])
					else :
						box_data.append([range(-80,80,1),[(best_threshold - best_psi[0]*x)/best_psi[1] for x in range(-80,80,1)]])

				else :
					if len(self.psi) == 1:
						if self.phi[0] == 1:
							print ("%f*y^2 = %f"%(self.psi,self.threshold))
						else:
							print ("%f*x^2 = %f"%(self.psi,self.threshold))
					else:
						print ("%f*x^2 + %f*xy + %f*y^2 = %f"%(self.psi[0][0],self.psi[0][1]+self.psi[1][0],self.psi[1][1],self.threshold))

				self.ltree.process_node(dimension,learner_type)
				self.rtree.process_node(dimension,learner_type)

			else :
				self.is_leaf = True

		else :
			self.is_leaf = True

	def evaluate_node(self,example): 

		#TODO For learner type

		if self.is_leaf :
			return self.membership
		else : 
			relevant_params = [example[x] for x in range(len(example)) if x in best_phi]
			evaluation = np.multiply(relevant_params,best_psi)

			if evaluation < best_threshold :
				return evaluation(self.ltree,example)
			else :
				return evaluation(self.rtree,example)

class DecisionTree():
	
	def __init__(self):
		self.root = Node()
		self.data = np.array([])

	def train_tree(self,train_dataset):
		self.data = np.array(dataset)
		self.root.process_node()

	def test_tree(self,test_dataset):
		return self.evaluate_node(example) # TODO For entire test dataset

def RandomForest():
	
	def __init__(self):
		self.no_of_trees = no_of_trees
		self.trees = [DecisionTree() for _ in range(no_of_trees)]

	def train_forest(self,train_dataset):

		for i in range(self.no_of_trees):
			self.trees[i].train_tree()

	def test_forest(self,test_dataset):

		membership = [0 for _ in range(no_of_trees)] 

		# TODO Membership

		for i in range(self.no_of_trees):

			prediction_from_tree = self.trees[i].test_tree


def main():
	global DATA,box_data
	root = Node()
	root.data = np.array(MOONS)
	root.process_node(2,'conic')
	plt.figure(2)
	for [x,y] in box_data:
		plt.plot(x,y)

	X = list(map(lambda x : x[0][0],MOONS))
	Y = list(map(lambda x : x[0][1],MOONS))
	L = list(map(lambda x : x[1],MOONS))
	plt.scatter(X,Y,c=L)
	plt.show()


if __name__ == '__main__':
	main()


