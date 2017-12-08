import pandas as pd 
import numpy as np
import math

#Data with features and target values
#Tutorial for Pandas is here - https://pandas.pydata.org/pandas-docs/stable/tutorials.html
#Helper functions are provided so you shouldn't need to learn Pandas
dataset = pd.read_csv("data.csv")

#========================================== Data Helper Functions ==========================================

#Normalize values between 0 and 1
#dataset: Pandas dataframe
#categories: list of columns to normalize, e.g. ["column A", "column C"]
#Return: full dataset with normalized values
def normalizeData(dataset, categories):
    normData = dataset.copy()
    col = dataset[categories]
    col_norm = (col - col.min()) / (col.max() - col.min())
    normData[categories] = col_norm
    return normData

#Encode categorical values as mutliple columns (One Hot Encoding)
#dataset: Pandas dataframe
#categories: list of columns to encode, e.g. ["column A", "column C"]
#Return: full dataset with categorical columns replaced with 1 column per category
def encodeData(dataset, categories):
	return pd.get_dummies(dataset, columns=categories)

#Split data between training and testing data
#dataset: Pandas dataframe
#ratio: number [0, 1] that determines percentage of data used for training
#Return: (Training Data, Testing Data)
def trainingTestData(dataset, ratio):
	tr = int(len(dataset)*ratio)
	return dataset[:tr], dataset[tr:]

#Convenience function to extract Numpy data from dataset
#dataset: Pandas dataframe
#Return: features numpy array and corresponding labels as numpy array
def getNumpy(dataset):
	features = dataset.drop(["can_id", "can_nam","winner"], axis=1).values
	labels = dataset["winner"].astype(int).values
	return features, labels

#Convenience function to extract data from dataset (if you prefer not to use Numpy)
#dataset: Pandas dataframe
#Return: features list and corresponding labels as a list
def getPythonList(dataset):
	f, l = getNumpy(dataset)
	return f.tolist(), l.tolist()

#Calculates accuracy of your models output.
#solutions: model predictions as a list or numpy array
#real: model labels as a list or numpy array
#Return: number between 0 and 1 representing your model's accuracy
def evaluate(solutions, real):
	predictions = np.array(solutions)
	labels = np.array(real)
	return (predictions == labels).sum() / labels.size

#===========================================================================================================

def preprocess():
	normdata = normalizeData(dataset, ['net_ope_exp', 'net_con', 'tot_loa'])
	encdata = encodeData(normdata, ['can_off', 'can_inc_cha_ope_sea'])
	return getNumpy(encdata)
		
class KNN:
	trdfeatures = [] 
	trdlabels = []
	k = 0
	
	def __init__(self):
		k = 5

	def train(self, features, labels):
		trdfeatures = features
		trdlabels = labels

	def predict(self, features):
		predictions = []
		for i in range(len(features)):
			neighbours = self.getNearestNeighbours(features[i])
			label = self.getPredictionLabel(neighbours)
			predictions.append(label)
		return predictions

	def getEuclideanDistance(self, row1, row2):
		return math.sqrt(np.sum(np.square(np.subtract(row1, row2))))

	def getNearestNeighbours(self, testrow):
		distances = []
		for i in range(len(trdfeatures)):
			ed = self.getEuclideanDistance(testrow, trdfeatures[i])
			distances.append((i, ed))
		distances.sort(key = lambda x: x[1])
		nearestnbrs = []
		for j in range(k):
			nearestnbrs.append(distances[j][0])
		return nearestnbrs

	def getPredictionLabel(self, neighbours):
		result = {}
		result[0] = 0
		result[1] = 0
		for i in range(len(neighbours)):
			label = trdlabels[neighbours[i]]
			if label == 1:
				result[1] += 1
			else:
				result[0] += 1
		predlabel = max(result, key=lambda k: result[k])
		return predlabel
			
	
class Perceptron:
	def __init__(self):
		#Perceptron state here
		#Feel free to add methods

	def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels

	def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features

class MLP:
	def __init__(self):
		#Multilayer perceptron state here
		#Feel free to add methods

	def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels

	def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features

class ID3:
	def __init__(self):
		#Decision tree state here
		#Feel free to add methods

	def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels

	def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features

def main():
	kNN = KNN()
	traindata, testdata = trainingTestData(dataset, 0.67)
	trfeat, trlab = preprocess(traindata)
	tsfeat, tslab = preprocess(testdata)
	kNN.train(trfeat, trlab)
	predictions = kNN.predict(tsfeat)
	accuracy = evaluate(predictions, tslab)
	print "Accuracy: " + str(accuracy)
	
if __name__ == "__main__": main()