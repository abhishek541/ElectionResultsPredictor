import pandas as pd 
import numpy as np
import math
import operator

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
    return (predictions == labels).sum() / float(labels.size)

#===========================================================================================================

#Use this as preprocess for KNN, Perceptron and MLP
def preprocess(data):
    normdata = normalizeData(data, ['net_ope_exp', 'net_con', 'tot_loa'])
    encdata = encodeData(normdata, ['can_off', 'can_inc_cha_ope_sea'])
    return getNumpy(encdata)

#Use this as preprocess for ID3
def preprocessID3(data):
    normdata = normalizeData(data, ['net_ope_exp', 'net_con', 'tot_loa'])
    return getNumpy(normdata)

		
class KNN:
    trdfeatures = [] 
    trdlabels = []
    k = 0

    def __init__(self):
        KNN.k = 15

    def train(self, features, labels):
        KNN.trdfeatures = features
        KNN.trdlabels = labels

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
        for i in range(len(KNN.trdfeatures)):
            ed = self.getEuclideanDistance(testrow, KNN.trdfeatures[i])
            distances.append((i, ed))
        distances.sort(key = lambda x: x[1])
        nearestnbrs = []
        for j in range(KNN.k):
            nearestnbrs.append(distances[j][0])
        return nearestnbrs

    def getPredictionLabel(self, neighbours):
        result = {}
        result[0] = 0
        result[1] = 0
        for i in range(len(neighbours)):
            label = KNN.trdlabels[neighbours[i]]
            if label == 1:
                result[1] += 1
            else:
                result[0] += 1
        predlabel = max(result, key=lambda k: result[k])
        return predlabel
			
	
class Perceptron:
    learn_rate = 0
    n_epoch = 0
    bias = 0.0
    weights = []

    def __init__(self):
        Perceptron.learn_rate = 0.01
        Perceptron.n_epoch = 5

    def train(self, features, labels):
        w_arr = [0.0 for i in range(len(features[0]))]
        b = 0.0
        for e in range(Perceptron.n_epoch):
            for j in range(len(features)):
                pred = self.getPredictLabel(features[j], b, w_arr)
                err = labels[j] - pred
                b = b + Perceptron.learn_rate * err
                for n in range(len(w_arr)):
                    w_arr[n] = w_arr[n] + Perceptron.learn_rate * err * features[j][n]
        Perceptron.bias = b
        Perceptron.weights = w_arr

    def predict(self, features):
        predictions = []
        for i in range(len(features)):
            predlabel = self.getPredictLabel(features[i], Perceptron.bias, Perceptron.weights)
            predictions.append(predlabel)
        return predictions

    def getPredictLabel(self, inputs, b, wts):
        val = b
        for i in range(len(inputs)):
            val += wts[i] * inputs[i]
        return 1.0 if val >= 0.0 else 0.0

class MLP:
    np.random.seed(1)
    w0 = 2*np.random.random((9, 10)) - 1
    w1 = 2*np.random.random((10, 1)) - 1
    
    def __init__(self):
        pass

    def train(self, features, labels):
        for i in range(features.shape[0]):
            l0 = features
            l1 = self.sygmoid(np.dot(l0, self.w0))
            l2 = self.sygmoid(np.dot(l1, self.w1))
            
            l2_error = labels.reshape(len(labels), 1) - l2     
            l2_delta = l2_error * self.sygmoid(l2, deriv=True)
            l1_error = l2_delta.dot(self.w1.T)
            l1_delta = l1_error * self.sygmoid(l1, deriv=True)
        
            self.w0 += l0.T.dot(l1_delta)
            self.w1 += l1.T.dot(l2_delta)

    def predict(self, features):
        hout = self.sygmoid(np.dot(features, self.w0))
        predictions = self.sygmoid(np.dot(hout, self.w1))
        return np.round(predictions, decimals=0)[0]
    
    def sygmoid(self, x , deriv = False):
        if (deriv==True):
            return x*(1-x)
        else: 
            return 1/(1+np.exp(-x))
        
class ID3:
    dtree = {}
    def __init__(self):
        pass
    
    def train(self, features, labels):
        colnames = ['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea']
        featuresnew = []
        for feat in features:
            featnew = []
            for i in range(len(feat)):
                if i < 3:
                    val = self.getBucket(feat[i])
                else:
                    val = feat[i]
                featnew.append(val)
            featuresnew.append(featnew)            
        ID3.dtree = self.tree(np.array(featuresnew), labels, colnames)

    def predict(self, features):
        featuresnew = []
        for feat in features:
            featnew = []
            for i in range(len(feat)):
                if i < 3:
                    val = self.getBucket(feat[i])
                else:
                    val = feat[i]
                featnew.append(val)
            featuresnew.append(featnew)
        predictions = []
        for nfeat in featuresnew:
            pred = self.getLeafValue(nfeat, ID3.dtree)
            predictions.append(pred)
        return predictions
        
    def getLeafValue(self, data, dtree):
        colnames = ['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea']
        for k, v in dtree.items():
            val = data[colnames.index(k)]
            if isinstance(v, dict):
                if str(val) not in list(v.keys()):
                    return 1
                v = v[str(val)]
                if isinstance(v, dict):
                    return self.getLeafValue(data, v)
                else:
                    return v
            else:
                return v
        
    def entropy(self, features, labels):
        numf = len(features)
        allclasses = {}
        for i in range(len(features)):
            label = labels[i]
            if label not in allclasses.keys():
                allclasses[label] = 0
            allclasses[label] += 1
        entropy = 0.0
        for key in allclasses:
            prob = float(allclasses[key])/numf
            entropy -= prob * math.log(prob,2)
        return entropy
    
    def getBucket(self, val):
        buckets = [0.2, 0.4, 0.6, 0.8, 1.0]
        if val <= buckets[0]:
            return buckets[0]
        elif val > buckets[0] and val <= buckets[1]:
            return buckets[1]
        elif val > buckets[1] and val <= buckets[2]:
            return buckets[2]
        elif val > buckets[2] and val <= buckets[3]:
            return buckets[3]
        elif val > buckets[3] and val <= buckets[4]:
            return buckets[4]
            
    def split(self, features, axis, val):
        if isinstance(features, np.ndarray):
            features = features.tolist()
        splitData = []
        for feat in features:
            if feat[axis] == val:
                splitFeat = feat[:axis]
                splitFeat.extend(feat[axis+1:])
                splitData.append(splitFeat)       
        return splitData
    
    def getBestFeat(self, features, labels):
        nfeat = len(features[0])
        baseEntropy = self.entropy(features, labels)
        bestInfoGain = 0.0;
        bestFeat = 0
        for i in range(nfeat):
            featList = [ex[i] for ex in features]
            uniqueVals = set(featList)
            newEntropy = 0.0
            for value in uniqueVals:
                newData = self.split(features, i, value)
                probability = len(newData)/float(len(features))
                newEntropy += probability * self.entropy(newData, labels)
            infoGain = baseEntropy - newEntropy
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeat = i
        return bestFeat
    
    def majority(self, classList):
        classCount={}
        for vote in classList:
            if vote not in classCount.keys(): classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]
    
    def tree(self, features, labels, colnames):
        if labels.tolist().count(labels[0]) == len(labels):
            return labels[0]
        if len(features[0]) == 1:
            return self.majority(labels)
        bestFeat = self.getBestFeat(features, labels)
        bestFeatLabel = colnames[bestFeat]
        theTree = {bestFeatLabel:{}}
        del(colnames[bestFeat])
        featValues = [ex[bestFeat] for ex in features]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = colnames[:]
            theTree[bestFeatLabel][value] = self.tree(self.split(features, bestFeat, value), labels, subLabels)
        return theTree

def main():
    kNN = KNN()
    traindata, testdata = trainingTestData(dataset, 0.7)
    trfeat, trlab = preprocess(traindata)
    tsfeat, tslab = preprocess(testdata)
    kNN.train(trfeat, trlab)
    predictions = kNN.predict(tsfeat)
    accuracy = evaluate(predictions, tslab)
    print ("Accuracy for KNN: " + str(accuracy))

    perc = Perceptron()
    traindata, testdata = trainingTestData(dataset, 0.84)
    trfeat, trlab = preprocess(traindata)
    tsfeat, tslab = preprocess(testdata)
    perc.train(trfeat, trlab)
    predictions = perc.predict(tsfeat)
    accuracy = evaluate(predictions, tslab)
    print ("Accuracy for Perceptron: " + str(accuracy))
    
    id3 = ID3()
    traindata, testdata = trainingTestData(dataset, 0.6)
    trfeat, trlab = preprocessID3(traindata)
    tsfeat, tslab = preprocessID3(testdata)
    id3.train(trfeat, trlab)
    predictions = id3.predict(tsfeat)
    accuracy = evaluate(predictions, tslab)
    print ("Accuracy for ID3: " + str(accuracy))
    
    mlp = MLP()
    traindata, testdata = trainingTestData(dataset, 0.84)
    trfeat, trlab = preprocess(traindata)
    tsfeat, tslab = preprocess(testdata)
    mlp.train(trfeat, trlab)
    predictions = mlp.predict(tsfeat)
    accuracy = evaluate(predictions, tslab)
    print ("Accuracy for MLP: " + str(accuracy))

if __name__ == "__main__": main()