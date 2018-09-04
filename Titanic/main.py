import DecisionTree
import ImportData
import randomForest
import random

# Setting Cross validation and bootstrap parameters
folds = 5 # folds number
treeNum = 100 # number of decision tree
resamplingSize = 100
cores = 2
feature_col = [1,2,3]
label_col = 4 #The

# Setting cross validation index
dataset = ImportData.loadCSV('fishiris.csv')
instanceNum = len(dataset)
index = [int(i/(instanceNum/folds)) for i in random.sample(range(instanceNum), instanceNum)]

trueValue = {}
predictValue = {}


for i in range(folds):

    # Initialize training and testing data
    training_index = [num for num in range(len(index)) if index[num] != i]
    testing_index = [num for num in range(len(index)) if index[num] == i]
    training = [dataset[num] for num in training_index]
    testing = [dataset[num] for num in testing_index]

    if treeNum == 1: # Train with Standard Decision Tree
        decisionTree = DecisionTree.decisionTree(training, impurity = "gini")
        func_predict = lambda x: DecisionTree.predict(x, decisionTree)
        trueValue[i] = [x[label_col] for x in testing]
        predictValue[i] = [[y for y in func_predict(x)][0] for x in testing]

        # print out error rate
        errorRate = sum([trueValue[i][x] == predictValue[i][x] for x in range(len(testing))]) / float(len(testing))
        print(errorRate)

    else: # train with random forest
        trees = randomForest.train(training, treeNum, resamplingSize, cores)
        trueValue[i] = [x[label_col] for x in testing]
        predictValue[i] = randomForest.predict(testing, trees, cores)
        errorRate = sum([trueValue[i][x] == predictValue[i][x] for x in range(len(testing))]) / float(len(testing))
        print(errorRate)




