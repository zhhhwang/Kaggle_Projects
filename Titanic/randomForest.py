import multiprocessing as mp
import random
import DecisionTree

def sample(input):
    datasets = input[0]
    resamplingSize = input[1]
    return [random.choice(datasets) for i in range(resamplingSize)]

def predictInstance(input):

    instance = input[0]
    trees = input[1]
    func_predict = lambda x: DecisionTree.predict(instance, x)
    prediction = map(func_predict, trees)
    #print(prediction)
    summarizeVoting = DecisionTree.labelCounts(prediction)
    #print(summarizeVoting)
    #print(max(summarizeVoting, key = summarizeVoting.get))
    return max(summarizeVoting, key = summarizeVoting.get)


def train(datasets, treeNum = 10, resamplingSize = 1000, cores=2):

    # Initialize parallel
    p = mp.Pool(cores)
    parallelList = [(datasets, resamplingSize) for i in range(treeNum)]

    pseudo = p.map(sample, parallelList)
    trees = p.map(DecisionTree.decisionTree, pseudo)
    return trees


def predict(datasets, trees, cores = 2):

    # Initial parallel
    p = mp.Pool(cores)
    parallelList = [(instance, trees) for instance in datasets]

    result = p.map(predictInstance, parallelList)

    return result