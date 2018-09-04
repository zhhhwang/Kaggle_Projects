import math

class Tree:

    def __init__(self, feature = -1, value = None, left = None, right = None, result = None):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.result = result

def separation(dataset, feature, value):
    """This function separates a data set into two
       based on a specific value of a feature.
    """

    if isinstance(value, int) or isinstance(value, float):
        sep = lambda x: x[feature] >= value
    else:
        sep = lambda x: x[feature] == value
    leftset = [instance for instance in dataset if sep(instance)]
    rightset = [instance for instance in dataset if not sep(instance)]
    return (leftset, rightset)


def entropy(dataset):

    result = 0.0
    labels = labelCounts(dataset)
    log2 = lambda x: math.log(x) / math.log(2)
    for i in labels:
        weight = float(labels[i]) / len(dataset)
        result -= weight * log2(weight)
    return result


def gini(dataset):
    total = len(dataset)
    counts = labelCounts(dataset)
    result = 0.0

    for k1 in counts:
        p1 = float(counts[k1]) / total
        for k2 in counts:
            if k1 == k2: continue
            p2 = float(counts[k2]) / total
            result += p1 * p2
    return result


def misclassification(dataset):
    result = {}
    labels = labelCounts(dataset)
    for i in labels:
        result[i] = float(labels[i]) / len(dataset)
    if len(result) == 0:
        return 0
    else:
        return (1 - max(result.values()))


def labelCounts(dataset):
    count = {}
    for instance in dataset:
        label = instance[-1]
        if label not in count: count[label] = 0
        count[label] += 1
    return count


def decisionTree(dataset, impurity = "entropy"):

    if len(dataset) == 0:
        return Tree()

    if impurity == "entropy":
        evalutionFunction = entropy
    elif impurity == "gini":
        evalutionFunction = gini
    else:
        evalutionFunction = misclassification

    initialScore = evalutionFunction(dataset)
    best_gain = 0.0
    attribute = None
    sets = None
    columnNum = len(dataset[0]) - 1

    for col in range(0, columnNum):
        columnValues = [instance[col] for instance in dataset]

        for value in columnValues:
            (leftset, rightset) = separation(dataset, col, value)

            weight = float(len(leftset)) / len(dataset)
            gain = initialScore - weight * evalutionFunction(leftset) - (1 - weight) * evalutionFunction(rightset)
            if gain > best_gain and len(leftset) > 0 and len(rightset) > 0:
                best_gain = gain
                attribute = (col, value)
                sets = (leftset, rightset)

    if best_gain > 0:
        left = decisionTree(sets[0])
        right = decisionTree(sets[1])
        return Tree(feature = attribute[0], value = attribute[1], left = left, right = right)
    else:
        return Tree(result = labelCounts(dataset).keys())


def predict(instance, tree):

    if tree.result != None:
        return tree.result
    else:
        v = instance[tree.feature]
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.left
            else:
                branch = tree.right
        else:
            if v == tree.value:
                branch = tree.left
            else:
                branch = tree.right
        return predict(instance, branch)
