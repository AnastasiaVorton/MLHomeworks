import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import pprint


# code is based on this resource - http://gabrielelanaro.github.io/blog/2016/03/03/decision-trees.html

# The task is to modify this algorithm, which is ID3, to be a C4.5 and to train in on Titanic dataset.

def partition(a):
    return {c: (a == c).nonzero()[0] for c in np.unique(a)}


def partition_continous(feature, threshold):
    """
    The function that partitions a feature with continuous data.
    :param feature: ndarray of values jf a feature
    :param threshold: the threshold by which to split
    :return: dict of 2 keys (< and >= to the threshold)
    """
    smaller, bigger = [], []
    for value in np.unique(feature):
        if value < threshold:
            smaller.append(np.where(feature == value)[0])
        else:
            bigger.append(np.where(feature == value)[0])
    seq_s = [array for array in smaller]
    seq_b = [array for array in bigger]
    if len(seq_b) == 1 and len(seq_s) > 1:
        return {f"<{threshold}": np.concatenate(seq_s), f">={threshold}": seq_b}
    elif len(seq_s) == 1 and len(seq_b) > 1:
        return {f"<{threshold}": seq_s, f">={threshold}": np.concatenate(seq_b)}
    elif len(seq_b) == 1 and len(seq_s) == 1:
        return {f"<{threshold}": seq_s, f">={threshold}": seq_b}
    elif len(seq_s) == 0 and len(seq_b) > 1:
        return {f"<{threshold}": [], f">={threshold}": np.concatenate(seq_b)}
    elif len(seq_b) == 0 and len(seq_s) > 1:
        return {f"<{threshold}": np.concatenate(seq_s), f">={threshold}": []}
    else:
        return {f"<{threshold}": np.concatenate(seq_s), f">={threshold}": np.concatenate(seq_b)}


def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float') / len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res


def information_gain(y, x):
    res = entropy(y)

    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float') / len(x)

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res


def intrinsic_value(feature, original_set):
    """
    A function to calculate the intrinsic value of a feature.
    :param feature: the feature column
    :param original_set: predictor set
    :return: float intrinsic value of a column
    """
    sum = 0
    val, counts = np.unique(feature, return_counts=True)
    freqs = counts.astype('float') / len(feature)

    for p, v in zip(freqs, val):
        sum += p * math.log(p, 2)

    sum *= -1
    return sum


def information_gain_ratio(feature, original_set):
    """
    A function that calculates information gain ration. In its essence is a ratio of information gain and intrinsic
    value of a feature
    :param feature: the feature column
    :param original_set: predictor set
    :return: float information gain ratio value
    """
    i_g = information_gain(original_set, feature)
    intr_val = intrinsic_value(feature, original_set)
    return i_g / intr_val


def is_pure(s):
    return len(set(s)) == 1


def is_categorical(feature):
    """
    Evaluates if the feacure is categorical
    :param feature: ndarray of feature values
    # :return: boolean
    # """
    values = np.unique(feature)
    if type(values[0]) == float and len(values) > 10:
        return False
    return True


def recursive_split(x, y, fields):
    # If there could be no split, just return the original set
    if is_pure(y) or len(y) == 0:
        return y

    # We get attribute that gives the highest information gain
    # gain = np.array([information_gain(y, x_attr) for x_attr in x.T])
    gain = np.array([information_gain_ratio(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(gain)

    # If there's no gain at all, nothing has to be done, just return the original set
    if np.all(gain < 1e-6):
        return y

    # We split using the selected attribute
    if is_categorical(x[:, selected_attr]):
        sets = partition(x[:, selected_attr])
    else:
        if selected_attr == 2:
            thr = np.amax(x[:, selected_attr])/2
            sets = partition(x[:, selected_attr])
            # sets = partition_continous(x[:, selected_attr], thr)

    res = {}

    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)

        res["{} = {}".format(fields[selected_attr], k)] = recursive_split(
            x_subset, y_subset, fields)

    return res


def find_majority(list):
    """
    I use this method to calculate the majority of the array values (used in prediction if we haven't seen the data yet-
    return the majority)
    :param list:
    :return:
    """
    map = {}
    maximum = ('', 0)  # (occurring element, occurrences)
    for n in list:
        if n in map:
            map[n] += 1
        else:
            map[n] = 1
        # Keep track of maximum on the go
        if map[n] > maximum[1]:
            maximum = (n, map[n])
    return maximum


def predict(tree, data_row, y):
    keys = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
    row = {}
    for i in range(len(data_row)):
        row[keys[i]] = data_row[i]
    # step 1: check if the key of the tree is in features
    for key in list(row.keys()):
        if key in list(tree.keys()):
            # step 2: see if the data we want to predict on was shown to thw model before.
            try:
                result = tree[key][row[key]]
            except:
                return find_majority(y)

            #  step 3: address the key
            result = tree[key][row[key]]
            # step 4: recursively do down the tree
            if isinstance(result, dict):
                return predict(row, result)
            else:
                return result
    return 0


def accuracy(predictions, y_test):
    """
    Evaluates the accuracy of the prediction
    """
    correct = 0
    for i in range(len(y_test)):
        if y_test[i] == predictions[i]:
            correct += 1
    return (correct / float(len(y_test))) * 100.0


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Changing this to be a Titanic dataset.
train_set = pd.read_csv(
    '/Users/anastasia/Documents/IU/Fall18/Machine learning/Labs/datasets/titanic.csv')
train_set = train_set.dropna(subset=['Age'])

# the last column of the data set (Survived) is teh target
X = train_set.iloc[:, :5].values
y = train_set.iloc[:, 6].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

fields = list(train_set.columns.values)
tree = recursive_split(X_train, y_train, fields)
# pprint.pprint(tree)

predicted = [predict(tree, row, y_test) for row in X_test]
print(accuracy(predicted, y_test))
