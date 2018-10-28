import numpy as np
import scipy.stats as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2


def accuracy(predictions, y_test):
    """ Evaluates the accuracy of the prediction """
    correct = 0
    for t in zip(predictions, y_test):
        if t[0] == t[1]:
            correct += 1
    return (correct / float(len(y_test))) * 100.0


def my_boost(iboost, X, y, sample_weight, n_est, estimator, estimator_weight, estimator_error):
    """ Implements sample weights update based on estimators prediction,
     weight and error on every iteration of Adaboost """
    estimator.fit(X, y, sample_weight=sample_weight)
    y_predict = estimator.predict(X)
    # Instances incorrectly classified
    incorrect = y_predict != y
    # Stop if classification is perfect
    if estimator_error <= 0:
        return sample_weight, 1., 0.
    # Only boost the weights if I will fit again
    if not iboost == n_est - 1:
        for i in range(len(sample_weight)):
            if incorrect[i]:
                sample_weight[i] *= np.exp(estimator_weight)
            else:
                sample_weight[i] *= np.exp(-1 * estimator_weight)
    return sample_weight / sum(sample_weight)


""" Preparing dataset"""
# reading the dataset
countries = pd.read_csv('/Users/anastasia/Documents/IU/Fall18/Machine learning/Labs/datasets/Countries.csv')
# resetting region values to be 0 or 1 (EUROPE - 0, otherwise - 1)
countries['Region'] = (countries['Region'] != 'EUROPE').astype(int)
# resetting countries from string to classes
le = LabelEncoder()
countries['Country'] = le.fit_transform(countries['Country'])
# filling in any nan values with a mean column value
features = list(countries.columns.values)
features.pop(1)
for column in features:
    mean = countries[column].mean()
    countries[column].fillna(mean, inplace=True)
# resetting index values
countries_dataset_cleaned = countries.reset_index(drop=True)
# selecting features and labels sets
X_1 = countries_dataset_cleaned.drop('Region', axis=1)
y_1 = countries_dataset_cleaned['Region']

""" Training adaboost classifier to find outliers """
# model preparation
d_tree = tree.DecisionTreeClassifier(max_leaf_nodes=4, max_features=5)
n_est = 50
model_for_outliers = AdaBoostClassifier(base_estimator=d_tree, n_estimators=n_est, algorithm='SAMME')
model_for_outliers = model_for_outliers.fit(X_1, y_1)

# process of finding outliers
sample_weight = np.empty(X_1.shape[0], dtype=np.float64)
sample_weight[:] = 1. / X_1.shape[0]
estimators = model_for_outliers.estimators_
estimator_weights = model_for_outliers.estimator_weights_
estimator_errors = model_for_outliers.estimator_errors_

# iterating to update sample weights
for iboost in range(n_est):
    sample_weight = my_boost(iboost, X_1, y_1, sample_weight, n_est, estimators[iboost],
                             estimator_weights[iboost], estimator_errors[iboost])
weights = sample_weight

""" Finding outliers """
# finding outliers
outliers_ind = []
# using standard score and a tuned constant that in my opinion performs well in our situation
scores = sp.zscore(weights)
for i in range(len(scores)):
    if scores[i] > 1.5:
        outliers_ind.append(countries['Country'][i])

# outliers plot
indices = [i for i in range(len(weights))]
plt.scatter(indices, weights, color='red')
plt.show()

# droping outliers
countries_cleaned = countries_dataset_cleaned.copy()
for i in outliers_ind:
    countries_cleaned = countries_cleaned[countries_cleaned['Country'] != i]
countries_cleaned = countries_cleaned.reset_index(drop=True)

""" Training the model with outliers """
# creating test and train sets
accuracies = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_1, y_1)
    n_est = 22
    model_with_outliers = AdaBoostClassifier(base_estimator=d_tree, n_estimators=n_est)
    model_with_outliers = model_with_outliers.fit(X_train, y_train)
    predictions = model_with_outliers.predict(X_test)
    acc = accuracy(predictions, y_test)
    accuracies.append(acc)
ind = [i for i in range(1, len(accuracies)+1)]
plt2.plot(ind, accuracies, color='blue', label='with outliers')


""" Training the model without outliers """
X_2 = countries_cleaned.drop('Region', axis=1)
y_2 = countries_cleaned['Region']
accuracies_after = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_2, y_2)

    model_without_outliers = AdaBoostClassifier(base_estimator=d_tree, n_estimators=n_est)
    model_without_outliers = model_without_outliers.fit(X_train, y_train)
    predictions = model_without_outliers.predict(X_test)
    acc_2 = accuracy(predictions, y_test)
    accuracies_after.append(acc_2)
plt2.plot(ind, accuracies_after, color='green', label='without outliers')
plt2.title("Accuracies compared")
plt2.legend()
plt2.show()

