import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt


def accuracy(predictions, y_test):
    """
    Evaluates the accuracy of the prediction
    """
    correct = 0
    for t in zip(predictions, y_test):
        if t[0] == t[1]:
            correct += 1
    return (correct / float(len(y_test))) * 100.0


"""
Preparing dataset
"""
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


"""
Training adaboost classifier to find outliers 
"""
d_tree = tree.DecisionTreeClassifier(max_leaf_nodes=4, max_features=5)
n_est = 22
model_for_outliers = AdaBoostClassifier(base_estimator=d_tree, n_estimators=n_est)
model_for_outliers, weights = model_for_outliers.fit(X_1, y_1)


"""
Finding outliers
"""
# finding outliers
average = sum(weights)/len(weights)
outliers_ind = []
for i in range(len(weights)):
    if weights[i] > average:
        outliers_ind.append(countries['Country'][i])

# outliers plot
indices = [i for i in range(len(weights))]
plt.scatter(indices, weights, color='red')

# droping outliers
countries_cleaned = countries_dataset_cleaned.copy()
for i in outliers_ind:
    countries_cleaned = countries_cleaned[countries_cleaned['Country'] != i]
countries_cleaned = countries_cleaned.reset_index(drop=True)

"""
Training the model with outliers 
"""
# creating test and train sets
X_train, X_test, y_train, y_test = train_test_split(X_1, y_1)
model_with_outliers = AdaBoostClassifier(base_estimator=d_tree, n_estimators=n_est)
model_with_outliers, weights = model_with_outliers.fit(X_train, y_train)
predictions = model_with_outliers.predict(X_test)
print("Before: ", accuracy(predictions, y_test))

"""
Training the model without outliers
"""
X_2 = countries_cleaned.drop('Region', axis=1)
y_2 = countries_cleaned['Region']
X_train, X_test, y_train, y_test = train_test_split(X_2, y_2)

model_without_outliers = AdaBoostClassifier(base_estimator=d_tree, n_estimators=n_est)
model_without_outliers, weights_2 = model_without_outliers.fit(X_train, y_train)
predictions = model_without_outliers.predict(X_test)
print("After: ", accuracy(predictions, y_test))

indices_2 = [i for i in range(len(weights_2))]
plt.scatter(indices_2, weights_2, color='blue')
plt.show()







