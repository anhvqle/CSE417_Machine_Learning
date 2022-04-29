#!/usr/bin/python3
# Homework 4 Code
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def bootstrap_sample(X_train, y_train):
    X_boot = np.zeros((X_train.shape[0], X_train.shape[1]))
    y_boot = np.zeros(y_train.shape[0])
    indexes = []

    for i in range(X_train.shape[0]):
        index = np.random.choice(X_train.shape[0])
        X_boot[i] = X_train[index]
        y_boot[i] = y_train[index]
        if index not in indexes:
            indexes.append(index)

    indexes = np.array(indexes)
    
    return X_boot, y_boot, indexes


def bagged_tree(X_train, y_train, X_test, y_test, num_bags):
    # The `bagged_tree` function learns an ensemble of numBags decision trees 
    # and also plots the  out-of-bag error as a function of the number of bags
    #
    # % Inputs:
    # % * `X_train` is the training data
    # % * `y_train` are the training labels
    # % * `X_test` is the testing data
    # % * `y_test` are the testing labels
    # % * `num_bags` is the number of trees to learn in the ensemble
    #
    # % Outputs:
    # % * `out_of_bag_error` is the out-of-bag classification error of the final learned ensemble
    # % * `test_error` is the classification error of the final learned ensemble on test data
    #
    # % Note: You may use sklearns 'DecisonTreeClassifier'
    # but **not** 'RandomForestClassifier' or any other bagging function

    #Get out of bag indexes
    OOB_indexes = []
    models = []

    for i in range(num_bags):
        X_boot, y_boot, indexes = bootstrap_sample(X_train, y_train)
        model = DecisionTreeClassifier(criterion='entropy')
        model.fit(X_boot, y_boot)
        models.append(model)

        OOB_index = np.setdiff1d(np.arange(X_train.shape[0]), indexes)
        OOB_indexes.append(OOB_index)

    # Calculate Test Error
    y_predictions = []
    for model in models:
        y_predictions.append(model.predict(X_test))

    y_predictions = stats.mode(y_predictions)[0][0]
    y_predictions = np.array(y_predictions)
    test_error = np.sum(y_predictions != y_test) / y_test.shape[0]

    # Calculate OOB Errors
    out_of_bag_predictions = []
    for i in range(X_train.shape[0]):
        predictions = []
        for bag, oob_i in enumerate(OOB_indexes):
            
            if np.any(oob_i == i):
                predictions.append(models[bag].predict(X_train[i].reshape(1, -1))[0])

        if len(predictions) != 0:
            out_of_bag_predictions.append(stats.mode(predictions)[0][0])
        else:
            out_of_bag_predictions.append(None)

    # out_of_bag_predictions = stats.mode(out_of_bag_predictions)[0][0]
    out_of_bag_predictions = np.array(out_of_bag_predictions)
    out_of_bag_error = np.sum(out_of_bag_predictions != y_train) / y_train.shape[0]

    return out_of_bag_error, test_error

def single_decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    train_error = np.sum(train_predictions != y_train) / y_train.shape[0]
    test_error = np.sum(test_predictions != y_test) / y_test.shape[0]
    
    return train_error, test_error

def main_hw4():
    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')

    num_bags = 200

    # Split data
    X_train = og_train_data[:, 1:] 
    y_train = og_train_data[:, 0]
    X_test = og_test_data[:, 1:] 
    y_test = og_test_data[:, 0]

    X_train = X_train[np.where((y_train == 1) | (y_train == 3))]
    y_train = y_train[np.where((y_train == 1) | (y_train == 3))]
    y_train = np.where(y_train == 1, 1, -1)

    X_test = X_test[np.where((y_test == 1) | (y_test == 3))]
    y_test = y_test[np.where((y_test == 1) | (y_test == 3))]
    y_test = np.where(y_test == 1, 1, -1)


    x_axis = list(range(1, num_bags + 1))
    y_axis = [0] * num_bags

    for num_bag in range(1, num_bags + 1):
        print('Iter:', num_bag)
        out_of_bag_error, test_error = bagged_tree(X_train, y_train, X_test, y_test, num_bag)
        y_axis[num_bag - 1] = out_of_bag_error

    # print(x_axis, y_axis)
    plt.plot(x_axis, y_axis)
    plt.title('OOB error for bagging decision trees (1 vs 3 Problem)')
    plt.xlabel('number of bags')
    plt.ylabel('OOB error')
    plt.show()


    # Run bagged trees
    out_of_bag_error, test_error = bagged_tree(X_train, y_train, X_test, y_test, num_bags)
    print("OOB Error", out_of_bag_error)
    print("Test Error", test_error)

    train_error, test_error = single_decision_tree(X_train, y_train, X_test, y_test)
    print("Train Error", train_error)
    print("Test Error", test_error)


if __name__ == "__main__":
    main_hw4()
