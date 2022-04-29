#!/usr/bin/python3
# Homework 5 Code
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def adaboost_trees(X_train, y_train, X_test, y_test, n_trees):
    # %AdaBoost: Implement AdaBoost using decision trees
    # %   using information gain as the weak learners.
    # %   X_train: Training set
    # %   y_train: Training set labels
    # %   X_test: Testing set
    # %   y_test: Testing set labels
    # %   n_trees: The number of trees to use

    N = X_train.shape[0]
    D_t = [1/N] * N
    D_t = np.array(D_t)
    models = []
    alphas = []

    for t in range(1, n_trees + 1):
        model = DecisionTreeClassifier(criterion='entropy', max_depth=1)
        model.fit(X_train, y_train, sample_weight=D_t)

        epsilon = 1 - model.score(X_train, y_train, sample_weight=D_t)
        alpha = 1/2 * np.log((1-epsilon) / epsilon)

        D_t = (D_t * np.exp(-alpha * y_train * model.predict(X_train))) / np.sum(D_t)

        alphas.append(alpha)
        models.append(model)


    train_error = []
    test_error = []

    y_train_pred = [0] * N
    y_test_pred = [0] * X_test.shape[0]

    for t, model in enumerate(models):
        train_pred = model.predict(X_train) * alphas[t]
        for i in range(N):
            y_train_pred[i] += train_pred[i]

        test_pred = model.predict(X_test) * alphas[t]
        for i in range(X_test.shape[0]):
            y_test_pred[i] += test_pred[i]

        y_train_normalized = np.array(y_train_pred) / t
        y_test_normalized = np.array(y_test_pred) / t

        train_error.append(np.sum(np.sign(y_train_normalized) != y_train) / y_train.shape[0])
        test_error.append(np.sum(np.sign(y_test_normalized) != y_test) / y_test.shape[0])

    return train_error, test_error


def main_hw5():
    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')

    num_trees = 200

    # Split data
    X_train = og_train_data[:, 1:] 
    y_train = og_train_data[:, 0]
    X_test = og_test_data[:, 1:] 
    y_test = og_test_data[:, 0]

    X_train = X_train[np.where((y_train == 3) | (y_train == 5))]
    y_train = y_train[np.where((y_train == 3) | (y_train == 5))]
    y_train = np.where(y_train == 3, 1, -1)

    X_test = X_test[np.where((y_test == 3) | (y_test == 5))]
    y_test = y_test[np.where((y_test == 3) | (y_test == 5))]
    y_test = np.where(y_test == 3, 1, -1)

    train_error, test_error = adaboost_trees(X_train, y_train, X_test, y_test, num_trees)
    # print("Train Error", train_error)
    # print("Test Error", test_error)

    x_axis = list(range(1, num_trees + 1))
    plt.plot(x_axis, train_error)
    plt.plot(x_axis, test_error)
    plt.legend(['Train Error', 'Test Error'])
    plt.xlabel('Number of Weak Hypotheses')
    plt.ylabel('Error')
    plt.title('3 vs 5 Problem')
    plt.show()


if __name__ == "__main__":
    main_hw5()