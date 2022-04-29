#!/usr/bin/python3
# Homework 2 Code
import numpy as np
import pandas as pd
import time

def sigmoid(s):
    return 1/(1+np.exp(-s))

def find_binary_error(w, X, y):
    # find_binary_error: compute the binary error of a linear classifier w on data set (X, y)
    # Inputs:
    #        w: weight vector
    #        X: data matrix (without an initial column of 1s)
    #        y: data labels (plus or minus 1)
    # Outputs:
    #        binary_error: binary classification error of w on the data set (X, y)
    #           this should be between 0 and 1.

    # Your code here, assign the proper value to binary_error:
    
    X = np.concatenate((np.ones([X.shape[0],1]), X), axis = 1)
    predictions = []
    for i in range(y.shape[0]):
        if sigmoid(np.dot(w, X[i])) > 0.5:
            predictions.append(1)
        else:
            predictions.append(-1)
    predictions = np.array(predictions)
    binary_error = np.sum(predictions != y) / y.shape[0]
    return binary_error


def logistic_reg(X, y, w_init, max_its, eta, grad_threshold):
    # logistic_reg learn logistic regression model using gradient descent
    # Inputs:
    #        X : data matrix (without an initial column of 1s)
    #        y : data labels (plus or minus 1)
    #        w_init: initial value of the w vector (d+1 dimensional)
    #        max_its: maximum number of iterations to run for
    #        eta: learning rate
    #        grad_threshold: one of the terminate conditions; 
    #               terminate if the magnitude of every element of gradient is smaller than grad_threshold
    # Outputs:
    #        t : number of iterations gradient descent ran for
    #        w : weight vector
    #        e_in : in-sample error (the cross-entropy error as defined in LFD)

    # Your code here, assign the proper values to t, w, and e_in:
    start_time = time.time()
    X = np.concatenate((np.ones([X.shape[0],1]), X), axis = 1)
    N, d = X.shape
    t = 0
    w = w_init
    while(t < max_its):
        t += 1
        gradient = np.zeros(d)
        for i in range(N):
            gradient += y[i] * X[i] / (1 + np.exp(y[i] * np.dot(w, X[i])))
        gradient *= (-1/N)
        w = w - eta * gradient
        if sum(np.abs(gradient) > grad_threshold) == 0:
            break
    
    e_in = 0
    for i in range(N):
        e_in += np.log(1 + np.exp(-y[i] * np.dot(w, X[i])))
    
    e_in /= N

    end_time = time.time()
    print(t, "iterations - finished in", end_time - start_time, "seconds")
    return t, w, e_in

def main():
    # Load training data
    train_data = pd.read_csv('clevelandtrain.csv')

    # Load test data
    test_data = pd.read_csv('clevelandtest.csv')

    # Your code here
    dataset = np.vectorize(lambda x: x if x == 1 else -1)
    x_train, y_train = train_data.iloc[:, :-1], dataset(train_data.iloc[:, -1].values)
    x_test, y_test = test_data.iloc[:, :-1], dataset(test_data.iloc[:, -1].values)

    for col in x_test.columns:
        # mean = x_train[col].mean()
        # std = x_train[col].std()
        x_train[col] = (x_train[col] - x_train[col].mean()) / x_train[col].std()
        x_test[col] = (x_test[col] - x_train[col].mean()) / x_train[col].std()

    x_train, x_test= x_train.values, x_test.values
    
    eta = 0.01
    grad_threshold = 10 ** (-3)
    N, d = x_train.shape
    w_init = np.zeros(d + 1)
    max_its = 10 ** 6
    t, w, e_in = logistic_reg(x_train, y_train, w_init, max_its, eta, grad_threshold)

    print('number of iteration required to terminate', t)
    print('E_in:', e_in)
    print('Binary classification on training data:', find_binary_error(w, x_train, y_train))
    print('Binary classification on testing data:', find_binary_error(w, x_test, y_test))


if __name__ == "__main__":
    main()