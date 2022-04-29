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


def logistic_reg(X, y, w_init, max_its, eta, grad_threshold, regularization_strength):
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

    X = np.concatenate((np.ones([X.shape[0],1]), X), axis = 1)
    N, d = X.shape
    t = 0
    w = w_init
    while t < max_its:
        t += 1
        gradient = np.zeros(d)
        for i in range(N):
            gradient += y[i] * X[i] / (1 + np.exp(y[i] * np.dot(w, X[i])))
            
        gradient *= (-1/N)
        
        #L1 Regularization
        additional_term = regularization_strength * np.sign(w)
        w_prime = w - eta * (additional_term + gradient)
        w = np.where((np.sign(w) != np.sign(w_prime)) & (w != 0), 0, w_prime)

        #L2 Regularization
        # additional_term = 2 * regularization_strength * w
        # w = w - eta * (additional_term + gradient)

        if sum(np.abs(gradient) > grad_threshold) == 0:
            break
    
    e_in = 0
    for i in range(N):
        e_in += np.log(1 + np.exp(-y[i] * np.dot(w, X[i])))
    
    e_in /= N

    return w

def main():
    # Load data
    X_train, X_test, y_train, y_test = np.load("digits_preprocess.npy", allow_pickle=True)
    y_train = np.where(y_train == 1, 1, -1)
    y_test = np.where(y_test == 1, 1, -1)

    for i in range(X_train.shape[1]):
        if np.sum(np.isclose(X_train[:, i], 0)) == X_train.shape[0]:
            continue
        X_train_i_mean = np.mean(X_train[:, i])
        X_train_i_std = np.std(X_train[:, i])
        X_train[:, i] = (X_train[:, i] - X_train_i_mean) / X_train_i_std
        X_test[:, i] = (X_test[:, i]- X_train_i_mean) / X_train_i_std
        

    regularization_strengths = [0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
    eta = 0.01
    grad_threshold = 10 ** (-6)
    N, d = X_train.shape
    w_init = np.zeros(d + 1)
    max_its = 10 ** 4

    for r in regularization_strengths:
        print('regularization_strength (lambda):', r)
        w = logistic_reg(X_train, y_train, w_init, max_its, eta, grad_threshold, r)
        #print('E_in:', e_in)
        #print('Binary classification on training data:', find_binary_error(w, X_train, y_train))
        print('Binary classification on testing data:', find_binary_error(w, X_test, y_test))
        print('Number of 0s in learned weight vector', np.sum(np.isclose(w, 0)))
        print('-----------------------------------------------------------------')

if __name__ == "__main__":
    main()