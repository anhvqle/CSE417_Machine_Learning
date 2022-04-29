#!/usr/bin/python3
# Homework 1 Code
import numpy as np
import matplotlib.pyplot as plt
import math

def perceptron_learn(data_in):
    # Run PLA on the input data
    #
    # Inputs: data_in: Assumed to be a matrix with each row representing an
    #                (x,y) pair, with the x vector augmented with an
    #                initial 1 (i.e., x_0), and the label (y) in the last column
    # Outputs: w: A weight vector (should linearly separate the data if it is linearly separable)
    #        iterations: The number of iterations the algorithm ran for

    # Your code here, assign the proper values to w and iterations:
    
    N = data_in.shape[0]
    d = data_in.shape[1] - 2
    x = data_in[:, :d+1]
    y = data_in[:, -1]
    w = np.ones([d + 1, 1])
    iterations = 0
    index = None

    while True:
        h = np.sign(x.dot(w))

        for i in range(len(h)):
            if h[i][0] != y[i]:
                index = i
                break
        
        if index == None:
            break

        w = w + (y[index] * x[index]).reshape(-1 , 1)
        index = None
        iterations += 1

    return w, iterations


def perceptron_experiment(N, d, num_exp):
    # Code for running the perceptron experiment in HW1
    # Implement the dataset construction and call perceptron_learn; repeat num_exp times
    #
    # Inputs: N is the number of training data points
    #         d is the dimensionality of each data point (before adding x_0)
    #         num_exp is the number of times to repeat the experiment
    # Outputs: num_iters is the # of iterations PLA takes for each experiment
    #          bounds_minus_ni is the difference between the theoretical bound and the actual number of iterations
    # (both the outputs should be num_exp long)

    # Initialize the return variables
    num_iters = np.zeros((num_exp,))
    bounds_minus_ni = np.zeros((num_exp,))
    # Your code here, assign the values to num_iters and bounds_minus_ni:

    rng = np.random.default_rng(54321)
    for i in range(num_exp):
        x = np.concatenate( (np.ones([N,1]), np.random.random([N, d])*2 - 1), axis = 1) 
        w_t = np.random.random([d + 1, 1])
        w_t[0] = 0
        y = np.sign(x.dot(w_t))
        D = np.concatenate((x, y), axis=1)
        w, num_iters[i] = perceptron_learn(D)

        rho = min(abs(x.dot(w_t)))

        R = -math.inf
        for x_i in x[:, 1:]:
            R = max(R, sum(x_i ** 2))

        bounds_minus_ni[i] = (sum(w_t**2) * R) / (rho**2) - num_iters[i]

    return num_iters, bounds_minus_ni


def main():
    print("Running the experiment...")
    num_iters, bounds_minus_ni = perceptron_experiment(100, 10, 1000)

    print("Printing histogram...")
    plt.hist(num_iters)
    plt.title("Histogram of Number of Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Count")
    plt.show()

    print("Printing second histogram")
    plt.hist(np.log(bounds_minus_ni))
    plt.title("Bounds Minus Iterations")
    plt.xlabel("Log Difference of Theoretical Bounds and Actual # Iterations")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    main()
