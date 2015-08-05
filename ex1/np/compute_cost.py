"""
Numpy implementation of the Solutions to Stanford Machine learning
course on Coursera.

By: Matt Hosack
August 5, 2015
"""

import numpy as np
import matplotlib.pyplot as plt


__author__ = "Matt Hosack"


def compute_cost(X, y, theta):
    """ Compute the cost function given
    a feature array, output vector, and
    theta array
    """

    m = y.shape[0]
    H = X.dot(theta)
    J = ((H - y) ** 2.).sum() / (2. * m)
    return J


def plot_data(X, y):
    """ Plot X and y scatter plot
    """

    plt.plot(X, y, 'rx', markersize=10)
    plt.show()


def gradient_descent(X, y, theta, alpha, n):
    """ Perform n iterations of gradient descent
    on X, y, theta using alpha as the learning factor

    Returns theta matrix, Cost function history
    """

    #m = number of features
    m = y.shape[0]
    #Cost function history can be a python list
    J_hist = list()

    for i in xrange(n):
        #h(x) equation
        H = X.dot(theta)
        #temp array for updating theta
        temp = np.zeros((theta.shape[0], 1))
        #calculate each temp value
        for j in xrange(len(temp)):
            temp[j] = theta[j] - alpha * (1./m) * ((H - y).T).dot(X[:, j])
        #update theta
        theta = temp
        #add cost function output to the history
        J_hist.append(compute_cost(X, y, theta))

    return theta, J_hist


def feature_normalize(X):
    cols = X.shape[1]
    mu = np.zeros(cols)
    sigma = np.zeros(cols)

    for i in xrange(cols):
        mu = float(np.mean(X[:, i]))
        sigma = float(np.std(X[:, i]))
        X[:, i] = (X[:, i] - float(mu)) / float(sigma)

    return X


def read_data(fname='../ex1data1.txt'):
    """ Read data from ex1data1.txt file
    """

    with open('../ex1data1.txt', 'r') as f:
        # read data from file
        x = list()
        y = list()
        try:
            for line in f:
                x.append(line.split(',')[0])
                y.append(line.split(',')[1])
            x = map(float, x)
            y = map(float, y)
        except ValueError:
            print 'Not able to convert data into numbers'
        except IndexError:
            print 'Data not formatted correctly ex: \'x,y\''

    # prepend a column of ones to x and stack y values vertically
    x = np.column_stack((np.ones(len(x)), np.array(x)))
    y = np.vstack(np.array(y))
    return x, y


def main():
    X, y = read_data()
    theta = np.array([[0], [0]])
    #print 'Cost: {}'.format(compute_cost(X, y, theta))
    #plot_data(X, y)
    #t, J = gradient_descent(X, y, theta, 0.01, 1500)
    #print 'Theta: {}'.format(t)
    #print 'J_hist: {}'.format(J)


if __name__ == '__main__':
    main()
