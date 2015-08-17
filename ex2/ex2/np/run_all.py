import os
import numpy as np
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
DATA1 = os.path.join(HERE, '..', 'ex2data1.txt')
DATA2 = os.path.join(HERE, '..', 'ex2data2.txt')


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def cost_function(theta, X, y):
    m, n = X.shape
    y = y.reshape(m, 1)

    X = np.hstack((np.ones((m, 1)), X))
    H = sigmoid(X.dot(theta))

    J = (-y.T.dot(np.log(H)) - (1. - y).T.dot(np.log(1. - H))) / m
    grad = (H - y).T.dot(X) / m

    return J, grad


def gradient_descent(X, y, theta, alpha, n):
    """ Perform n iterations of gradient descent
    on X, y, theta using alpha as the learning factor

    Returns theta matrix, Cost function history
    """

    #number of features
    m = y.shape[0]
    X = np.hstack((np.ones((m, 1)), X))
    y = y.reshape((y.shape[0], 1))

    for i in xrange(n):
        #Hypothesis
        H = sigmoid(X.dot(theta))
        #temp array for updating theta
        temp = np.zeros(theta.shape)
        #calculate each temp value
        for j in xrange(len(temp)):
            temp[j] = theta[j] - alpha * ((H - y).T).dot(X[:, j])
        #update theta
        theta = temp

    return theta


def predict(theta, X):
    return np.round(sigmoid(X.dot(theta)))


def plot_data(X, y):
    pos = np.where(y > 0.5)[0]
    neg = np.where(y < 0.5)[0]

    plt.scatter(X[pos, 0], X[pos, 1], marker='+', color='k')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', color='y')

    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.legend(['Admitted', 'Not Admitted'])

    plt.show()


def main():
    data1, data2 = [], []
    with open(DATA1, 'r') as f:
        data1 = f.readlines()
    with open(DATA2, 'r') as f:
        data2 = f.readlines()

    X, y = [], []
    for line in data1:
        x1, x2, y1 = line.split(',')
        X.append([float(x1), float(x2)])
        y.append(y1)
    y = map(float, y)

    X = np.array(X)
    y = np.array(y)

    plot_data(X, y)

    m, n = X.shape
    initial_theta = np.zeros((n+1, 1))
    J, grad = cost_function(initial_theta, X, y)

    #theta = gradient_descent(X, y, initial_theta, 0.01, 1500)

if __name__ == '__main__':
    main()
