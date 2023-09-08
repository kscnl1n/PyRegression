#! /usr/bin/python

import sys
import numpy as np
import plotly as py
import matplotlib.pyplot as plt

from Exceptions import *

class LinearRegression(object):
    """ 
    LinearRegression class contains methods to load, normalize, plot data; and predict outputs.
    """

    def __init__(self):
        self.norm = False #Data has not been normalized
        self.gd = False #Gradient descent has not been carried out

    def load_data(self, *files):
        """
        LOADING METHOD:
        If 2 files are given, the first one is assumed to contain x (input) values and the second one y (output) values.
        If 1 file is given the first column is assumed to contain x values and the second one y values.
        """
        if len(files) == 1:
            dataXY = np.matrix(np.genfromtxt(files[0], delimiter = ','))
            pos = dataXY.shape[1]
            y = dataXY[:, pos -1]
            X = dataXY[:, :pos - 1]
        else:
            X = np.matrix(np.genfromtxt(files[0], delimiter = ','))
            y = np.matrix(np.genfromtxt(files[1], delimiter = ','))
        ones = np.matrix(np.ones(shape = (X.shape[0], 1)))
        self.X = np.hstack((ones, X))
        self.y = y
        self.theta = np.matrix(np.zeros(shape = (self.X.shape[1], 1)))
        self.norm = False
        self.gd = False
        self.gradient_descent()

    def plot(self):
        """
        PLOTTING METHOD:
        Plots the loaded data and the regression line. Throws DataHandlingException if more than 1 lx-label 
        and NoDataException if no data is found.
        """
        if not hasattr(self, 'X'): # To raise an exception is data is not loaded
            raise NoDataException()
        elif self.X.shape[1] > 2: # To raise an exception if there way to many exceptions
            raise DataHandlingException()
        else:
            X = self.X[:,1]
            y = self.X * self.theta
            plt.plot(self.X[:,1], self.y, 'rx', X, y, 'g-')
            plt.show()

    def normalize(self):
        """
        NORMALIZATION METHOD:
        Normalizes the data such that the mean is 0 and the data is between -0.5-0.5.
        Stores the mean range of all x features aside from 1s added for vectorization. 
        """
        if not self.norm: # Prevents from normalizing again
            self.norm = True
            if not hasattr(self, 'X'): # Raises exception if data not loaded
                raise NoDataException()
            else:
                self.X_mean = np.matrix(np.zeros(shape=(1,self.X.shape[1] - 1)))
                self.X_range = np.matrix(np.zeros(shape=(1,self.X.shape[1] - 1)))
                for i in range(self.X.shape[1]):
                    if not i == 0: # Since 1st column contains the 1s for vectorization
                        tempX = self.X[:, i]
                        meanX = np.mean(tempX)
                        self.X_mean[0, i - 1] = meanX
                        rangeX = max(tempX) - min(tempX)
                        self.X_range[0, i - 1] = rangeX
                        tempX = (tempX - meanX)/rangeX
                        self.X[:, i] = tempX
            pass
        pass

    def compute_cost(self):
        """
        COMPUTE_COST METHOD:
        Calculates cost Function of the Linear Regression algorithm
        """
        m = self.X.shape[0]
        preds = self.X * self.theta
        errors = preds - self.y
        sq_errors = np.square(errors)
        summation = np.sum(sq_errors)
        J = ((1.0/(2*m))*summation)
        return J

    def gradient_descent(self):
        """
        GRADIENT_DESCENT METHOD:
        Carries out gradient descent to compute the
        values of theta such that the error is reduced 
        """
        self.normalize()
        if not self.gd:
            self.gd = True
            num_iters = 7500
            alpha = 0.01
            m = self.X.shape[0]
            n = self.X.shape[1]
            history = np.matrix(np.zeros(shape = (1, num_iters)))
            for i in range(num_iters):
                delta = np.matrix(np.zeros(shape = (n, 1)))
                for j in range(m):
                    Xi = self.X[j,:].T
                    pred = self.theta.T * Xi
                    error = pred - self.y[j,0]
                    delta += (error[0,0] * Xi)
                self.theta = self.theta - ((alpha/m)*delta)
                history[0,i] = self.compute_cost()
            self.history = history

    def predict(self, x):
        """ 
        PREDICT METHOD:
        Consumes an npmatrix of shape (1xn) where n
        is the number of features (x labels) AKA user data.
        The function then predicts an output
        based on data.
        """
        pred_x = np.matrix(np.zeros(shape = x.shape))
        for i in range(x.shape[1]):
            pred_x[0,i] = ((x[0, i] - self.X_mean[0,i])/(self.X_range[0,i]))
        ones = np.matrix('1')
        pred_x = np.hstack((ones, pred_x))
        pred = self.theta.T * pred_x.T
        return pred[0,0]

    def can_plot(self):
        """
        CAN_PLOT METHOD:
        If the data can be plotted, can_plot will return true.
        If not it will return false. We will call this function later to give the user an error message
        //TO UPDATE - KAY 
        """
        if not hasattr(self, 'X'):
            return False
        elif self.X.shape[1] > 2:
            return False
        else:
            return True
