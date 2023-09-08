#! /usr/bin/python

import LinearRegression as lr
import numpy as np
import sys

def main(filename):
    classifier = lr.LinearRegression()
    classifier.load_data(filename)
    print "Cost: ", classifier.compute_cost()
    classifier.gradient_descent()
    print "Theta values: ", classifier.theta()
    print "New computed cost: ", classifier.compute_cost()
    print "Prediction for 10000: ", classifier.predict(np.matrix('1.0')) * 10000
    print "Prediction for 20000: ", classifier.predict(np.matrix('2.0')) * 10000
    print "Prediction for 30000: ", classifier.predict(np.matrix('3.0')) * 10000
    print "Prediction for 500000: ", classifier.predict(np.matrix('50.0')) * 10000
    if classifier.can_plot():
        classifier.plot()

if __name__ == '__main__':
    f = sys.argv[1]
    main(f)