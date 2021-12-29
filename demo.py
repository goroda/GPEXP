# Copyright (c) 2013-2016, Massachusetts Institute of Technology
# Copyright (c) 2016-2022, Alex Gorodetsky
#
# This file is part of GPEXP:
# Author: Alex Gorodetsky alex@alexgorodetsky
#
# GPEXP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# GPEXP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GPEXP.  If not, see <http://www.gnu.org/licenses/>.

# Code

import numpy as np
import matplotlib.pyplot as plt

# Mine
from gpExp.kernels import KernelSquaredExponential
from gpExp.experimentalDesign import * 
from gpExp.gp import GP
from gpExp.approximation import Space


# FUNCTION TO APPROXIMATE
def func(x):

    #out = np.sin(2.0*np.pi*x) + np.sin(4.0*np.pi*x) + np.sin(16.0*np.pi*x)
    out = np.sin(2.0*np.pi*x)

    if len(out.shape) == 2:
        return out[:,0]
    return out


#DEMO FUNCTION CALLED BY __main__ below
def demo():
    
    print("\n\n******************************************************")
    print("Starting a one dimensional demo of the GPEXP toolbox! ")
    print("******************************************************")
    dimension = 1

    #Setup a squared exponential covariance Kernel
    correlationLength = [0.3]
    signalSize = 1.0
    kernel = KernelSquaredExponential(correlationLength, signalSize, dimension)
    
    #Setup GP
    noise = 0.0
    gpT = GP(kernel, noise)
    
    #Update Hyperparameters
    #Can call make the following call to get a list of hyperparameters

    print("\nUsing GP Squared Exponential Kernels with initial values")
    paramNames = gpT.getHypParamNames()
    for p in paramNames:
        print("\t" + p + " = {:3.5f}".format(gpT.kernel.hyperParam[p]))
    print("\n")

    #Generate Training Points 
    xTrain = np.array([-0.8, 0.2, 0.3, -0.1]).reshape((4,1)) #must be an $N x dim$ vector
    yTrain = func(xTrain)+np.sqrt(noise)*np.random.randn(len(xTrain)) #get noisy function values
    logLike = gpT.computeLogLike(xTrain, yTrain)
    print("Initial log likelihood of training points is {:3.5f}".format(logLike))
    
    
    print("\n**************************************************************")
    print("Optimizing kernel hyperparameters by maximizing marginal log-likelihood\n")
    lbCL = 1e-2
    ubCL = 1e10
    lbSigSize = 9e-1
    ubSigSize = 2e0
    guess = dict({'cl0':1e-1, 'signalSize':1e0}) #initial guess
    lb = dict({'cl0':lbCL, 'signalSize':lbSigSize}) #lowerb bounds for search
    ub = dict({'cl0':ubCL, 'signalSize':ubSigSize}) #upperBounds for search
    
    #optParams = gpT.findOptParamsLogLike(xTrain, yTrain, guess,lb,ub)
    optParams = gpT.findOptParamsLogLike(xTrain, yTrain)
    print("\nNew params are: ")
    for p, val in optParams[0].items():
        print("\t" + p + " = {:3.5f}".format(val))
    print("With new hyperparameters log-likelihood is {:3.5f}".format(optParams[1]))
    #Train New GP
    gpT.train(xTrain, yTrain)
    # can print out the coefficients by uncommenting below line
    #print "coeff ", gpT.coeff

    #Get Points for testing
    xDemo = np.linspace(-1,1,1000).reshape((1000,1))

    #evaluate mean and variance of gp at zDemo
    m, var = gpT.evaluate(xDemo,compvar=1)
    stddev = np.sqrt(var)
    
    #plot
    fig = plt.figure(1)
    plt.fill_between(xDemo[:,0], m-2*stddev, m+2*stddev, facecolor=[0.7,0.7,0.7])
    plt.plot(xTrain, yTrain, 'ko', ms=5, )
    plt.plot(xDemo, func(xDemo), 'k--', label='True Function')
    plt.plot(xDemo,m, label='Posterior Mean')
    plt.title('Posterior')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(loc=0)

    #######################################################################
    #Experimental Design with IVAR
    ######################################################################
    #NOTE UNIFORM NOISE IS USED HERE

    #Specify sampler and density function
    sampler = lambda size : np.random.rand(size[0],size[1])*2.0-1.0 #[-1,1]
    distrib = lambda points : (np.abs(points)<1.0)*0.5 #uniform distibution on [-1, 1]
    space = Space(dimension, sampler, distrib)

    #Specify cost function
    nPointsAdd = 4 #number of experimental desing points to add
    nTrainPointsHave = len(xTrain) #number of I already have
    costFunction = costFunctionGP_IVAR(gpT, nPointsAdd+nTrainPointsHave, space)

    #xperimental design procedure
    exp = ExperimentalDesignDerivative(costFunction, nPointsAdd+nTrainPointsHave, dimension)
    
    #Set lb or ub of new points
    #MAKE SURE THAT FIRST LB AND UB ARE EQUAL TO EXISTING TRAINING POINTS
    lbNewPoints = -np.ones((nPointsAdd))
    ubNewPoints = np.ones((nPointsAdd))
    lb = np.concatenate((xTrain.flatten(), lbNewPoints))
    ub = np.concatenate((xTrain.flatten(), ubNewPoints))

    #newTrainPoints contains previous experiments and new ones
    print("\n***************************************************************")
    print("Beginning experimental design of new points\n")

    newTrainPoints = exp.beginWithVarGreedy(nodesKeep=xTrain,lbounds=lb, rbounds=ub)

    #Train New GP
    newTrainValues = func(newTrainPoints)
    gpT.train(newTrainPoints, newTrainValues)

    #Evaluate GP
    m, var = gpT.evaluate(xDemo,compvar=1)
    stddev = np.sqrt(var)

    #plot
    fig = plt.figure(2)
    plt.fill_between(xDemo[:,0], m-2*stddev, m+2*stddev, facecolor=[0.7,0.7,0.7])
    plt.plot(newTrainPoints, newTrainValues, 'ko', ms=5)
    plt.plot(xDemo, func(xDemo), 'k--', label='True Function')
    plt.plot(xDemo,m, label='Posterior Mean')
    plt.title('Posterior')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(loc=0)
    
if __name__ == "__main__":
    demo()
    plt.show()
    plt.close('all')
    #plt.show()

    

