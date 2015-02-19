#Copyright (c) 2013-2014, Massachusetts Institute of Technology
#
#This file is part of GPEXP:
#Author: Alex Gorodetsky goroda@mit.edu
#
#GPEXP is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#GPEXP is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with GPEXP.  If not, see <http://www.gnu.org/licenses/>.

#Code

from experimentalDesign import costFunctionGP_IVAR
from experimentalDesign import ExperimentalDesignNoDerivative
from experimentalDesign import ExperimentalDesignDerivative
from experimentalDesign import ExperimentalDesignGreedyWithNoDerivatives, performGreedyVarExperimentalDesign
from experimentalDesign import performExpDesignWithContinuation, performExpDesignWithHypContinuation
from gp import GP
import copy
import numpy as np

class Space:

    """ The Space class describes the space and measure in which we are interested """
    dimension = None
    inBoundsBool = None
    sample = None # function
    probDensity = None
    noiseFunc = None

    def __init__(self, dimensionIn,  samplerIn, probDensityIn, noise=None):
        """ Initialize Space class """
        self.dimension = dimensionIn
        self.sample = samplerIn
        self.probDensity = probDensityIn
        self.noiseFunc = noise
        
class Approximation:
    
    """ This is the Approximation it contains information about the approximation method
        and the space on which we will be performing the approximation """

    space = None #class of Space type
    approxMethod = None #class containing approx method

    def __init__(self, spaceIn, approxMethodIn):
        
        self.space = spaceIn
        self.approxMethod = approxMethodIn
    

    def trainWithPoints(self, pointsIn, *argc):
        
        self.approxMethod.train(pointsIn)

    def trainWithPointsWeights(self, pointsIn, weightsIn):

        self.approxMethod.train(pointsIn, weightsIn)

class AdaptiveGPapprox(object):

    def __init__(self, spaceIn, gp, hyperParamInfo, functionIn, innerConverg, outerConverg, writeDir):
        self.space = spaceIn
        self.gp = copy.copy(gp)
        self.kernel = self.gp.kernel
        self.hyperParamInfo = hyperParamInfo
        self.dimension = self.kernel.dimension
        self.noiseIn = self.gp.noise
        self.function = functionIn
        self.dirSave = writeDir
        self.innerLoopConvergenceCriteria = innerConverg
        self.outerLoopConvergenceCriteria = outerConverg
        super(AdaptiveGPapprox, self).__init__()

    
    def innerLoop(self, nodesStart, nBatch, diagnostics, expType, numRestarts, continuationType, **kwargs):
        
        if diagnostics==1:
            costList = []
            nodesList = []

        nNodesStart = len(nodesStart)
        nodesBefore = nodesStart[:]
        nNodesBefore = nNodesStart
        innerIter = 0
        innerConverged = 0
        while innerConverged == 0:
            nNodes = nNodesBefore + nBatch
            
            costFunction = costFunctionGP_IVAR(self.gp, nNodes, self.space)
            if expType == 'GreedyVar':
                assert nBatch == 1, 'for greedy var can only use batch sizes of 1'

                mcPointsUse = kwargs['mcPoints']
                startVals = np.concatenate((nodesBefore, mcPointsUse), axis=0)
                mcPointsWeight = self.space.probDensity(startVals)
                #really nodes gVar
                #nodes= performGreedyVarExperimentalDesign(self.kernel, startVals, 
                #            nNodes, self.dimension, weights=mcPointsWeight,
                #            indKeepStart=np.arange(0,len(nodesBefore)).tolist())
                nodes= performGreedyVarExperimentalDesign(self.kernel, startVals, 
                            nNodes, self.dimension, 
                            indKeepStart=np.arange(0,len(nodesBefore)).tolist())



            else:
                startVals = [ np.concatenate((nodesBefore, self.space.sample((nBatch,self.dimension))),axis=0) for ii in xrange(numRestarts)]
                if expType == 'derivative':
                    exp = ExperimentalDesignDerivative(costFunction, nNodes, self.dimension)
                elif expType == 'noDerivative':
                    exp = ExperimentalDesignNoDerivative(costFunction, nNodes, self.dimension)

                lb = -100.0*np.ones((nNodes*self.dimension))
                ub = 100.0*np.ones((nNodes*self.dimension))
                lb[0:nNodesBefore*self.dimension] = nodesBefore.reshape((nNodesBefore*self.dimension))
                ub[0:nNodesBefore*self.dimension] = nodesBefore.reshape((nNodesBefore*self.dimension))
                if continuationType == 'hyp':
                    contExp = performExpDesignWithHypContinuation(exp, kwargs['hypList'])
                    nodes = contExp.begin(startVals, lb, ub)
                elif continuationType == 'noise':
                    contExp = performExpDesignWithContinuation(1e-1, exp)
                    nodes = contExp.begin(startVals, lb, ub)
                else:
                    #print "start: "
                    #print "points ", startVals

                    #print "lb ", lb
                    #print "ub ", ub
                    #nodes = exp.begin(startVals, lb, ub)
                    nodes = exp.beginWithVarGreedy(nodesKeep=nodesBefore,
                                lbounds=lb, rbounds=ub)
                    #print "stop "

            cost = costFunction.evaluate(nodes)
            nNodesBefore = nNodes
            if diagnostics == 1:
                nodesList.append(nodes)
                costList.append(cost)

            innerIter += 1
            if 'IVARlevel' in self.innerLoopConvergenceCriteria.keys():
                print "cost ", cost, " ivarLevel ", self.innerLoopConvergenceCriteria['IVARlevel']
                if cost < self.innerLoopConvergenceCriteria['IVARlevel']:
                    innerConverged = 1
            elif 'numIters' in self.innerLoopConvergenceCriteria.keys():
                #print "here ", self.innerLoopConvergenceCriteria['numIters']
                if innerIter >= self.innerLoopConvergenceCriteria['numIters']:
                    innerConverged = 1
            #print self.innerLoopConvergenceCriteria
            #print "innerIter ", innerIter, cost, innerConverged
            nodesBefore = nodes[:]

        if diagnostics == 1:
            return nodes, nodesList, costList
        else:
            return nodes, cost
        
    def adapt(self, nBatch, startPointsIn=np.array([]), diagnostics=1, expType='noDerivative', numRestarts=1, continuationType=0, **kwargs):
        MAXITER=100
        outerConverged = 0
        outerIter = 0
        
        if len(startPointsIn) == 0:
            startPoints = np.zeros((0, self.dimension))
            nNodes = 0
        else:
            nNodes = len(startPoints)
            startPoints = startPointsIn[:]
        
        if diagnostics == 1:
            nodesList = []
            hyperParam= []
            costList = []
            logLike = []

        oldParams = copy.copy(self.gp.kernel.hyperParam)   
        while outerConverged == 0:
            
            if diagnostics == 1:
                startPoints, newNList, newCList = self.innerLoop(startPoints, 
                        nBatch, diagnostics, expType, numRestarts, 
                        continuationType, **kwargs)
                nodesList = nodesList + newNList
                costList = costList + newCList
                hyperParam = hyperParam + len(newNList)*[oldParams]
                cost = costList[-1]
            else:
                startPoints, cost = innerLoop(startPoints, nBatch, 
                                        diagnostics, expType, numRestarts, 
                                        continuationType, **kwargs)
            
            self.hyperParamInfo['start'] =copy.copy(self.gp.kernel.hyperParam)
            self.gp.addNodesAndComputeCovariance(startPoints)
            print "starting HyperParamsters ", self.hyperParamInfo['start']
            newParams, optValue = \
                self.gp.findOptParamsLogLike(startPoints, 
                        self.function(startPoints), 
                        self.hyperParamInfo['start'], 
                        self.hyperParamInfo['lower'], 
                        self.hyperParamInfo['upper'])
            self.gp.updateKernelParams(newParams)
            #print "ending HyperParameters ", self.gp.kernel.hyperParam
            if diagnostics == 1:
                logLike.append(-optValue)
            
            print "Number of Points: ", str(len(startPoints)), " COST: ", cost
            print "New Parameters ", newParams, " logLike: ", -optValue
            print "New Noise ", self.gp.noise
            outerIter +=1
            if 'paramTol' in self.outerLoopConvergenceCriteria.keys():
                diff = 0.0
                for keys in oldParams.keys():
                    diff = diff + (oldParams[keys]-newParams[keys])**2.0
                print "paramTol ", self.outerLoopConvergenceCriteria['paramTol']
                print "Converged Indicator ", diff, cost
                if diff < self.outerLoopConvergenceCriteria['paramTol'] and \
                        cost < 1e-4:
                    outerConverged = 1
            elif 'numIters' in self.outerLoopConvergenceCriteria.keys():
                if outerIter >= self.outerLoopConvergenceCriteria['numIters']:
                    outerConverged = 1
            if outerIter > MAXITER:
                outerConverged = 1

            oldParams = copy.copy(newParams)
             
        if diagnostics == 1:
            return nodesList, costList, logLike, hyperParam


