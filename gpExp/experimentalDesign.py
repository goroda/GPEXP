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


import numpy as np
import scipy.optimize as optimize
import scipy.stats as spstats
import nlopt as nlopt
import sys
import parallel_utilities
import gp_kernel_utilities
import copy
import itertools

class costFunctionBase(object):
    
    def __init__(self, nInputs, space):
        
        self.numInputs = nInputs
        self.space = space

class costFuncGPUCbound(costFunctionBase):

    def __init__(self, gaussianProcess, kappa, xTrain, yTrain, nInputs, space,  **kwargs):
        super(costFuncGPUCbound, self).__init__(nInputs, space)
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.kappa = kappa # kappa is tunable to balance exploit vs explore
        self.gaussianProcess = copy.copy(gaussianProcess)
        self.gaussianProcess.train(xTrain, yTrain)

    def evaluate(self, trainPoints):
        """ 
        Cost function for bayesian optimization- GP Upper Confidence Bound

        Parameters
        ----------
        inputPoints: ndarray 
            nPoints x dimension - array of points.
        
        Returns
        -------   
        cost : float
            Cost function value of experiment with inputPoints.
                    
        Notes
        -----    
        Snoek 2014
        
        """
        fBest = np.max(self.yTrain)
        newPoint = np.reshape(trainPoints[-1,:], (1,self.space.dimension))
        predMean, predvar = self.gaussianProcess.evaluate(newPoint,compvar=1)
        predstd = np.sqrt(predvar)
        cost = -float(predMean - self.kappa*predstd)
        return cost
 
class costFuncPI(costFunctionBase):

    def __init__(self, gaussianProcess, xTrain, yTrain, nInputs, space,  **kwargs):
        super(costFuncPI, self).__init__(nInputs, space)
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.gaussianProcess = copy.copy(gaussianProcess)
        self.gaussianProcess.train(xTrain, yTrain)

    def evaluate(self, trainPoints):
        """ 
        Cost function for bayesian optimization- probability of improvement

        Parameters
        ----------
        inputPoints: ndarray 
            nPoints x dimension - array of points.
        
        Returns
        -------   
        cost : float
            Cost function value of experiment with inputPoints.
                    
        Notes
        -----    
        Snoek 2014
        
        """
        fBest = np.max(self.yTrain)
        newPoint = np.reshape(trainPoints[-1,:], (1,self.space.dimension))
        predMean, predvar = self.gaussianProcess.evaluate(newPoint,compvar=1)
        predstd = np.sqrt(predvar)
        gamma = (fBest - predMean)/predstd
        phiGamma = spstats.norm.cdf(gamma)
        cost = -float(phiGamma)
        return cost
       
class costFuncEI(costFunctionBase):

    def __init__(self, gaussianProcess, xTrain, yTrain, nInputs, space,  **kwargs):
        super(costFuncEI, self).__init__(nInputs, space)
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.gaussianProcess = copy.copy(gaussianProcess)
        self.gaussianProcess.train(xTrain, yTrain)

    def evaluate(self, trainPoints):
        """ 
        Cost function for bayesian optimization- expected improvement

        Parameters
        ----------
        inputPoints: ndarray 
            nPoints x dimension - array of points.
        
        Returns
        -------   
        cost : float
            Cost function value of experiment with inputPoints.
                    
        Notes
        -----    
        Snoek 2014
        
        """
        fBest = np.max(self.yTrain)
        newPoint = np.reshape(trainPoints[-1,:], (1,self.space.dimension))
        predMean, predvar = self.gaussianProcess.evaluate(newPoint,compvar=1)
        predstd = np.sqrt(predvar)
        gamma = (fBest - predMean)/predstd
        phiGamma = spstats.norm.cdf(gamma)
        probGamma = spstats.norm.pdf(gamma)
        cost = -predstd*(gamma*phiGamma + probGamma)
        return cost[0]
        
class costFunctionGP_IVAR(costFunctionBase):
     
    def __init__(self, gaussianProcess, nInputs, space, version=1, **kwargs):
        super(costFunctionGP_IVAR, self).__init__(nInputs, space)
        self.gaussianProcess = copy.copy(gaussianProcess)
        self.version = version
        if self.version == 1:
            if 'mcPoints' in kwargs:
                #self.nMC = kwargs['nMC']
                self.mcPoints = kwargs['mcPoints']
                self.nMC = len(self.mcPoints)
            else:
                self.nMC = 10000
                #self.nMC = 1000
                self.mcPoints = space.sample((self.nMC, space.dimension)) 
            #self.mcPointWeights = space.probDensity(self.mcPoints)
        else:
            if space.noiseFunc is not None:
                raise TypeError, "cost function with basis functions not implemented when a variable noise function is specified"
    def evaluate(self, inputPoints):
        """ 
        Cost function for experimental design - minimizes integrated variance

        Parameters
        ----------
        inputPoints: ndarray 
            nPoints x dimension - array of points.
        
        mcPoints : (nMC, dim) ndarray
                   array of MC points to be used for computation in version 1
        Returns
        -------   
        cost : float
            Cost function value of experiment with inputPoints.
                    
        Notes
        -----    
        For the algorithm check out my paper
        
        """
        
        assert inputPoints.shape == (self.numInputs, self.space.dimension), \
                ("inputPoints are the wrong size: ", inputPoints.shape)
        
 
        if self.version == 1:
            if self.space.noiseFunc is None:
                self.gaussianProcess.addNodesAndComputeCovariance(inputPoints)      
                varMC = self.gaussianProcess.evaluateVariance(self.mcPoints)
                cost = 1.0/float(self.nMC)*np.sum(varMC)         
            else:
                addNoise = self.space.noiseFunc(inputPoints)
                self.gaussianProcess.addNodesAndComputeCovariance(inputPoints, addNoise)    
                varMC = self.gaussianProcess.evaluateVariance(self.mcPoints)
                cost = 1.0/float(self.nMC)*np.sum(varMC)         

            #print "cost ", cost                             
            return np.abs(cost)        
        
        elif self.version == 0:
            #compute inverse covariance
            self.gaussianProcess.addNodesAndComputeCovariance(inputPoints)
            lenEigs = len(self.gaussianProcess.kernel.eigenvalues)
            
            def forloop(ii, fn=None):
                out = np.zeros((np.shape(ii)[0]))
                ii = ii.reshape((len(ii)))
                overallIter = 0
                for jj in ii:
                    evals = self.gaussianProcess.kernel.evaluateKernelBasis(jj,inputPoints)
                    out[overallIter] = self.gaussianProcess.kernel.eigenvalues[jj]*(1.0 - \
                        self.gaussianProcess.kernel.eigenvalues[jj]*  \
                        np.dot(evals.T,np.dot(self.gaussianProcess.precisionMatrix, evals)))
                    overallIter = overallIter + 1
                return out
            
            p = np.arange(lenEigs).reshape((lenEigs,1))
            out = parallel_utilities.parallelizeMcForLoop(forloop, None, p)   
            cost = np.sum(out)
            #cost = 0.0 
            #for jj in xrange(lenEigs):
            #    evals = self.gaussianProcess.kernel.evaluateKernelBasis(jj,inputPoints)
            #    cost = cost + self.gaussianProcess.kernel.eigenvalues[jj]*(1.0 - \
            #            self.gaussianProcess.kernel.eigenvalues[jj]*  \
            #            np.dot(evals.T,np.dot(self.gaussianProcess.precisionMatrix, evals)))
        
            return np.abs(cost) #**2.0
    
    def derivative(self, inputPoints):
        """ 
        Cost function derivative for experimental design 
        minimizes cost function in terms of explicit basis

        Parameters
        ----------
        inputPoints: ndarray 
            nPoints x dimension - array of points.
            
        Returns
        -------   
        deriv : 1darray
            derivative of costFunctions (nPoints x dimension) x 1
                    
        Notes
        -----    
        For the algorithm check out my paper
        
        """
        if self.version == 1:
            if self.space.noiseFunc is None:
                self.gaussianProcess.addNodesAndComputeCovariance(inputPoints)
                out = self.gaussianProcess.evaluateVarianceDerivative(self.mcPoints)
                out = np.sum(out,axis=1)/float(self.nMC)
            else:
                addNoise = self.space.noiseFunc(inputPoints)
                self.gaussianProcess.addNodesAndComputeCovariance(inputPoints, noiseIn=addNoise)
                out = self.gaussianProcess.evaluateVarianceDerivative(self.mcPoints, noiseFunc=self.space.noiseFunc)
                out = np.sum(out, axis=1)/float(self.nMC)

            return out
        elif self.version == 0:
            #compute inverse covariance
            self.gaussianProcess.addNodesAndComputeCovariance(inputPoints)
            lenEigs = len(self.gaussianProcess.kernel.eigenvalues)
  
            nPoints = inputPoints.shape[0]
            
            #compute dphi_ii(x_i)
            derivPhiMatrix = np.zeros((nPoints, lenEigs ))
            phiMatrix = np.zeros((nPoints, lenEigs))
            for ii in xrange(lenEigs):
    #            print "shape of interest ", np.shape(self.kernel.evaluateKernelBasisDeriv(ii, inputPoints))
                derivPhiMatrix[:,ii] = \
                    np.reshape(self.gaussianProcess.kernel.evaluateKernelBasisDeriv(ii, inputPoints), (nPoints))
                phiMatrix[:,ii] = \
                    self.gaussianProcess.kernel.evaluateKernelBasis(ii, inputPoints)
            
            dMat = np.dot(self.gaussianProcess.precisionMatrix, phiMatrix)
            out = np.zeros((nPoints*self.space.dimension))
            
            def computeForEachmm(ii, fn):
                out = np.zeros((np.shape(ii)[0]))
                ii = ii.reshape((len(ii)))
                overallIter = 0
                for itera in ii:
                    out[overallIter] = 0.0;
                    c = np.zeros((nPoints))
                    c[itera] = 1.0
                    Amat = np.zeros((nPoints,nPoints))
                    for kk in xrange(lenEigs):
                        cphi = np.outer(c, phiMatrix[:,kk].T)
                        Amat = Amat + self.gaussianProcess.kernel.eigenvalues[kk] * derivPhiMatrix[itera, kk]*(cphi + cphi.T)
                    for jj in xrange(lenEigs):
                        out[overallIter] = out[overallIter] +  -self.gaussianProcess.kernel.eigenvalues[jj]**2.0 * ( np.dot(-dMat[:,jj].T, np.dot(Amat, dMat[:,jj]) ) + 2.0 * derivPhiMatrix[itera,jj] * dMat[itera,jj])   
                    overallIter = overallIter+1
                #print "out ", out
                return out
           
            p = np.arange(nPoints).reshape((out.shape[0],1))
            out = parallel_utilities.parallelizeMcForLoop(computeForEachmm, None, p)
            
            return out
 
class costFunctionGP_MI(costFunctionBase):

    def __init__(self, gaussianProcess, nInputs, space, square=False):
        super(costFunctionGP_MI, self).__init__(nInputs, space)
        self.gaussianProcess = gaussianProcess
        if space.dimension == 2 and square==True:
            x = np.linspace(-1,1,10)
            self.nMC = len(x)*len(x)
            self.mcPoints = np.array(list(itertools.product(x,x)))
        else:
            self.nMC = 200
            self.mcPoints = space.sample((self.nMC, space.dimension))
        
        self.gaussianProcess.addNodesAndComputeCovariance(self.mcPoints)
        self.cov = copy.copy(self.gaussianProcess.covarianceMatrix)
        self.invcov = copy.copy(self.gaussianProcess.precisionMatrix)

    def evaluate(self, index, indexAdded): 
        """
        index: int
               index of MC points whihc one is considering to add
        indexAdded: list of indices
                    indices already added 
        """
        point = self.mcPoints[index,:].reshape((1,self.space.dimension))
        var = self.gaussianProcess.kernel.evaluate(point, point)
        pointsAdded = self.mcPoints[indexAdded,:].reshape((len(indexAdded), self.space.dimension))

        
        #Numerator
        numpointVsAdded = self.gaussianProcess.kernel.evaluate(pointsAdded, point)
        covNum = gp_kernel_utilities.calculateCovarianceMatrix(self.gaussianProcess.kernel, 
                        pointsAdded, self.gaussianProcess.noise)
        invCovNum = np.linalg.pinv(covNum)
        numerator = var - np.dot(numpointVsAdded.T, np.dot(invCovNum, numpointVsAdded))

        #Compute variance of all locations other than those provided in indexAdded and index
        leftIndices = np.setdiff1d(np.arange(self.nMC),indexAdded)
        leftIndices = np.setdiff1d(leftIndices, [index])
        pointsLeft = self.mcPoints[leftIndices,:]

        #Denominator
        numpointVsNotAdded = self.gaussianProcess.kernel.evaluate(pointsLeft, point)
        covDen = gp_kernel_utilities.calculateCovarianceMatrix(self.gaussianProcess.kernel, 
                        pointsLeft, self.gaussianProcess.noise)
        invCovDen = np.linalg.pinv(covDen)
        denominator = var - np.dot(numpointVsNotAdded.T, np.dot(invCovDen, numpointVsNotAdded))
        
        
        out = numerator/denominator
        return out

class costFunctionGP_VAR(costFunctionBase):

    def __init__(self, gaussianProcess, nInputs, space):
        super(costFunctionGP_VAR, self).__init__(nInputs, space)
        self.gaussianProcess = gaussianProcess
        
    def evaluate(self, ):
        pass
     
class ExperimentalDesign(object):
    """ Experimental Design Class """
    nMCpoints = 10000
    def __init__(self, costFunction, nPoints, nDims, **kwargs):
        """ Initializes the experimental design class 
        
            input:
               costFunction
               boundFunction
        """
        
        self.costFunction = costFunction
        self.nPoints = nPoints
        self.nDims = nDims
        super(ExperimentalDesign,self).__init__()
        
    def boundsFunction(self, optPoints):
        """ 
        This function outputs probability of each optPoints. In the case that the probability
        is zero it outputs -1e5. It is intended for use in the optimization routine 
         
        Parameters
        ----------    
        optPoints : 1darray 
            (nPoints x dimension) x 1 - array of points
        
        Returns
        -------      
        out : 1darray
            Array of probability values. If probability is == 0 returns -1e5
         
        Notes
        -----             
        
        """
        if len(np.shape(optPoints)) == 1:
            optPoints = np.reshape(optPoints, (len(optPoints)/self.nDims, self.nDims))
            
        out = self.costFunction.space.probDensity(optPoints)
        #out[out< 1e-20] = -1e0 
        out[out == 0.0] = -1e0 
                 
        if np.min(out) < 0.0:
            return -1e0
        else:
            return 1e0
        
        return out
    
#class ExperimentalDesignGreedy(ExperimentalDesign):
#    pass

class ExperimentalDesignDerivative(ExperimentalDesign):
    
    def __init__(self, costFunction, nPoints, nDims):
        self.addObj = lambda x: 0
        self.addGrad = lambda x: 0
        super(ExperimentalDesignDerivative, self).__init__(costFunction, nPoints, nDims)
    
    def addPenaltyToObjective(self, addObj, addGrad):
        self.addObj = addObj
        self.addGrad = addGrad

    def objFunc(self, in1, gradIn):
        #print "begin Obj Exp Design ", self.costFunction.gaussianProcess.kernel.hyperParam
        #print "Length of grad in, ", len(gradIn)
        in0 = np.reshape(in1, (len(in1)/self.nDims, self.nDims))
        n = None
        if gradIn.size > 0:
            gradIn[:] = self.costFunction.derivative(in0)
            #print gradIn
            #print "norm of grad ", np.linalg.norm(gradIn)
            gradIn[:] = gradIn[:]# + self.addGrad(in0)
            #n = np.linalg.norm(gradIn)
            #if n > 10:
            #    print "potential Error "

        out = self.costFunction.evaluate(in0) - 10.0*np.min(np.array([self.boundsFunction(in0), 0.0]))
        #print "bounds ", 
        #print "in0 ", in0.shape
        #add = self.addObj(in0)
        #print "adding ", add
        #out = out + add
        sys.stdout.write("\r Optimization (Cost = %s, ||g||= %s) OK " % (str(out), str(n)) )
        sys.stdout.flush()
        #print "end obj exp design", out, n
        return out                       
    
    def constraint(self, in0, gradIn):
        if gradIn.size > 0:
            gradIn[:] = -optimize.approx_fprime(in0, self.boundsFunction, 1e-8)
            #print "norm of constraint grad ", np.linalg.norm(gradIn)
        out = -self.boundsFunction(in0)
        #print "Is out of bounds ", out>0
        return out
    
    def beginWithVarGreedy(self, nodesKeep=None, lbounds = [], rbounds = []):
        """
        Initialize experimental design with greedy entropy nodes

        """ 
        
        kTemp = copy.copy(self.costFunction.gaussianProcess.kernel)
        if nodesKeep is not None:
            mcPoints = np.concatenate((nodesKeep, self.costFunction.mcPoints), axis=0)
            indKeep = np.arange(len(nodesKeep)).tolist()
        else:
            try:
                mcPoints = self.costFunction.mcPoints[:]
            except AttributeError: #no mcPoints were given
                nMC = 1000
                mcPoints = self.costFunction.space.sample((nMC, self.costFunction.space.dimension)) 
            indKeep = []
        
        #print "begin var greedy"
        startVals = performGreedyVarExperimentalDesign(kTemp, mcPoints, 
                        self.nPoints, self.nDims, indKeepStart=indKeep)
        
        #print "cost Of Start ", self.costFunction.evaluate(startVals)
        endVals = self.begin([startVals], lbounds, rbounds)
        #print "cost Of end", self.costFunction.evaluate(endVals)
        return endVals

    def begin(self, startValues, lbounds = [], rbounds = []):
        """ 
        Performs experimental design by minimizing costFunction
        
        Parameters
        ----------      
        startValuesIn : ndarray
            Starting values for experimental design.
        
        Returns
        -------
        endVals : ndarray
            Ending values for experimental design.
            
        Notes
        -----   

        """

        #local_opt = nlopt.opt(nlopt.LD_MMA, len(startValues[0])*self.nDims)
        local_opt = nlopt.opt(nlopt.LD_SLSQP, len(startValues[0])*self.nDims)
        #local_opt = nlopt.opt(nlopt.LD_LBFGS, len(startValues[0])*self.nDims)
        #local_opt = nlopt.opt(nlopt.LD_TNEWTON_PRECOND_RESTART, len(startValues[0])*self.nDims)
        local_opt.set_ftol_rel(1e-6)
        local_opt.set_ftol_abs(1e-6)
        local_opt.set_xtol_rel(1e-6)

        #local_opt.set_xtol_rel(1e-12)
        #local_opt.set_ftol_rel(1e-12)

        #print "HERE"
        #local_opt.set_maxtime(50); #seconds
        #local_opt.set_maxeval(20);
        #local_opt.add_inequality_constraint(constraint, 0.0)
        
        #opt = local_opt
        if len(lbounds)==0:
            local_opt.set_lower_bounds(-100.0*np.ones((len(startValues[0])*self.nDims)))
            local_opt.set_upper_bounds(100.0* np.ones((len(startValues[0])*self.nDims)))
        else:
            local_opt.set_lower_bounds(lbounds)
            local_opt.set_upper_bounds(rbounds)
            
        #local_opt.add_inequality_constraint(self.constraint, 0.0)
        sol = []
        obj = np.zeros((len(startValues)))
        optResults = np.zeros((len(startValues)))
        #print "begin optimization of designs"
        for ii in xrange(len(startValues)):
            opt = copy.copy(local_opt)
            opt.set_min_objective(self.objFunc)

            #print "Start objective "#, self.objFunc(startValues[ii], np.zeros((0)))   
            #print startValues[ii]   
            pts = opt.optimize(startValues[ii].reshape((len(startValues[ii])*self.nDims)))
            #print "optimization of designs successful "
            optResults[ii] = opt.last_optimize_result()
            sol.append(pts)
            #print "\nEnd objective\n"#, self.objFunc(sol[ii], np.zeros((0)))
            #obj[ii] = self.objFunc(sol[ii], np.zeros((0))) 
            obj[ii] = opt.last_optimum_value()
            #print "\nEnd objective ", obj[-1]
        
        indBest = np.argmin(obj)    
        #print sol
        endVals = np.reshape(sol[indBest], (len(sol[indBest])/self.nDims, self.nDims))
        finalGrad = self.costFunction.derivative(endVals)
        #print "IVAR at minimum  ", obj[indBest], " ||dIVAR/dPts|| at min is ", \
        #            np.linalg.norm(finalGrad)
        #print "IVAR optimizer Result ", optResults[indBest]
                                
        
        return endVals
    
class ExperimentalDesignNoDerivative(ExperimentalDesign):
    
    def __init__(self, costFunction, nPoints, nDims):
        super(ExperimentalDesignNoDerivative, self).__init__(costFunction, nPoints, nDims)
    
    def objFunc(self, in0, gradIn):
        #print "Length of grad in, ", len(gradIn)
        
        in0 = np.reshape(in0, (len(in0)/self.nDims, self.nDims))
        if gradIn.size > 0:
            assert 1==0, "Trying to evaluate derivatives in the NoDerivative Exp design"
            gradIn[:] = []
            print "norm of grad ", np.linalg.norm(gradIn)
        
        out = self.costFunction.evaluate(in0) - 10.0*np.min(np.array([self.boundsFunction(in0), 0.0]))
        #print "evaluated", out
        #print "bounds ", 
        sys.stdout.write("\r Optimization (Cost): (%s) " % str(out))
        sys.stdout.flush()
        return out
    
    def constraint(self, in0, gradIn):
        if gradIn.size > 0:
            gradIn[:] = -optimize.approx_fprime(in0, self.boundsFunction, 1e-10)
            #print "norm of constraint grad ", np.linalg.norm(gradIn)
        return -self.boundsFunction(in0)
    
    def beginWithVarGreedy(self, nodesKeep=None, lbounds = [], rbounds = []):
        """
        Initialize experimental design with greedy entropy nodes

        """ 
        
        kTemp = copy.copy(self.costFunction.gaussianProcess.kernel)
        if nodesKeep is not None:
            mcPoints = np.concatenate((nodesKeep, self.costFunction.mcPoints), axis=0)
            indKeep = np.arange(len(nodesKeep)).tolist()
        else:
            mcPoints = self.costFunction.mcPoints[:]
            indKeep = []
        
        print "startGreedy"
        startVals = performGreedyVarExperimentalDesign(kTemp, mcPoints, 
                        self.nPoints, self.nDims, indKeepStart=indKeep)
        
        print "cost Of Start ", self.costFunction.evaluate(startVals)
        endVals = self.begin([startVals], lbounds, rbounds)
        print "cost Of end", self.costFunction.evaluate(endVals)
        return endVals

                         
    def begin(self, startValues, lbounds = [], rbounds = [], ):   
        #print "here "
        local_opt = nlopt.opt(nlopt.LN_COBYLA, len(startValues[0])*self.nDims)
        #local_opt = nlopt.opt(nlopt.LN_NELDERMEAD, len(startValues[0])*self.nDims)
        local_opt.set_ftol_rel(1e-12)
        local_opt.set_xtol_rel(1e-12)
        local_opt.set_ftol_abs(1e-12)
        #local_opt.set_maxtime(120);
        local_opt.set_maxtime(40)
        local_opt.set_maxeval(200)
        
        if len(lbounds)==0:
            local_opt.set_lower_bounds(-100.0*np.ones((len(startValues[0])*self.nDims)))
            local_opt.set_upper_bounds(100.0* np.ones((len(startValues[0])*self.nDims)))
        else:
            local_opt.set_lower_bounds(lbounds)
            local_opt.set_upper_bounds(rbounds)

        #local_opt.add_inequality_constraint(self.constraint, 0.0)
        opt = copy.copy(local_opt)
        opt.set_min_objective(self.objFunc)
        sol = []
        obj = np.zeros((len(startValues)))
        for ii in xrange(len(startValues)):
            #print "Start objective ", #self.objFunc(startValues[ii].reshape((len(startValues[ii])*self.nDims)))   
            #print startValues[ii]   
            pts = startValues[ii].reshape((len(startValues[ii])*self.nDims))
            #print "pts ", pts
            sol.append(opt.optimize(pts))
            #print "End objective\n"#, self.objFunc(sol[ii], np.zeros((0)))
            obj[ii] = self.objFunc(sol[ii], np.zeros((0))) 
            #print "End objective ", obj[-1]
        
        indBest = np.argmin(obj)    
        #print sol
        endVals = np.reshape(sol[indBest], (len(sol[indBest])/self.nDims, self.nDims))
       
        return endVals        


class performExpDesignWithHypContinuation(object):

    def __init__(self, experiment, hypList):
        super(performExpDesignWithHypContinuation, self).__init__()
        self.experiment = copy.copy(experiment)
        self.hypList = hypList
        self.origHyperParams = copy.copy(experiment.costFunction.gaussianProcess.kernel.hyperParam)
        
    def begin(self, startValues, lbounds=[], rbounds=[], **kwargs):

        saveList = None
        if 'returnAll' in kwargs:
            saveList = []
            for pts in startValues:
                saveList.append([100000, pts])

        endValsList = startValues # keeps track of all ending points
        for hyperParams in self.hypList:

            newList = []
            currCost = []
            for vals in endValsList:
                
                self.experiment.costFunction.gaussianProcess.updateKernelParams(hyperParams)
                endVals = self.experiment.begin(startValues, lbounds, rbounds)
                #reset Parameters to original
                self.experiment.costFunction.gaussianProcess.updateKernelParams(self.origHyperParams)
                currCostv = self.experiment.costFunction.evaluate(endVals)
                currCost.append(currCostv)
                newList.append(endVals[:])

            if saveList is not None:
                saveList.append(newList[np.argmin(np.array(currCost))])
                print "best Current cost ", np.min(currCost)

            endValsList = newList;
        
        costs = np.zeros((len(startValues)))
        ii = 0
        for vals in endValsList:
            costs[ii] = self.experiment.costFunction.evaluate(vals)
            ii = ii+1

        bestMin = np.argmin(costs)
        if saveList is not None:
            return endValsList[bestMin], saveList
        else:
            return endValsList[bestMin]


class performExpDesignWithContinuation(object):

    def __init__(self, noiseStart, experiment):
        super(performExpDesignWithContinuation, self).__init__()
        self.reduceFactor = 0.1
        self.noiseStart = noiseStart
        self.experiment = copy.copy(experiment)
        self.noiseEnd = self.experiment.costFunction.gaussianProcess.noise
    
    def begin3(self, startValues, lbounds = [], rbounds = [],**kwargs):
       
        # One step at a higher level and just evaluate at desired level
        print "Noise Start ", self.noiseStart       
        currentPenalty = self.noiseStart
        endValsList = startValues
        saveList = None
        if 'returnAll' in kwargs:
            saveList = []
            for pts in startValues:
                saveList.append([100000, pts])

        newList = []
        currCost = []
        for vals in endValsList:
            endVals = self.innerOpt(currentPenalty, [vals], lbounds, rbounds)
            currCost.append(self.experiment.costFunction.evaluate(endVals))
            newList.append(endVals[:])
        if saveList is not None:
            saveList.append([currentPenalty, newList[np.argmin(np.array(currCost))]])
        
        endValsList = newList;
        self.experiment.costFunction.gaussianProcess.noise = self.noiseEnd

        costs = np.zeros((len(startValues)))
        ii = 0
        for vals in endValsList:
            costs[ii] = self.experiment.costFunction.evaluate(vals)
            ii = ii+1

        bestMin = np.argmin(costs)
        if saveList is not None:
            return endValsList[bestMin], saveList
        else:
            return endValsList[bestMin]

    def begin2(self, startValues, lbounds = [], rbounds = [], **kwargs):

        currentPenalty = self.noiseStart
        endValsList = startValues
        saveList = None
        if 'returnAll' in kwargs:
            saveList = []
            for pts in startValues:
                saveList.append([100000, pts])

        while currentPenalty > 1e-8:
            newList = []
            currCost = []
            for vals in endValsList:
                endVals = self.innerOpt2(currentPenalty, [vals], lbounds, rbounds)
                currCost.append(self.experiment.costFunction.evaluate(endVals))
                newList.append(endVals[:])
            if saveList is not None:
                saveList.append([currentPenalty, newList[np.argmin(np.array(currCost))]])
            currentPenalty= currentPenalty*self.reduceFactor
            print "Current Penalty Level", currentPenalty
            #print "point and cost ", newList[np.argmin(np.array(currCost))], np.min(np.array(currCost))
            #print "\n"
            endValsList = newList;
        
        costs = np.zeros((len(startValues)))
        ii = 0
        for vals in endValsList:
            costs[ii] = self.experiment.costFunction.evaluate(vals)
            ii = ii+1

        bestMin = np.argmin(costs)
        if saveList is not None:
            return endValsList[bestMin], saveList
        else:
            return endValsList[bestMin]

    def begin(self, startValues, lbounds = [], rbounds = [], **kwargs):
        
        currentNoiseLevel = self.noiseStart
        endValsList = startValues
        saveList = None
        if 'returnAll' in kwargs:
            saveList = []
            for pts in startValues:
                saveList.append([100000, pts])

        while currentNoiseLevel > self.noiseEnd:
            newList = []
            currCost = []
            for vals in endValsList:
                endVals = self.innerOpt(currentNoiseLevel, [vals], lbounds, rbounds)
                currCost.append(self.experiment.costFunction.evaluate(endVals))
                newList.append(endVals[:])
            if saveList is not None:
                saveList.append([currentNoiseLevel, newList[np.argmin(np.array(currCost))]])
            currentNoiseLevel = currentNoiseLevel*self.reduceFactor
            print "Current Noise Level", self.experiment.costFunction.gaussianProcess.noise, self.noiseEnd
            #print "point and cost ", newList[np.argmin(np.array(currCost))], np.min(np.array(currCost))
            #print "\n"
            endValsList = newList;
        
        self.experiment.costFunction.gaussianProcess.noise = self.noiseEnd

        costs = np.zeros((len(startValues)))
        ii = 0
        for vals in endValsList:
            costs[ii] = self.experiment.costFunction.evaluate(vals)
            ii = ii+1

        bestMin = np.argmin(costs)
        if saveList is not None:
            return endValsList[bestMin], saveList
        else:
            return endValsList[bestMin]

    
    def innerOpt2(self, penaltyIn, startValues, lbounds = [], rbounds =[]):
        
        #ADD THE GRADIENT
        addObj = lambda in0: np.sum(penaltyIn*self.experiment.costFunction.gaussianProcess.evaluateVariance(in0), axis=0)
        addGrad = lambda in0: penaltyIn*self.experiment.costFunction.gaussianProcess.evaluateVarianceDerivWRTnewpt(in0)

        self.experiment.addPenaltyToObjective(addObj, addGrad)
        endVals = self.experiment.begin(startValues, lbounds, rbounds)
        return endVals

    def innerOpt(self, noiseIn, startValues, lbounds = [], rbounds = []):
        
        self.experiment.costFunction.gaussianProcess.noise = noiseIn
        endVals = self.experiment.begin(startValues, lbounds, rbounds)

        return endVals

class ExperimentalDesignGreedyWithNoDerivatives(ExperimentalDesignNoDerivative):
    
    def __init__(self, costFunction, nPoints, nPointsBatch, nDims, **kwargs):
        super(ExperimentalDesignGreedyWithNoDerivatives, self).__init__(costFunction, nPoints, nDims)
        self.nPointsBatch = nPointsBatch
        self.useContinuation = 0
        if "useCont" in kwargs:
            self.useContinuation = kwargs['useCont']
            
    
    def begin(self, startValues=np.array([])):
        nPointsAdded = len(startValues)
        if len(startValues>0):
            points = startValues.copy()
        else:
            points = np.zeros((0,self.nDims))
        
        overallPointsTrace = []
        tol = 1e-16
        err = 10000
        while (nPointsAdded < self.nPoints) and (err > tol):
            print "Number of points so far ", nPointsAdded
            nPointsAdded = nPointsAdded + self.nPointsBatch
            lbounds = -1.0*np.ones((nPointsAdded*self.nDims))
            rbounds = 1.0*np.ones((nPointsAdded*self.nDims))
           
            nPointsPrev = nPointsAdded - self.nPointsBatch
            lbounds[0:nPointsPrev*self.nDims] = points.reshape((nPointsPrev*self.nDims))
            rbounds[0:nPointsPrev*self.nDims] = points.reshape((nPointsPrev*self.nDims))
            #check several starting points
            startVals = []
            for ii in xrange(1):
                startVals.append(np.concatenate((points, 
                         self.costFunction.space.sample((self.nPointsBatch,self.nDims))),
                                                axis=0))
            
            currCost = costFunctionGP_IVAR(self.costFunction.gaussianProcess, 
                    nPointsAdded, self.costFunction.space, 
                    self.costFunction.version, nMC=self.costFunction.nMC, 
                    mcPoints=self.costFunction.mcPoints)
            expCurr = ExperimentalDesignNoDerivative(currCost, nPointsAdded, self.nDims)
            if self.useContinuation != 0:
                exp = performExpDesignWithContinuation(self.useContinuation, expCurr)
            
                points, pointsTrace = exp.begin(startVals, lbounds, rbounds, returnAll=1)
                overallPointsTrace.append(pointsTrace)
                #print "pointsTrace, ", pointsTrace
            else:
                points = expCurr.begin(startVals, lbounds, rbounds)
            err = currCost.evaluate(points)
            print "Current Error ", err
        
        if self.useContinuation != 0:
            return points, overallPointsTrace
        else:
            return points

class ExperimentalDesignGreedyWithDerivatives(ExperimentalDesignDerivative):
    
    def __init__(self, costFunction, nPoints, nPointsBatch, nDims, **kwargs):
        super(ExperimentalDesignGreedyWithDerivatives, self).__init__(costFunction, nPoints, nDims)
        self.nPointsBatch = nPointsBatch
        self.useContinuation = 0
        if "useCont" in kwargs:
            self.useContinuation = kwargs['useCont']
            if 'hypParamList' in kwargs:
                self.hypList = kwargs['hypParamList']
            
    
    def begin(self, startValues=np.array([])):
        nPointsAdded = len(startValues)
        if len(startValues>0):
            points = startValues.copy()
        else:
            points = np.zeros((0,self.nDims))
        
        overallPointsTrace = []
        tol = 1e-16
        err = 10000
        while (nPointsAdded < self.nPoints) and (err > tol):
            print "Number of points so far ", nPointsAdded
            nPointsAdded = nPointsAdded + self.nPointsBatch
            lbounds = -100.0*np.ones((nPointsAdded*self.nDims))
            rbounds = 100.0*np.ones((nPointsAdded*self.nDims))
           
            nPointsPrev = nPointsAdded - self.nPointsBatch
            lbounds[0:nPointsPrev*self.nDims] = points.reshape((nPointsPrev*self.nDims))
            rbounds[0:nPointsPrev*self.nDims] = points.reshape((nPointsPrev*self.nDims))
            #check several starting points
            startVals = []
            for ii in xrange(2):
                startVals.append(np.concatenate((points, 
                         self.costFunction.space.sample((self.nPointsBatch,self.nDims))),
                                                axis=0))
            
            currCost = costFunctionGP_IVAR(self.costFunction.gaussianProcess, 
                    nPointsAdded, self.costFunction.space, 
                    self.costFunction.version, nMC=self.costFunction.nMC, 
                    mcPoints=self.costFunction.mcPoints)
            expCurr = ExperimentalDesignDerivative(currCost, nPointsAdded, self.nDims)
            if self.useContinuation != 0:
                #exp = performExpDesignWithContinuation(self.useContinuation, expCurr)
                exp = performExpDesignWithHypContinuation(expCurr, self.hypList)
               
                #points, pointsTrace = exp.begin3(startVals, lbounds, rbounds, returnAll=1)
                points, pointsTrace = exp.begin(startVals, lbounds, rbounds, returnAll=1)
                #points, pointsTrace = exp.begin2(startVals, lbounds, rbounds, returnAll=1)
                overallPointsTrace.append(pointsTrace)
                #print "pointsTrace, ", pointsTrace
            else:
                #points = expCurr.begin(startVals, lbounds, rbounds)
                points = expCurr.beginWithVarGreedy(nodesKeep=points, lbounds=lbounds, rbounds=rbounds)
            err = currCost.evaluate(points)
            print "Current Error ", err
        
        if self.useContinuation != 0:
            return points, overallPointsTrace
        else:
            return points

def performGreedyMIExperimentalDesign(costFuncMI, nPoints, start=0):

    """
    Performs greedy experimental design by minimizing mutual information by choosing points among
    all available MC points obtained from kernel
 
    Parameters
    ----------
    nPoints : int
        number of points to start with
 
    Returns
    -------
    endVals : ndarray
        final point set
 
    """
     
    indKeep = [start]
    indexOptions = np.setdiff1d(np.arange(costFuncMI.nMC), indKeep)
    for ii in xrange(len(indKeep),nPoints):
        
        out = np.zeros((len(indexOptions)))
        for jj, ind in enumerate(indexOptions):
            out[jj] = costFuncMI.evaluate(ind, indKeep)
        
        #Index to add
        indNew = indexOptions[np.argmax(out)]
        indKeep.append(indNew)
        #Remove index from options
        indexOptions = np.setdiff1d(indexOptions, indNew) #remove this index

    return costFuncMI.mcPoints[indKeep,:]

def performGreedyVarExperimentalDesign(kernel, mcPoints, nPoints, dimension, weights=None, indKeepStart=[]):
    """
    Performs greedy experimental design by minimizing variance by choosing points among
    all available MC points obtained from kernel
 
    Parameters
    ----------
    nPoints : int
        number of points to start with
 
    Returns
    -------
    endVals : ndarray
        final point set
 
    """
       
    pointsHave = 0
    if indKeepStart == []:
        indKeep = []
    else:
        indKeep = indKeepStart
        pointsHave = len(indKeep)

    while pointsHave < nPoints:
 
        ptsChooseFrom = mcPoints.copy()
        if pointsHave == 0:
               
            k = kernel.evaluate(ptsChooseFrom, ptsChooseFrom)
            if weights is not None:
                k = k * weights
            indKeep.append(np.argmax(k))
            #ptsChooseFrom = np.delete(ptsChooseFrom, (indKeep[-1]), axis=0)
 
        else:
            
            covMat = gp_kernel_utilities.calculateCovarianceMatrix(kernel, mcPoints[indKeep,:])
            invMat = np.linalg.pinv(covMat)
            kernVals = np.zeros((pointsHave, len(ptsChooseFrom)))
               
            for ii in xrange(pointsHave):
                pt = mcPoints[indKeep[ii],:].reshape((1,dimension))
                kernVals[ii,:] = kernel.evaluate(ptsChooseFrom, pt)
               
            k = np.zeros((len(ptsChooseFrom)))
            for ii in xrange(len(ptsChooseFrom)):
                pt = ptsChooseFrom[ii].reshape((1,dimension))
                k[ii] = kernel.evaluate(pt, pt) - \
                        np.dot(kernVals[:,ii].T, np.dot(invMat, kernVals[:,ii]))

            if weights is not None:
                k = k * weights
            indKeep.append(np.argmax(k))
            #print "indKeep ", indKeep
            #ptsChooseFrom = np.delete(ptsChooseFrom, (indKeep[-1]), axis=0)
 
        pointsHave = pointsHave+1
        #print indKeep[-1]
        #print self.kernel.kernelMCpoints[indKeep[-1],:]
 
    return mcPoints[indKeep,:]

