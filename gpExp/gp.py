#Copyright (c) 2013-2015, Massachusetts Institute of Technology
#
#This file is part of GPEXP:
#Author: Alex Gorodetsky goroda@mit.edu
#
#GPEXP is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 2 of the License, or
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

try:
    import nlopt as nlopt
    NLOPT = True
except ImportError:
    NLOPT = False
 
#NLOPT = False
if NLOPT is False:
    try:
        from  scipy.optimize import fmin_slsqp as slsqp
        from  scipy.optimize import fmin_cobyla as cobyla
        from  scipy.optimize import fmin_l_bfgs_b as bfgs
    except ImportError:
        print "Warning: no optimization package found!"

import multiprocessing as mp
from gp_kernel_utilities import calculateCovarianceMatrix
import parallel_utilities
import gp_kernel_utilities
import copy
import itertools

class GP(object):
    """ This is the GP class with prior mean is zero """
    coeff = None
    precisionMatrix = None
    covarianceMatrix = None
    noise = None
    pts = None
    FITC = None
    fitcnodes = None

    def __init__(self, kernel_in, noiseIn, **kwargs):
        """ Initialize the GP class """
        try:
            self.kernel = copy.deepcopy(kernel_in) # this is a kernel class
        except:
            print "warning "
            self.kernel = copy.copy(kernel_in)

        self.noise = noiseIn #std of noise
        if 'FITC' in kwargs:
            self.FITC = kwargs['FITC']
        super(GP,self).__init__()
   
    def gpPriorMean(self, pts):
        return np.zeros((pts.shape[0]))
     
    def train(self, pts, evalsIn, noiseIn=None):
        """
        Computes GP coefficients
        
        Parameters
        ----------
        pts     : ndarray
                  Array of training points
        evalsIn : ndarray
                  Array of functionValues
        noiseIn : None if use default noise (self.noise)
                  ndarray if use separate noise for each function value
        Returns
        -------
        Nothing - stores coefficients in class
        
        Notes
        -----
        """
        #remove mean function
        assert len(evalsIn.shape) == 1, "evaluations must be an (N,) array for training GP"
        
        evals = evalsIn-self.gpPriorMean(pts)
        self.addNodesAndComputeCovariance(pts,noiseIn)
        self.fVals = evals.copy()       
        self.coeff = np.dot(self.precisionMatrix, evals)
     
    def evaluate(self, newpt, compvar=0):
        """
        Evaluate GP mean and variance
        
        Parameters
        ----------
        newpt : ndarray
            locations at which to evaluate GP
        compvar : int (default = 0)
            if 0 then dont compute variance
            if 1 then compute variance
            if 2 then compute covariance
            
        Returns
        -------
        out : 1darray
            values of mean function
        var : 1darray
            values of variance

        Notes
        -----
        """
        
        assert newpt.shape[1] == self.kernel.dimension, "evaluation points for GP is incorrect shape"
        
        numNewPoints = newpt.shape[0]
        numTrainPoints = self.pts.shape[0]
        
        kernelvals = np.zeros((numNewPoints, numTrainPoints))
        for jj in xrange(numTrainPoints):
            pt = np.reshape(self.pts[jj,:], (1,self.kernel.dimension))
            kernelvals[:,jj] = self.kernel.evaluate(newpt, pt)
        
        out = np.dot(kernelvals, self.coeff) + self.gpPriorMean(newpt)
            
        if compvar == 1:
            var_newpt = self.kernel.evaluate(newpt, newpt)
            var = np.zeros((numNewPoints))
            for jj in xrange(numNewPoints):
                var[jj] = var_newpt[jj] - \
                    np.dot(kernelvals[jj,:], np.dot(self.precisionMatrix, kernelvals[jj,:].T))
            return out, np.abs(var)
        elif compvar == 2:
            covar_newpt = np.zeros((numNewPoints, numNewPoints))
            for jj in xrange(numNewPoints):
                pt = np.reshape(newpt[jj,:], (1,self.kernel.dimension))
                covar_newpt[:,jj] = self.kernel.evaluate(newpt, pt)
            covar = covar_newpt - np.dot(kernelvals, np.dot(self.precisionMatrix, kernelvals.T))
            return out, covar
        else:
            return out
        
    def addNodesAndComputeCovariance(self, nodes, noiseIn=None):
        """ 
        Add nodes and compute covariance 
        
        Parameters
        ----------
        nodes : ndarray
        
        noiseIn : None if use default noise (self.noise)
                  ndarray if use separate noise for each function value

        Returns
        -------
        Nothing
        
        Note
        ----
        This function is useful if one doesnt want to train right away
        or if one just wants to evaluate the posterior variance later
        """
        if self.FITC == None:
            if noiseIn is None:
                self.covarianceMatrix = calculateCovarianceMatrix(self.kernel, nodes, self.noise)
            else:
                self.covarianceMatrix = calculateCovarianceMatrix(self.kernel, nodes, noiseIn)
            self.precisionMatrix = np.linalg.pinv(self.covarianceMatrix)
        else:
            if noiseIn is None:
                
                nNodes = len(nodes)
                nu = int(np.floor(nNodes*self.FITC))
                if self.fitcnodes == None:
                    indu = np.random.permutation(nNodes)[0:nu]
                    snodes = nodes[indu]
                    self.fitcnodes = np.array(snodes, dtype=float)
                else:
                    snodes = self.fitcnodes
                Quu = calculateCovarianceMatrix(self.kernel, snodes, self.noise)
                invQuu = np.linalg.pinv(Quu)
                kernelvals = np.zeros((nu, nNodes))
                for jj in xrange(nu):
                    pt = np.reshape(snodes[jj], (1,self.kernel.dimension))
                    kernelvals[jj,:] = self.kernel.evaluate(pt, nodes)
                Q = np.dot(kernelvals.T, np.dot(invQuu, kernelvals))
                K = calculateCovarianceMatrix(self.kernel, nodes, self.noise)
                g = np.diag(K-Q)
                G = np.diag(g)
                self.covarianceMatrix = Q+G
                invG = np.diag(1.0/(g+1e-12))
                self.precisionMatrix = invG - np.dot(invG, np.dot(kernelvals.T, 
                                        np.dot(np.linalg.inv(Quu+ np.dot(kernelvals,
                                                np.dot(invG, kernelvals.T))), np.dot(
                                                kernelvals, invG))))
            else:
                print "NOT IMPLEMENTED YET"
        self.pts = nodes.copy()
          
    def evaluateVariance(self, newpt, parallel=1):
        
        """
        Evaluate GP posterior variance
        
        Parameters
        ----------
        newpt : ndarray
            locations at which to evaluate GP
        compvar : int (default = 0)
            if 0 then dont compute variance
            if 1 then compute variance
            
        Returns
        -------
        out : 1darray
            values of mean function
        var : 1darray
            values of variance

        Notes
        -----
        """
        
        assert self.pts is not None, "must specify training points before running this" 
        assert newpt.shape[1] == self.kernel.dimension, "evaluation points for GP is incorrect shape"
        
        numNewPoints = newpt.shape[0]
        numTrainPoints = self.pts.shape[0]
        #print "Evaluate Variance "
        nThreshForParallel = 500001 # at this level both parallel and serial seem to perform similarly
        if numNewPoints < nThreshForParallel or parallel==0:
            kernelvals = np.zeros((numNewPoints, numTrainPoints))
            for jj in xrange(numTrainPoints):
                pt = np.reshape(self.pts[jj,:], (1,self.kernel.dimension))
                kernelvals[:,jj] = self.kernel.evaluate(newpt, pt)
                   
            var_newpt = self.kernel.evaluate(newpt, newpt)
            var = np.zeros((numNewPoints))
            for jj in xrange(numNewPoints):
                var[jj] = var_newpt[jj] - \
                    np.dot(kernelvals[jj,:], np.dot(self.precisionMatrix, kernelvals[jj,:].T))
            return var
        else:
            var =  parallel_utilities.parallelizeMcForLoop(self.evaluateVariance, 0, newpt)
            return var
    
    def evaluateVarianceDerivWRTnewpt(self, newpt):
        """ evaluate derivative of posterior variance at newpt """

        assert self.pts is not None, "must specify training points before running this" 
        assert newpt.shape[1] == self.kernel.dimension, "evaluation points for GP is incorrect shape"
        
        out = np.zeros((newpt.shape))
        derivs = np.zeros((newpt.shape[0], newpt.shape[1], len(self.pts)))
        evals = np.zeros((newpt.shape[0], self.pts.shape[0]))
        for ii, pt in enumerate(self.pts):
            p = pt.reshape((1,self.kernel.dimension))
            derivs[:,:,ii] = self.kernel.derivative(newpt,p)
            evals[:,ii] = self.kernel.evaluate(p, newpt)
        
        evalSigma = np.dot(self.precisionMatrix, evals.T) #N_{GP} X N
        for ii, pt in enumerate(newpt):
            out[ii,:] =  -2.0 *np.dot(derivs[ii,:,:], evalSigma[:,ii])

        outout = out.reshape((np.prod(newpt.shape)))
        return outout
        
    def evaluateVarianceDerivative(self, newpt, noiseFunc=None):
        """ 
        evaluate posterior variance derivative wrt training points
        
        input:
            newpt : ndarray (NN x dim)
                    location at which variance is evaluated

        output:
            out : ndarray (ND,NN)
                  i.e. if k < N l < D then
                  out[k*l, jj] = dC(newpt[jj,:], newpt[jj,:]) / d self.pts[k,l],
                  
        """
        # Returns len(self.pts)*dim \times newpt array
        #NOTE there is no loop over newpt!

        assert self.pts is not None, "must specify training points before running this" 
        assert newpt.shape[1] == self.kernel.dimension, "evaluation points for GP is incorrect shape"

        outout = np.zeros((len(self.pts)*self.kernel.dimension,len(newpt)))
        derivCovTotal = np.zeros((len(self.pts),len(self.pts),self.kernel.dimension))
        totEvals = np.zeros((len(newpt), len(self.pts)))
        
        #preprocessing
        derivTotal = []
        for zz in xrange(len(self.pts)):
            
            p = self.pts[zz,:].reshape((1,self.kernel.dimension))
            indUse = np.array([np.linalg.norm(pp - p) < 1e-10 for pp in self.pts])
            
            derivCovTotal[zz,:,:] = self.kernel.derivative(self.pts, p)
            totEvals[:,zz] = self.kernel.evaluate(p, newpt)
            derivTotal.append(-self.kernel.derivative(newpt, p))
            if noiseFunc is not None:
                indUse = np.tile(indUse.reshape((len(self.pts),1)), self.kernel.dimension)
                derivCovTotal[zz,:,:] += indUse*noiseFunc.deriv(self.pts)
                if np.linalg.norm(p-newpt) < 1e-10:
                   totEvals[:,zz] += noiseFunc(p)
                   derivTotal[-1] -= noiseFunc.deriv(p)

        evalsDotPrec = np.dot(totEvals, self.precisionMatrix)
        
        #form together the derivative
        out2 = np.zeros((len(self.pts)*self.kernel.dimension,len(newpt)))
        out1 = np.zeros((len(self.pts)*self.kernel.dimension,len(newpt)))
        for jj in xrange(len(self.pts)):
            for kk in xrange(self.kernel.dimension):
                #temp = np.zeros((len(self.pts), len(newpt)))
                #temp[jj,:] = derivTotal[jj][:.kk]
                out1[jj*self.kernel.dimension+kk,:] =2.0*evalsDotPrec[:,jj]*derivTotal[jj][:,kk]

                dSigdXreal = np.zeros((len(self.pts),len(self.pts)))
                dSigdXreal[jj,:] = derivCovTotal[:,jj,kk]
                dSigdXreal[:,jj] = derivCovTotal[:,jj,kk]
                v1 = np.dot(evalsDotPrec, dSigdXreal)
                out2[jj*self.kernel.dimension+kk,:] = -np.sum(v1* evalsDotPrec, axis=1)
        
        out = -(out1 + out2)
        return out

    def generateSamples(self, x, noise=1e-10):
        """ 
        Generates samples of the gaussian process

        Parameter
        ---------
        x : ndarray 
            locations at which to generate samples
        noise : float
            variance of noise to add to gp samples
        Returns
        -------
        y : 1darray
            samples at x
            
        Notes
        -----
        """
        
        nugget =  0.0
        covarianceMatrix = \
            gp_kernel_utilities.calculateCovarianceMatrix(self.kernel, x, nugget)
        u, s, vt = np.linalg.svd(covarianceMatrix)
        L = np.dot(u, np.diag(np.sqrt(s)))
        #print L
        y = self.gpPriorMean(x) + np.dot(L, np.random.randn(len(x))) + \
            np.sqrt(noise)*np.random.randn(len(x))
        
        return y
    
    def computeLogLike(self, pts, evals):
        """ 
        Compute the Marginal Log-Likelihood of the GP with pts and evals

        Parameter
        ---------
        pts : ndarray 
            locations at which to generate samples
        evals : data at the locations
                variance of noise to add to gp samples
        Returns
        -------
        y : 1darray
            samples at x
            
        Notes
        -----
        """

        return self.loglikeParams(pts, evals)

    def loglikeParams(self, pts, evals, returnDeriv=0,noiseIn=None):
        #print "start "
        if noiseIn is None:
            if self.FITC ==  None:
                covMat = gp_kernel_utilities.calculateCovarianceMatrix(self.kernel, 
                        pts, self.noise)
                invMat = np.linalg.pinv(covMat)
            else:
                nodes = pts
                nNodes = len(nodes)
                nu = int(np.floor(nNodes*self.FITC))
                if self.fitcnodes == None:
                    indu = np.random.permutation(nNodes)[0:nu]
                    snodes = nodes[indu]
                    self.fitcnodes = np.array(snodes, dtype=float)
                else:
                    snodes = self.fitcnodes
                Quu = calculateCovarianceMatrix(self.kernel, snodes, self.noise)
                invQuu = np.linalg.pinv(Quu)
                kernelvals = np.zeros((nu, nNodes))
                for jj in xrange(nu):
                    pt = np.reshape(snodes[jj], (1,self.kernel.dimension))
                    kernelvals[jj,:] = self.kernel.evaluate(pt, nodes)
                Q = np.dot(kernelvals.T, np.dot(invQuu, kernelvals))
                K = calculateCovarianceMatrix(self.kernel, nodes, self.noise)
                g = np.diag(K-Q)
                G = np.diag(g)
                covMat = Q+G
                invG = np.diag(1.0/(g+1e-12))
                invMat = invG - np.dot(invG, np.dot(kernelvals.T, 
                                        np.dot(np.linalg.inv(Quu+ np.dot(kernelvals,
                                                np.dot(invG, kernelvals.T))), np.dot(
                                                kernelvals, invG))))

        else:
            covMat = gp_kernel_utilities.calculateCovarianceMatrix(self.kernel, 
                        pts, noiseIn=noiseIn)
            invMat = np.linalg.pinv(covMat)
        

        sdet, logdet = np.linalg.slogdet(covMat)
        precEval = np.dot(invMat, evals)

        firstTerm = -0.5 * np.dot(evals, precEval)
        secondTerm =  -0.5 * logdet
        thirdTerm = -len(evals)/2.0 * np.log(2.0*np.pi) 
        out = firstTerm + secondTerm + thirdTerm
        
        keys = self.kernel.hyperParam.keys() + ['noise']
        #Compute Derivative
        if returnDeriv == 1:
            derivMat = np.zeros((len(pts), len(pts), len(keys)))
            nDim = pts.shape[1]
            for ii, pt in enumerate(pts):
                p1 = pt.reshape((1,nDim))
                for jj, pt2 in enumerate(pts):
                    p2 = pt2.reshape((1,nDim))
                    analyDeriv = self.kernel.derivativeWrtHypParams(p1,p2)
                    if ii == jj:
                        analyDeriv['noise'] = 1.0#self.noise
                    else:
                        analyDeriv['noise'] = 0.0
                    for kk, key in enumerate(keys): #loop over hyParams
                        derivMat[ii,jj,kk] = analyDeriv[key]

            term1 = np.outer(precEval, precEval)- invMat
            outD = dict({})
            for ii in xrange(len(keys)):
                outD[keys[ii]] =  0.5 * np.trace(np.dot( term1, derivMat[:,:,ii])) 
                if keys[ii] == 'noise':
                    outD[keys[ii]] *= self.noise*2.0
            
            return out, outD
        else:
            return out
    
    def getHypParamNames(self):
        
        return self.kernel.hyperParam.keys()

    def updateKernelParams(self, paramsIn):
        """ 
        Set new hyperparameters

        Parameter
        ---------
        paramsIn : dictionary
                   new hyperparameters
        Returns
        -------
        y : 1darray
            samples at x
            
        Notes
        -----
        """

        #print "params ", params 
        params = copy.copy(paramsIn)
        if 'noise' in params.keys():
            self.noise = copy.copy(params['noise'])
            del params['noise']
        self.kernel.updateHyperParameters(params)
#     
    def findOptParamsLogLike(self, pts, evals, paramsStart=None, paramLowerBounds=None, paramUpperBounds=None,useNoise=None, maxiter=40):
        """ 
        Compute the optimal hyperparameters by maximizing the marginal log likelihood

        Parameter
        ---------
        pts : ndarray 
            locations at which to generate samples
        evals : data at the locations
                variance of noise to add to gp samples
        paramsStart : dictionary
                      hyperpameters which initialize the optimization
                      defaults to current GP hyperpameters
        paramsLowerBounds : dictionary
                            lower bounds for parameters 
                            defaults to max([x/10, 1e-3] where x
                            is each hyperpameter
        paramsUpperBounds : dictionary
                            upper bounds for parameters 
                            defaults to max([x*10, 10] where x
                            is each hyperpameter

        useNoise : None for noise learning bounds are [10^-12, 1e-2]
                 : float for common noise for each data point
                 : ndarray for separate noise for each data point

        maxIter : maximum number of optimization iterations

        Returns
        -------
        y : 1darray
            samples at x
            
        Notes
        -----
        """
        #return -logLike
        dimension = np.shape(pts)[1]
        
        if paramsStart is None:
            paramsStart = copy.deepcopy(self.kernel.hyperParam)

        if paramLowerBounds is None:
            paramLowerBounds = dict({})
            for k,v in paramsStart.iteritems():
                paramLowerBounds[k] = np.max([v/10.0,1e-3])
        if paramUpperBounds is None:
            paramUpperBounds = dict({})
            for k,v in paramsStart.iteritems():
                paramUpperBounds[k] = np.min([v*10.0,10.0])

        keys = []
        paramVals = []
        paramLb = []
        paramUb = []
        for k, v in paramsStart.iteritems():
            keys.append(k)
            paramVals.append(v)
            paramLb.append(paramLowerBounds[k])
            paramUb.append(paramUpperBounds[k])
        
        #LAST KEY IS NOISE
        if useNoise is None:
            keys.append('noise')
            paramLb.append(1e-12)
            paramUb.append(1e-2)
            paramVals.append(1e-5) #Start noise

        def objFunc(in0, gradIn):
            self.updateKernelParams(dict(zip(keys,in0)))
            if gradIn.size > 0:
                margLogLike, derivs= self.loglikeParams(pts, evals, returnDeriv=1)
                outD = np.zeros((len(keys)))
                for ii in xrange(len(keys)):
                    outD[ii] = derivs[keys[ii]]
                gradIn[:] = -outD
                
            else:
                margLogLike= self.loglikeParams(pts, evals, returnDeriv=0)
            
            out = -margLogLike
            # record last point evaluation
            objFunc.last_x_value = in0.copy() 
            objFunc.last_f_value = out 
            return out 
        
       
        paramsOut, optValue = self.chooseParams(paramLb, paramUb, paramVals, objFunc, maxiter=maxiter)
        bestParams = np.zeros(np.shape(paramsOut))
        
        params = dict(zip(keys,paramsOut))
        self.updateKernelParams(params)
        return params, optValue
               
    def chooseParams(self, paramLowerBounds, paramUpperBounds, startValues, costFunction,maxiter=40):
        
        if NLOPT is True:
             local_opt = nlopt.opt(nlopt.LN_COBYLA, len(startValues))

             local_opt.set_xtol_rel(1e-3)
             local_opt.set_ftol_rel(1e-3)
             local_opt.set_ftol_abs(1e-3)
             local_opt.set_maxtime(10);
             local_opt.set_maxeval(50*len(startValues)); 
                
             local_opt.set_lower_bounds(paramLowerBounds)
             local_opt.set_upper_bounds(paramUpperBounds)

             try:
                local_opt.set_min_objective(costFunction)       
                sol = local_opt.optimize(startValues)
             except nlopt.RoundoffLimited:
                return costFunction.last_x_value, costFunction.last_f_value
             return sol, local_opt.last_optimum_value()
        else:
            maxeval = 100
            bounds = zip(paramLowerBounds, paramUpperBounds)
            objFunc = lambda x: costFunction(x,np.empty(0))
            #sol = slsqp(objFunc, np.array(startValues), bounds=bounds,
            #            iter=maxeval)
            

            #print "startValuies ", len(startValues), len(paramLowerBounds)
            objFunc = lambda x : costFunction(x,np.empty(0))
            def const(x):
                good = 1.0
                for ii in xrange(len(x)):
                    if (x[ii] < paramLowerBounds[ii]):
                        return -1.0
                    elif (x[ii] > paramUpperBounds[ii]):
                        return -1.0
                return good

            #sol = cobyla(objFunc, np.array(startValues), cons=(const), maxfun=maxeval)
            sol_bfgs = bfgs(objFunc, np.array(startValues), bounds=bounds, approx_grad=True, factr=1e10, maxfun=maxiter)
            sol = sol_bfgs[0]
            #print "sol ", np.round(sol,4)
            val = objFunc(sol)
            return sol,val
#===============================================================================
        
