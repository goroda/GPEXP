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
import nlopt as nlopt
import multiprocessing as mp
from gp_kernel_utilities import calculateCovarianceMatrix, calculateCovarianceMatrixFITC
import parallel_utilities
import gp_kernel_utilities
import copy
import itertools

class GP(object):
    """ This is the GP class with prior mean is zero """
    
    x = None # training nodes
    y = None # training values
    yn = None # noise on training values
    u = None; s = None; v = None  #usv = covariance matrix
    precMat = None
    coeff = None
    noise = None
    pts = None

    def __init__(self, kernel_in, meanType=0, **kwargs):
        """ 
        Initialize the GP class 
            
        Inputs
        ======
        kernel_in : kernel class
                    Covariance kernel of Gaussian process
        meanType : int
                   0 for zero mean
                   1 for training sample mean
                   2 for linear regression mean
        """
        self.kernel = copy.copy(kernel_in) # this is a kernel class
        self.meanType = meanType #0 for zero mean, 1 for mean of training samples 2 for lin
        super(GP,self).__init__()
   
    def getDimension(self):
        return self.kernel.dimension
    def getHypParam(self):
        return self.kernel.hyperParam
    def addTrainXY(self, x, y, noise):
        """
        Add Training Points and Values

        Inputs
        ======
        x : ndarray (nPts x dim)
            Training Nodes
        y : 1darray (nPts)
            function values at training Nodes
        
        noise :: 1darray (nPts) or float 
                 noise on function values
        """

        assert len(x.shape) == 2, "x must be 2darray"
        self.x = copy.copy(x)
        self.y = copy.copy(y)
        self.yn = noise

    def computeCov(self, noise=None, fitc=False):
        """ 
        Compute and store covariance matrix
        
        Parameters
        ----------
        noise : float or ndarray

        fitc : approximation FITC  (A unifying View of Sparse Approximate GP 2005 Rasmussen)
               float   : use a random subset of fitc% of the training set
               ndarray : pts to use

        Returns
        -------
        fitcnodes if fitc==True
        Computes either the svd of the covariance matrix in the fitc=False case
        or the the precision matrix in the fitc=True case
        
        Note
        ----
        This function is useful if one doesnt want to train right away
        or if one just wants to evaluate the posterior variance later
        """
        if fitc == False:
            covm = calculateCovarianceMatrix(self.kernel, self.x, self.yn)
            self.u, self.s, self.v = np.linalg.svd(covm)
        else:
            self.precMat, fitcnodes = calculateCovarianceMatrixFITC(self.kernel, self.x, 
                                                                    self.yn, fitc)
            return fitcnodes

    def gpPriorMean(self, pts, vals=None):
        """
        Evaluate Prior Mean
        
        Inputs
        ======
        pts :: np
        """
        if self.meanType == 0:
            return np.zeros((pts.shape[0]))

    def train(self) :
        """
        Computes GP coefficients
        
        Returns
        -------
        Nothing - stores coefficients in class
        
        Notes
        -----
        """
        #remove mean function
        
        evals = self.y-self.gpPriorMean(self.x)
        try: 
            self.coeff = np.dot(self.precMat, evals)
        except TypeError:
            self.coeff = np.dot(self.u, np.dot(np.diag(self.s**(-1.0)), np.dot(self.v,evals)))
    
    def multPrec(self, rhs):
        
        try:
            out = np.dot(self.u, np.dot(np.diag(self.s**(-1.0)), np.dot(self.v, rhs)))
        except ValueError:
            out = np.dot(self.precMat, rhs)

        return out

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
        numTrainPoints = self.x.shape[0]
        
        kernelvals = np.zeros((numNewPoints, numTrainPoints))
        for jj in xrange(numTrainPoints):
            pt = np.reshape(self.x[jj,:], (1,self.kernel.dimension))
            kernelvals[:,jj] = self.kernel.evaluate(newpt, pt)
        
        out = np.dot(kernelvals, self.coeff) + self.gpPriorMean(newpt)
            
        # below code should be parallelized
        
        if compvar:
            var_newpt = self.kernel.evaluate(newpt, newpt)
            var = np.zeros((numNewPoints))
            for jj in xrange(numNewPoints):
                var[jj] = var_newpt[jj] - \
                    np.dot(kernelvals[jj,:],self.multPrec(kernelvals[jj,:].T))
            return out, np.abs(var)
        else:
            return out
        
        
    def evaluateVariance(self, newpt, parallel=1):
        """
        Evaluate GP posterior variance at a new point
        
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
        
        assert self.pts != None, "must specify training points before running this" 
        assert newpt.shape[1] == self.kernel.dimension, "evaluation points for GP is incorrect shape"
        
        numNewPoints = newpt.shape[0]
        numTrainPoints = self.pts.shape[0]
        #print "Evaluate Variance "
        nThreshForParallel = 500001 # at this level both parallel and serial seem to perform similarly
        if numNewPoints < nThreshForParallel or parallel==0:
            #print "Not parallel ", numNewPoints
            kernelvals = np.zeros((numNewPoints, numTrainPoints))
            for jj in xrange(numTrainPoints):
                pt = np.reshape(self.pts[jj,:], (1,self.kernel.dimension))
                kernelvals[:,jj] = self.kernel.evaluate(newpt, pt)
                   
            # below code should be parallelized
            var_newpt = self.kernel.evaluate(newpt, newpt)
            var = np.zeros((numNewPoints))
            for jj in xrange(numNewPoints):
                var[jj] = var_newpt[jj] - \
                    np.dot(kernelvals[jj,:], self.multPrec(kernelvals[jj,:].T))
            #print "Done "
            return var
        else:
            var =  parallel_utilities.parallelizeMcForLoop(self.evaluateVariance, 0, newpt)
            return var
    
    def evaluateVarianceDerivWRTnewpt(self, newpt, returnFlattened=True):
        """ 
        Evaluate derivative of posterior variance at newpt 
        
        Inputs
        ======
        newpt : ndarray N x D
                Array at which we are obtaining deriviative
        
        Outputs
        =======
        outout : 1darray ND if returnFlattened=True
                 N x D if returnFlattened=False
                 Array of derivatives
        Notes
        =====
        Recall that the variance at x* is K(x*, x*) - K(x,x*)^T Cov^{-1} K(x, x*)
        Thus the derivative dVar/dx* can be found as 
        dVar/dx* = dK/dx* - dK(x,x*)/dx* ^T Cov^{-1} K(x, x*) + K(x,x*) Cov^{-1} dK(x,x*)dx*
                 = dK/dx* - 2 dK(x,x*)/dx* ^T Cov^{-1} K(x, x*)

        Where the dimensions are
        dVar/dx* \in reals^D
        dK/dx& \in reals^D
        dK(x,x*)/dx \in reals^{Ntrain x d}
        
        and the second equality comes from the symmetry of Cov
        """

        assert self.x != None, "must specify training points before running this" 
        assert newpt.shape[1] == self.kernel.dimension, "evaluation points for GP is incorrect shape"
        
        #Allocate output
        out = np.zeros((newpt.shape))
        derivs = np.zeros((newpt.shape[0], newpt.shape[1], len(self.x)))
        evals = np.zeros((newpt.shape[0], self.x.shape[0]))
        for ii, pt in enumerate(self.x):
            p = pt.reshape((1,self.kernel.dimension))
            derivs[:,:,ii] = self.kernel.derivative(newpt,p)
            evals[:,ii] = self.kernel.evaluate(p, newpt)
        
        #Can I use tensorDot
        evalSigma = self.multPrec(evals.T) #N_{GP} X N
        for ii, pt in enumerate(newpt):
            out[ii,:] =  -2.0 *np.dot(derivs[ii,:,:], evalSigma[:,ii])
        
        if returnFlattened is True:
            outout = out.flatten()
            return outout
        return out
        
    def evaluateVarianceDerivative(self, newpt, noiseFunc=None):
        """ 
        Evaluate Posterior Variance Derivative WRT training points at locations newpt
        
        Inputs
        ======
        newpt : ndarray N x D
                Array at which we are obtaining deriviative
        
        Outputs
        =======
            out : ndarray (ND,NN)
                  i.e. if k < N l < D then
                  out[k*l, jj] = dC(newpt[jj,:], newpt[jj,:]) / d self.pts[k,l],
        
        Notes
        =====
        Recall that the variance at x* is K(x*, x*) - K(x,x*)^T Cov^{-1}(x,x) K(x, x*)
        Thus the derivative with respect to training point i dVar/dx_i can be found as 
        dVar/dx_i = -[dK(x,x*)/dx_i ^T Cov^{-1}(x,x) K(x, x*) + K(x,x*) Cov^{-1} dK(x,x*)dx +
                       - K(x,x*)^T Cov(x,x)^{-1} dCov(x,x)dx_i Cov(x,x)^{-1} K(x,x)
                 = - 2dK(x,x*)/dx_i ^T Cov^{-1} K(x, x*) + K(x,x*)^T Cov(x,x)^{-1} dCov(x,x)dx_i Cov(x,x)^{-1} K(x,x)

        Where the dimensions are
        dVar/dx_i \in reals^D
        dK/dx_i \in reals^D
        dK(x,x*)/dx_i \in reals^{Ntrain x D}
        dCov(x,x)/dx_i \in reals^{Ntrain x Ntrain x D}

        """
        # Returns len(self.pts)*dim \times newpt array
        #NOTE there is no loop over newpt!

        assert self.x != None, "must specify training points before running this" 
        assert newpt.shape[1] == self.kernel.dimension, "evaluation points for GP is incorrect shape"

        outout = np.zeros((len(self.x)*self.kernel.dimension,len(newpt)))
        derivCovTotal = np.zeros((len(self.x),len(self.x),self.kernel.dimension))
        totEvals = np.zeros((len(newpt), len(self.x)))
        
        #preprocessing
        derivTotal = []
        for zz in xrange(len(self.x)):
            
            p = self.x[zz,:].reshape((1,self.kernel.dimension))
            indUse = np.array([np.linalg.norm(pp - p) < 1e-10 for pp in self.pts])
            
            derivCovTotal[zz,:,:] = self.kernel.derivative(self.x, p) #compute the derivative of each training point with each other
            totEvals[:,zz] = self.kernel.evaluate(p, newpt) #compute all required kernel evaluations
            derivTotal.append(-self.kernel.derivative(newpt, p))
            if noiseFunc is not None: # add derivatives due to noise if noise variance is a function
                indUse = np.tile(indUse.reshape((len(self.pts),1)), self.kernel.dimension)
                derivCovTotal[zz,:,:] += indUse*noiseFunc.deriv(self.pts)
                if np.linalg.norm(p-newpt) < 1e-10:
                   totEvals[:,zz] += noiseFunc(p)
                   derivTotal[-1] -= noiseFunc.deriv(p)

        evalsDotPrec = self.precMult(totEvals.T).T #computes K(x,x*)^Tcov(x,x)^{-1}
        
        #form together the derivative
        out2 = np.zeros((len(self.pts)*self.kernel.dimension,len(newpt)))
        out1 = np.zeros((len(self.pts)*self.kernel.dimension,len(newpt)))
        #can probably do this better without for loops
        for jj in xrange(len(self.pts)):
            for kk in xrange(self.kernel.dimension):
               
                out1[jj*self.kernel.dimension+kk,:] =2.0*evalsDotPrec[:,jj]*derivTotal[jj][:,kk] #term without cov deriv

                dSigdXreal = np.zeros((len(self.pts),len(self.pts)))
                #because symmetrix they have same derivatives
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
    
    def computeLogLike(self, *args):
        return self.loglikeParams(self.x,self.y,args)

    def loglikeParams(self, pts, evals, returnDeriv=0, fitc=False):
        #print "start "
        if fitc is False:
            covMat = calculateCovarianceMatrix(self.kernel, pts, self.yn)
            invMat = np.linalg.pinv(covMat)
        else:
            covMat, invMat, fitcnodes = calculateCovarianceMatrixFITC(self.kernel, pts,
                                                self.yn, fitc, returnCov=True)
        
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
                    outD[keys[ii]] *= self.yn*2.0
            

            return out, outD
        else:
            #print "end"
            return out
    
    def getHypParamNames(self):
        return self.kernel.hyperParams.keys()

    def updateKernelParams(self, paramsIn):
        #print "params ", params 
        params = copy.copy(paramsIn)
        if 'noise' in params.keys():
            self.yn = copy.copy(params['noise'])
            del params['noise']
        self.kernel.updateHyperParameters(params)
#     
    def findOptParamsLogLike(self, pts, evals, paramsStart, paramLowerBounds, paramUpperBounds,useNoise=None, **kwargs):
        """
        useNoise = None for noise learning
                 = float for common noise for each
                 = ndarray for separate noise
        """
        #return -logLike
        dimension = np.shape(pts)[1]
       
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
        if useNoise == None:
            keys.append('noise')
            paramLb.append(1e-12)
            paramUb.append(1e-2)
            paramVals.append(1e-5) #Start noise

        # Want to maximize logLIke -> minimize - logLike
        #print "pts ", pts
        #print "evals ", evals

        if 'FITC' in kwargs:
            nNodes = pts.shape[0]
            nu = int(np.floor(nNodes*kwargs['FITC']))
            indu = np.random.permutation(nNodes)[0:nu]
            snodes = np.array(pts[indu], dtype=float)

        nCalls = 0
        def objFunc(in0, gradIn):
            #print "begin obj max like"
            #print "here ", in0
            self.updateKernelParams(dict(zip(keys,in0)))
            if gradIn.size > 0:
                if 'FITC' in kwargs:
                    margLogLike, derivs= self.loglikeParams(pts, evals, returnDeriv=1,fitc=snodes)
                else:
                    margLogLike, derivs= self.loglikeParams(pts, evals, returnDeriv=1)
                outD = np.zeros((len(keys)))
                for ii in xrange(len(keys)):
                    outD[ii] = derivs[keys[ii]]
                gradIn[:] = -outD
                #print "params ", in0
                #print "val ",  margLogLike
                #print "gradIn ", gradIn
                #print self.kernel.hyperParam, self.noise, -margLogLike, np.linalg.norm(gradIn[:])
                
            else:
                margLogLike= self.loglikeParams(pts, evals, returnDeriv=0)
            
            out = -margLogLike
            print " logLike: ",out
            #print "woop ", 
            return out 
        
       
        paramsOut, optValue = self.chooseParams(paramLb, paramUb, paramVals, objFunc)
        #print "ok"
        bestParams = np.zeros(np.shape(paramsOut))
        
        #print "keys ", keys
        #print "paramsOut ", paramsOut
        #print "zip ", zip(keys, paramsOut)
        params = dict(zip(keys,paramsOut))
        #print "New Parameters ", params
        self.updateKernelParams(params)
        return params, optValue
               
    def chooseParams(self, paramLowerBounds, paramUpperBounds, startValues, costFunction):
        
         #print "length of StartValues ", len(startValues)
         #print "paramLb and Ub ", paramLowerBounds, paramUpperBounds
            
         #opt = nlopt.opt(nlopt.GD_STOGO, len(startValues))
         #opt = nlopt.opt(nlopt.G_MLSL_LDS, len(startValues))
         #opt.set_lower_bounds(paramLowerBounds)
         #opt.set_upper_bounds(paramUpperBounds)
         #opt.set_min_objective(costFunction)

#        #print "startValues ", startValues
         local_opt = nlopt.opt(nlopt.LN_COBYLA, len(startValues))
         #local_opt = nlopt.opt(nlopt.LD_MMA, len(startValues))
        
         #local_opt = nlopt.opt(nlopt.LD_LBFGS, len(startValues))
         #local_opt = nlopt.opt(nlopt.LD_TNEWTON_PRECOND_RESTART, len(startValues))
         #local_opt = nlopt.opt(nlopt.LD_TNEWTON, len(startValues))
         #local_opt = nlopt.opt(nlopt.LD_SLSQP, len(startValues))
         
         #local_opt = nlopt.opt(nlopt.LN_NELDERMEAD, len(startValues))
         #local_opt.set_ftol_rel(1e-2)
         local_opt.set_xtol_rel(1e-3)
         #local_opt.set_ftol_rel(1e-3)
         #local_opt.set_xtol_abs(1e-40)
         #local_opt.set_ftol_rel(1e-40)
         #local_opt.set_ftol_abs(1e-15)
        # local_opt.set_maxtime(10);
         local_opt.set_maxeval(20); #200
#            
         local_opt.set_lower_bounds(paramLowerBounds)
         local_opt.set_upper_bounds(paramUpperBounds)
        
         try:
            local_opt.set_min_objective(costFunction)       
            sol = local_opt.optimize(startValues)
         except nlopt.RoundoffLimited:
            return startValues, None
         #opt.set_local_optimizer(local_opt)
         #opt.set_population(1)
         #sol = opt.optimize(startValues)
         #sol = opt.optimize()
         #print "Negative LogLikeOpt ", local_opt.last_optimum_value()
         #print "Log Like Optimizer Result ", local_opt.last_optimize_result()
#        #print "sol ", sol
         return sol, local_opt.last_optimum_value()
#===============================================================================
        
