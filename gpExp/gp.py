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
            
        # below code should be parallelized
        
        if compvar:
            var_newpt = self.kernel.evaluate(newpt, newpt)
            var = np.zeros((numNewPoints))
            for jj in xrange(numNewPoints):
                var[jj] = var_newpt[jj] - \
                    np.dot(kernelvals[jj,:], np.dot(self.precisionMatrix, kernelvals[jj,:].T))
            return out, np.abs(var)
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
            if noiseIn == None:
                self.covarianceMatrix = calculateCovarianceMatrix(self.kernel, nodes, self.noise)
            else:
                self.covarianceMatrix = calculateCovarianceMatrix(self.kernel, nodes, noiseIn)
            self.precisionMatrix = np.linalg.pinv(self.covarianceMatrix)
        else:
            if noiseIn == None:
                
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
        #print "Condition Number of Covariance Matrix ", np.linalg.cond(self.covarianceMatrix)

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
                    np.dot(kernelvals[jj,:], np.dot(self.precisionMatrix, kernelvals[jj,:].T))
            #print "Done "
            return var
        else:
            var =  parallel_utilities.parallelizeMcForLoop(self.evaluateVariance, 0, newpt)
            return var
    
    def evaluateVarianceDerivWRTnewpt(self, newpt):
        """ evaluate derivative of posterior variance at newpt """

        assert self.pts != None, "must specify training points before running this" 
        assert newpt.shape[1] == self.kernel.dimension, "evaluation points for GP is incorrect shape"
        
        out = np.zeros((newpt.shape))
        derivs = np.zeros((newpt.shape[0], newpt.shape[1], len(self.pts)))
        evals = np.zeros((newpt.shape[0], self.pts.shape[0]))
        for ii, pt in enumerate(self.pts):
            p = pt.reshape((1,self.kernel.dimension))
            derivs[:,:,ii] = self.kernel.derivative(newpt,p)
            evals[:,ii] = self.kernel.evaluate(p, newpt)
        
        evalSigma = np.dot(self.precisionMatrix, evals.T) #N_{GP} X N
        #FIX THIS
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

        assert self.pts != None, "must specify training points before running this" 
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
        return self.loglikeParams(pts, evals)

    def loglikeParams(self, pts, evals, returnDeriv=0,noiseIn=None):
        #print "start "
        if noiseIn == None:
            if self.FITC ==  None:
                covMat = gp_kernel_utilities.calculateCovarianceMatrix(self.kernel, 
                        pts, self.noise)
                invMat = np.linalg.pinv(covMat)
            else:
                print "HERE"
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
        
        #print "covMat ", covMat 

        u, s, v = np.linalg.svd(covMat)
        #Ensure Good Conditioning
        indGood = s>1e-7
        u = u[:,indGood]
        v = v[indGood,:]
        s = s[indGood]

       # print "diffCovmat ", np.linalg.norm(covMat- 
       #         np.dot(u,np.dot(np.diag(s), v)))
       
        sdet, logdet = np.linalg.slogdet(covMat)
        #print "logDet ", logdet
        
        #precEval = np.dot(v.T, np.dot(np.diag(s**-1), np.dot(u.T, evals)))
        precEval = np.dot(invMat, evals)

        firstTerm = -0.5 * np.dot(evals, precEval)
        secondTerm =  -0.5 * logdet
        thirdTerm = -len(evals)/2.0 * np.log(2.0*np.pi) 
        #print "terms ", firstTerm, secondTerm, thirdTerm
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

            #outD = np.zeros((len(keys)))
            #invMat = np.dot(v.T, np.dot(np.diag(s**-1), u.T))
            term1 = np.outer(precEval, precEval)- invMat
            #print "term1 ", term1           
            outD = dict({})
            for ii in xrange(len(keys)):
                #print "derivMat ", derivMat[:,:,ii]
                outD[keys[ii]] =  0.5 * np.trace(np.dot( term1, derivMat[:,:,ii])) 
                if keys[ii] == 'noise':
                    #print "self.noise ", self.noise
                    #print "traceQ ", outD[keys[ii]]*2.0
                    outD[keys[ii]] *= self.noise*2.0
                    #print "outD[keys[ii]] ", outD[keys[ii]]
            

            #print "end"
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
            self.noise = copy.copy(params['noise'])
            del params['noise']
        self.kernel.updateHyperParameters(params)
#     
    def findOptParamsLogLike(self, pts, evals, paramsStart=None, paramLowerBounds=None, paramUpperBounds=None,useNoise=None):
        """
        useNoise = None for noise learning
                 = float for common noise for each
                 = ndarray for separate noise
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
        if useNoise == None:
            keys.append('noise')
            paramLb.append(1e-12)
            paramUb.append(1e-2)
            paramVals.append(1e-5) #Start noise

        # Want to maximize logLIke -> minimize - logLike
        #print "pts ", pts
        #print "evals ", evals
        def objFunc(in0, gradIn):
            #print "begin obj max like"
            #print "here ", in0
            self.updateKernelParams(dict(zip(keys,in0)))
            if gradIn.size > 0:
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
            #print "loglike ", out
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
         #local_opt = nlopt.opt(nlopt.LD_SLSQP, len(startValues))
         
         #local_opt = nlopt.opt(nlopt.LN_NELDERMEAD, len(startValues))
         #local_opt.set_ftol_rel(1e-2)
         local_opt.set_xtol_rel(1e-3)
         local_opt.set_ftol_rel(1e-3)
         #local_opt.set_xtol_abs(1e-40)
         #local_opt.set_ftol_rel(1e-40)
         local_opt.set_ftol_abs(1e-3)
         local_opt.set_maxtime(10);
         local_opt.set_maxeval(40); #200
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
        
