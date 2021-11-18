#Copyright (c) 2013-2016, Massachusetts Institute of Technology
#Copyright (c) 2016-2022, Alex Gorodetsky
#
#This file is part of GPEXP:
#Author: Alex Gorodetsky alex@alexgorodetsky
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
from scipy.sparse.linalg import LinearOperator
#import scipy.sparse.linalg.eigen.arpack as sparpack
import scipy.sparse.linalg.eigen as sparpack
from . import parallel_utilities
"""
#################################
# Utilities
#################################       
"""                       
def calculateCovarianceMatrix(kernel, points, nugget=0.0):
        """ 
        Calculate covariance matrix

        Parameters
        ---------- 
        points : ndarray 
            Locations at which to evaluate covariance matrix
            
        nugget : float or ndarray
            nugget for covariance matrix. default=0
            
        Returns
        -------
        covarianceMatrix : ndarray 
            Covariance matrix 
        
        Notes
        -----
        """
        size_of_mat, dim = points.shape
        covarianceMatrix = np.zeros((size_of_mat,size_of_mat))
        for jj in range(size_of_mat):
            point = np.reshape(points[jj,:] , (1, dim) )
            covarianceMatrix[jj,:] = \
                kernel.evaluate(points, point)
            covarianceMatrix[jj,jj] = covarianceMatrix[jj,jj] 

        if isinstance(nugget, float):
            dadd = nugget*np.ones((size_of_mat))
        elif isinstance(nugget, np.ndarray):
            dadd = nugget[:]
        
        covarianceMatrix = covarianceMatrix + np.diag(dadd)
        return covarianceMatrix

def calculateCovarianceMatrixFITC(kernel, nodes, nugget, fitc, returnCov=False):
    nNodes = len(nodes)
    if isinstance(fitc, float):
        nu = int(np.floor(nNodes*fitc))
        indu = np.random.permutation(nNodes)[0:nu]
        snodes = np.array(nodes[indu], dtype=float)
    else:
        #print fitc.shape
        nu = len(fitc)
        snodes = fitc

    Quu = calculateCovarianceMatrix(kernel, snodes, nugget)
    invQuu = np.linalg.pinv(Quu)
    kernelvals = np.zeros((nu, nNodes))
    for jj in range(nu):
        pt = np.reshape(snodes[jj], (1,kernel.dimension))
        kernelvals[jj,:] = kernel.evaluate(pt, nodes)
    Q = np.dot(kernelvals.T, np.dot(invQuu, kernelvals))
    K = calculateCovarianceMatrix(kernel, nodes, nugget)
    g = np.diag(K-Q)
    G = np.diag(g)
    covmat = Q+G
    invG = np.diag(1.0/(g+1e-12))
    precMat = invG - np.dot(invG, np.dot(kernelvals.T, 
                        np.dot(np.linalg.inv(Quu+ np.dot(kernelvals,
                                np.dot(invG, kernelvals.T))), np.dot(
                                kernelvals, invG))))
    if returnCov == False:
        if fitc is not False:
            return precMat, snodes
        return precMat
    else:
        if fitc is not False:
            return covmat, precMat, snodes
        return covmat, precMat


def covTimesV(b, kernel, mcPoints):
    """ 
    Performs the operation of covariance matrix.
    
    Parameter
    ---------
    weights : 1darray
    
    b : 1darray
        array which we are left multiplying by covariance matrix
    Returns
    -------
    out : 1darray
        Covariance times matrix
        
    Notes
    -----
    Used for linear operator in eigenvalue solver
    """
    print("Calculate ")
    #counter = counter+1
    out = np.zeros((b.shape))
    
    def computeForEachmm(ii, fn):
        out = np.zeros((np.shape(ii)[0]))
        ii = ii.reshape((len(ii)))
        overallIter = 0
        for itera in ii:
            out[overallIter] = np.dot(kernel.evaluate(mcPoints,np.reshape(mcPoints[itera,:], (1,mcPoints.shape[1]))),b)
            overallIter = overallIter+1
        #print "out ", out
        return out
    
    p = np.arange(b.shape[0]).reshape((out.shape[0],1))
    out = parallel_utilities.parallelizeMcForLoop(computeForEachmm, None, p)    
    return out
 
def calculateKernelBasisFunctionsMC(kernel, numBasis, mcPoints):
    """ 
    Calculates basis functions using Monte Carlo and Nystrom method.
    
    Parameters
    ---------- 
    numBasis : int 
        calculate numBasis eigenvectors
    mcPoints : ndarray 
        Monte Carlo points
    
    Returns
    -------
    eigv : 1darray 
        eigenvalues numBasisx1
    eigve : ndarray  
        eigenvectors mcPoints x numBasis

    Notes
    -----

    """
    linearOpVersion = 1
    nMC = mcPoints.shape[0]   
    numberEigenVectors = int(min(numBasis,nMC))        

    #print "looking for ", numberEigenVectors, "eigenvalues"e
    if linearOpVersion:
        #covm = calculateCovarianceMatrix(kernel, mcPoints)
        covop = lambda v, k=kernel, p=mcPoints: covTimesV(v, k, p)
        A = LinearOperator( (nMC, nMC), matvec=covop, \
            dtype=float)
        eigv, eigve= sparpack.eigsh(A,
                 k=numberEigenVectors, maxiter=10*numberEigenVectors)
    else:
        #print "calculateCovariance Matrix "
        covm= calculateCovarianceMatrix(kernel, mcPoints)
        eigv, eigve= sparpack.eigsh(covm,
                 k=numberEigenVectors, maxiter = 50*numberEigenVectors)

    
    #print "completed eigenvalue decomposition "
    #sort in decending order
    eigv = eigv[::-1]
    eigve = eigve[:, ::-1]
    
    # perform nystrom
    eigve = eigve*np.sqrt(float(nMC))
    eigv = eigv/float(nMC)
    
    return eigv, eigve

def efunc(x, ptsin, eigve, kern):
    """  
    Very simple function taking in coefficients and outputting a GP approximation
        
    Parameters
    ----------
    x : ndarray
        Points at which we will compute the approximation
    ptsin : ndarray
        Points used to train approximation
    eigve : 1darray or ndarray
        coefficients of approximation
        
    Returns
    -------
    outh : 1darray
        Interpolation of eigve at x.
        
    Notes
    -----
    Used for Nystrom approximation
    
    See Also
    --------
    calculateBasisFunctionsMonteCarlo
    """
    #nystrom method
    nPtsL = np.shape(x)[0]
    dim = np.shape(ptsin)[1]
    outh = np.zeros((nPtsL))
    for qq in range(nPtsL):

        evals = kern.evaluateKernel(ptsin, np.reshape(x[qq,:], 
                                (1, dim)))
        outh[qq] = np.dot(eigve, evals)

    return outh   
