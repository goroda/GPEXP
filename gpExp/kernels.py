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
import parallel_utilities
import math

class Kernel(object):
    """ This is a Kernel Class """
    
    #kernelBasis = None
    
    #nugget = 5.0e-1
    nugget = 0.0
    hyperParam = dict({})
    def __init__(self, hyperParam, dimension, *argc):
        """ Initializes Kernel class """
        # really is a constant kernel = 0
        
        self.dimension = dimension
        self.hyperParam = hyperParam
        super(Kernel,self).__init__()
    
    def updateHyperParameters(self, hyperParamNew):
        
        for key in hyperParamNew.keys():
            assert key in self.hyperParam.keys(), (key, " is not a valid hyperParameter")
        self.hyperParam = hyperParamNew

    def evaluate(self, x1, x2):
          
        assert len(x2.shape) > 1 and len(x1.shape) > 1, "Must supply nd arrays to evaluation function"
        
        nPointsx1 = x1.shape[0]
        nPointsx2 = x2.shape[0]
        assert x1.shape[1] == self.dimension and x2.shape[1] == self.dimension, \
               ( " Incorrect dimension of input points fed to kernel ", x1.shape, x2.shape)
                
        if nPointsx1 > nPointsx2:
            out = self.evaluateF(x1, np.tile(x2, (nPointsx1, 1)))
        elif nPointsx1 < nPointsx2:
            out = self.evaluateF(np.tile(x1, (nPointsx2, 1)), x2)
        else:
            out = self.evaluateF(x1,x2)
        
        return out
    
    def evaluateF(self, x1, x2):
        """ Default kernel is constant 0 """

        return 0
 
class KernelIsoMatern(Kernel):

    def __init__(self, rho, signalSize, dimension, nu=3.0/2.0):
       
        #note nu is not treated as hyperparamter
        hyperParam = dict({'rho':rho, 'signalSize':signalSize})
        self.nu = nu
        super(KernelIsoMatern, self).__init__(hyperParam, dimension)
         
    def evaluateF(self, x1, x2):
        """ Private evaluate method in which x1 and x2 are the same shape """
        assert x1.shape == x2.shape, "__evaluate() received non-equal shaped point sets"

        if np.abs(1.5-self.nu) < 1e-10:
            #d = np.linalg.norm(x1-x2)#np.sqrt(np.sum((x1-x2)**2.0,axis=1))
            d = np.sqrt(np.sum((x1-x2)**2.0,axis=1))
            term = np.sqrt(3)*d/self.hyperParam['rho']
            out = self.hyperParam['signalSize']* (1.0+term)* np.exp (-term)
        #super(KernelIsoMatern, self).evaluateF(x1,x2)
        return out
    
    def derivativeWrtHypParams(self, x1, x2):
        #Assume that derivative is taken at current hyperparameters

        assert x1.shape == x2.shape, "__evaluate() received non-equal shaped point sets"
        pass
        """
        cl = np.zeros((self.dimension))
        for ii in xrange(self.dimension):
            cl[ii] = self.hyperParam['cl'+str(ii)]
        
        out = {}
        evals = self.evaluateF(x1, x2)
        for key in self.hyperParam.keys():
            if key == 'signalSize':
                out[key] = np.exp (-0.5 * np.sum((x1-x2)**2.0*np.tile(cl**-2.0, (x1.shape[0],1)),axis=1) )
            else:
                direction = float(key[2:])# which direction
                out[key] = evals*(x1[:,direction]-x2[:,direction])**2.0 \
                                / cl[direction]**3.0
        """
        #return out



class KernelSquaredExponential(Kernel):
    """  exp(- (x-x')^2/(2*l^2) """

    def __init__(self,  correlationLength, signalSize, dimension):
       
        hyperParam = dict({})
        if len(correlationLength) == 1:
            correlationLength = np.tile( correlationLength, (dimension))     
        for ii in xrange(len(correlationLength)):
            hyperParam['cl'+str(ii)] = correlationLength[ii]

        hyperParam['signalSize'] = signalSize 
        super(KernelSquaredExponential, self).__init__(hyperParam, dimension)
         
    def evaluateF(self, x1, x2):
        """ Private evaluate method in which x1 and x2 are the same shape """
        assert x1.shape == x2.shape, "__evaluate() received non-equal shaped point sets"
        cl = np.zeros((self.dimension))
        for ii in xrange(self.dimension):
            cl[ii] = self.hyperParam['cl'+str(ii)]

        out = self.hyperParam['signalSize']* \
            np.exp (-0.5 * np.sum((x1-x2)**2.0*np.tile(cl**-2.0, (x1.shape[0],1)),axis=1) )
        #super(KernelSquaredExponential, self).evaluateF(x1,x2)
        return out
    
    def derivativeWrtHypParams(self, x1, x2):
        #Assume that derivative is taken at current hyperparameters

        assert x1.shape == x2.shape, "__evaluate() received non-equal shaped point sets"

        cl = np.zeros((self.dimension))
        for ii in xrange(self.dimension):
            cl[ii] = self.hyperParam['cl'+str(ii)]
        
        out = {}
        evals = self.evaluateF(x1, x2)
        for key in self.hyperParam.keys():
            if key == 'signalSize':
                out[key] = np.exp (-0.5 * np.sum((x1-x2)**2.0*np.tile(cl**-2.0, (x1.shape[0],1)),axis=1) )
            else:
                direction = float(key[2:])# which direction
                out[key] = evals*(x1[:,direction]-x2[:,direction])**2.0 \
                                / cl[direction]**3.0

        return out

    def derivative(self, x1, x2, version=0):
        """ Squared Exponential
            input:
                point1 : ndarray 
                point2 : 1xdimension

            output:
                if version == 0
                    evaluation : float or ndarray 
                        derivative of Gaussian function around point2     
                        out[jj, ii] = dK(point1[jj,:], point2) / d point1[jj,ii]
                if version == 1:
                    evaluation :float or ndarray
                        out[jj, ii] = dK(point2, point1[jj,:])/ point2[ii]
                        
            This function defines the Gaussian kernel Derivative
            
            Note:
            If we define K(x_1) = K(x_1, x_2) 
            then this function computes dK(x_1)/dx_1 which is vector valued (size of dimension

        """  
        assert len(x2.shape) > 1 and len(x1.shape) > 1, "Must supply nd arrays to evaluation function"
        assert x2.shape[0] == 1 and x2.shape[1] == self.dimension, "x2 not in correct shape"
        assert x1.shape[0] > 0 and x1.shape[1] == self.dimension, "x1 not in correct shape"
        cl = np.zeros((self.dimension))
        for ii in xrange(self.dimension):
            cl[ii] = self.hyperParam['cl'+str(ii)]

        if version == 0 or version == 1:
            nPointsx1 = x1.shape[0]
            rEvals = self.evaluate(x1,x2)
            out = -self.hyperParam['signalSize']*0.5*2 * (x1 - np.tile(x2, (nPointsx1,1)))/ \
                    np.tile(cl**2.0, (nPointsx1,1))* \
                    np.tile(np.reshape(rEvals, (x1.shape[0],1)), ((1,self.dimension))) 
        return out

class KernelHermiteND(Kernel):

    def __init__(self, tIn, dimension):
        hyperParam = dict({})
        self.oneDKern = []
        for ii in xrange(dimension):
            hyperParam[ii] = tIn[ii]
            self.oneDKern.append(KernelHermite1D(tIn[ii],1))
        super(KernelHermiteND, self).__init__(hyperParam, dimension)
    
    def updateHyperParameters(self, params):
        for keys in self.hyperParam.keys():
            self.hyperParam[keys] = params[keys]

        for ii in xrange(self.dimension):
            self.oneDKern[ii].updateHyperParameters(dict({'t':self.hyperParam[ii]}))

    def evaluateF(self, x1, x2):
        """ 
        Parameter
        --------    
        x1 : 1darray 
            n x dimension
        x2 : 1darray  
            n x dimension

        Returns
        ------- 
        evaluation : float or 1darray      
                
        Notes
        ----- 

        """
        
        assert x1.shape == x2.shape, "__evaluate() received non-equal shaped point sets"
        nPoints = x1.shape[0]
        out = np.ones((x1.shape[0]))
        #print "x1 ", x1
        #print "x2 ", x2
        for ii, kern in enumerate(self.oneDKern):
            newVal = kern.evaluate(x1[:,ii].reshape((nPoints,1)),x2[:,ii].reshape((nPoints,1)))
            
            #print "newVal ", newVal
            out = out*newVal
        return out

    def derivative(self, x1, x2):
        """ Derivative of Mehler 2D kernel
        input:
            point1 : ndarray 
            point2 : 1xdimension

        output: 
            evaluation : float or ndarray 
                derivative of Gaussian function around point2     
                out[jj, ii] = dK(point1[jj,:], point2) / d point1[jj,ii]
                    
        This function defines the Mehler Kernel Derivative
        
        Note:
        If we define K(x_1) = K(x_1, x_2) 
        then this function computes dK(x_1)/dx_1 which is vector valued (size of dimension

        """  
        out = np.zeros(point1.shape)
        evals = np.zeros((point1.shape))
        nPoints1 = point1.shape[0]
        dim = point1.shape[1]
        pass
 
class KernelHermite2D(Kernel):

    def __init__(self, tIn, dimension):
        assert dimension==2, "2DMehler Kernel"
        hyperParam = dict({})
        self.oneDKern = []
        for ii in xrange(dimension):
            hyperParam[ii] = tIn[ii]
            self.oneDKern.append(KernelHermite1D(tIn[ii],1))
        super(KernelHermite2D, self).__init__(hyperParam, dimension)
    
    def updateHyperParameters(self, params):
        for keys in self.hyperParam.keys():
            self.hyperParam[keys] = params[keys]
            self.oneDKern[keys].updateHyperParameters(dict({'t':params[keys]}))

    def evaluateF(self, x1, x2):
        """ 
        Parameter
        --------    
        x1 : 1darray 
            n x dimension
        x2 : 1darray  
            n x dimension

        Returns
        ------- 
        evaluation : float or 1darray      
                
        Notes
        ----- 

        """
        
        assert x1.shape == x2.shape, "__evaluate() received non-equal shaped point sets"
        nPoints = x1.shape[0]
        out = np.ones((x1.shape[0]))
        #print "x1 ", x1
        #print "x2 ", x2
        for ii, kern in enumerate(self.oneDKern):
            newVal = kern.evaluate(x1[:,ii].reshape((nPoints,1)),x2[:,ii].reshape((nPoints,1)))
            #print "newVal ", newVal
            out = out*newVal
        return out

    def derivative(self, x1, x2):
        """ Derivative of Mehler 2D kernel
        input:
            point1 : ndarray 
            point2 : 1xdimension

        output: 
            evaluation : float or ndarray 
                derivative of Gaussian function around point2     
                out[jj, ii] = dK(point1[jj,:], point2) / d point1[jj,ii]
                    
        This function defines the Mehler Kernel Derivative
        
        Note:
        If we define K(x_1) = K(x_1, x_2) 
        then this function computes dK(x_1)/dx_1 which is vector valued (size of dimension

        """  
        out = np.zeros(point1.shape)
        evals = np.zeros((point1.shape))
        nPoints1 = point1.shape[0]
        dim = point1.shape[1]
        pass
        #for ii, kern in enumerate(self.oneDKern):
        #    evals[:,ii] = kern.evaluate(x1[:,ii].reshape((nPoints1,1)),x2[:,ii].reshape((1,1)))

        #for ii, kern in enumerate(self.oneDKern):
        #    temp = np.ones((point1.shape))
        #    for jj, kern2 in enumerate(self.oneDKern):

        #        if ii == jj:
        #            temp2 = kern.derivative(x1[:,ii].reshape((nPoints1,1)),x2[:,ii].reshape((1,1)))
        #        else:
        #            temp = temp*evals[:,ii]
        #    out = out + np.dot(temp, temp2)
    
class KernelHermite1D(Kernel):
    """ This function defines the 1d mehler hermite kernel. """

    def __init__(self, tIn, dimension):
        assert dimension==1, "Mehler Hermite Kernel is only one dimensional"
        hyperParam = dict({})
        hyperParam['t'] = tIn
        super(KernelHermite1D, self).__init__(hyperParam, dimension)
           
    def evaluateF(self, x1, x2):
        """ 
        
        Parameter
        --------    
        x1 : 1darray 
            n x dimension
        x2 : 1darray  
            n x dimension

        Returns
        ------- 
        evaluation : float or 1darray      
                
        Notes
        ----- 

        """
        
        assert x1.shape[1]== 1 and x2.shape[1]==1, \
            "Hermite1d kernel only accepts one dimensional points"
        assert x1.shape == x2.shape, "__evaluate() received non-equal shaped point sets"
        
        out =  (1.0 - self.hyperParam['t']**2.0)**(-1.0/2.0) * \
            np.exp( - ( x1**2.0*self.hyperParam['t']**2.0  - 
                2.0 * self.hyperParam['t'] * x1 * x2 + x2**2.0 * self.hyperParam['t']**2.0) \
                / (2.0 * (1.0 - self.hyperParam['t']**2.0)))     
        #out = (2.0*np.pi)**-1.0 * out

        if math.isnan(out[0,0]):
            print "xs ", x1[0,:], x2[0,:]
            print "t", self.hyperParam['t']
            print 'NAN in kernel hermi1d exiting'
            exit()
        return  np.reshape(out, (x1.shape[0]))

    def derivative(self, x1, x2):
        """ Derivative of Mehler 1D kernel
        input:
            point1 : ndarray 
            point2 : 1xdimension

        output: 
            evaluation : float or ndarray 
                derivative of Gaussian function around point2     
                out[jj, ii] = dK(point1[jj,:], point2) / d point1[jj,ii]
                    
        This function defines the Mehler Kernel Derivative
        
        Note:
        If we define K(x_1) = K(x_1, x_2) 
        then this function computes dK(x_1)/dx_1 which is vector valued (size of dimension

        """  
        assert len(x2.shape) > 1 and len(x1.shape) > 1, "Must supply nd arrays to evaluation function"
        assert x2.shape[0] == 1 and x2.shape[1] == self.dimension, "x2 not in correct shape"
        assert x1.shape[0] > 0 and x1.shape[1] == self.dimension, "x1 not in correct shape"
        
        nPointsx1 = x1.shape[0]
        rEvals = self.evaluate(x1,x2)
        out = -0.5 * ( 2.0* x1 * self.hyperParam['t']**2.0 -\
                        2.0*self.hyperParam['t']*np.tile(x2, (nPointsx1,1)))/ \
                        (1.0-self.hyperParam['t']**2.0) *\
                         np.reshape(rEvals, (nPointsx1,1)) 
        
        return out
 
class KernelExplicitBasis(Kernel):
    """ This is the KernelExplicitBasis Class """
    
    def __init__(self, polynomial, eigenvalues, dimension, *args):
                     
        """ Initialize KernelExplicitBasis class """        
        self.polynomial = polynomial # needs to have evalPoly function
        self.eigenvalues = eigenvalues
        hyperParam = dict({})
        hyperParam['eigenvalues'] = eigenvalues
        # python standard i think
        super(KernelExplicitBasis, self).__init__(hyperParam, dimension)
 
    def evaluateKernelBasis(self, basis, points):
        return self.polynomial.evalPoly(basis, points)

    def evaluateKernelBasisDeriv(self, basis, points):
        return self.polynomial.evalPolyDeriv(basis, points)
    
    def evaluateF(self, x1, x2):
        """ 
        This function evaluates the kernel at the sets of points
        corresponding to point1 and point2. 
        
        Parameter
        ---------
        point1 : ndarray 
            evaluations x dimension
        point2 : ndarray 
            evaluations x dimension

        Returns
        ------- 
        
        evaluations : float or 1darray 
               evaluations      
                
        Notes
        -----
        
        """
        assert x1.shape[1]== self.dimension and x2.shape[1]==self.dimension, \
            "Dimension of points to evaluate is incorrect"
        assert x1.shape == x2.shape, "__evaluate() received non-equal shaped point sets"
        
       
        def forloop(ii, fn=None):
            out = np.zeros((np.shape(ii)[0], x1.shape[0]))
            ii = ii.reshape((len(ii)))
            overallIter = 0
            for jj in ii:
                out[overallIter,:] = self.eigenvalues[jj] * self.polynomial.evalPoly(jj, x1) *\
                                    self.polynomial.evalPoly(jj,x2)
                overallIter = overallIter + 1
            #print np.shape(out)
            return out
        
        evaluations = parallel_utilities.parallelizeMcForLoop(forloop, None,
                                 np.arange(len(self.eigenvalues)).reshape((len(self.eigenvalues),1)))

        evaluations = np.sum(evaluations,axis=0)
        #print "length of eigenvlaues ", len(self.eigenvalues)
        #print "shape of evaluations ", evaluations.shape
        #evaluations = 0.0
        #for jj in xrange(len(self.eigenvalues)):
        #    evaluations = evaluations + self.eigenvalues[jj]* \
        #        self.polynomial.evalPoly(jj, x1) * \
        #        self.polynomial.evalPoly(jj, x2)
        
        return evaluations 
    
    def derivative(self, x1, x2):
        """ Derivative of explicit basis kernel
        input:
            point1 : ndarray 
            point2 : 1xdimension

        output: 
            evaluation : float or ndarray 
                derivative of Gaussian function around point2     
                out[jj, ii] = dK(point1[jj,:], point2) / d point1[jj,ii]
                    
        This function defines the Explicit Basis Kernel
        
        Note:
        If we define K(x_1) = K(x_1, x_2) 
        then this function computes dK(x_1)/dx_1 which is vector valued (size of dimension

        """  
        assert len(x2.shape) > 1 and len(x1.shape) > 1, "Must supply nd arrays to evaluation function"
        assert x2.shape[0] == 1 and x2.shape[1] == self.dimension, "x2 not in correct shape"
        assert x1.shape[0] > 0 and x1.shape[1] == self.dimension, "x1 not in correct shape"

        nPointsx1 = x1.shape[0]
        x2tiled = np.tile(x2, (nPointsx1,1))
        out = np.zeros(x1.shape)
        for ii in xrange(len(self.eigenvalues)):
            out = out + self.eigenvalues[ii]*self.polynomial.evalPolyDeriv(ii, x1).reshape((x1.shape))* \
                    np.reshape(self.polynomial.evalPoly(ii, x2tiled), (nPointsx1,self.dimension))
        return out
