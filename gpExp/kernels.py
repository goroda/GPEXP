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
from . import parallel_utilities
import math

class Kernel(object):
    """ This is a Kernel Class """
    
    nugget = 0.0
    hyperParam = dict({})
    def __init__(self, hyperParam, dimension, *argc):
        """ Initializes Kernel class """
        
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
    """ Matern Kernel """
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
        raise AttributeError("derivativeWrtHypParams not implemented for KernelIsoMatern")


class KernelSquaredExponential(Kernel):
    """  exp(- (x-x')^2/(2*l^2) """

    def __init__(self,  correlationLength, signalSize, dimension):
       
        hyperParam = dict({})
        if len(correlationLength) == 1:
            correlationLength = np.tile( correlationLength, (dimension))     
        for ii in range(len(correlationLength)):
            hyperParam['cl'+str(ii)] = correlationLength[ii]

        hyperParam['signalSize'] = signalSize 
        super(KernelSquaredExponential, self).__init__(hyperParam, dimension)
         
    def evaluateF(self, x1, x2):
        """ Private evaluate method in which x1 and x2 are the same shape """
        assert x1.shape == x2.shape, "__evaluate() received non-equal shaped point sets"
        cl = np.zeros((self.dimension))
        for ii in range(self.dimension):
            cl[ii] = self.hyperParam['cl'+str(ii)]

        out = self.hyperParam['signalSize']* \
            np.exp (-0.5 * np.sum((x1-x2)**2.0*np.tile(cl**-2.0, (x1.shape[0],1)),axis=1) )
        return out
    
    def derivativeWrtHypParams(self, x1, x2):
        #Assume that derivative is taken at current hyperparameters

        assert x1.shape == x2.shape, "__evaluate() received non-equal shaped point sets"

        cl = np.zeros((self.dimension))
        for ii in range(self.dimension):
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
        for ii in range(self.dimension):
            cl[ii] = self.hyperParam['cl'+str(ii)]

        if version == 0 or version == 1:
            nPointsx1 = x1.shape[0]
            rEvals = self.evaluate(x1,x2)
            out = -self.hyperParam['signalSize']*0.5*2 * (x1 - np.tile(x2, (nPointsx1,1)))/ \
                    np.tile(cl**2.0, (nPointsx1,1))* \
                    np.tile(np.reshape(rEvals, (x1.shape[0],1)), ((1,self.dimension))) 
        return out

class KernelMehlerND(Kernel):
    """ Mehler kernel in ND """
    def __init__(self, tIn, dimension):
        hyperParam = dict({})
        self.oneDKern = []
        for ii in range(dimension):
            hyperParam[ii] = tIn[ii]
            self.oneDKern.append(KernelMehler1D(tIn[ii],1))
        super(KernelMehlerND, self).__init__(hyperParam, dimension)
    
    def updateHyperParameters(self, params):
        for keys in self.hyperParam.keys():
            self.hyperParam[keys] = params[keys]

        for ii in range(self.dimension):
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
        raise AttributeError("derivative of KernelMehlerND not yet implemented")

class KernelMehler1D(Kernel):
    """ This function defines the 1d mehler hermite kernel. """

    def __init__(self, tIn, dimension):
        assert dimension==1, "Mehler Hermite Kernel is only one dimensional"
        hyperParam = dict({})
        hyperParam['t'] = tIn
        super(KernelMehler1D, self).__init__(hyperParam, dimension)
           
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
            print("xs ", x1[0,:], x2[0,:])
            print("t", self.hyperParam['t'])
            print('NAN in kernel hermi1d exiting')
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

