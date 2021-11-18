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
import math

def sampleLHS(numPts):
    #1D
    segSize = 1.0/float(numPts)
    pointVal = np.zeros(numPts)
    for ii in range(numPts):
        segMin = float(ii) * segSize
        point = segMin + (np.random.random() * segSize)
        pointVal[ii] = point #(point #* (1 - -1)) + -1 
    return pointVal


def genSample2DCircle(size, radius):
    #2Dcircle centered at 0
    #implemented using rejection sampling
    numPoints = size[0]
    numDims = size[1]
    samples = np.zeros(size)
    for ii in range(numPoints):
        notValid = 1
        while notValid:
            s = np.random.uniform(-1, 1, numDims)
            if np.sqrt(s[0]**2.0 + s[1]**2.0) < radius:
                notValid = 0
        
        samples[ii,:] = s

    return samples

def genSampleNDCircle(size, radius):

    numPoints = size[0]
    numDims = size[1]
    samples = np.zeros(size)
    for ii in range(numPoints):
        notValid = 1
        while notValid:
            s = np.random.uniform(-1,1,numDims)
            if np.sqrt(np.sum(s**2.0)) < radius:
                notValid = 0

        samples[ii,:] = s
    return samples

def distribFunc2DCircle(points, radius):
    area = np.pi*2.0*radius**2
    prob = 1.0/area
    out = np.zeros((len(points)))
    out[points[:,0]**2.0 + points[:,1]**2.0 < radius**2.0] = prob
    
    return out
  
def distribFuncNDCircle(points, radius):
    nDim = points.shape[1]
    area = np.pi**(nDim/2.0)/math.gamma(nDim/2.0 + 1)*radius**nDim
    prob = 1.0/area
    out = np.zeros((len(points)))
    out[points[:,0]**2.0 + points[:,1]**2.0 < radius**2.0] = prob

    return out

def genSampleUniform(size):
    # uniform on [-1, 1]
    return np.random.rand(size[0],size[1])*2.0 - 1.0

def distribFuncUniform(points):
    
    out = np.array(np.fabs(points)<1) / 2.0
    return out
 
def distribFuncUniform2D(points):
    out = np.array(np.fabs(points)<1) / 4.0   
    return out

def distribFunctionUniformND(points, dim):
    out = np.array(np.fabs(points)<1) / (2.0**dim)
    return out
    
def genSampleMickey(size):
    centersIn = [  (0.5,0),  (-0.5,0), (0.0,-0.5) ]
    radiusesIn = [ 0.3, 0.3, 0.5 ]
    
    centersOut = [ (0.0,-0.5) , (0.2, -0.3 ) , (-0.2, -0.3 ) ]
    radiusesOut = [ 0.05, 0.1, 0.1]
    
    #circles = { (0.5,0) : 0.3, (-0.5,0) : 0.3, (0.0,-0.5) : 0.5, (0.0, -0.5): 0.05 ,
    #           (0.2, -0.3 ) : 0.1, (-0.2, -0.3 ): 0.1}
    
    ellipseCenters = [ 0, -0.75]
    ellipseRad = [ 0.2, 0.1]
    
    
    nSamples = size[0]
    
    samplesKeep = np.zeros((0,2))
    while len(samplesKeep) < nSamples:
        samples = np.random.rand(nSamples,2)*2.0-1.0
        samplesOk = np.zeros((0,2))
        #now need to do rejection sampling
        
        #algorithm will check if its in any of the circles
        for center, rad in zip(centersIn, radiusesIn):
            rSquared = (samples[:,0]- center[0])**2.0 + (samples[:,1] - center[1])**2.0
            #print "rsquared ", np.nonzero(rSquared< rad**2)
            if len(rSquared) > 0:
                ptsok = samples[rSquared < rad**2]
                samplesOk = np.concatenate((samplesOk, ptsok), axis=0)
                samples = samples[rSquared > rad**2]
        
        for center, rad in zip(centersOut, radiusesOut):
            rSquared = (samplesOk[:,0]- center[0])**2.0 + (samplesOk[:,1] - center[1])**2.0
            if len(rSquared) > 0:
                samplesIn = samplesOk[rSquared < rad**2.0]
                samples = np.concatenate((samples, samplesIn ), axis=0)
                samplesOk = samplesOk[rSquared > rad**2]
        
        #check ellipse
        checkIn = (samplesOk[:,0] - ellipseCenters[0])**2.0/ellipseRad[0]**2 + \
             (samplesOk[:,1] - ellipseCenters[1])**2.0/ellipseRad[1]**2
        if len(checkIn) > 0:
            samplesIn = samplesOk[checkIn<1]
            samplesOk = samplesOk[checkIn>1]
            samples = np.concatenate((samples, samplesIn), axis=0)
        
        samplesKeep = np.concatenate((samplesKeep, samplesOk), axis=0)
        
    samplesKeep = samplesKeep[0:nSamples,:]   
    return samplesKeep

def distribFuncMickey(samplesInAAA):
    #really slow but works
    area = 1.172
    
    centersIn = [  (0.5,0),  (-0.5,0), (0.0,-0.5) ]
    radiusesIn = [ 0.3, 0.3, 0.5 ]
    
    centersOut = [ (0.0,-0.5) , (0.2, -0.3 ) , (-0.2, -0.3 ) ]
    radiusesOut = [ 0.05, 0.1, 0.1]
    
    #circles = { (0.5,0) : 0.3, (-0.5,0) : 0.3, (0.0,-0.5) : 0.5, (0.0, -0.5): 0.05 ,
    #           (0.2, -0.3 ) : 0.1, (-0.2, -0.3 ): 0.1}
    
    ellipseCenters = [ 0, -0.75]
    ellipseRad = [ 0.2, 0.1]
    samplesOk = np.zeros((0,2))
    samples = samplesInAAA.copy()
    #now need to do rejection sampling
    
    #algorithm will check if its in any of the circles
    for center, rad in zip(centersIn, radiusesIn):
        rSquared = (samples[:,0]- center[0])**2.0 + (samples[:,1] - center[1])**2.0
        if len(rSquared < rad**2) > 0:
            ptsok = samples[rSquared < rad**2]
            #ptsok = samples(list(np.nonzero(rSquared < rad**2)))
            samplesOk = np.concatenate((samplesOk, ptsok), axis=0)
        if len(rSquared > rad**2) > 0:
            samples = samples[rSquared > rad**2]
    
    for center, rad in zip(centersOut, radiusesOut):
        rSquared = (samplesOk[:,0]- center[0])**2.0 + (samplesOk[:,1] - center[1])**2.0
        if len(rSquared) > 0:
            samplesIn = samplesOk[rSquared < rad**2.0]
            samples = np.concatenate((samples, samplesIn ), axis=0)
            samplesOk = samplesOk[rSquared > rad**2]
    
    #check ellipse
    checkIn = (samplesOk[:,0] - ellipseCenters[0])**2.0/ellipseRad[0]**2 + \
         (samplesOk[:,1] - ellipseCenters[1])**2.0/ellipseRad[1]**2
    if len(checkIn) > 0:
        samplesIn = samplesOk[checkIn<1]
        samplesOk = samplesOk[checkIn>1]
        samples = np.concatenate((samples, samplesIn), axis=0)
    
    
    indGood = []
    for ii in range(len(samplesInAAA)):
        #bad sample
        for jj in range(len(samplesOk)):
            if np.linalg.norm(samplesInAAA[ii,:]-samplesOk[jj,:]) < 1e-15:
                indGood.append(ii)
                break
            
    out = np.zeros((len(samplesInAAA)))
    out[indGood] = 1.0/area        
    return out
    
def genSampleTriangle(size):
    #gen samples 2DTriangle with slope y = -x
    #size is number of samples
    numPoints = size[0]
    numDims = size[1]
    samples = np.zeros(size)
    for ii in range(numPoints):
        notValid = 1
        while notValid:
            s = np.random.uniform(-1, 1, numDims)
            if s[0] > -s[1]:
                notValid = 0
        
        samples[ii,:] = s

    return samples

def genSampleDonut(size, radius):
    # size is size of samples
    # radius is inner radius
    # donut with outer radius  = 1
    numPoints = size[0]
    numDims = size[1]
    samples = np.zeros(size)
    radius = 0.7;
    for ii in range(numPoints):
        notValid = 1
        while notValid:
            s = np.random.uniform(-1, 1, numDims)
            if np.sqrt(s[0]**2.0 + s[1]**2.0) > radius:
                notValid = 0
        
        samples[ii,:] = s

    return samples

def onedGaussDistrib(x):
    out = 1.0/(2.0*np.pi)**0.5 * np.exp(-0.5*x**2.0)
    return out

def twodGaussDistrib(x):
    #isotropic 2d gaussian
    out =  1.0/(2.0*np.pi) * np.exp(-0.5*x[:,0]**2.0 -0.5*x[:,1]**2.0)
    return out

