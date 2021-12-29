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

from gpExp.experimentalDesign import *
from gpExp.approximation import Space
import gpExp.utilities as utilities

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def plotCircle(xp ,mP, addYlabel=0):
    # Plotting setup
    golden_mean = (np.sqrt(5)-1.0)/2.0
    #figWidth = 1.6 # inches (so that cna fit 3 figures on a line)
    figWidth = 1.3 # inches (so that cna fit 4 figures on a line)
    figHeight = figWidth #figWidth*golden_mean
    fig_size = [figWidth,figHeight]
    #params = {'backend': 'ps',
    params = {'axes.labelsize':9,
              'text.fontsize': 9,
              'legend.fontsize':9,
              'xtick.labelsize':7,
              'ytick.labelsize':7,
              'text.usetex': False,
              'figure.figsize':fig_size}
    matplotlib.rcParams.update(params)

    fig = plt.figure()
    if addYlabel==1:
            plt.axes([0.39,0.35,0.95-0.39,0.95-0.35])
            plt.ylabel('$x^{(2)}$')
    else:
        plt.axes([0.3,0.35,0.95-0.32,0.95-0.35])

    plt.plot(xp[:,0], xp[:,1], 'ok',ms=2)#+colors[iteration])
    circle1 = plt.Circle((0,0), radius, color='r', fill=False )
    fig.gca().add_artist(circle1)
    plt.xlabel('$x^{(1)}$')
    plt.axis('equal')
    plt.axis([-1, 1, -1, 1])
 
#####################################
# Define Space
#####################################
dimension = 2
radius = 0.7
sampleFunction = lambda size, r=radius : utilities.genSample2DCircle(size, r)
distributionFunction = lambda points, r=radius : utilities.distribFunc2DCircle(points, r)
space = Space(dimension, sampleFunction, distributionFunction)

n = 20
nMC = 3000#max(10000,50*n)
mcPoints = sampleFunction((nMC,dimension))


pts = performLJExperimentalDesign(mcPoints,n)

fig = plt.figure()
plt.plot(pts[:,0], pts[:,1], 'ok',ms=2)#+colors[iteration])
circle1 = plt.Circle((0,0), radius, color='r', fill=False )
fig.gca().add_artist(circle1)
plt.xlabel('$x^{(1)}$')
plt.axis('equal')
plt.axis([-1, 1, -1, 1])
plt.show()
 
