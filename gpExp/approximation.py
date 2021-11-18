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

class Space:

    """ The Space class describes the space and weighting in which we are interested """
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
