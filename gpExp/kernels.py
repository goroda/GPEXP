"""Kernels."""

# Code

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

import numpy as np
import abc


class Kernel(abc.ABC):
    """This is a Kernel Class.

    This is an abstract base class from which specific 
    kernels need to inherit and implement the `_evaluateF` function
    """

    def __init__(self, hyperParam, dimension, *argc):
        """Initialize Kernel class."""
        self.nugget = 0.0
        self.dimension = dimension
        self._hyperParam = hyperParam

    @property
    def hyper_params(self):
        """Get hyperparameters."""
        return self._hyperParam

    def updateHyperParameters(self, hyperParamNew):
        """Update hyperparameters of hyperparameters.

        Parameters
        ----------
        hyperParamNew : dict
                  A dictionary with a subset of existing hyperparameters
        """
        for key in hyperParamNew:
            assert key in self._hyperParam.keys(), \
                (key, " is not a valid hyperParameter")
            self._hyperParam[key] = hyperParamNew

    def evaluate(self, x1, x2):
        """Evaluate the kernel."""
        assert len(x2.shape) > 1 and len(x1.shape) > 1, \
            "Must supply nd arrays to evaluation function"

        nPointsx1 = x1.shape[0]
        nPointsx2 = x2.shape[0]
        assert x1.shape[1] == self.dimension and \
            x2.shape[1] == self.dimension, \
            f" Incorrect dimension of input: {x1.shape, x2.shape}"

        if nPointsx1 > nPointsx2:
            out = self._evaluateF(x1, np.tile(x2, (nPointsx1, 1)))
        elif nPointsx1 < nPointsx2:
            out = self._evaluateF(np.tile(x1, (nPointsx2, 1)), x2)
        else:
            out = self._evaluateF(x1, x2)

        return out

    @abc.abstractmethod
    def _evaluateF(self, x1, x2):
        """Evaluate the kernel.

        Parameters
        ----------
        x1 : ndarray (n, d)
        x2 : ndarray (n, d)
        """
        assert x1.shape == x2.shape, \
            "__evaluate() received non-equal shaped point sets"

    @abc.abstractmethod
    def derivativeWrtHypParams(self, x1, x2):
        """Compute derivative with respect to hyperparmaeters."""
        assert x1.shape == x2.shape, \
            "__evaluate() received non-equal shaped point sets"

    @abc.abstractmethod
    def derivative(self, x1, x2):
        """Compute derivative with respect to the first argument.

        Parameters
        ----------
        x1 : ndarray (N, d)
        x2 : ndarray (d)

        Returns
        -------
        evaluation : float or ndarray
            derivative of Gaussian function around point2
             out[jj, ii] = dK(point1[jj,:], point2) / d point1[jj,ii]
                if version == 1:
                    evaluation :float or ndarray
                        out[jj, ii] = dK(point2, point1[jj,:])/ point2[ii]
            This function defines the Gaussian kernel Derivative

        Notes
        If we define K(x_1) = K(x_1, x_2)
        Then this function computes dK(x_1)/dx_1 which is vector valued
        """
        assert x1.ndim == 2 and x2.ndim == 1, \
            "Must supply nd arrays to derivative function"
        assert x2.shape[0] == self.dimension, "x2 not in correct shape"
        assert x1.shape[0] > 0 and x1.shape[1] == self.dimension, \
            "x1 not in correct shape"


class KernelIsoMatern(Kernel):
    """Isotropic Matern Kernel."""

    def __init__(self, rho, signalSize, dimension, nu=3.0/2.0):
        """Initialize the isotropic matern kernel."""
        # nu is not treated as hyperparamter
        hyperParam = dict({'rho': rho, 'signalSize': signalSize})
        self.nu = nu
        super().__init__(hyperParam, dimension)

    def _evaluateF(self, x1, x2):
        """Private evaluate method in which x1 and x2 are the same shape."""
        super()._evaluateF(x1, x2)

        if np.abs(1.5-self.nu) < 1e-10:
            d = np.sqrt(np.sum((x1-x2)**2.0, axis=1))
            term = np.sqrt(3.0)*d/self._hyperParam['rho']
            out = self._hyperParam['signalSize'] * (1.0+term) * np.exp(-term)
        return out

    def derivativeWrtHypParams(self, x1, x2):
        """Compute Derivative with respect to hyperparameters."""
        # super().derivativeWrtHypParams(x1, x2)
        raise NotImplementedError(
            "derivativeWrtHypParams not implemented for"
            "KernelIsoMatern")

    def derivative(self, x1, x2):
        """Compute Derivative."""
        raise NotImplementedError(
            "derivative not implemented for"
            "KernelIsoMatern")


class KernelSquaredExponential(Kernel):
    """Squared Exponential Kernel exp(- (x-x')^2/(2*l^2).

    This is non-isotropic kernel with a different correlation parameter
    for each dimension
    """

    def __init__(self,  correlation_length, signal_size, dimension):
        """Initialize squared exponential kernel."""
        hyperParam = dict({})
        if len(correlation_length) == 1:
            correlation_length = np.tile(correlation_length, (dimension))
        for ii in range(len(correlation_length)):
            hyperParam['cl'+str(ii)] = correlation_length[ii]

        hyperParam['signalSize'] = signal_size
        super().__init__(hyperParam, dimension)

    def _evaluateF(self, x1, x2):
        """Evaluate."""
        super()._evaluateF(x1, x2)

        cl = np.zeros((self.dimension))
        for ii in range(self.dimension):
            cl[ii] = self._hyperParam['cl'+str(ii)]

        out = self._hyperParam['signalSize'] * \
            np.exp(-0.5 * np.sum((x1-x2)**2.0 *
                                 np.tile(cl**-2.0,
                                         (x1.shape[0], 1)),
                                 axis=1))
        return out

    def derivativeWrtHypParams(self, x1, x2):
        """Compute Derivative w.r.t hyperparameters."""
        super().derivativeWrtHypParams(x1, x2)

        cl = np.zeros((self.dimension))
        for ii in range(self.dimension):
            cl[ii] = self._hyperParam['cl'+str(ii)]

        out = {}
        evals = self._evaluateF(x1, x2)
        for key in self._hyperParam.keys():
            if key == 'signalSize':
                out[key] = np.exp(-0.5 *
                                  np.sum((x1-x2)**2.0 *
                                         np.tile(cl**-2.0, (x1.shape[0], 1)),
                                         axis=1))
            else:
                direction = int(key[2:])  # which direction
                out[key] = evals*(x1[:, direction] - x2[:, direction])**2.0 / \
                    cl[direction]**3.0

        return out

    def derivative(self, x1, x2):
        """Compute Derivative w.r.t first input."""
        super().derivative(x1, x2)

        cl = np.zeros((self.dimension))
        for ii in range(self.dimension):
            cl[ii] = self._hyperParam['cl'+str(ii)]

        # if version == 0 or version == 1:
        nPointsx1 = x1.shape[0]
        rEvals = self.evaluate(x1, x2[np.newaxis, :])
        x2t = np.tile(x2, (nPointsx1, 1))
        out = -self._hyperParam['signalSize'] * (x1 - x2t) / \
            np.tile(cl**2.0, (nPointsx1, 1)) * \
            np.tile(np.reshape(rEvals, (x1.shape[0], 1)),
                    ((1, self.dimension)))
        return out


class KernelMehler1D(Kernel):
    """One dimensional Mehler Hermite Kernel."""

    def __init__(self, tIn):
        """Initialize."""
        if (tIn < -1 or tIn > 1):
            raise ValueError("Mehler parameter must be between (-1,1)")
        hyperParam = dict({'t': tIn})
        super().__init__(hyperParam, 1)

    def _evaluateF(self, x1, x2):
        """Evaluate.

        Parameter
        --------
        x1 : 1darray
            n x dimension
        x2 : 1darray
            n x dimension

        Returns
        -------
        evaluation : float or 1darray
        """
        assert x1.shape[1] == 1 and x2.shape[1] == 1, \
            "Hermite1d kernel only accepts one dimensional points"

        out = (1.0 - self._hyperParam['t']**2.0)**(-1.0/2.0) * \
            np.exp(-(x1**2.0*self._hyperParam['t']**2.0 -
                     2.0 * self._hyperParam['t'] * x1 * x2 +
                     x2**2.0 * self._hyperParam['t']**2.0) / 
                   (2.0 * (1.0 - self._hyperParam['t']**2.0)))

        out = out.flatten()
        if any(np.isnan(out)):
            print("xs ", x1[0, :], x2[0, :])
            print("t", self._hyperParam['t'])
            print('NAN in kernel hermi1d exiting')
            raise RuntimeError("Mehler1D kernel returns NAN")

        return out

    def derivative(self, x1, x2):
        """Compute Derivative of Mehler 1D kernel."""
        super().derivative(x1, x2)

        nPointsx1 = x1.shape[0]
        rEvals = self.evaluate(x1, x2[np.newaxis, :])[:, np.newaxis]
        x2t = np.tile(x2, (nPointsx1, 1))
        num = 2.0 * x1 * self._hyperParam['t']**2.0 - \
            2.0 * self._hyperParam['t'] * x2t
        out = -0.5 * num / (1.0 - self._hyperParam['t']**2.0) * rEvals

        return out

    def derivativeWrtHypParams(self, x1, x2):
        """Compute Derivative with respect to hyperparameters."""
        # super().derivativeWrtHypParams(x1, x2)
        raise NotImplementedError(
            "derivativeWrtHypParams not implemented for"
            "KernelIsoMatern")


class KernelMehlerND(Kernel):
    """Mehler kernel in N dimensions."""

    def __init__(self, tIn):
        """Initialize."""
        hyperParam = dict({})
        self.oneDKern = []
        for ii in range(len(tIn)):
            hyperParam[ii] = tIn[ii]
            self.oneDKern.append(KernelMehler1D(tIn[ii]))
        super(KernelMehlerND, self).__init__(hyperParam, len(tIn))

    def updateHyperParameters(self, params):
        """Update Hyperparameters."""
        for keys in self._hyperParam:
            self._hyperParam[keys] = params[keys]

        for ii in range(self.dimension):
            self.oneDKern[ii].updateHyperParameters(
                {'t': self._hyperParam[ii]})

    def _evaluateF(self, x1, x2):
        super()._evaluateF(x1, x2)

        nPoints = x1.shape[0]
        out = np.ones((x1.shape[0]))
        for ii, kern in enumerate(self.oneDKern):
            newVal = kern.evaluate(x1[:, ii].reshape((nPoints, 1)),
                                   x2[:, ii].reshape((nPoints, 1)))

            out = out * newVal
        return out

    def derivative(self, x1, x2):
        """Compute the derivative."""
        super().derivative(x1, x2)
        raise NotImplementedError(
            "derivative of KernelMehlerND not yet implemented")

    def derivativeWrtHypParams(self, x1, x2):
        """Compute Derivative with respect to hyperparameters."""
        # super().derivativeWrtHypParams(x1, x2)
        raise NotImplementedError(
            "derivativeWrtHypParams not implemented for"
            "KernelIsoMatern")
