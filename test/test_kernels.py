"""Test Kernels."""

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

import unittest

import numpy as np
import gpExp.kernels as kernels


class TestKernelSquaredExponential(unittest.TestCase):

    def setUp(self):
        dimension = 1
        self.cor_length = np.array([0.5])
        self.signal_size = 0.5
        self.kernel = kernels.KernelSquaredExponential(self.cor_length,
                                                       self.signal_size,
                                                       dimension)

    def test_sqexp(self):

        x1 = np.array([[0.2]])
        x2 = np.array([[0.8]])
        val = self.kernel.evaluate(x1, x2)
        self.assertEqual(1, val.ndim, "val is not the correct size")
        self.assertEqual(1, val.shape[0], "not correct num elems")
        should_be = self.signal_size * \
            np.exp(-0.5 * (0.2 - 0.8)**2 / self.cor_length[0]**2)
        self.assertAlmostEqual(should_be, val[0], 7, "not equal!")

    def test_sqexp_multi(self):

        x1 = np.array([[0.2], [0.5]])
        x2 = np.array([[0.7]])

        val = self.kernel.evaluate(x1, x2)
        self.assertEqual(1, val.ndim, "val is not the correct size")
        self.assertEqual(2, val.shape[0], "not correct num elems")

        should_be1 = self.signal_size * \
            np.exp(-0.5 * (x1[0, 0] - x2[0, 0])**2 / self.cor_length[0]**2)
        should_be2 = self.signal_size * \
            np.exp(-0.5 * (x1[1, 0] - x2[0, 0])**2 / self.cor_length[0]**2)

        self.assertAlmostEqual(should_be1, val[0], 7, "not equal!")
        self.assertAlmostEqual(should_be2, val[1], 7, "not equal!")

    def test_sqexp_derivative_wrt_hyp(self):
        """Test the derviative with respect to the hyperparameters."""

        dim = 2
        cl = np.array([0.5, 0.3])
        sig_size = 0.9
        kernel = kernels.KernelSquaredExponential(cl, sig_size, dim)

        self.assertIn('cl0', kernel.hyper_params)
        self.assertIn('cl1', kernel.hyper_params)
        self.assertIn('signalSize', kernel.hyper_params)

        x1 = np.array([[2.0, 3.0], [4.0, 5.0]])
        x2 = np.array([[0.2, 0.4], [-0.2, 0.3]])

        derivs = kernel.derivativeWrtHypParams(x1, x2)

        self.assertEqual(2, derivs['signalSize'].shape[0])

        diff = (x1 - x2)**2
        diff[:, 0] /= -2.0 * cl[0]**2
        diff[:, 1] /= -2.0 * cl[1]**2
        diff = np.sum(diff, axis=1)
        should_be = np.exp(diff)
        self.assertAlmostEqual(should_be[0], derivs['signalSize'][0], 10)
        self.assertAlmostEqual(should_be[1], derivs['signalSize'][1], 10)

        self.assertIs('cl0' in derivs, True)
        self.assertIs('cl1' in derivs, True)
        should_be = sig_size * np.exp(diff) * \
            (-0.5 * (x1[:, 0] - x2[:, 0])**2 * -2 / cl[0]**3)
        self.assertAlmostEqual(should_be[0], derivs['cl0'][0], 10)
        self.assertAlmostEqual(should_be[1], derivs['cl0'][1], 10)

        should_be = sig_size * np.exp(diff) * \
            (-0.5 * (x1[:, 1] - x2[:, 1])**2 * -2 / cl[1]**3)
        self.assertAlmostEqual(should_be[0], derivs['cl1'][0], 10)
        self.assertAlmostEqual(should_be[1], derivs['cl1'][1], 10)

    def test_sqexp_derivative(self):
        """Test the derivative with respect to the first argument."""
        dim = 2
        cl = np.array([0.5, 0.3])
        sig_size = 0.9
        kernel = kernels.KernelSquaredExponential(cl, sig_size, dim)

        x1 = np.array([[2.0, 3.0], [4.0, 5.0]])
        x2 = np.array([0.2, 0.4])

        derivs = kernel.derivative(x1, x2)

        self.assertEqual(2, derivs.ndim)
        self.assertEqual(2, derivs.shape[1])
        self.assertEqual(2, derivs.shape[0])

        x2t = np.tile(x2[np.newaxis, :], (2, 1))
        diff = (x1 - x2t)**2
        diff[:, 0] /= -2.0 * cl[0]**2
        diff[:, 1] /= -2.0 * cl[1]**2
        diff = np.sum(diff, axis=1)
        should_be = sig_size * np.exp(diff) * (-0.5 / cl**2 * 2 * (x1 - x2t))

        self.assertAlmostEqual(should_be[0, 0], derivs[0, 0], 10)
        self.assertAlmostEqual(should_be[0, 1], derivs[0, 1], 10)
        self.assertAlmostEqual(should_be[1, 0], derivs[1, 0], 10)
        self.assertAlmostEqual(should_be[1, 1], derivs[1, 1], 10)


class TestKernelMehler1D(unittest.TestCase):

    def setUp(self):
        self.rho = 0.5
        self.kernel = kernels.KernelMehler1D(self.rho)

    def test_mehler(self):

        drho = 1 - self.rho**2
        pre = 1.0 / np.sqrt(drho)
        den = 2 * drho

        x1 = np.array([[0.2]])
        x2 = np.array([[0.8]])

        val = self.kernel.evaluate(x1, x2)
        self.assertEqual(1, val.ndim, "val is not the correct size")
        self.assertEqual(1, val.shape[0], "not correct num elems")

        num = (self.rho**2 * (x1**2 + x2**2) - 2 * self.rho * x1 * x2)
        should_be = pre * np.exp(-num / den).flatten()
        self.assertAlmostEqual(should_be[0], val[0], 7, "not equal!")

    @unittest.skip("not yet implemented")
    def test_mehler_derivative_wrt_hyp(self):
        """Test the derviative with respect to the hyperparameters."""
        pass

    def test_mehler_derivative(self):
        """Test the derivative with respect to the first argument."""

        drho = 1 - self.rho**2
        pre = 1.0 / np.sqrt(drho)
        den = 2 * drho

        x1 = np.array([[2.0], [4.0]])
        x2 = np.array([0.2])

        derivs = self.kernel.derivative(x1, x2)

        self.assertEqual(2, derivs.ndim)
        self.assertEqual(1, derivs.shape[1])
        self.assertEqual(2, derivs.shape[0])

        num = (self.rho**2 * (x1**2 + x2**2) - 2 * self.rho * x1 * x2)
        vals = pre * np.exp(-num / den).flatten()

        num_two = -(self.rho**2 * 2 * x1.flatten() - 2 * self.rho * x2.flatten())
        should_be = vals * num_two / den

        self.assertAlmostEqual(should_be[0], derivs[0, 0], 10)
        self.assertAlmostEqual(should_be[1], derivs[1, 0], 10)


class TestKernelMehlerND(unittest.TestCase):

    def setUp(self):
        self.rho = np.array([0.5, 0.3])
        self.kernel = kernels.KernelMehlerND(self.rho)

    def test_mehler(self):

        drho = 1 - self.rho**2
        pre = 1.0 / np.sqrt(drho)
        den = 2 * drho

        x1 = np.array([[0.2, 0.7]])
        x2 = np.array([[0.8, -0.2]])

        val = self.kernel.evaluate(x1, x2)
        self.assertEqual(1, val.ndim, "val is not the correct size")
        self.assertEqual(1, val.shape[0], "not correct num elems")

        num = (self.rho**2 * (x1**2 + x2**2) - 2 * self.rho * x1 * x2)
        should_be = pre * np.exp(-num / den)
        self.assertAlmostEqual(np.prod(should_be), val[0], 7, "not equal!")

    @unittest.skip("not yet implemented")
    def test_mehler_derivative_wrt_hyp(self):
        """Test the derviative with respect to the hyperparameters."""
        pass

    @unittest.skip("not yet implemented")    
    def test_mehler_derivative(self):
        """Test the derivative with respect to the first argument."""

        drho = 1 - self.rho**2
        pre = 1.0 / np.sqrt(drho)
        den = 2 * drho

        x1 = np.array([[2.0], [4.0]])
        x2 = np.array([0.2])

        derivs = self.kernel.derivative(x1, x2)

        self.assertEqual(2, derivs.ndim)
        self.assertEqual(1, derivs.shape[1])
        self.assertEqual(2, derivs.shape[0])

        num = (self.rho**2 * (x1**2 + x2**2) - 2 * self.rho * x1 * x2)
        vals = pre * np.exp(-num / den).flatten()

        num_two = -(self.rho**2 * 2 * x1.flatten() - 2 * self.rho * x2.flatten())
        should_be = vals * num_two / den

        self.assertAlmostEqual(should_be[0], derivs[0, 0], 10)
        self.assertAlmostEqual(should_be[1], derivs[1, 0], 10)
        
