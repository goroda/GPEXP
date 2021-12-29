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

    def test_sqexp(self):
        dimension = 1
        cor_length = np.array([0.5])
        signal_size = 0.5
        kernel = kernels.KernelSquaredExponential(cor_length,
                                                  signal_size,
                                                  dimension)

        x1 = np.array([[0.2]])
        x2 = np.array([[0.8]])
        val = kernel.evaluate(x1, x2)
        self.assertEqual(1, val.ndim, "val is not the correct size")
        self.assertEqual(1, val.shape[0], "not correct num elems")
        should_be = signal_size * \
            np.exp(-0.5 * (0.2 - 0.8)**2 / cor_length[0]**2)
        self.assertAlmostEqual(should_be, val[0], 7, "not equal!")
