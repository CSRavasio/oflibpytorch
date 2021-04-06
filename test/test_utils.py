#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: 2021, Claudio S. Ravasio
# License: LGPL 2.1
# Author: Claudio S. Ravasio, PhD student at University College London (UCl), supervised by:
#   Dr Christos Bergeles, PI of the Robotics and Vision in Medicine (RViM) lab in the School of Biomedical Engineering &
#       Imaging Sciences (BMEIS) at King's College London (KCL)
#   Prof Lyndon Da Cruz, consultant ophthalmic surgeon, Moorfields Eye Hospital, London UK
#
# This file is part of oflibpytorch

import torch
import unittest
import numpy as np
import math
from oflibpytorch.utils import get_valid_ref, get_valid_padding, validate_shape, get_valid_device, to_numpy, \
    flow_from_matrix


class TestValidityChecks(unittest.TestCase):
    def test_get_valid_ref(self):
        self.assertEqual(get_valid_ref(None), 't')
        self.assertEqual(get_valid_ref('s'), 's')
        self.assertEqual(get_valid_ref('t'), 't')
        with self.assertRaises(TypeError):
            get_valid_ref(0)
        with self.assertRaises(ValueError):
            get_valid_ref('test')

    def test_get_valid_device(self):
        self.assertEqual(get_valid_device(None), 'cpu')
        self.assertEqual(get_valid_device('cpu'), 'cpu')
        with self.assertRaises(ValueError):
            get_valid_device(0)
        with self.assertRaises(ValueError):
            get_valid_device('test')
        if torch.cuda.is_available():
            self.assertEqual(get_valid_device('cuda'), 'cuda')
        else:
            with self.assertRaises(ValueError):
                get_valid_device('cuda')
                # NOTE: Unsure of how to get cuda.is_available() to return False for testing purposes. Setting CUDA
                #   visibility to "" or "-1" still has cuda.is_available() return True.

    def test_get_valid_padding(self):
        with self.assertRaises(TypeError):
            get_valid_padding(100)
        with self.assertRaises(ValueError):
            get_valid_padding([10, 20, 30, 40, 50])
        with self.assertRaises(ValueError):
            get_valid_padding([10., 20, 30, 40])
        with self.assertRaises(ValueError):
            get_valid_padding([-10, 10, 10, 10])

    def test_validate_shape(self):
        with self.assertRaises(TypeError):
            validate_shape('test')
        with self.assertRaises(ValueError):
            validate_shape([10, 10, 10])
        with self.assertRaises(ValueError):
            validate_shape([-1, 10])
        with self.assertRaises(ValueError):
            validate_shape([10., 10])


class TestFlowFromMatrix(unittest.TestCase):
    # All numerical values in calculated manually and independently
    def test_identity(self):
        # No transformation, equals passing identy matrix, to 200 by 300 flow field
        shape = [200, 300]
        matrix = torch.eye(3)
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_equal(to_numpy(flow), 0))
        self.assertIsNone(np.testing.assert_equal(flow.shape[1:], shape))

    def test_translation(self):
        # Translation of 10 horizontally, 20 vertically, to 200 by 300 flow field
        shape = [200, 300]
        matrix = torch.tensor([[1, 0, 10], [0, 1, 20], [0, 0, 1]])
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_equal(to_numpy(flow[0]), 10))
        self.assertIsNone(np.testing.assert_equal(to_numpy(flow[1]), 20))
        self.assertIsNone(np.testing.assert_equal(flow.shape[1:], shape))

    def test_rotation(self):
        # Rotation of 30 degrees counter-clockwise, to 200 by 300 flow field
        shape = [200, 300]
        matrix = torch.tensor([[math.sqrt(3) / 2, .5, 0], [-.5, math.sqrt(3) / 2, 0], [0, 0, 1]])
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_equal(to_numpy(flow[:, 0, 0]), [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[:, 0, 299]), [-40.0584042685, -149.5], rtol=1e-6))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[:, 199, 0]), [99.5, -26.6609446469], rtol=1e-6))
        self.assertIsNone(np.testing.assert_equal(flow.shape[1:], shape))

    def test_rotation_with_shift(self):
        # Rotation of 30 degrees clockwise around point [10, 50] (hor, ver), to 200 by 300 flow field
        shape = [200, 300]
        matrix = torch.tensor([[math.sqrt(3) / 2, -.5, 26.3397459622],
                               [.5, math.sqrt(3) / 2, 1.69872981078],
                               [0, 0, 1]])
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_array_almost_equal(to_numpy(flow[:, 50, 10]), [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[:, 50, 299]), [-38.7186583063, 144.5], rtol=1e-6))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[:, 199, 10]), [-74.5, -19.9622148361], rtol=1e-6))
        self.assertIsNone(np.testing.assert_equal(flow.shape[1:], shape))

    def test_scaling(self):
        # Scaling factor 0.8, to 200 by 300 flow field
        shape = [200, 300]
        matrix = torch.tensor([[.8, 0, 0], [0, .8, 0], [0, 0, 1]])
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_equal(to_numpy(flow[:, 0, 0]), [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[:, 0, 100]), [-20, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[:, 100, 0]), [0, -20]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[1:], shape))

    def test_scaling_with_shift(self):
        # Scaling factor 2 around point [20, 30] (hor, ver), to 200 by 300 flow field
        shape = [200, 300]
        matrix = torch.tensor([[2, 0, -20], [0, 2, -30], [0, 0, 1]])
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_array_almost_equal(to_numpy(flow[:, 30, 20]), [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[:, 30, 70]), [50, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[:, 80, 20]), [0, 50]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[1:], shape))

    def test_device(self):
        device = ['cpu', 'cuda']
        if torch.cuda.is_available():
            expected_device = ['cpu', 'cuda']
        else:
            expected_device = ['cpu', 'cpu']
        shape = [200, 300]
        matrix = torch.eye(3)
        for dev, expected_dev in zip(device, expected_device):
            flow = flow_from_matrix(matrix.to(dev), shape)
            self.assertEqual(flow.device.type, expected_dev)


if __name__ == '__main__':
    unittest.main()
