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
from oflibpytorch.utils import get_valid_vecs, get_valid_ref, get_valid_padding, validate_shape, get_valid_device, \
    to_numpy, flow_from_matrix, matrix_from_transform, matrix_from_transforms, normalise_coords
from oflibpytorch.flow_class import Flow


class TestValidityChecks(unittest.TestCase):
    def test_get_valid_vecs(self):
        # Different valid vector inputs
        vecs_np_2hw = np.zeros((2, 100, 200))
        vecs_np_hw2 = np.zeros((100, 200, 2))
        vecs_pt_2hw = torch.zeros((2, 100, 200))
        vecs_pt_hw2 = torch.zeros((100, 200, 2))
        for vecs in [vecs_np_2hw, vecs_np_hw2, vecs_pt_2hw, vecs_pt_hw2]:
            self.assertIsInstance(get_valid_vecs(vecs), torch.Tensor)
            self.assertIsNone(np.testing.assert_equal(to_numpy(get_valid_vecs(vecs)), vecs_np_2hw))

        # Wrong vector type or shape
        with self.assertRaises(TypeError):
            get_valid_vecs('test')
        with self.assertRaises(ValueError):
            get_valid_vecs(np.zeros((2, 100, 200, 1)))
        with self.assertRaises(ValueError):
            get_valid_vecs(torch.ones(2, 100, 200, 1))
        with self.assertRaises(ValueError):
            get_valid_vecs(np.zeros((3, 100, 200)))
        with self.assertRaises(ValueError):
            get_valid_vecs(torch.ones(3, 100, 200))
        with self.assertRaises(ValueError):
            get_valid_vecs(torch.ones(2, 100, 200), desired_shape=(110, 200))

        # Invalid vector values
        vectors = np.random.rand(100, 200, 2)
        vectors[10, 10] = np.NaN
        vectors[20, 20] = np.Inf
        vectors[30, 30] = -np.Inf
        with self.assertRaises(ValueError):
            get_valid_vecs(vectors)
        vectors = torch.tensor(vectors)
        with self.assertRaises(ValueError):
            get_valid_vecs(vectors)

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


class TestMatrixFromTransforms(unittest.TestCase):
    # All numerical values in desired_matrix calculated manually
    def test_combined_transforms(self):
        transforms = [
            ['translation', -100, -100],
            ['rotation', 0, 0, 30],
            ['translation', 100, 100]
        ]
        actual_matrix = matrix_from_transforms(transforms)
        desired_matrix = matrix_from_transform('rotation', [100, 100, 30])
        self.assertIsInstance(matrix_from_transforms(transforms), torch.Tensor)
        self.assertIsNone(np.testing.assert_equal(to_numpy(actual_matrix), to_numpy(desired_matrix)))


class TestMatrixFromTransform(unittest.TestCase):
    # All numerical values in desired_matrix calculated manually
    def test_translation(self):
        # Translation of 15 horizontally, 10 vertically
        desired_matrix = np.eye(3)
        transform = 'translation'
        values = [15, 10]
        desired_matrix[0, 2] = 15
        desired_matrix[1, 2] = 10
        self.assertIsInstance(matrix_from_transform(transform, values), torch.Tensor)
        self.assertIsNone(np.testing.assert_equal(desired_matrix, matrix_from_transform(transform, values)))

    def test_rotation(self):
        # Rotation of 30 degrees counter-clockwise
        desired_matrix = np.eye(3)
        transform = 'rotation'
        values = [0, 0, 30]
        desired_matrix[:2, :2] = [[math.sqrt(3) / 2, .5], [-.5, math.sqrt(3) / 2]]
        self.assertIsInstance(matrix_from_transform(transform, values), torch.Tensor)
        self.assertIsNone(np.testing.assert_allclose(desired_matrix, matrix_from_transform(transform, values)))

        # Rotation of 45 degrees clockwise
        desired_matrix = np.eye(3)
        transform = 'rotation'
        values = [0, 0, -45]
        desired_matrix[:2, :2] = [[1 / math.sqrt(2), -1 / math.sqrt(2)], [1 / math.sqrt(2), 1 / math.sqrt(2)]]
        self.assertIsInstance(matrix_from_transform(transform, values), torch.Tensor)
        self.assertIsNone(np.testing.assert_allclose(desired_matrix, matrix_from_transform(transform, values)))

    def test_rotation_with_shift(self):
        # Rotation of 30 degrees clockwise around point [10, 50] (hor, ver)
        desired_matrix = np.eye(3)
        transform = 'rotation'
        values = [10, 50, -30]
        desired_matrix[:2, :2] = [[math.sqrt(3) / 2, -.5], [.5, math.sqrt(3) / 2]]
        desired_matrix[0, 2] = 26.3397459622
        desired_matrix[1, 2] = 1.69872981078
        self.assertIsInstance(matrix_from_transform(transform, values), torch.Tensor)
        self.assertIsNone(np.testing.assert_allclose(desired_matrix, matrix_from_transform(transform, values),
                                                     rtol=1e-6))

        # Rotation of 45 degrees counter-clockwise around point [-20, -30] (hor, ver)
        desired_matrix = np.eye(3)
        transform = 'rotation'
        values = [-20, -30, 45]
        desired_matrix[:2, :2] = [[1 / math.sqrt(2), 1 / math.sqrt(2)], [-1 / math.sqrt(2), 1 / math.sqrt(2)]]
        desired_matrix[0, 2] = 15.3553390593
        desired_matrix[1, 2] = -22.9289321881
        self.assertIsInstance(matrix_from_transform(transform, values), torch.Tensor)
        self.assertIsNone(np.testing.assert_allclose(desired_matrix, matrix_from_transform(transform, values)))

    def test_scaling(self):
        # Scaling factor 0.8
        desired_matrix = np.eye(3)
        transform = 'scaling'
        values = [0, 0, 0.8]
        desired_matrix[0, 0] = 0.8
        desired_matrix[1, 1] = 0.8
        self.assertIsInstance(matrix_from_transform(transform, values), torch.Tensor)
        self.assertIsNone(np.testing.assert_allclose(desired_matrix, matrix_from_transform(transform, values)))

        # Scaling factor 2
        desired_matrix = np.eye(3)
        transform = 'scaling'
        values = [0, 0, 2]
        desired_matrix[0, 0] = 2
        desired_matrix[1, 1] = 2
        self.assertIsInstance(matrix_from_transform(transform, values), torch.Tensor)
        self.assertIsNone(np.testing.assert_allclose(desired_matrix, matrix_from_transform(transform, values)))

    def test_scaling_with_shift(self):
        # Scaling factor 0.8 around point [10, 50] (hor, ver)
        desired_matrix = np.eye(3)
        transform = 'scaling'
        values = [10, 50, 0.8]
        desired_matrix[0, 0] = 0.8
        desired_matrix[1, 1] = 0.8
        desired_matrix[0, 2] = 2
        desired_matrix[1, 2] = 10
        self.assertIsInstance(matrix_from_transform(transform, values), torch.Tensor)
        self.assertIsNone(np.testing.assert_allclose(desired_matrix, matrix_from_transform(transform, values)))

        # Scaling factor 2 around point [20, 30] (hor, ver)
        desired_matrix = np.eye(3)
        transform = 'scaling'
        values = [20, 30, 2]
        desired_matrix[0, 0] = 2
        desired_matrix[1, 1] = 2
        desired_matrix[0, 2] = -20
        desired_matrix[1, 2] = -30
        self.assertIsInstance(matrix_from_transform(transform, values), torch.Tensor)
        self.assertIsNone(np.testing.assert_allclose(desired_matrix, matrix_from_transform(transform, values)))


class TestNormaliseCoords(unittest.TestCase):
    def test_normalise(self):
        coords = torch.tensor([[0, 0],
                               [-1, 11],
                               [10, 5],
                               [21, 11],
                               [20, 10]])
        shape = (11, 21)
        normalised_coords = normalise_coords(coords, shape)
        expected_normalised_coords = torch.tensor([[-1, -1],
                                                   [-1.1, 1.2],
                                                   [0, 0],
                                                   [1.1, 1.2],
                                                   [1, 1]])
        self.assertIsNone(np.testing.assert_allclose(to_numpy(normalised_coords), to_numpy(expected_normalised_coords),
                                                     rtol=1e-6))


if __name__ == '__main__':
    unittest.main()
