#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: 2021, Claudio S. Ravasio
# License: MIT (https://opensource.org/licenses/MIT)
# Author: Claudio S. Ravasio, PhD student at University College London (UCL), research assistant at King's College
# London (KCL), supervised by:
#   Dr Christos Bergeles, PI of the Robotics and Vision in Medicine (RViM) lab in the School of Biomedical Engineering &
#       Imaging Sciences (BMEIS) at King's College London (KCL)
#   Prof Lyndon Da Cruz, consultant ophthalmic surgeon, Moorfields Eye Hospital, London UK
#
# This file is part of oflibpytorch

import torch
import unittest
import cv2
import numpy as np
import math
import sys
sys.path.append('..')
from src.oflibpytorch.utils import get_valid_vecs, get_valid_shape, get_valid_ref, get_valid_mask, get_valid_padding, \
    validate_shape, \
    get_valid_device, to_numpy, move_axis, flow_from_matrix, matrix_from_transform, matrix_from_transforms, \
    reverse_transform_values, normalise_coords, apply_flow, threshold_vectors, from_matrix, from_transforms,  \
    load_kitti, load_sintel, load_sintel_mask, resize_flow, is_zero_flow, track_pts
from src.oflibpytorch.flow_class import Flow


class TestMoveAxis(unittest.TestCase):
    def test_move_axis(self):
        ip_tensor = torch.tensor(np.ones((1, 2, 3, 4)))
        ip_shape = ip_tensor.shape
        for i in range(4):
            for j in range(4):
                op_tensor = move_axis(ip_tensor, i, j)
                ip_shape_copy = list(ip_shape)
                active_dim = ip_shape[i]
                ip_shape_copy.pop(i)
                expected_shape = ip_shape_copy[:j] + [active_dim] + ip_shape_copy[j:]
                self.assertEqual(list(op_tensor.shape), expected_shape)

        for i in range(4):
            for j in range(4):
                op_tensor = move_axis(ip_tensor, -i - 1, -j - 1)
                ip_shape_copy = list(reversed(ip_shape))
                active_dim = ip_shape_copy[i]
                ip_shape_copy.pop(i)
                expected_shape = ip_shape_copy[:j] + [active_dim] + ip_shape_copy[j:]
                self.assertEqual(list(op_tensor.shape), list(reversed(expected_shape)))


class TestValidityChecks(unittest.TestCase):
    def test_get_valid_vecs(self):
        # Valid 3-dim vector inputs
        np_12hw = np.zeros((1, 2, 100, 200))
        np_2hw = np.zeros((2, 100, 200))
        np_hw2 = np.zeros((100, 200, 2))
        pt_2hw = torch.zeros((2, 100, 200))
        pt_hw2 = torch.zeros((100, 200, 2))
        for vecs in [np_2hw, np_hw2, pt_2hw, pt_hw2]:
            for desired_shape in [[100, 200], [1, 100, 200]]:
                self.assertIsInstance(get_valid_vecs(vecs, desired_shape), torch.Tensor)
                self.assertIsNone(np.testing.assert_equal(to_numpy(get_valid_vecs(vecs, desired_shape)), np_12hw))
            for desired_shape in [[110, 200], [5, 100, 200]]:
                with self.assertRaises(ValueError):
                    get_valid_vecs(vecs, desired_shape)

        # Valid 4-dim vector inputs
        np_n2hw = np.zeros((5, 2, 100, 200))
        np_nhw2 = np.zeros((5, 100, 200, 2))
        pt_n2hw = torch.zeros((5, 2, 100, 200))
        pt_nhw2 = torch.zeros((5, 100, 200, 2))
        for vecs in [np_n2hw, np_nhw2, pt_n2hw, pt_nhw2]:
            self.assertIsInstance(get_valid_vecs(vecs, [5, 100, 200]), torch.Tensor)
            self.assertIsNone(np.testing.assert_equal(to_numpy(get_valid_vecs(vecs, [5, 100, 200])), np_n2hw))
            for desired_shape in [[110, 200], [100, 200], [1, 100, 200]]:
                with self.assertRaises(ValueError):
                    get_valid_vecs(vecs, desired_shape)

        # Wrong vector type or shape
        with self.assertRaises(TypeError):
            get_valid_vecs('test')
        with self.assertRaises(ValueError):
            get_valid_vecs(torch.ones(2, 100, 200, 1))
        with self.assertRaises(ValueError):
            get_valid_vecs(np.zeros((3, 100, 200)))
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

    def test_get_valid_shape(self):
        with self.assertRaises(TypeError):
            get_valid_shape('test')
        with self.assertRaises(ValueError):
            get_valid_shape([10])
        with self.assertRaises(ValueError):
            get_valid_shape([10, 10, 10, 10])
        with self.assertRaises(ValueError):
            get_valid_shape([-1, 10])
        with self.assertRaises(ValueError):
            get_valid_shape([5, 0])
        with self.assertRaises(ValueError):
            get_valid_shape([10., 10])
        self.assertEqual(get_valid_shape([2, 3]), (1, 2, 3))
        self.assertEqual(get_valid_shape([4, 2, 3]), (4, 2, 3))
        self.assertEqual(get_valid_shape((2, 3)), (1, 2, 3))
        self.assertEqual(get_valid_shape((4, 2, 3)), (4, 2, 3))

    def test_get_valid_ref(self):
        self.assertEqual(get_valid_ref(None), 't')
        self.assertEqual(get_valid_ref('s'), 's')
        self.assertEqual(get_valid_ref('t'), 't')
        with self.assertRaises(TypeError):
            get_valid_ref(0)
        with self.assertRaises(ValueError):
            get_valid_ref('test')

    def test_get_valid_mask(self):
        # Valid 2-dim mask inputs
        np_1hw = np.zeros((1, 100, 200))
        np_hw = np.zeros((100, 200))
        pt_hw = torch.zeros((100, 200))
        for mask in [np_hw, pt_hw]:
            for desired_shape in [[100, 200], [1, 100, 200]]:
                self.assertIsInstance(get_valid_mask(mask, desired_shape), torch.Tensor)
                self.assertIsNone(np.testing.assert_equal(to_numpy(get_valid_mask(mask, desired_shape)), np_1hw))
            for desired_shape in [[110, 200], [5, 100, 200]]:
                with self.assertRaises(ValueError):
                    get_valid_mask(mask, desired_shape)

        # Valid 3-dim mask inputs
        np_nhw = np.zeros((5, 100, 200))
        pt_nhw = torch.zeros((5, 100, 200))
        for vecs in [np_nhw, pt_nhw]:
            self.assertIsInstance(get_valid_mask(vecs, [5, 100, 200]), torch.Tensor)
            self.assertIsNone(np.testing.assert_equal(to_numpy(get_valid_mask(vecs, [5, 100, 200])), np_nhw))
            for desired_shape in [[110, 200], [100, 200], [1, 100, 200]]:
                with self.assertRaises(ValueError):
                    get_valid_mask(vecs, desired_shape)

        with self.assertRaises(TypeError):  # mask input not numpy array or torch tensor
            get_valid_mask('test')
        with self.assertRaises(ValueError):  # mask numpy array input wrong number of dimensions
            get_valid_mask(np.zeros((1, 2, 100, 200)))
        with self.assertRaises(ValueError):  # mask torch tensor input wrong number of dimensions
            get_valid_mask(torch.zeros((2, 100, 200, 1)))
        with self.assertRaises(ValueError):  # mask numpy array height H does not match desired shape
            get_valid_mask(np.zeros((101, 200)), desired_shape=(100, 200))
        with self.assertRaises(ValueError):  # mask torch tensor width W does not match desired shape
            get_valid_mask(torch.ones(100, 201), desired_shape=(100, 200))
        with self.assertRaises(ValueError):  # mask numpy array values not either 0 or 1
            get_valid_mask(np.ones((100, 200)) * 20)
        with self.assertRaises(ValueError):  # mask torch tensor values not either 0 or 1
            get_valid_mask(torch.ones(100, 200) * 10)

    def test_get_valid_device(self):
        self.assertEqual(get_valid_device(None), torch.device('cpu'))
        self.assertEqual(get_valid_device('cpu'), torch.device('cpu'))
        self.assertEqual(get_valid_device('cuda'), torch.device('cuda:0'))
        self.assertEqual(get_valid_device(0), torch.device('cuda:0'))
        with self.assertRaises(ValueError):
            get_valid_device('test')
        if torch.cuda.is_available():
            self.assertEqual(get_valid_device('cuda'), torch.device('cuda:0'))
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
        # No transformation, equals passing identity matrix, to 200 by 300 flow field
        shape = [5, 200, 300]
        matrix = torch.eye(3).unsqueeze(0).repeat(5, 1, 1)
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_equal(to_numpy(flow), 0))
        self.assertIsNone(np.testing.assert_equal(flow.shape[0], shape[0]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[2:], shape[1:]))

    def test_translation(self):
        # Translation of 10 horizontally, 20 vertically, to 200 by 300 flow field
        shape = [1, 200, 300]
        matrix = torch.tensor([[1, 0, 10], [0, 1, 20], [0, 0, 1]]).unsqueeze(0)
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_equal(to_numpy(flow[:, 0]), 10))
        self.assertIsNone(np.testing.assert_equal(to_numpy(flow[:, 1]), 20))
        self.assertIsNone(np.testing.assert_equal(flow.shape[0], shape[0]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[2:], shape[1:]))

    def test_rotation(self):
        # Rotation of 30 degrees counter-clockwise, to 200 by 300 flow field
        shape = [1, 200, 300]
        matrix = torch.tensor([[math.sqrt(3) / 2, .5, 0], [-.5, math.sqrt(3) / 2, 0], [0, 0, 1]]).unsqueeze(0)
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_equal(to_numpy(flow[0, :, 0, 0]), [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 0, 299]), [-40.0584042685, -149.5], rtol=1e-6))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 199, 0]), [99.5, -26.6609446469], rtol=1e-6))
        self.assertIsNone(np.testing.assert_equal(flow.shape[0], shape[0]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[2:], shape[1:]))

    def test_rotation_with_shift(self):
        # Rotation of 30 degrees clockwise around point [10, 50] (hor, ver), to 200 by 300 flow field
        shape = [1, 200, 300]
        matrix = torch.tensor([[math.sqrt(3) / 2, -.5, 26.3397459622],
                               [.5, math.sqrt(3) / 2, 1.69872981078],
                               [0, 0, 1]]).unsqueeze(0)
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_array_almost_equal(to_numpy(flow[0, :, 50, 10]), [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 50, 299]), [-38.7186583063, 144.5], rtol=1e-6))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 199, 10]), [-74.5, -19.9622148361], rtol=1e-6))
        self.assertIsNone(np.testing.assert_equal(flow.shape[0], shape[0]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[2:], shape[1:]))

    def test_scaling(self):
        # Scaling factor 0.8, to 200 by 300 flow field
        shape = [1, 200, 300]
        matrix = torch.tensor([[.8, 0, 0], [0, .8, 0], [0, 0, 1]]).unsqueeze(0)
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_equal(to_numpy(flow[0, :, 0, 0]), [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 0, 100]), [-20, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 100, 0]), [0, -20]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[0], shape[0]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[2:], shape[1:]))

    def test_scaling_with_shift(self):
        # Scaling factor 2 around point [20, 30] (hor, ver), to 200 by 300 flow field
        shape = [2, 200, 300]
        matrix = torch.stack((torch.eye(3), torch.tensor([[2, 0, -20], [0, 2, -30], [0, 0, 1]])), dim=0)
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_array_almost_equal(to_numpy(flow[1, :, 30, 20]), [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[1, :, 30, 70]), [50, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[1, :, 80, 20]), [0, 50]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[0], shape[0]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[2:], shape[1:]))

    def test_device(self):
        device = ['cpu', 'cuda']
        if torch.cuda.is_available():
            expected_device = ['cpu', 'cuda']
        else:
            expected_device = ['cpu', 'cpu']
        shape = [1, 200, 300]
        matrix = torch.eye(3).unsqueeze(0)
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


class TestReverseTransformValues(unittest.TestCase):
    def test_reverse_transform_values(self):
        transform_list = [
            ['translation', 10, 20],
            ['rotation', 100, 43, -30],
            ['scaling', 44, 12, 1.25]
        ]
        reversed_transform_list = [
            ['translation', -10, -20],
            ['rotation', 100, 43, 30],
            ['scaling', 44, 12, .8]
        ]
        self.assertEqual(reverse_transform_values(transform_list), reversed_transform_list)


class TestNormaliseCoords(unittest.TestCase):
    def test_normalise(self):
        coords = torch.tensor([[0, 0],
                               [-1, 11],
                               [10, 5],
                               [21, 11],
                               [20, 10]])
        shape = (11, 21)
        exp_coords = torch.tensor([[-1, -1],
                                   [-1.1, 1.2],
                                   [0, 0],
                                   [1.1, 1.2],
                                   [1, 1]])
        coord_list = [coords, coords.unsqueeze(0), coords.unsqueeze(0).unsqueeze(0).repeat(4, 3, 1, 1)]
        exp_coord_list = [exp_coords, exp_coords.unsqueeze(0), exp_coords.unsqueeze(0).unsqueeze(0).repeat(4, 3, 1, 1)]
        for c, e_c in zip(coord_list, exp_coord_list):
            self.assertIsNone(np.testing.assert_allclose(to_numpy(normalise_coords(c, shape)),
                                                         to_numpy(e_c),
                                                         rtol=1e-6))


class TestApplyFlow(unittest.TestCase):
    def test_rotation(self):
        img = cv2.imread('smudge.png')
        img = torch.tensor(np.moveaxis(img, -1, 0))
        for dev in ['cpu', 'cuda']:
            for ref in ['t', 's']:
                flow = Flow.from_transforms([['rotation', 255.5, 255.5, 90]], img.shape[1:], ref).vecs
                warped_img = apply_flow(flow.to(dev), img, ref)
                desired_img = img.transpose(1, 2).flip(1)
                self.assertIsNone(np.testing.assert_equal(to_numpy(warped_img), to_numpy(desired_img)))

    def test_translation(self):
        img = cv2.imread('smudge.png')
        img = cv2.resize(img, None, fx=1, fy=.5)
        img = torch.tensor(np.moveaxis(img, -1, 0))
        for dev in ['cpu', 'cuda']:
            for ref in ['s', 't']:
                flow = Flow.from_transforms([['translation', 10, -20]], img.shape[1:], ref).vecs
                warped_img = apply_flow(flow.to(dev), img, ref)
                self.assertIsNone(np.testing.assert_equal(to_numpy(warped_img[:, :-20, 10:]),
                                                          to_numpy(img[:, 20:, :-10])))

    def test_2d_target(self):
        img = cv2.imread('smudge.png', 0)
        img = cv2.resize(img, None, fx=1, fy=.5)
        img = torch.tensor(img)
        for dev in ['cpu', 'cuda']:
            for ref in ['s', 't']:
                flow = Flow.from_transforms([['translation', 10, -20]], img.shape, ref).vecs
                warped_img = apply_flow(flow.to(dev), img, ref)
                self.assertIsNone(np.testing.assert_equal(to_numpy(warped_img[:-20, 10:]),
                                                          to_numpy(img[20:, :-10])))

    def test_batch_sizes(self):
        img = cv2.imread('smudge.png')
        img = cv2.resize(img, None, fx=1, fy=.5)
        i_chw = torch.tensor(np.moveaxis(img, -1, 0))
        i_hw = torch.tensor(np.moveaxis(img, -1, 0))[0]
        i_1chw = i_chw.unsqueeze(0)
        i_11hw = i_1chw[:, 0:1]
        i_nchw = i_1chw.repeat(4, 1, 1, 1)
        i_n1hw = i_11hw.repeat(4, 1, 1, 1)
        for ref in ['s', 't']:
            for f in [
                Flow.from_transforms([['translation', 10, -20]], img.shape[:2], ref).vecs,
                Flow.from_transforms([['translation', 10, -20]], img.shape[:2], ref).vecs.repeat(4, 1, 1, 1),
            ]:
                for i in [i_1chw, i_11hw, i_nchw, i_n1hw]:
                    warped_i = apply_flow(f, i, ref)
                    self.assertEqual(warped_i.shape[0], max(f.shape[0], i.shape[0]))
                    for w_ind, i_ind in zip(warped_i, i):
                        self.assertIsNone(np.testing.assert_equal(to_numpy(w_ind[:-20, 10:]),
                                                                  to_numpy(i_ind[20:, :-10])))
        f = Flow.from_transforms([['translation', 10, -20]], img.shape[:2], 't').vecs.repeat(4, 1, 1, 1)
        warped_i = apply_flow(f, i_hw, 't')
        self.assertEqual(warped_i.shape, (4, i_hw.shape[0], i_hw.shape[1]))
        warped_i = apply_flow(f, i_chw, 't')
        self.assertEqual(warped_i.shape, (4, 3, i_hw.shape[0], i_hw.shape[1]))

    def test_apply_flow_failed(self):
        flow = Flow.from_transforms([['translation', 2, 0]], (10, 10)).vecs
        with self.assertRaises(TypeError):  # target is not torch tensor
            apply_flow(flow, target='test', ref='t')
        with self.assertRaises(ValueError):  # target torch tensor too few dimensions
            apply_flow(flow, target=torch.zeros(5), ref='t')
        with self.assertRaises(ValueError):  # target torch tensor too many dimensions
            apply_flow(flow, target=torch.zeros((1, 1, 1, 1, 5)), ref='t')
        with self.assertRaises(ValueError):  # target torch tensor shape does not match flow shape
            apply_flow(flow, target=torch.zeros((11, 10)), ref='t')
        with self.assertRaises(ValueError):  # target torch tensor shape does not match flow shape
            apply_flow(flow, target=torch.zeros((1, 11, 10)), ref='t')
        with self.assertRaises(ValueError):  # target torch tensor shape does not match flow shape
            apply_flow(flow, target=torch.zeros((1, 1, 11, 10)), ref='t')
        with self.assertRaises(ValueError):  # target torch tensor batch size does not match flow batch size
            apply_flow(flow=torch.zeros((3, 1, 11, 10)), target=torch.zeros((2, 1, 11, 10)), ref='t')


class TestThresholdVectors(unittest.TestCase):
    def test_threshold(self):
        vecs = torch.zeros((5, 2, 10, 1))
        vecs[2, 0, 0, 0] = -1e-5
        vecs[2, 0, 1, 0] = 1e-4
        vecs[2, 0, 2, 0] = -1e-3
        vecs[2, 0, 3, 0] = 1
        for use_mag in [True, False]:
            thresholded = threshold_vectors(vecs, threshold=1e-3, use_mag=use_mag)
            thresholded = to_numpy(thresholded[2, 0, :4, 0])
            self.assertIsNone(np.testing.assert_allclose(thresholded, [0, 0, -1e-3, 1]))
            thresholded = threshold_vectors(vecs, threshold=1e-4, use_mag=use_mag)
            thresholded = to_numpy(thresholded[2, 0, :4, 0])
            self.assertIsNone(np.testing.assert_allclose(thresholded, [0, 1e-4, -1e-3, 1]))
            thresholded = threshold_vectors(vecs, threshold=1e-5, use_mag=use_mag)
            thresholded = to_numpy(thresholded[2, 0, :4, 0])
            self.assertIsNone(np.testing.assert_allclose(thresholded, [-1e-5, 1e-4, -1e-3, 1]))


class TestFromMatrix(unittest.TestCase):
    def test_from_matrix(self):
        # With reference 's', this simply corresponds to using flow_from_matrix, tested in test_utils.
        # With reference 't':
        # Rotation of 30 degrees clockwise around point [10, 50] (hor, ver)
        matrix = np.array([[math.sqrt(3) / 2, -.5, 26.3397459622],
                           [.5, math.sqrt(3) / 2, 1.69872981078],
                           [0, 0, 1]])
        shape = [200, 300]
        flow = from_matrix(matrix, shape, 't')
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 50, 10]), [0, 0], atol=1e-4))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 50, 299]), [38.7186583063, 144.5], rtol=1e-4))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 199, 10]), [-74.5, 19.9622148361], rtol=1e-4))
        self.assertIsNone(np.testing.assert_equal(flow.shape[0], 1))
        self.assertIsNone(np.testing.assert_equal(flow.shape[2:], shape))

        # With and without inverse matrix for ref 't'
        matrix = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]])
        inv_matrix = np.array([[1, 0, -10], [0, 1, -20], [0, 0, 1]])
        vecs = to_numpy(from_matrix(matrix, shape, ref='t', matrix_is_inverse=False), switch_channels=True)
        inv_vecs = to_numpy(from_matrix(inv_matrix, shape, ref='t', matrix_is_inverse=True), switch_channels=True)
        self.assertIsNone(np.testing.assert_allclose(vecs, inv_vecs, rtol=1e-3))

    def test_failed_from_matrix(self):
        with self.assertRaises(ValueError):  # Invalid shape size
            from_matrix(torch.eye(3), [2, 10, 10], 't')
        with self.assertRaises(TypeError):  # Invalid matrix type
            from_matrix('test', [10, 10], 't')
        with self.assertRaises(ValueError):  # Invalid matrix shape
            from_matrix(np.eye(4), [10, 10], 't')
        with self.assertRaises(TypeError):  # Invalid matrix_is_inverse type
            from_matrix(torch.eye(3), [10, 10], matrix_is_inverse='test')
        with self.assertRaises(ValueError):  # matrix_is_inverse True despite ref = 's'
            from_matrix(torch.eye(3), [10, 10], ref='s', matrix_is_inverse=True)


class TestFromTransforms(unittest.TestCase):
    def test_from_transforms_rotation(self):
        shape = [200, 300]
        transforms = [['rotation', 10, 50, -30]]
        flow = from_transforms(transforms, shape, 't')
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 50, 10]), [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 50, 299]), [38.7186583063, 144.5]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 199, 10]), [-74.5, 19.9622148361], rtol=1e-6))
        self.assertIsNone(np.testing.assert_equal(flow.shape[0], 1))
        self.assertIsNone(np.testing.assert_equal(flow.shape[2:], shape))

    def test_from_transforms_scaling(self):
        shape = [200, 300]
        transforms = [['scaling', 20, 30, 2]]
        flow = from_transforms(transforms, shape, 's')
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 30, 20]), [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 30, 70]), [50, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 80, 20]), [0, 50]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[0], 1))
        self.assertIsNone(np.testing.assert_equal(flow.shape[2:], shape))

    def test_from_transforms_multiple_s(self):
        shape = [200, 300]
        transforms = [
            ['translation', -20, -30],
            ['scaling', 0, 0, 2],
            ['translation', 20, 30]
        ]
        flow = from_transforms(transforms, shape, 's')
        self.assertIsNone(np.testing.assert_equal(to_numpy(flow[0, :, 30, 20]), [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 30, 70]), [50, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 80, 20]), [0, 50]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[0], 1))
        self.assertIsNone(np.testing.assert_equal(flow.shape[2:], shape))

    def test_from_transforms_multiple_t(self):
        shape = [200, 300]
        transforms = [
            ['translation', -10, -50],
            ['rotation', 0, 0, -30],
            ['translation', 10, 50]
        ]
        flow = from_transforms(transforms, shape, 't')
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 50, 10]), [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 50, 299]), [38.7186583063, 144.5]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 199, 10]), [-74.5, 19.9622148361], rtol=1e-6))
        self.assertIsNone(np.testing.assert_equal(flow.shape[0], 1))
        self.assertIsNone(np.testing.assert_equal(flow.shape[2:], shape))

    def test_failed_from_transforms(self):
        shape = [200, 300]
        transforms = 'test'
        with self.assertRaises(TypeError):  # transforms not a list
            from_transforms(transforms, shape, 't')
        transforms = ['test']
        with self.assertRaises(TypeError):  # transform not a list
            from_transforms(transforms, shape, 't')
        transforms = [['translation', 20, 10], ['rotation']]
        with self.assertRaises(ValueError):  # transform missing information
            from_transforms(transforms, shape, 't')
        transforms = [['translation', 20, 10], ['rotation', 1]]
        with self.assertRaises(ValueError):  # transform with incomplete information
            from_transforms(transforms, shape, 't')
        transforms = [['translation', 20, 10], ['rotation', 1, 'test', 10]]
        with self.assertRaises(ValueError):  # transform with invalid information
            from_transforms(transforms, shape, 't')
        transforms = [['translation', 20, 10], ['test', 1, 1, 10]]
        with self.assertRaises(ValueError):  # transform type invalid
            from_transforms(transforms, shape, 't')


class TestFromKITTI(unittest.TestCase):
    def test_load(self):
        output = load_kitti('kitti.png')
        self.assertIsInstance(output, torch.Tensor)
        desired_flow = np.arange(0, 10)[:, np.newaxis] * np.arange(0, 20)[np.newaxis, :]
        self.assertIsNone(np.testing.assert_equal(to_numpy(output[0, ...]), desired_flow))
        self.assertIsNone(np.testing.assert_equal(to_numpy(output[1, ...]), 0))
        self.assertIsNone(np.testing.assert_equal(to_numpy(output[2, :, 0]), 1))
        self.assertIsNone(np.testing.assert_equal(to_numpy(output[2, :, 10]), 0))

    def test_failed_load(self):
        with self.assertRaises(ValueError):  # Wrong path
            load_kitti('test')
        with self.assertRaises(ValueError):  # Wrong flow shape
            load_kitti('kitti_wrong.png')


class TestFromSintel(unittest.TestCase):
    def test_load_flow(self):
        f = load_sintel('sintel.flo')
        self.assertIsInstance(f, torch.Tensor)
        desired_flow = np.arange(0, 10)[:, np.newaxis] * np.arange(0, 20)[np.newaxis, :]
        self.assertIsNone(np.testing.assert_equal(to_numpy(f[0, ...]), desired_flow))
        self.assertIsNone(np.testing.assert_equal(to_numpy(f[1, ...]), 0))

    def test_failed_load_flow(self):
        with self.assertRaises(TypeError):  # Path not a string
            load_sintel(0)
        with self.assertRaises(ValueError):  # Wrong tag
            load_sintel('sintel_wrong.flo')

    def test_load_mask(self):
        m = load_sintel_mask('sintel_invalid.png')
        self.assertIsInstance(m, torch.Tensor)
        self.assertIsNone(np.testing.assert_equal(to_numpy(m[:, 0]), True))
        self.assertIsNone(np.testing.assert_equal(to_numpy(m[:, 10]), False))

    def test_failed_load_mask(self):
        with self.assertRaises(TypeError):  # Path not a string
            load_sintel_mask(0)
        with self.assertRaises(ValueError):  # File does not exist
            load_sintel_mask('test.png')


class TestResizeFlow(unittest.TestCase):
    def test_resize(self):
        shape = [20, 10]
        ref = 's'
        flow = Flow.from_transforms([['rotation', 30, 50, 30]], shape, ref).vecs
        for f in [flow, flow.squeeze(0)]:
            # Different scales
            scales = [.2, .5, 1, 1.5, 2, 10]
            for scale in scales:
                resized_flow = resize_flow(f, scale)
                resized_shape = scale * np.array(f.shape[-2:])
                self.assertIsNone(np.testing.assert_equal(resized_flow.shape[-2:], resized_shape))
                self.assertIsNone(np.testing.assert_allclose(to_numpy(resized_flow[..., 0, 0]),
                                                             to_numpy(f[..., 0, 0]) * scale, rtol=.1))
                self.assertEqual(len(f.shape), len(resized_flow.shape))

            # Scale list
            scale = [.5, 2]
            resized_flow = resize_flow(f, scale)
            resized_shape = np.array(scale) * np.array(f.shape[-2:])
            self.assertIsNone(np.testing.assert_equal(resized_flow.shape[-2:], resized_shape))
            self.assertIsNone(np.testing.assert_allclose(to_numpy(resized_flow[..., 0, 0]),
                                                         to_numpy(f[..., 0, 0]) * np.array(scale)[::-1], rtol=.1))
            self.assertEqual(len(f.shape), len(resized_flow.shape))

            # Scale tuple
            scale = (2, .5)
            resized_flow = resize_flow(f, scale)
            resized_shape = np.array(scale) * np.array(f.shape[-2:])
            self.assertIsNone(np.testing.assert_equal(resized_flow.shape[-2:], resized_shape))
            self.assertIsNone(np.testing.assert_allclose(to_numpy(resized_flow[..., 0, 0]),
                                                         to_numpy(f[..., 0, 0]) * np.array(scale)[::-1], rtol=.1))
            self.assertEqual(len(f.shape), len(resized_flow.shape))

    def test_resize_on_fields(self):
        # Check scaling is performed correctly based on the actual flow field
        ref = 't'
        flow_small = Flow.from_transforms([['rotation', 0, 0, 30]], (50, 80), ref).vecs_numpy
        flow_large = Flow.from_transforms([['rotation', 0, 0, 30]], (150, 240), ref).vecs
        flow_resized = to_numpy(resize_flow(flow_large, 1 / 3), switch_channels=True)
        self.assertIsNone(np.testing.assert_allclose(flow_resized, flow_small, atol=1, rtol=.1))

    def test_failed_resize(self):
        flow = Flow.from_transforms([['rotation', 30, 50, 30]], [20, 10], 's').vecs
        with self.assertRaises(TypeError):  # Wrong shape type
            resize_flow(flow, 'test')
        with self.assertRaises(ValueError):  # Wrong shape values
            resize_flow(flow, ['test', 0])
        with self.assertRaises(ValueError):  # Wrong shape shape
            resize_flow(flow, [1, 2, 3])
        with self.assertRaises(ValueError):  # Shape is 0
            resize_flow(flow, 0)
        with self.assertRaises(ValueError):  # Shape below 0
            resize_flow(flow, -0.1)


class TestIsZeroFlow(unittest.TestCase):
    def test_is_zero_flow(self):
        flow = np.zeros((10, 10, 2), 'float32')
        self.assertEqual(is_zero_flow(flow, thresholded=True), True)
        self.assertEqual(is_zero_flow(flow, thresholded=False), True)

        flow[:3, :, 0] = 1e-4
        self.assertEqual(is_zero_flow(flow, thresholded=True), True)
        self.assertEqual(is_zero_flow(flow, thresholded=False), False)

        flow[:3, :, 1] = -1e-3
        self.assertEqual(is_zero_flow(flow, thresholded=True), False)
        self.assertEqual(is_zero_flow(flow, thresholded=False), False)

        flow[0, 0] = 10
        self.assertEqual(is_zero_flow(flow, thresholded=True), False)
        self.assertEqual(is_zero_flow(flow, thresholded=False), False)

    def test_failed_is_zero_flow(self):
        with self.assertRaises(TypeError):  # Wrong thresholded type
            is_zero_flow(np.zeros((10, 10, 2)), 'test')


class TestTrackPts(unittest.TestCase):
    def test_track_pts(self):
        f_s = Flow.from_transforms([['rotation', 0, 0, 30]], (512, 512), 's').vecs
        f_t = Flow.from_transforms([['rotation', 0, 0, 30]], (512, 512), 't').vecs
        pts = torch.tensor([[20.5, 10.5], [8.3, 7.2], [120.4, 160.2]])
        desired_pts = [
            [12.5035207776, 19.343266740],
            [3.58801085141, 10.385382907],
            [24.1694586156, 198.93726969]
        ]

        # Reference 's'
        pts_tracked_s = track_pts(f_s, 's', pts)
        self.assertIsInstance(pts_tracked_s, torch.Tensor)
        self.assertIsNone(np.testing.assert_allclose(to_numpy(pts_tracked_s), desired_pts, rtol=1e-6))

        # Reference 't'
        pts_tracked_t = track_pts(f_t, 't', pts)
        self.assertIsInstance(pts_tracked_t, torch.Tensor)
        self.assertIsNone(np.testing.assert_allclose(to_numpy(pts_tracked_t), desired_pts))

        # Reference 't', integer output
        pts_tracked_t = track_pts(f_t, 't', pts, int_out=True)
        self.assertIsInstance(pts_tracked_t, torch.Tensor)
        self.assertIsNone(np.testing.assert_equal(to_numpy(pts_tracked_t), np.round(desired_pts)))
        self.assertEqual(pts_tracked_t.dtype, torch.long)

        # Test tracking for 's' flow and int pts (checked via debugger)
        f = Flow.from_transforms([['translation', 10, 20]], (512, 512), 's').vecs
        pts = np.array([[20, 10], [8, 7]])
        desired_pts = [[40, 20], [28, 17]]
        pts_tracked_s = track_pts(f, 's', torch.tensor(pts))
        self.assertIsNone(np.testing.assert_equal(to_numpy(pts_tracked_s), desired_pts))

    def test_device_track_pts(self):
        for d1 in ['cpu', 'cuda']:
            for d2 in ['cpu', 'cuda']:
                f = Flow.from_transforms([['translation', 10, 20]], (512, 512), 's', device=d1).vecs
                pts = torch.tensor([[20, 10], [8, 7]]).to(d2)
                pts_tracked = track_pts(f, 's', pts)
                self.assertEqual(pts_tracked.device, f.device)

    def test_failed_track_pts(self):
        pts = torch.tensor([[20, 10], [20, 10], [8, 7]])
        flow = torch.zeros((2, 10, 10))
        with self.assertRaises(TypeError):  # Wrong pts type
            track_pts(flow, 's', pts='test')
        with self.assertRaises(ValueError):  # Wrong pts shape
            track_pts(flow, 's', pts=torch.zeros((10, 10, 2)))
        with self.assertRaises(ValueError):  # Pts channel not of size 2
            track_pts(flow, 's', pts=pts.transpose(0, -1))
        with self.assertRaises(TypeError):  # Wrong int_out type
            track_pts(flow, 's', pts, int_out='test')


if __name__ == '__main__':
    unittest.main()
