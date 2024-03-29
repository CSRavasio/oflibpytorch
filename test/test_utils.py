#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: 2022, Claudio S. Ravasio
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
    get_valid_device, to_numpy, move_axis, flow_from_matrix, matrix_from_transform, matrix_from_transforms, \
    reverse_transform_values, normalise_coords, apply_flow, threshold_vectors, from_matrix, from_transforms,  \
    load_kitti, load_sintel, load_sintel_mask, resize_flow, is_zero_flow, track_pts, get_flow_endpoints, \
    grid_from_unstructured_data, apply_s_flow, get_pure_pytorch, set_pure_pytorch, unset_pure_pytorch, to_tensor
from src.oflibpytorch.flow_class import Flow
from src.oflibpytorch.flow_operations import batch_flows


class TestPurePytorch(unittest.TestCase):
    def test_set_pure_pytorch(self):
        unset_pure_pytorch()
        set_pure_pytorch(warn=True)
        self.assertEqual(get_pure_pytorch(), True)

    def test_unset_pure_pytorch(self):
        set_pure_pytorch()
        unset_pure_pytorch(warn=True)
        self.assertEqual(get_pure_pytorch(), False)


class TestTensorNumpy(unittest.TestCase):
    def test_to_numpy(self):
        pt = torch.zeros((3, 2, 20, 20), requires_grad=True)
        arr = to_numpy(pt, switch_channels=True)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (3, 20, 20, 2))
        if torch.cuda.is_available():
            pt = pt.to('cuda')
            arr = to_numpy(pt, switch_channels=True)
            self.assertIsInstance(arr, np.ndarray)
            self.assertEqual(arr.shape, (3, 20, 20, 2))

    def test_to_tensor(self):
        arr = np.zeros((3, 20, 20, 2))
        pt = to_tensor(arr)
        self.assertIsInstance(pt, torch.Tensor)
        self.assertEqual(pt.shape, (3, 20, 20, 2))
        pt = to_tensor(arr, switch_channels='batched')
        self.assertIsInstance(pt, torch.Tensor)
        self.assertEqual(pt.shape, (3, 2, 20, 20))
        pt = to_tensor(arr[0], switch_channels='single')
        self.assertIsInstance(pt, torch.Tensor)
        self.assertEqual(pt.shape, (2, 20, 20))
        if torch.cuda.is_available():
            pt = to_tensor(arr, device='cuda')
            self.assertEqual(pt.device.type, 'cuda')


class TestMoveAxis(unittest.TestCase):
    def test_move_axis(self):
        ip_tensor = torch.tensor(np.ones((1, 2, 3, 4)), requires_grad=True)
        ip_shape = ip_tensor.shape
        for i in range(4):
            for j in range(4):
                op_tensor = move_axis(ip_tensor, i, j)
                self.assertIsNotNone(op_tensor.grad_fn)
                ip_shape_copy = list(ip_shape)
                active_dim = ip_shape[i]
                ip_shape_copy.pop(i)
                expected_shape = ip_shape_copy[:j] + [active_dim] + ip_shape_copy[j:]
                self.assertEqual(list(op_tensor.shape), expected_shape)

        for i in range(4):
            for j in range(4):
                op_tensor = move_axis(ip_tensor, -i - 1, -j - 1)
                self.assertIsNotNone(op_tensor.grad_fn)
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
        pt_2hw = torch.zeros((2, 100, 200), requires_grad=True)
        pt_hw2 = torch.zeros((100, 200, 2), requires_grad=True)
        for vecs in [np_2hw, np_hw2, pt_2hw, pt_hw2]:
            for desired_shape in [[100, 200], [1, 100, 200]]:
                v = get_valid_vecs(vecs, desired_shape)
                self.assertIsInstance(v, torch.Tensor)
                self.assertIsNone(np.testing.assert_equal(to_numpy(v), np_12hw))
                if isinstance(vecs, torch.Tensor):
                    self.assertIsNotNone(v.grad_fn)
            for desired_shape in [[110, 200], [5, 100, 200]]:
                with self.assertRaises(ValueError):
                    get_valid_vecs(vecs, desired_shape)

        # Valid 4-dim vector inputs
        np_n2hw = np.zeros((5, 2, 100, 200))
        np_nhw2 = np.zeros((5, 100, 200, 2))
        pt_n2hw = torch.zeros((5, 2, 100, 200))
        pt_nhw2 = torch.zeros((5, 100, 200, 2))
        for vecs in [np_n2hw, np_nhw2, pt_n2hw, pt_nhw2]:
            v = get_valid_vecs(vecs, [5, 100, 200])
            self.assertIsInstance(v, torch.Tensor)
            self.assertIsNone(np.testing.assert_equal(to_numpy(v), np_n2hw))
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
        with self.assertRaises(ValueError):
            get_valid_vecs(vectors[..., 0])
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
                m = get_valid_mask(mask, desired_shape)
                self.assertIsInstance(m, torch.Tensor)
                self.assertIsNone(np.testing.assert_equal(to_numpy(m), np_1hw))
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


class TestFlowFromMatrix(unittest.TestCase):
    # All numerical values in calculated manually and independently
    def test_identity(self):
        # No transformation, equals passing identity matrix, to 200 by 300 flow field
        shape = [5, 200, 300]
        matrix = torch.eye(3, requires_grad=True).unsqueeze(0).expand(5, -1, -1)
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_equal(to_numpy(flow), 0))
        self.assertIsNone(np.testing.assert_equal(flow.shape[0], shape[0]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[2:], shape[1:]))
        self.assertIsNotNone(flow.grad_fn)

    def test_translation(self):
        # Translation of 10 horizontally, 20 vertically, to 200 by 300 flow field
        shape = [1, 200, 300]
        matrix = torch.tensor([[1, 0, 10], [0, 1, 20], [0, 0, 1.]], requires_grad=True).unsqueeze(0)
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_equal(to_numpy(flow[:, 0]), 10))
        self.assertIsNone(np.testing.assert_equal(to_numpy(flow[:, 1]), 20))
        self.assertIsNone(np.testing.assert_equal(flow.shape[0], shape[0]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[2:], shape[1:]))
        self.assertIsNotNone(flow.grad_fn)

    def test_rotation(self):
        # Rotation of 30 degrees counter-clockwise, to 200 by 300 flow field
        shape = [1, 200, 300]
        matrix = torch.tensor([
            [math.sqrt(3) / 2, .5, 0], [-.5, math.sqrt(3) / 2, 0], [0, 0, 1]
        ], requires_grad=True).unsqueeze(0)
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_equal(to_numpy(flow[0, :, 0, 0]), [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 0, 299]), [-40.0584042685, -149.5], rtol=1e-6))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 199, 0]), [99.5, -26.6609446469], rtol=1e-6))
        self.assertIsNone(np.testing.assert_equal(flow.shape[0], shape[0]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[2:], shape[1:]))
        self.assertIsNotNone(flow.grad_fn)

    def test_rotation_with_shift(self):
        # Rotation of 30 degrees clockwise around point [10, 50] (hor, ver), to 200 by 300 flow field
        shape = [1, 200, 300]
        matrix = torch.tensor([[math.sqrt(3) / 2, -.5, 26.3397459622],
                               [.5, math.sqrt(3) / 2, 1.69872981078],
                               [0, 0, 1]], requires_grad=True).unsqueeze(0)
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_array_almost_equal(to_numpy(flow[0, :, 50, 10]), [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 50, 299]), [-38.7186583063, 144.5], rtol=1e-6))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 199, 10]), [-74.5, -19.9622148361], rtol=1e-6))
        self.assertIsNone(np.testing.assert_equal(flow.shape[0], shape[0]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[2:], shape[1:]))
        self.assertIsNotNone(flow.grad_fn)

    def test_scaling(self):
        # Scaling factor 0.8, to 200 by 300 flow field
        shape = [1, 200, 300]
        matrix = torch.tensor([[.8, 0, 0], [0, .8, 0], [0, 0, 1]], requires_grad=True).unsqueeze(0)
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_equal(to_numpy(flow[0, :, 0, 0]), [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 0, 100]), [-20, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 100, 0]), [0, -20]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[0], shape[0]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[2:], shape[1:]))
        self.assertIsNotNone(flow.grad_fn)

    def test_scaling_with_shift(self):
        # Scaling factor 2 around point [20, 30] (hor, ver), to 200 by 300 flow field
        shape = [2, 200, 300]
        matrix = torch.stack((
            torch.eye(3, requires_grad=True), torch.tensor([[2., 0, -20], [0, 2, -30], [0, 0, 1]], requires_grad=True)
        ), dim=0)
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_array_almost_equal(to_numpy(flow[1, :, 30, 20]), [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[1, :, 30, 70]), [50, 0]))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[1, :, 80, 20]), [0, 50]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[0], shape[0]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[2:], shape[1:]))
        self.assertIsNotNone(flow.grad_fn)

    def test_device(self):
        device = ['cpu', 'cuda']
        if torch.cuda.is_available():
            expected_device = ['cpu', 'cuda']
        else:
            expected_device = ['cpu', 'cpu']
        shape = [1, 200, 300]
        matrix = torch.eye(3, requires_grad=True).unsqueeze(0)
        for dev, expected_dev in zip(device, expected_device):
            flow = flow_from_matrix(matrix.to(dev), shape)
            self.assertEqual(flow.device.type, expected_dev)
            self.assertIsNotNone(flow.grad_fn)


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
                               [20, 10.]], requires_grad=True)
        shape = (11, 21)
        exp_coords = torch.tensor([[-1, -1],
                                   [-1.1, 1.2],
                                   [0, 0],
                                   [1.1, 1.2],
                                   [1, 1]])
        coord_list = [coords, coords.unsqueeze(0), coords.unsqueeze(0).unsqueeze(0).expand(4, 3, -1, -1)]
        exp_coord_list = [exp_coords, exp_coords.unsqueeze(0),
                          exp_coords.unsqueeze(0).unsqueeze(0).expand(4, 3, -1, -1)]
        for c, e_c in zip(coord_list, exp_coord_list):
            n_c = normalise_coords(c, shape)
            self.assertIsNone(np.testing.assert_allclose(to_numpy(n_c), to_numpy(e_c), rtol=1e-6))
            self.assertIsNotNone(n_c.grad_fn)

        with self.assertRaises(ValueError):
            normalise_coords(coord_list[0], [1, 2, 3])


class TestApplyFlow(unittest.TestCase):
    def test_zero_flow(self):
        img = cv2.imread('smudge.png')
        img = cv2.resize(img, None, fx=0.125, fy=.125)
        img = torch.tensor(np.moveaxis(img, -1, 0), dtype=torch.float, requires_grad=True)
        flow = Flow.zero(img.shape[1:])
        img_warped = flow.apply(img)
        self.assertIsNone(np.testing.assert_equal(to_numpy(img), to_numpy(img_warped)))

    def test_rotation(self):
        img = cv2.imread('smudge.png')[:, :480]
        img = torch.tensor(np.moveaxis(img, -1, 0), dtype=torch.float, requires_grad=True)
        desired_img = img.transpose(1, 2).flip(1)
        for dev in ['cpu', 'cuda']:
            for ref in ['t', 's']:
                flow = Flow.from_transforms([['rotation', 239.5, 239.5, 90]], img.shape[1:], ref).vecs
                set_pure_pytorch()
                warped_img = apply_flow(flow.to(dev), img, ref)
                self.assertIsNone(np.testing.assert_allclose(to_numpy(warped_img), to_numpy(desired_img), atol=5e-3))
                self.assertIsNotNone(warped_img.grad_fn)
                unset_pure_pytorch()
                warped_img = apply_flow(flow.to(dev), img, ref)
                self.assertIsNone(np.testing.assert_allclose(to_numpy(warped_img), to_numpy(desired_img), atol=5e-3))
                self.assertIsNotNone(warped_img.grad_fn) if ref == 't' else self.assertIsNone(warped_img.grad_fn)

    def test_translation(self):
        img = cv2.imread('smudge.png')
        img = cv2.resize(img, None, fx=0.125, fy=.125)
        img = torch.tensor(np.moveaxis(img, -1, 0), dtype=torch.float, requires_grad=True)
        for dev in ['cpu', 'cuda']:
            for ref in ['s', 't']:
                flow = Flow.from_transforms([['translation', 10, -20]], img.shape[1:], ref).vecs
                set_pure_pytorch()
                warped_img = apply_flow(flow.to(dev), img, ref)
                self.assertIsNone(np.testing.assert_allclose(to_numpy(warped_img[:, :-20, 10:]),
                                                             to_numpy(img[:, 20:, :-10]), atol=5e-3))
                self.assertIsNotNone(warped_img.grad_fn)
                unset_pure_pytorch()
                warped_img = apply_flow(flow.to(dev), img, ref)
                self.assertIsNone(np.testing.assert_allclose(to_numpy(warped_img[:, :-20, 10:]),
                                                             to_numpy(img[:, 20:, :-10]), atol=5e-3))
                self.assertIsNotNone(warped_img.grad_fn) if ref == 't' else self.assertIsNone(warped_img.grad_fn)

    def test_2d_target(self):
        img = cv2.imread('smudge.png', 0)
        img = cv2.resize(img, None, fx=0.125, fy=.125)
        img = torch.tensor(img, dtype=torch.float, requires_grad=True)
        for dev in ['cpu', 'cuda']:
            for ref in ['s', 't']:
                flow = Flow.from_transforms([['translation', 10, -20]], img.shape, ref).vecs
                set_pure_pytorch()
                warped_img = apply_flow(flow.to(dev), img, ref)
                self.assertIsNone(np.testing.assert_allclose(to_numpy(warped_img[:-20, 10:]),
                                                             to_numpy(img[20:, :-10]), atol=5e-3))
                self.assertIsNotNone(warped_img.grad_fn)
                unset_pure_pytorch()
                warped_img = apply_flow(flow.to(dev), img, ref)
                self.assertIsNone(np.testing.assert_allclose(to_numpy(warped_img[:-20, 10:]),
                                                             to_numpy(img[20:, :-10]), atol=5e-3))
                self.assertIsNotNone(warped_img.grad_fn) if ref == 't' else self.assertIsNone(warped_img.grad_fn)

    def test_batch_sizes(self):
        img = cv2.imread('smudge.png')
        img = cv2.resize(img, None, fx=.125, fy=.125)
        i_chw = torch.tensor(np.moveaxis(img, -1, 0), dtype=torch.float, requires_grad=True)
        i_hw = i_chw[0]
        i_1chw = i_chw.unsqueeze(0)
        i_11hw = i_1chw[:, 0:1]
        i_nchw = i_1chw.expand(4, -1, -1, -1)
        i_n1hw = i_11hw.expand(4, -1, -1, -1)
        for ref in ['s', 't']:
            for f in [
                Flow.from_transforms([['translation', 10, -20]], img.shape[:2], ref).vecs,
                Flow.from_transforms([['translation', 10, -20]], img.shape[:2], ref).vecs.expand(4, -1, -1, -1),
            ]:
                for i in [i_1chw, i_11hw, i_nchw, i_n1hw]:
                    set_pure_pytorch()
                    warped_i = apply_flow(f, i, ref)
                    self.assertIsNotNone(warped_i.grad_fn)
                    self.assertEqual(warped_i.shape[0], max(f.shape[0], i.shape[0]))
                    for w_ind, i_ind in zip(warped_i, i):
                        self.assertIsNone(np.testing.assert_allclose(to_numpy(w_ind[:-20, 10:]),
                                                                     to_numpy(i_ind[20:, :-10]), atol=5e-3))
                    unset_pure_pytorch()
                    warped_i = apply_flow(f, i, ref)
                    self.assertIsNotNone(warped_i.grad_fn) if ref == 't' else self.assertIsNone(warped_i.grad_fn)
                    self.assertEqual(warped_i.shape[0], max(f.shape[0], i.shape[0]))
                    for w_ind, i_ind in zip(warped_i, i):
                        self.assertIsNone(np.testing.assert_allclose(to_numpy(w_ind[:-20, 10:]),
                                                                     to_numpy(i_ind[20:, :-10]), atol=5e-3))
        f = Flow.from_transforms([['translation', 10, -20]], img.shape[:2], 't').vecs.expand(4, -1, -1, -1)
        set_pure_pytorch()
        warped_i = apply_flow(f, i_hw, 't')
        self.assertIsNotNone(warped_i.grad_fn)
        self.assertEqual(warped_i.shape, (4, i_hw.shape[0], i_hw.shape[1]))
        warped_i = apply_flow(f, i_chw, 't')
        self.assertIsNotNone(warped_i.grad_fn)
        self.assertEqual(warped_i.shape, (4, 3, i_hw.shape[0], i_hw.shape[1]))
        unset_pure_pytorch()
        warped_i = apply_flow(f, i_hw, 't')
        self.assertIsNotNone(warped_i.grad_fn)
        self.assertEqual(warped_i.shape, (4, i_hw.shape[0], i_hw.shape[1]))
        warped_i = apply_flow(f, i_chw, 't')
        self.assertIsNotNone(warped_i.grad_fn)
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
            apply_flow(flow=torch.ones((3, 2, 11, 10)), target=torch.zeros((2, 1, 11, 10)), ref='t')


class TestThresholdVectors(unittest.TestCase):
    def test_threshold(self):
        vecs = torch.zeros((5, 2, 10, 1))
        vecs[2, 0, 0, 0] = -1e-5
        vecs[2, 0, 1, 0] = 1e-4
        vecs[2, 0, 2, 0] = -1e-3
        vecs[2, 0, 3, 0] = 1
        vecs.requires_grad_()
        for use_mag in [True, False]:
            thresholded = threshold_vectors(vecs, threshold=1e-3, use_mag=use_mag)
            self.assertIsNotNone(thresholded.grad_fn)
            thresholded = to_numpy(thresholded[2, 0, :4, 0])
            self.assertIsNone(np.testing.assert_allclose(thresholded, [0, 0, -1e-3, 1]))
            thresholded = threshold_vectors(vecs, threshold=1e-4, use_mag=use_mag)
            self.assertIsNotNone(thresholded.grad_fn)
            thresholded = to_numpy(thresholded[2, 0, :4, 0])
            self.assertIsNone(np.testing.assert_allclose(thresholded, [0, 1e-4, -1e-3, 1]))
            thresholded = threshold_vectors(vecs, threshold=1e-5, use_mag=use_mag)
            self.assertIsNotNone(thresholded.grad_fn)
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
        set_pure_pytorch()
        flow = from_matrix(matrix, shape, 't')
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 50, 10]), [0, 0], atol=1e-4))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 50, 299]), [38.7186583063, 144.5], rtol=1e-4))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 199, 10]), [-74.5, 19.9622148361], rtol=1e-4))
        self.assertIsNone(np.testing.assert_equal(flow.shape[0], 1))
        self.assertIsNone(np.testing.assert_equal(flow.shape[2:], shape))
        matrix = torch.tensor(matrix, requires_grad=True)
        flow = from_matrix(matrix, shape, 't')
        self.assertIsNotNone(flow.grad_fn)
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 50, 10]), [0, 0], atol=1e-4))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 50, 299]), [38.7186583063, 144.5], rtol=1e-4))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 199, 10]), [-74.5, 19.9622148361], rtol=1e-4))
        self.assertIsNone(np.testing.assert_equal(flow.shape[0], 1))
        self.assertIsNone(np.testing.assert_equal(flow.shape[2:], shape))
        unset_pure_pytorch()
        matrix = np.array([[math.sqrt(3) / 2, -.5, 26.3397459622],
                           [.5, math.sqrt(3) / 2, 1.69872981078],
                           [0, 0, 1]])
        flow = from_matrix(matrix, shape, 't')
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 50, 10]), [0, 0], atol=1e-4))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 50, 299]), [38.7186583063, 144.5], rtol=1e-4))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 199, 10]), [-74.5, 19.9622148361], rtol=1e-4))
        self.assertIsNone(np.testing.assert_equal(flow.shape[0], 1))
        self.assertIsNone(np.testing.assert_equal(flow.shape[2:], shape))
        matrix = torch.tensor(matrix, requires_grad=True)
        flow = from_matrix(matrix, shape, 't')
        self.assertIsNotNone(flow.grad_fn)
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 50, 10]), [0, 0], atol=1e-4))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 50, 299]), [38.7186583063, 144.5], rtol=1e-4))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0, :, 199, 10]), [-74.5, 19.9622148361], rtol=1e-4))
        self.assertIsNone(np.testing.assert_equal(flow.shape[0], 1))
        self.assertIsNone(np.testing.assert_equal(flow.shape[2:], shape))

        # With and without inverse matrix for ref 't'
        matrix = torch.tensor([[1., 0, 10], [0, 1, 20], [0, 0, 1]], requires_grad=True)
        inv_matrix = torch.tensor([[1., 0, -10], [0, 1, -20], [0, 0, 1]], requires_grad=True)
        v = from_matrix(matrix, shape, ref='t', matrix_is_inverse=False)
        self.assertIsNotNone(v.grad_fn)
        vecs = to_numpy(v, switch_channels=True)
        v = from_matrix(inv_matrix, shape, ref='t', matrix_is_inverse=True)
        self.assertIsNotNone(v.grad_fn)
        inv_vecs = to_numpy(v, switch_channels=True)
        self.assertIsNone(np.testing.assert_allclose(vecs, inv_vecs, rtol=1e-3))

    def test_failed_from_matrix(self):
        with self.assertRaises(ValueError):  # Invalid shape size
            from_matrix(torch.eye(3), [2, 10, 10], 't')
        with self.assertRaises(TypeError):  # Invalid matrix type
            from_matrix('test', [10, 10], 't')
        with self.assertRaises(ValueError):  # Invalid matrix shape
            from_matrix(np.eye(4), [10, 10], 't')
        with self.assertRaises(ValueError):  # Invalid matrix ndim
            from_matrix(np.ones((1, 1, 3, 3)), [10, 10], 't')
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

    def test_from_transforms_padding(self):
        shape = [100, 150]
        padding = [5, 10, 10, 20]
        padded_shape = [shape[0] + sum(padding[0:2]), shape[1] + sum(padding[2:4])]
        transforms = [['rotation', 10, 50, -30], ['scaling', 20, 30, 2], ['translation', 20, 10]]
        flow = from_transforms(transforms[0:1], shape, 's')
        flow_padded = from_transforms(transforms[0:1], shape, 's', padding)
        self.assertEqual(list(flow_padded.shape[2:]), padded_shape)
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow[0]),
                                                     to_numpy(flow_padded[0, :, padding[0]:-padding[1],
                                                              padding[2]:-padding[3]]), atol=1e-4))
        flow = from_transforms(transforms[1:2], shape, 't')
        flow_padded = from_transforms(transforms[1:2], shape, 't', padding)
        self.assertEqual(list(flow_padded.shape[2:]), padded_shape)
        self.assertIsNone(np.testing.assert_equal(to_numpy(flow[0]),
                                                     to_numpy(flow_padded[0, :, padding[0]:-padding[1],
                                                              padding[2]:-padding[3]])))
        flow = from_transforms(transforms[2:3], shape, 't')
        flow_padded = from_transforms(transforms[2:3], shape, 't', padding)
        self.assertEqual(list(flow_padded.shape[2:]), padded_shape)
        self.assertIsNone(np.testing.assert_equal(to_numpy(flow[0]),
                                                  to_numpy(flow_padded[0, :, padding[0]:-padding[1],
                                                           padding[2]:-padding[3]])))

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
        with self.assertRaises(ValueError):  # rotation missing information
            transforms = [['translation', 20, 10], ['rotation']]
            from_transforms(transforms, shape, 't')
        with self.assertRaises(ValueError):  # rotation with incomplete information
            transforms = [['translation', 20, 10], ['rotation', 1]]
            from_transforms(transforms, shape, 't')
        with self.assertRaises(ValueError):  # rotation with invalid information
            transforms = [['translation', 20, 10], ['rotation', 1, 'test', 10]]
            from_transforms(transforms, shape, 't')
        with self.assertRaises(ValueError):  # translation missing information
            transforms = [['translation', 20, 10], ['translation']]
            from_transforms(transforms, shape, 't')
        with self.assertRaises(ValueError):  # translation with incomplete information
            transforms = [['translation', 20, 10], ['translation', 1]]
            from_transforms(transforms, shape, 't')
        with self.assertRaises(ValueError):  # translation with invalid information
            transforms = [['translation', 20, 10], ['translation', 1, 'test']]
            from_transforms(transforms, shape, 't')
        with self.assertRaises(ValueError):  # scaling missing information
            transforms = [['translation', 20, 10], ['scaling']]
            from_transforms(transforms, shape, 't')
        with self.assertRaises(ValueError):  # scaling with incomplete information
            transforms = [['translation', 20, 10], ['scaling', 1]]
            from_transforms(transforms, shape, 't')
        with self.assertRaises(ValueError):  # scaling with invalid information
            transforms = [['translation', 20, 10], ['scaling', 1, 'test', 2]]
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
        flow = Flow.from_transforms([['rotation', 30, 50, 30]], shape, ref).vecs.requires_grad_()
        for f in [flow, flow.squeeze(0)]:
            # Different scales
            scales = [.2, .5, 1, 1.5, 2, 10]
            for scale in scales:
                resized_flow = resize_flow(f, scale)
                self.assertIsNotNone(resized_flow.grad_fn)
                resized_shape = scale * np.array(f.shape[-2:])
                self.assertIsNone(np.testing.assert_equal(resized_flow.shape[-2:], resized_shape))
                self.assertIsNone(np.testing.assert_allclose(to_numpy(resized_flow[..., 0, 0]),
                                                             to_numpy(f[..., 0, 0]) * scale, rtol=.1))
                self.assertEqual(len(f.shape), len(resized_flow.shape))

            # Scale list
            scale = [.5, 2]
            resized_flow = resize_flow(f, scale)
            self.assertIsNotNone(resized_flow.grad_fn)
            resized_shape = np.array(scale) * np.array(f.shape[-2:])
            self.assertIsNone(np.testing.assert_equal(resized_flow.shape[-2:], resized_shape))
            self.assertIsNone(np.testing.assert_allclose(to_numpy(resized_flow[..., 0, 0]),
                                                         to_numpy(f[..., 0, 0]) * np.array(scale)[::-1], rtol=.1))
            self.assertEqual(len(f.shape), len(resized_flow.shape))

            # Scale tuple
            scale = (2, .5)
            resized_flow = resize_flow(f, scale)
            self.assertIsNotNone(resized_flow.grad_fn)
            resized_shape = np.array(scale) * np.array(f.shape[-2:])
            self.assertIsNone(np.testing.assert_equal(resized_flow.shape[-2:], resized_shape))
            self.assertIsNone(np.testing.assert_allclose(to_numpy(resized_flow[..., 0, 0]),
                                                         to_numpy(f[..., 0, 0]) * np.array(scale)[::-1], rtol=.1))
            self.assertEqual(len(f.shape), len(resized_flow.shape))

    def test_resize_on_fields(self):
        # Check scaling is performed correctly based on the actual flow field
        ref = 't'
        flow_small = Flow.from_transforms([['rotation', 0, 0, 30]], (50, 80), ref).vecs_numpy
        flow_large = Flow.from_transforms([['rotation', 0, 0, 30]], (150, 240), ref).vecs.requires_grad_()
        flow_resized = resize_flow(flow_large, 1 / 3)
        self.assertIsNotNone(flow_resized.grad_fn)
        self.assertIsNone(np.testing.assert_allclose(to_numpy(flow_resized, switch_channels=True),
                                                     flow_small, atol=1, rtol=.1))

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

        flow = np.zeros((3, 10, 10, 2), 'float32')
        flow[1:, 3, :, 0] = 1e-4
        self.assertEqual(is_zero_flow(flow, thresholded=True).tolist(), [True, True, True])
        self.assertEqual(is_zero_flow(flow, thresholded=False).tolist(), [True, False, False])

    def test_failed_is_zero_flow(self):
        with self.assertRaises(TypeError):  # Wrong thresholded type
            is_zero_flow(np.zeros((10, 10, 2)), 'test')


class TestTrackPts(unittest.TestCase):
    def test_track_pts(self):
        set_pure_pytorch()
        f_s = Flow.from_transforms([['rotation', 0, 0, 30]], (200, 210), 's').vecs
        f_t = Flow.from_transforms([['rotation', 0, 0, 30]], (200, 210), 't').vecs
        pts = torch.tensor([
            [20.5, 10.5],
            [8.3, 7.2],
            [120.4, 160.2]
        ], requires_grad=True)
        desired_pts = [
            [12.5035207776, 19.343266740],
            [3.58801085141, 10.385382907],
            [24.1694586156, 198.93726969]
        ]
        # Gradient propagation when flow / points / both require grad
        p = torch.tensor([[20.5, 10.5]])
        p_g = torch.tensor([[20.5, 10.5]], requires_grad=True)
        f_s_g = f_s.clone().requires_grad_()
        f_t_g = f_t.clone().requires_grad_()
        tracked_list = [
            track_pts(f_s, 's', p_g),
            track_pts(f_s_g, 's', p),
            track_pts(f_s_g, 's', p_g),
            track_pts(f_t, 't', p_g),
            track_pts(f_t_g, 't', p),
            track_pts(f_t_g, 't', p_g),
        ]
        for tracked in tracked_list:
            self.assertIsNotNone(tracked.grad_fn)

        # Zero flow
        p_zero = track_pts(torch.zeros_like(f_s), 's', pts)
        self.assertIsNone(np.testing.assert_equal(to_numpy(pts), to_numpy(p_zero)))

        # Reference 's'
        pts_tracked_s = track_pts(f_s, 's', pts)
        self.assertIsNotNone(pts_tracked_s.grad_fn)
        self.assertIsInstance(pts_tracked_s, torch.Tensor)
        self.assertIsNone(np.testing.assert_allclose(to_numpy(pts_tracked_s), desired_pts, rtol=1e-6))
        pts_tracked_s = track_pts(f_s, 's', pts.unsqueeze(0))
        self.assertIsInstance(pts_tracked_s, torch.Tensor)
        self.assertIsNone(np.testing.assert_allclose(to_numpy(pts_tracked_s)[0], desired_pts, rtol=1e-6))

        set_pure_pytorch()
        # Reference 't'
        pts_tracked_t = track_pts(f_t, 't', pts)
        self.assertIsNotNone(pts_tracked_s.grad_fn)
        self.assertIsInstance(pts_tracked_t, torch.Tensor)
        self.assertIsNone(np.testing.assert_allclose(to_numpy(pts_tracked_t), desired_pts, rtol=5e-3))

        # Reference 't', integer output
        pts_tracked_t = track_pts(f_t, 't', pts, int_out=True)
        self.assertIsInstance(pts_tracked_t, torch.Tensor)
        self.assertIsNone(np.testing.assert_allclose(to_numpy(pts_tracked_t), np.round(desired_pts), atol=1))
        self.assertEqual(pts_tracked_t.dtype, torch.long)

        unset_pure_pytorch()
        # Reference 't'
        pts_tracked_t = track_pts(f_t, 't', pts)
        self.assertIsInstance(pts_tracked_t, torch.Tensor)
        self.assertIsNone(np.testing.assert_allclose(to_numpy(pts_tracked_t), desired_pts))

        # Reference 't', integer output
        pts_tracked_t = track_pts(f_t, 't', pts, int_out=True)
        self.assertIsInstance(pts_tracked_t, torch.Tensor)
        self.assertIsNone(np.testing.assert_equal(to_numpy(pts_tracked_t), np.round(desired_pts)))
        self.assertEqual(pts_tracked_t.dtype, torch.long)

        # Test tracking for 's' flow and int pts, also batched points on batched flows
        set_pure_pytorch()
        f_trans = Flow.from_transforms([['translation', 10, 20]], (200, 210), 's')
        f_rot = Flow.from_transforms([['rotation', 0, 0, 30]], (200, 210), 's')
        f_s = batch_flows((f_trans, f_rot, f_rot))
        f_s_g = f_s.vecs.clone().requires_grad_()
        pts = np.stack(([[20, 10], [8, 7], [12, 15]],
                        [[18, 9], [6, 5], [110, 150]],
                        [[6, 5], [110, 150], [18, 9]]), axis=0)
        desired_pts = [
            [[40, 20], [28, 17], [32, 25]],
            [
                [11.088456153869629, 16.794227600097656],
                [2.6961512565612793, 7.330124855041504],
                [20.262794494628906, 184.90380859375]
            ],
            [
                [2.6961512565612793, 7.330124855041504],
                [20.262794494628906, 184.90380859375],
                [11.088456153869629, 16.794227600097656]
            ]
        ]
        pts_tracked_s = track_pts(f_s_g, 's', torch.tensor(pts))
        self.assertIsNotNone(pts_tracked_s.grad_fn)
        for i in range(pts_tracked_s.shape[0]):
            self.assertIsNone(np.testing.assert_allclose(to_numpy(pts_tracked_s[i]), desired_pts[i], rtol=1e-6))
        pts_tracked_s = track_pts(f_s_g, 's', torch.tensor(pts)[0:1])
        self.assertIsNotNone(pts_tracked_s.grad_fn)
        self.assertIsNone(np.testing.assert_allclose(to_numpy(pts_tracked_s[0]), desired_pts[0], rtol=1e-6))
        pts_tracked_s = track_pts(f_s_g, 's', torch.tensor(pts).to(torch.float))
        self.assertIsNotNone(pts_tracked_s.grad_fn)
        for i in range(pts_tracked_s.shape[0]):
            self.assertIsNone(np.testing.assert_allclose(to_numpy(pts_tracked_s[i]), desired_pts[i], rtol=1e-6))
        f_trans = Flow.from_transforms([['translation', 10, 20]], (200, 210), 't')
        f_rot = Flow.from_transforms([['rotation', 0, 0, 30]], (200, 210), 't')
        f_t = batch_flows((f_trans, f_rot, f_rot))
        pts_tracked_t = track_pts(f_t.vecs.requires_grad_(), 't', torch.tensor(pts).to(torch.float))
        self.assertIsNotNone(pts_tracked_t.grad_fn)
        for i in range(pts_tracked_t.shape[0]):
            self.assertIsNone(np.testing.assert_allclose(to_numpy(pts_tracked_s[i]), desired_pts[i], rtol=1e-6))

    def test_device_track_pts(self):
        for d1 in ['cpu', 'cuda']:
            for d2 in ['cpu', 'cuda']:
                f = Flow.from_transforms([['translation', 10, 20]], (512, 512), 's', device=d1).vecs.requires_grad_()
                pts = torch.tensor([[20, 10], [8, 7]]).to(d2)
                set_pure_pytorch()
                pts_tracked = track_pts(f, 's', pts)
                self.assertIsNotNone(pts_tracked.grad_fn)
                self.assertEqual(pts_tracked.device, f.device)
                unset_pure_pytorch()
                pts_tracked = track_pts(f, 's', pts)
                self.assertIsNotNone(pts_tracked.grad_fn)
                self.assertEqual(pts_tracked.device, f.device)

    def test_failed_track_pts(self):
        pts = torch.tensor([[20, 10], [20, 10], [8, 7]])
        flow = torch.zeros((2, 10, 10))
        with self.assertRaises(TypeError):  # Wrong pts type
            track_pts(flow, 's', pts='test')
        with self.assertRaises(ValueError):  # Wrong pts shape: too many dims
            track_pts(flow, 's', pts=torch.zeros((1, 2, 10, 2)))
        with self.assertRaises(ValueError):  # Wrong pts shape: batch size not equal flow
            track_pts(flow, 's', pts=torch.zeros((5, 10, 2)))
        with self.assertRaises(ValueError):  # Pts channel not of size 2
            track_pts(flow, 's', pts=pts.transpose(0, -1))
        with self.assertRaises(TypeError):  # Wrong int_out type
            track_pts(flow, 's', pts, int_out='test')


class TestGetFlowEndpoints(unittest.TestCase):
    def test_get_flow_endpoints(self):
        # ref 's'
        flow = Flow.from_transforms([['translation', -2, 2]], (6, 6), 's')
        x, y = get_flow_endpoints(flow.vecs.requires_grad_(), 's')
        self.assertIsNotNone(x.grad_fn)
        self.assertIsNotNone(y.grad_fn)
        self.assertIsNone(np.testing.assert_equal(to_numpy(x[0, 0]), [-2, -1, 0, 1, 2, 3]))
        self.assertIsNone(np.testing.assert_equal(to_numpy(x[0, :, 0]), [-2, -2, -2, -2, -2, -2]))
        self.assertIsNone(np.testing.assert_equal(to_numpy(y[0, 0]), [2, 2, 2, 2, 2, 2]))
        self.assertIsNone(np.testing.assert_equal(to_numpy(y[0, :, 0]), [2, 3, 4, 5, 6, 7]))
        # ref 't'
        flow = Flow.from_transforms([['translation', -2, 2]], (6, 6), 't')
        x, y = get_flow_endpoints(flow.vecs.requires_grad_(), 't')
        self.assertIsNotNone(x.grad_fn)
        self.assertIsNotNone(y.grad_fn)
        self.assertIsNone(np.testing.assert_equal(to_numpy(x[0, 0]), [2, 3, 4, 5, 6, 7]))
        self.assertIsNone(np.testing.assert_equal(to_numpy(x[0, :, 0]), [2, 2, 2, 2, 2, 2]))
        self.assertIsNone(np.testing.assert_equal(to_numpy(y[0, 0]), [-2, -2, -2, -2, -2, -2]))
        self.assertIsNone(np.testing.assert_equal(to_numpy(y[0, :, 0]), [-2, -1, 0, 1, 2, 3]))


class TestGridFromUnstructuredData(unittest.TestCase):
    def test_grid_from_unstructured_data(self):
        flow = Flow.from_transforms([['rotation', 50, 75, -20]], (100, 150), ref='s')
        flow = batch_flows((flow, flow))
        vecs = flow.vecs.requires_grad_()
        flow_rev = Flow.from_transforms([['rotation', 50, 75, 20]], (100, 150), ref='s')
        flow_rev = batch_flows((flow_rev, flow_rev))
        x, y = get_flow_endpoints(vecs, flow.ref)
        data, mask = grid_from_unstructured_data(x, y, vecs)
        self.assertIsNotNone(data.grad_fn)
        self.assertIsNotNone(mask.grad_fn)
        flow_approx = Flow(-data, flow.ref)
        mask = to_numpy(mask.squeeze(1) > 0.95)
        self.assertIsNone(np.testing.assert_allclose(flow_rev.vecs_numpy[mask],
                                                     flow_approx.vecs_numpy[mask],
                                                     atol=5e-2))
        self.assertEqual(np.count_nonzero(mask), 24664)

        # Check differentiability
        vecs = flow.vecs
        vecs_g = vecs.clone().requires_grad_()
        vecs_list = [vecs, vecs_g, vecs_g]
        data_list = [vecs_g, vecs, vecs_g]
        for v, d in zip(vecs_list, data_list):
            x, y = get_flow_endpoints(v, 's')
            data, mask = grid_from_unstructured_data(x, y, d)
            self.assertIsNotNone(data.grad_fn)
            self.assertIsNotNone(mask.grad_fn)


class TestApplySFlow(unittest.TestCase):
    def test_masked_s_flow(self):
        # Mask     Flow     m-fl     m-fl
        # 0000     0110     0000     ----
        # 0100     0110     0100     -1--
        # 0010     0000     0000     --0-
        # 0000     0000     0000     ----
        #
        img = cv2.imread('smudge.png')  # 480x512 pixels
        shape = img.shape[:2]
        img = torch.tensor(np.moveaxis(img, -1, 0), dtype=torch.float).unsqueeze(0)
        mask = torch.zeros_like(img[:, 0], dtype=torch.bool)
        mask[:, 100:-100, 100:-100] = True
        mask[:, 300:, :250] = False
        mask[:, :300, 250:] = False
        flow_base = Flow.from_transforms([['scaling', 256, 240, 1.3]], shape, 't')
        img_base = to_numpy(flow_base.apply(img))
        flow = Flow.from_transforms([['scaling', 256, 240, 1.3]], shape, 's', mask)
        flow._vecs[:, :, 300:] = 0
        flow._vecs[:, :, :, :100] = 0
        flow._vecs[:, :, :, -100:] = 0
        vecs_g = flow._vecs.clone().requires_grad_()
        img_g = img.clone().requires_grad_()
        flow_list = [vecs_g, vecs_g, flow._vecs]
        img_list = [img, img_g, img_g]
        for f, i in zip(flow_list, img_list):
            # masked, not occluding zero flow: output should match baseline within mask
            img_w_true, dens_true = apply_s_flow(f, i, flow._mask, occlude_zero_flow=False)
            self.assertIsNotNone(img_w_true.grad_fn)
            mask = np.zeros(shape, np.bool)
            mask[100-42:300+18, 100-47:250-2] = True
            mask[300:-100, 250:-100] = True
            # Check the mask from density matches what it should be
            self.assertEqual(np.count_nonzero(dens_true*255 - mask*255), 0)
            mask[300:] = False
            n = np.count_nonzero(mask) * 3
            # Check the warped area matches approximately
            img_w_true = to_numpy(img_w_true)
            self.assertGreater(np.count_nonzero(np.abs(img_w_true[0, :, mask] - img_base[0, :, mask]) < 1) / n, 0.55)
            self.assertGreater(np.count_nonzero(np.abs(img_w_true[0, :, mask] - img_base[0, :, mask]) < 5) / n, 0.9)
            self.assertGreater(np.count_nonzero(np.abs(img_w_true[0, :, mask] - img_base[0, :, mask]) < 10) / n, 0.98)
            # Means: >55% of image pixels (and channel elements) within 1, >90% within 5, >98% within 10
            # Check the non-warped area matches exactly
            mask2 = np.zeros(shape, np.bool)
            mask2[300:-100, 250:-100] = True
            self.assertIsNone(np.testing.assert_equal(img_w_true[0, :, mask2], to_numpy(img)[0, :, mask2]))

            # masked, occluding zero flow: output should be the same as before
            img_w_true2, dens_true2 = apply_s_flow(f, i, flow._mask, occlude_zero_flow=True)
            self.assertIsNotNone(img_w_true2.grad_fn)
            self.assertIsNone(np.testing.assert_equal(img_w_true, to_numpy(img_w_true2)))
            dens_true[0, 300:-100, 250:-100] = False
            self.assertIsNone(np.testing.assert_equal(dens_true, to_numpy(dens_true2)))

            # not masked, not occluding zero flow: output should match baseline within mask
            img_w_false, dens_false = apply_s_flow(f, i, occlude_zero_flow=False)
            self.assertIsNotNone(img_w_false.grad_fn)
            # Check the mask from density matches what it should be
            self.assertEqual(np.count_nonzero(dens_false*255 - 255), 0)
            mask = np.zeros(shape, np.bool)
            mask[100:300, 100:-100] = True
            n = np.count_nonzero(mask) * 3
            # Check the warped area matches approximately
            img_w_false = to_numpy(img_w_false)
            self.assertGreater(np.count_nonzero(np.abs(img_w_false[0, :, mask] - img_base[0, :, mask]) < 1) / n, 0.65)
            self.assertGreater(np.count_nonzero(np.abs(img_w_false[0, :, mask] - img_base[0, :, mask]) < 5) / n, 0.94)
            self.assertGreater(np.count_nonzero(np.abs(img_w_false[0, :, mask] - img_base[0, :, mask]) < 10) / n, 0.98)
            # Means: >65% of image pixels (and channel elements) within 1, >94% within 5, >98% within 10
            # Check the non-warped area matches exactly
            mask2 = np.ones(shape, np.bool)
            mask2[:300+18, 100-47:-100+47] = False
            self.assertIsNone(np.testing.assert_equal(img_w_false[0, :, mask2], to_numpy(img)[0, :, mask2]))

            # not masked, occluding zero flow: output should match baseline within mask
            img_w_false, dens_false = apply_s_flow(f, i, occlude_zero_flow=True)
            self.assertIsNotNone(img_w_false.grad_fn)
            mask = np.zeros(shape, np.bool)
            mask[:300+18, 100-47:-100+47] = True
            mask[240, 256] = False
            # Check the mask from density matches what it should be
            self.assertEqual(np.count_nonzero(dens_false*255 - mask*255), 0)
            n = np.count_nonzero(mask) * 3
            # Check the warped area matches approximately
            img_w_false = to_numpy(img_w_false)
            self.assertGreater(np.count_nonzero(np.abs(img_w_false[0, :, mask] - img_base[0, :, mask]) < 1) / n, 0.65)
            self.assertGreater(np.count_nonzero(np.abs(img_w_false[0, :, mask] - img_base[0, :, mask]) < 5) / n, 0.95)
            self.assertGreater(np.count_nonzero(np.abs(img_w_false[0, :, mask] - img_base[0, :, mask]) < 10) / n, 0.99)
            # Means: >65% of image pixels (and channel elements) within 1, >95% within 5, >99% within 10
            # Check the non-warped area matches exactly
            self.assertIsNone(np.testing.assert_equal(img_w_false[0, :, ~mask], to_numpy(img)[0, :, ~mask]))


if __name__ == '__main__':
    unittest.main()
