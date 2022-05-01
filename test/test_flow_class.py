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

import unittest
import torch
import cv2
import numpy as np
import math
import sys
sys.path.append('..')
from src.oflibpytorch.flow_class import Flow
from src.oflibpytorch.utils import to_numpy, apply_flow, matrix_from_transforms, resize_flow


class FlowTest(unittest.TestCase):
    def test_flow(self):
        if torch.cuda.is_available():
            expected_device_list = ['cpu', 'cuda', 'cpu']
        else:
            expected_device_list = ['cpu', 'cpu', 'cpu']

        # 3-dim vec inputs
        np_12hw = np.zeros((1, 2, 100, 200))
        np_1hw2 = np.zeros((1, 100, 200, 2))
        np_2hw = np.zeros((2, 100, 200))
        np_hw2 = np.zeros((100, 200, 2))
        pt_2hw = torch.zeros((2, 100, 200))
        pt_hw2 = torch.zeros((100, 200, 2))
        mask_empty = None
        mask_1np = np.ones((1, 100, 200), 'bool')
        mask_np = np.ones((100, 200), 'bool')
        mask_pt = torch.ones(100, 200).to(torch.bool)
        for vecs in [np_2hw, np_hw2, pt_2hw, pt_hw2]:
            for ref, ref_expected in zip(['t', 's', None], ['t', 's', 't']):
                for mask in [mask_empty, mask_np, mask_pt]:
                    for device, device_expected in zip(['cpu', 'cuda', None], expected_device_list):
                        flow = Flow(vecs, ref=ref, mask=mask, device=device)
                        self.assertIsNone(np.testing.assert_equal(to_numpy(flow.vecs), np_12hw))
                        self.assertIsNone(np.testing.assert_equal(flow.vecs_numpy, np_1hw2))
                        self.assertEqual(flow.shape, [1, 100, 200])
                        self.assertEqual(flow.ref, ref_expected)
                        self.assertIsNone(np.testing.assert_equal(to_numpy(flow.mask), mask_1np))
                        self.assertEqual(flow.device.type, device_expected)
                        self.assertEqual(flow.vecs.device.type, device_expected)
                        self.assertEqual(flow.mask.device.type, device_expected)

        # 4-dim vec inputs
        np_n2hw = np.zeros((5, 2, 100, 200))
        np_nhw2 = np.zeros((5, 100, 200, 2))
        pt_n2hw = torch.zeros((5, 2, 100, 200))
        pt_nhw2 = torch.zeros((5, 100, 200, 2))
        mask_empty = None
        mask_np_nhw = np.ones((5, 100, 200), 'bool')
        mask_pt_nhw = torch.ones(5, 100, 200).to(torch.bool)
        for vecs in [np_n2hw, np_nhw2, pt_n2hw, pt_nhw2]:
            for ref, ref_expected in zip(['t', 's', None], ['t', 's', 't']):
                for mask in [mask_empty, mask_np_nhw, mask_pt_nhw]:
                    for device, device_expected in zip(['cpu', 'cuda', None], expected_device_list):
                        flow = Flow(vecs, ref=ref, mask=mask, device=device)
                        self.assertIsNone(np.testing.assert_equal(to_numpy(flow.vecs), np_n2hw))
                        self.assertIsNone(np.testing.assert_equal(flow.vecs_numpy, np_nhw2))
                        self.assertEqual(flow.shape, [5, 100, 200])
                        self.assertEqual(flow.ref, ref_expected)
                        self.assertIsNone(np.testing.assert_equal(to_numpy(flow.mask), mask_np_nhw))
                        self.assertEqual(flow.device.type, device_expected)
                        self.assertEqual(flow.vecs.device.type, device_expected)
                        self.assertEqual(flow.mask.device.type, device_expected)

        # tensor to cuda, test cuda
        if torch.cuda.is_available():
            expected_device_list = ['cpu', 'cuda', 'cuda']
        else:
            expected_device_list = ['cpu', 'cpu', 'cpu']
        vecs_pt_cuda = torch.zeros((2, 100, 200)).to('cuda')
        for ref, ref_expected in zip(['t', 's', None], ['t', 's', 't']):
            for mask in [mask_empty, mask_np, mask_pt]:
                for device, device_expected in zip(['cpu', 'cuda', None], expected_device_list):
                    flow = Flow(vecs_pt_cuda, ref=ref, mask=mask, device=device)
                    self.assertIsNone(np.testing.assert_equal(to_numpy(flow.vecs), np_12hw))
                    self.assertIsNone(np.testing.assert_equal(flow.vecs_numpy, np_1hw2))
                    self.assertEqual(flow.ref, ref_expected)
                    self.assertIsNone(np.testing.assert_equal(to_numpy(flow.mask), mask_1np))
                    self.assertEqual(flow.device.type, device_expected)
                    self.assertEqual(flow.vecs.device.type, device_expected)
                    self.assertEqual(flow.mask.device.type, device_expected)

        # Wrong flow vector type or shape
        with self.assertRaises(TypeError):
            Flow('test')
        with self.assertRaises(ValueError):
            Flow(np.zeros((2, 100, 200, 1)))
        with self.assertRaises(ValueError):
            Flow(torch.ones(2, 100, 200, 1))
        with self.assertRaises(ValueError):
            Flow(np.zeros((3, 100, 200)))
        with self.assertRaises(ValueError):
            Flow(torch.ones(3, 100, 200))

        # Invalid flow vector values
        vectors = np.random.rand(100, 200, 2)
        vectors[10, 10] = np.NaN
        vectors[20, 20] = np.Inf
        vectors[30, 30] = -np.Inf
        with self.assertRaises(ValueError):
            Flow(vectors)
        vectors = torch.tensor(vectors)
        with self.assertRaises(ValueError):
            Flow(vectors)

        # Wrong mask shape
        vecs = torch.zeros((2, 100, 200))
        with self.assertRaises(TypeError):
            Flow(vecs, mask='test')
        with self.assertRaises(ValueError):
            Flow(vecs, mask=np.zeros((2, 100, 200)))
        with self.assertRaises(ValueError):
            Flow(vecs, mask=torch.ones(2, 100, 200))
        with self.assertRaises(ValueError):
            Flow(vecs, mask=np.zeros((101, 200)))
        with self.assertRaises(ValueError):
            Flow(vecs, mask=torch.ones(100, 201))
        with self.assertRaises(ValueError):
            Flow(vecs, mask=np.ones((100, 200)) * 20)
        with self.assertRaises(ValueError):
            Flow(vecs, mask=torch.ones(100, 200) * 10)

    def test_zero(self):
        if torch.cuda.is_available():
            expected_device_list = ['cpu', 'cuda']
        else:
            expected_device_list = ['cpu', 'cpu']
        shape_list = [[200, 300], [5, 200, 300]]
        exp_shape_list = [[1, 200, 300], [5, 200, 300]]
        for shape, exp_shape in zip(shape_list, exp_shape_list):
            zero_flow = Flow.zero(shape)
            self.assertIsNone(np.testing.assert_equal(zero_flow.shape, exp_shape))
            self.assertIsNone(np.testing.assert_equal(zero_flow.vecs_numpy, 0))
            self.assertIs(zero_flow.ref, 't')
            zero_flow = Flow.zero(shape, 's')
            self.assertIs(zero_flow.ref, 's')
            for device, expected_device in zip(['cpu', 'cuda'], expected_device_list):
                flow = Flow.zero(shape, device=device)
                self.assertEqual(flow.vecs.device.type, expected_device)
                self.assertEqual(flow.mask.device.type, expected_device)

    def test_from_matrix(self):
        # With reference 's', this simply corresponds to using flow_from_matrix, tested in test_utils.
        # With reference 't':
        # Rotation of 30 degrees clockwise around point [10, 50] (hor, ver)
        matrix_np = np.array([[math.sqrt(3) / 2, -.5, 26.3397459622],
                              [.5, math.sqrt(3) / 2, 1.69872981078],
                              [0, 0, 1]])
        matrix_pt = torch.tensor(matrix_np)
        shape = [200, 300]
        matrix_device_list = ['cpu', 'cuda']
        flow_device_list = ['cpu', 'cuda', None]
        if torch.cuda.is_available():
            flow_expected_device_list = ['cpu', 'cuda', None]
        else:
            flow_expected_device_list = ['cpu', 'cpu', 'cpu']
        for matrix in [matrix_pt, matrix_np]:
            for matrix_device in matrix_device_list:
                for flow_device, flow_expected_device in zip(flow_device_list, flow_expected_device_list):
                    if isinstance(matrix, torch.Tensor):
                        matrix = matrix.to(matrix_device)
                    flow = Flow.from_matrix(matrix, shape, 't', device=flow_device)
                    if flow_expected_device is None:  # If no device passed, expect same device as the matrix passed in
                        flow_expected_device = matrix.device.type if isinstance(matrix, torch.Tensor) else 'cpu'
                    self.assertEqual(flow.device.type, flow_expected_device)
                    self.assertIsNone(np.testing.assert_allclose(flow.vecs_numpy[0, 50, 10], [0, 0], atol=1e-4))
                    self.assertIsNone(np.testing.assert_allclose(flow.vecs_numpy[0, 50, 299], [38.7186583063, 144.5],
                                                                 atol=1e-4, rtol=1e-4))
                    self.assertIsNone(np.testing.assert_allclose(flow.vecs_numpy[0, 199, 10], [-74.5, 19.9622148361],
                                                                 atol=1e-4, rtol=1e-4))
                    self.assertIsNone(np.testing.assert_equal(flow.shape[1:], shape))

    def test_from_transforms(self):
        shape = [10, 20]

        transforms = [['rotation', 5, 10, -30]]
        for device in ['cpu', 'cuda', None]:
            flow = Flow.from_transforms(transforms, shape, device=device)
            expected_device = device if torch.cuda.is_available() and device is not None else 'cpu'
            self.assertEqual(flow.device.type, expected_device)

    def test_from_kitti(self):
        path = 'kitti.png'
        f = Flow.from_kitti(path, load_valid=True)
        desired_flow = np.arange(0, 10)[:, np.newaxis] * np.arange(0, 20)[np.newaxis, :]
        self.assertIsNone(np.testing.assert_equal(f.vecs_numpy[0, ..., 0], desired_flow))
        self.assertIsNone(np.testing.assert_equal(f.vecs_numpy[0, ..., 1], 0))
        self.assertIsNone(np.testing.assert_equal(f.mask_numpy[0, :, 0], True))
        self.assertIsNone(np.testing.assert_equal(f.mask_numpy[0, :, 10], False))
        f = Flow.from_kitti(path, load_valid=False)
        self.assertIsNone(np.testing.assert_equal(f.mask_numpy, True))

        with self.assertRaises(TypeError):  # Wrong load_valid type
            Flow.from_kitti(path, load_valid='test')
        with self.assertRaises(ValueError):  # Wrong path
            Flow.from_kitti('test')
        with self.assertRaises(ValueError):  # Wrong flow shape
            Flow.from_kitti('kitti_wrong.png')

    def test_from_sintel(self):
        path = 'sintel.flo'
        f = Flow.from_sintel(path)
        desired_flow = np.arange(0, 10)[:, np.newaxis] * np.arange(0, 20)[np.newaxis, :]
        self.assertIsNone(np.testing.assert_equal(f.vecs_numpy[0, ..., 0], desired_flow))
        self.assertIsNone(np.testing.assert_equal(f.mask_numpy, True))
        f = Flow.from_sintel(path, 'sintel_invalid.png')
        self.assertIsNone(np.testing.assert_equal(f.mask_numpy[0, :, 0], True))
        self.assertIsNone(np.testing.assert_equal(f.mask_numpy[0, :, 10], False))

        with self.assertRaises(ValueError):  # Wrong tag
            Flow.from_sintel('sintel_wrong.flo')
        with self.assertRaises(ValueError):  # Wrong mask path
            Flow.from_sintel(path, 'test.png')
        with self.assertRaises(ValueError):  # Wrong mask shape
            Flow.from_sintel(path, 'sintel_invalid_wrong.png')

    def test_copy(self):
        vectors = np.random.rand(200, 200, 2)
        mask = np.random.rand(200, 200) > 0.5
        for ref in ['t', 's']:
            for device in ['cpu', 'cuda']:
                flow = Flow(vectors, ref, mask, device)
                flow_copy = flow.copy()
                self.assertIsNone(np.testing.assert_equal(flow.vecs_numpy, flow_copy.vecs_numpy))
                self.assertIsNone(np.testing.assert_equal(flow.mask_numpy, flow_copy.mask_numpy))
                self.assertEqual(flow.ref, flow_copy.ref)
                self.assertEqual(flow.device, flow_copy.device)
                self.assertNotEqual(id(flow), id(flow_copy))

    def test_to_device(self):
        vectors = np.random.rand(200, 200, 2)
        mask = np.random.rand(200, 200) > 0.5
        for ref in ['t', 's']:
            for start_device in ['cpu', 'cuda']:
                for target_device in ['cpu', 'cuda']:
                    flow = Flow(vectors, ref, mask, start_device)
                    f = flow.to_device(target_device)
                    self.assertIsNone(np.testing.assert_equal(flow.vecs_numpy, f.vecs_numpy))
                    self.assertIsNone(np.testing.assert_equal(flow.mask_numpy, f.mask_numpy))
                    self.assertEqual(flow.ref, f.ref)
                    self.assertEqual(f.device.type, target_device)

    def test_str(self):
        flow = Flow.zero(shape=(100, 200), ref='s', device='cuda')
        self.assertEqual(str(flow)[:54],
                         "Flow object, reference s, shape 100*200, device cuda:0")

    def test_getitem(self):
        vectors = np.random.rand(200, 200, 2)
        flow = Flow(vectors)
        indices = np.random.randint(0, 150, size=(20, 2))
        for i in indices:
            # Cutting a number of elements
            self.assertIsNone(np.testing.assert_allclose(flow[i].vecs_numpy, vectors[i]))
            # Cutting a specific item
            self.assertIsNone(np.testing.assert_allclose(flow[i[0]:i[0] + 1, i[1]:i[1] + 1].vecs_numpy,
                                                         vectors[i[0]:i[0] + 1, i[1]:i[1] + 1]))
            # Cutting an area
            self.assertIsNone(np.testing.assert_allclose(flow[i[0]:i[0] + 40, i[1]:i[1] + 40].vecs_numpy,
                                                         vectors[i[0]:i[0] + 40, i[1]:i[1] + 40]))
        # Make sure the device hasn't changed
        for device in ['cpu', 'cuda']:
            flow = Flow(vectors, device=device)
            expected_device = device if torch.cuda.is_available() else 'cpu'
            self.assertEqual(flow[10:20].device.type, expected_device)

    def test_add(self):
        mask1 = np.ones((100, 200), 'bool')
        mask1[:40] = 0
        mask2 = np.ones((100, 200), 'bool')
        mask2[60:] = 0
        vecs1 = np.random.rand(100, 200, 2)
        vecs2 = np.random.rand(100, 200, 2)
        vecs2_np_2hw = np.random.rand(2, 100, 200)
        vecs2_pt_2hw = torch.rand(2, 100, 200)
        vecs2_pt_hw2 = torch.rand(100, 200, 2)
        vec_list = [vecs2, vecs2_np_2hw, vecs2_pt_2hw, vecs2_pt_hw2]
        if torch.cuda.is_available():
            vec_list.append(torch.rand(2, 100, 200).to('cuda'))
            vec_list.append(torch.rand(100, 200, 2).to('cuda'))
        vecs3 = np.random.rand(200, 200, 2)
        flow1 = Flow(vecs1, mask=mask1)
        flow2 = Flow(vecs2, mask=mask2)
        flow3 = Flow(vecs3)

        # Addition
        for vecs in vec_list:
            if isinstance(vecs, torch.Tensor):
                v = to_numpy(vecs)
            else:
                v = vecs
            if v.shape[0] == 2:
                v = np.moveaxis(v, 0, -1)
            self.assertIsNone(np.testing.assert_allclose((flow1 + vecs).vecs_numpy, vecs1 + v,
                                                         rtol=1e-6, atol=1e-6))
            self.assertEqual((flow1 + vecs).device, flow1.vecs.device)
            self.assertEqual((flow1 + vecs).device, flow1.mask.device)
        self.assertIsNone(np.testing.assert_allclose((flow1 + flow2).vecs_numpy, vecs1 + vecs2, rtol=1e-6, atol=1e-6))
        self.assertIsNone(np.testing.assert_equal(np.sum(to_numpy((flow1 + flow2).mask)), (60 - 40) * 200))
        with self.assertRaises(TypeError):
            flow1 + 'test'
        with self.assertRaises(ValueError):
            flow1 + flow3
        with self.assertRaises(ValueError):
            flow1 + vecs3

    def test_sub(self):
        mask1 = np.ones((100, 200), 'bool')
        mask1[:40] = 0
        mask2 = np.ones((100, 200), 'bool')
        mask2[60:] = 0
        vecs1 = np.random.rand(100, 200, 2)
        vecs2 = np.random.rand(100, 200, 2)
        vecs2_np_2hw = np.random.rand(2, 100, 200)
        vecs2_pt_2hw = torch.rand(2, 100, 200)
        vecs2_pt_hw2 = torch.rand(100, 200, 2)
        vec_list = [vecs2, vecs2_np_2hw, vecs2_pt_2hw, vecs2_pt_hw2]
        if torch.cuda.is_available():
            vec_list.append(torch.rand(2, 100, 200).to('cuda'))
            vec_list.append(torch.rand(100, 200, 2).to('cuda'))
        vecs3 = np.random.rand(200, 200, 2)
        flow1 = Flow(vecs1, mask=mask1)
        flow2 = Flow(vecs2, mask=mask2)
        flow3 = Flow(vecs3)

        # Subtraction
        for vecs in vec_list:
            if isinstance(vecs, torch.Tensor):
                v = to_numpy(vecs)
            else:
                v = vecs
            if v.shape[0] == 2:
                v = np.moveaxis(v, 0, -1)
            self.assertIsNone(np.testing.assert_allclose((flow1 - vecs).vecs_numpy, vecs1 - v,
                                                         rtol=1e-6, atol=1e-6))
            self.assertEqual((flow1 + vecs).device, flow1.vecs.device)
            self.assertEqual((flow1 + vecs).device, flow1.mask.device)
        self.assertIsNone(np.testing.assert_allclose((flow1 - flow2).vecs_numpy, vecs1 - vecs2, rtol=1e-6, atol=1e-6))
        self.assertIsNone(np.testing.assert_equal(np.sum(to_numpy((flow1 - flow2).mask)), (60 - 40) * 200))
        with self.assertRaises(TypeError):
            flow1 - 'test'
        with self.assertRaises(ValueError):
            flow1 - flow3
        with self.assertRaises(ValueError):
            flow1 - vecs3

    def test_mul(self):
        vecs1 = np.random.rand(100, 200, 2)
        vecs2 = np.random.rand(100, 200, 2)
        flow1 = Flow(vecs1)

        # Multiplication
        ints = np.random.randint(-10, 10, 100)
        floats = (np.random.rand(100) - .5) * 20
        # ... using ints and floats
        for i, f in zip(ints, floats):
            self.assertIsNone(np.testing.assert_allclose((flow1 * i).vecs_numpy, vecs1 * i, rtol=1e-6, atol=1e-6))
            self.assertIsNone(np.testing.assert_allclose((flow1 * f).vecs_numpy, vecs1 * f, rtol=1e-6, atol=1e-6))
        # ... using a list of length 2
        int_list = np.random.randint(-10, 10, (100, 2))
        for li in int_list:
            v = vecs1.astype('f')
            v[..., 0] *= li[0]
            v[..., 1] *= li[1]
            self.assertIsNone(np.testing.assert_allclose((flow1 * list(li)).vecs_numpy, v, rtol=1e-6, atol=1e-6))
        # ... using a numpy array of size 2
        int_list = np.random.randint(-10, 10, (100, 2))
        for li in int_list:
            v = vecs1.astype('f')
            v[..., 0] *= li[0]
            v[..., 1] *= li[1]
            self.assertIsNone(np.testing.assert_allclose((flow1 * li).vecs_numpy, v, rtol=1e-6, atol=1e-6))
        # ... using a numpy array and torch tensor of the same shape as the flow
        vecs2_pt_hw2 = torch.rand(100, 200, 2)
        vecs_list = [vecs2, vecs2_pt_hw2]
        if torch.cuda.is_available():
            vecs_list.append(torch.rand(100, 200, 2).to('cuda'))
        for vecs in vecs_list:
            if isinstance(vecs, torch.Tensor):
                v = to_numpy(vecs)
            else:
                v = vecs
            self.assertIsNone(np.testing.assert_allclose((flow1 * vecs[..., 0]).vecs_numpy, vecs1 * v[..., :1],
                                                         rtol=1e-6, atol=1e-6))
            self.assertEqual((flow1 * vecs[..., 0]).device, flow1.vecs.device)
            self.assertEqual((flow1 * vecs[..., 0]).device, flow1.mask.device)
        # ... using numpy arrays and torch tensors of the same shape as the flow vectors
        vecs2_np_2hw = np.random.rand(2, 100, 200)
        vecs2_pt_2hw = torch.rand(2, 100, 200)
        vecs2_pt_hw2 = torch.rand(100, 200, 2)
        vecs_list = [vecs2, vecs2_np_2hw, vecs2_pt_2hw, vecs2_pt_hw2]
        if torch.cuda.is_available():
            vecs_list.append(torch.rand(2, 100, 200).to('cuda'))
            vecs_list.append(torch.rand(100, 200, 2).to('cuda'))
        for vecs in vecs_list:
            if isinstance(vecs, torch.Tensor):
                v = to_numpy(vecs)
            else:
                v = vecs
            if v.shape[0] == 2:
                v = np.moveaxis(v, 0, -1)
            self.assertIsNone(np.testing.assert_allclose((flow1 * vecs).vecs_numpy, vecs1 * v,
                                                         rtol=1e-6, atol=1e-6))
            self.assertEqual((flow1 * vecs).device, flow1.vecs.device)
            self.assertEqual((flow1 * vecs).device, flow1.mask.device)
        # ... using a list of the wrong length
        with self.assertRaises(ValueError):
            flow1 * [0, 1, 2]
        # ... using a numpy array of the wrong size
        with self.assertRaises(ValueError):
            flow1 * np.array([0, 1, 2])
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 * np.random.rand(200, 200)
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 * np.random.rand(200, 200, 2)
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 * np.random.rand(100, 200, 3)
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 * np.random.rand(200, 200, 2, 1)

    def test_div(self):
        vecs1 = np.random.rand(100, 200, 2) + .5
        vecs2 = -np.random.rand(100, 200, 2) - .5
        flow1 = Flow(vecs1)

        # Division
        ints = np.random.randint(-10, 10, 100)
        floats = (np.random.rand(100) - .5) * 20
        # ... using ints and floats
        for i, f in zip(ints, floats):
            if i < -1e-5 or i > 1e-5:
                self.assertIsNone(np.testing.assert_allclose((flow1 / i).vecs_numpy, vecs1 / i, rtol=1e-6, atol=1e-6))
            if f < -1e-5 or f > 1e-5:
                self.assertIsNone(np.testing.assert_allclose((flow1 / f).vecs_numpy, vecs1 / f, rtol=1e-6, atol=1e-6))
        # ... using a list of length 2
        int_list = np.random.randint(-10, 10, (100, 2))
        for li in int_list:
            if li[0] != 0 and li[1] != 0:
                v = vecs1.astype('f')
                v[..., 0] /= li[0]
                v[..., 1] /= li[1]
                self.assertIsNone(np.testing.assert_allclose((flow1 / list(li)).vecs_numpy, v, rtol=1e-6, atol=1e-6))
        # ... using a numpy array of size 2
        int_list = np.random.randint(-10, 10, (100, 2))
        for li in int_list:
            if li[0] != 0 and li[1] != 0:
                v = vecs1.astype('f')
                v[..., 0] /= li[0]
                v[..., 1] /= li[1]
                self.assertIsNone(np.testing.assert_allclose((flow1 / li).vecs_numpy, v, rtol=1e-6, atol=1e-6))
        # ... using a numpy array and torch tensor of the same shape as the flow
        vecs2_pt_hw2 = torch.rand(100, 200, 2) + .5
        vecs_list = [vecs2, vecs2_pt_hw2]
        if torch.cuda.is_available():
            vecs_list.append(torch.rand(100, 200, 2).to('cuda') + .5)
        for vecs in vecs_list:
            if isinstance(vecs, torch.Tensor):
                v = to_numpy(vecs)
            else:
                v = vecs
            self.assertIsNone(np.testing.assert_allclose((flow1 / vecs[..., 0]).vecs_numpy, vecs1 / v[..., :1],
                                                         rtol=1e-6, atol=1e-6))
            self.assertEqual((flow1 / vecs[..., 0]).device, flow1.vecs.device)
            self.assertEqual((flow1 / vecs[..., 0]).device, flow1.mask.device)
        # ... using numpy arrays and torch tensors of the same shape as the flow vectors
        vecs2_np_2hw = np.random.rand(2, 100, 200) + .5
        vecs2_pt_2hw = torch.rand(2, 100, 200) + .5
        vecs2_pt_hw2 = torch.rand(100, 200, 2) + .5
        vecs_list = [vecs2, vecs2_np_2hw, vecs2_pt_2hw, vecs2_pt_hw2]
        if torch.cuda.is_available():
            vecs_list.append(torch.rand(2, 100, 200).to('cuda') + .5)
            vecs_list.append(torch.rand(100, 200, 2).to('cuda') + .5)
        for vecs in vecs_list:
            if isinstance(vecs, torch.Tensor):
                v = to_numpy(vecs)
            else:
                v = vecs
            if v.shape[0] == 2:
                v = np.moveaxis(v, 0, -1)
            self.assertIsNone(np.testing.assert_allclose((flow1 / vecs).vecs_numpy, vecs1 / v,
                                                         rtol=1e-6, atol=1e-6))
            self.assertEqual((flow1 / vecs).device, flow1.vecs.device)
            self.assertEqual((flow1 / vecs).device, flow1.mask.device)
        # ... using a list of the wrong length
        with self.assertRaises(ValueError):
            flow1 / [1, 2, 3]
        # ... using a numpy array of the wrong size
        with self.assertRaises(ValueError):
            flow1 / np.array([1, 2, 3])
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 / np.ones((200, 200))
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 / np.ones((200, 200, 2))
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 / np.ones((100, 200, 3))
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 / np.ones((200, 200, 2, 1))

    def test_pow(self):
        vecs1 = np.random.rand(100, 200, 2)
        vecs2 = np.random.rand(100, 200, 2)
        flow1 = Flow(vecs1)

        # Exponentiation
        ints = np.random.randint(-2, 2, 100)
        floats = (np.random.rand(100) - .5) * 4
        # ... using ints and floats
        for i, f in zip(ints, floats):
            self.assertIsNone(np.testing.assert_allclose((flow1 ** i).vecs_numpy, vecs1 ** i, rtol=1e-6, atol=1e-6))
            self.assertIsNone(np.testing.assert_allclose((flow1 ** f).vecs_numpy, vecs1 ** f, rtol=1e-6, atol=1e-6))
        # ... using a list of length 2
        int_list = np.random.randint(-5, 5, (100, 2))
        for li in int_list:
            v = vecs1.astype('f')
            v[..., 0] **= li[0]
            v[..., 1] **= li[1]
            self.assertIsNone(np.testing.assert_allclose((flow1 ** list(li)).vecs_numpy, v, rtol=1e-6, atol=1e-6))
        # ... using a numpy array of size 2
        int_list = np.random.randint(-5, 5, (100, 2))
        for li in int_list:
            v = vecs1.astype('f')
            v[..., 0] **= li[0]
            v[..., 1] **= li[1]
            self.assertIsNone(np.testing.assert_allclose((flow1 ** li).vecs_numpy, v, rtol=1e-6, atol=1e-6))
        # ... using a numpy array and torch tensor of the same shape as the flow
        vecs2_pt_hw2 = torch.rand(100, 200, 2)
        vecs_list = [vecs2, vecs2_pt_hw2]
        if torch.cuda.is_available():
            vecs_list.append(torch.rand(100, 200, 2).to('cuda'))
        for vecs in vecs_list:
            if isinstance(vecs, torch.Tensor):
                v = to_numpy(vecs)
            else:
                v = vecs
            self.assertIsNone(np.testing.assert_allclose((flow1 ** vecs[..., 0]).vecs_numpy, vecs1 ** v[..., :1],
                                                         rtol=1e-6, atol=1e-6))
            self.assertEqual((flow1 ** vecs[..., 0]).device, flow1.vecs.device)
            self.assertEqual((flow1 ** vecs[..., 0]).device, flow1.mask.device)
        # ... using numpy arrays and torch tensors of the same shape as the flow vectors
        vecs2_np_2hw = np.random.rand(2, 100, 200)
        vecs2_pt_2hw = torch.rand(2, 100, 200)
        vecs2_pt_hw2 = torch.rand(100, 200, 2)
        vecs_list = [vecs2, vecs2_np_2hw, vecs2_pt_2hw, vecs2_pt_hw2]
        if torch.cuda.is_available():
            vecs_list.append(torch.rand(2, 100, 200).to('cuda'))
            vecs_list.append(torch.rand(100, 200, 2).to('cuda'))
        for vecs in vecs_list:
            if isinstance(vecs, torch.Tensor):
                v = to_numpy(vecs)
            else:
                v = vecs
            if v.shape[0] == 2:
                v = np.moveaxis(v, 0, -1)
            self.assertIsNone(np.testing.assert_allclose((flow1 ** vecs).vecs_numpy, vecs1 ** v,
                                                         rtol=1e-6, atol=1e-6))
            self.assertEqual((flow1 ** vecs).device, flow1.vecs.device)
            self.assertEqual((flow1 ** vecs).device, flow1.mask.device)
        # ... using a list of the wrong length
        with self.assertRaises(ValueError):
            flow1 ** [0, 1, 2]
        # ... using a numpy array of the wrong size
        with self.assertRaises(ValueError):
            flow1 ** np.array([0, 1, 2])
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 ** np.random.rand(200, 200)
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 ** np.random.rand(200, 200, 2)
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 ** np.random.rand(100, 200, 3)
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 ** np.random.rand(200, 200, 2, 1)

    def test_neg(self):
        vecs1 = np.random.rand(100, 200, 2)
        flow1 = Flow(vecs1)
        self.assertIsNone(np.testing.assert_allclose((-flow1).vecs_numpy, -vecs1))

    def test_resize(self):
        shape = [20, 10]
        ref = 's'
        flow = Flow.from_transforms([['rotation', 30, 50, 30]], shape, ref)

        # Different scales
        scales = [.2, .5, 1, 1.5, 2, 10]
        for scale in scales:
            self.assertIsNone(np.testing.assert_equal(flow.resize(scale).vecs_numpy,
                                                      to_numpy(resize_flow(flow.vecs, scale), switch_channels=True)))
        # Scale mask
        shape_small = (20, 40)
        shape_large = (30, 80)
        mask_small = np.ones(shape_small, 'bool')
        mask_small[:6, :20] = 0
        mask_large = np.ones(shape_large, 'bool')
        mask_large[:9, :40] = 0
        flow_small = Flow.from_transforms([['rotation', 0, 0, 30]], shape_small, 't', mask_small)
        flow_large = flow_small.resize((1.5, 2))
        self.assertIsNone(np.testing.assert_equal(to_numpy(flow_large.mask), mask_large))

        # Check scaling is performed correctly based on the actual flow field
        ref = 't'
        flow_small = Flow.from_transforms([['rotation', 0, 0, 30]], (50, 80), ref)
        flow_large = Flow.from_transforms([['rotation', 0, 0, 30]], (150, 240), ref)
        flow_resized = flow_large.resize(1/3)
        self.assertIsNone(np.testing.assert_allclose(flow_resized.vecs_numpy, flow_small.vecs_numpy, atol=1, rtol=.1))

    def test_pad(self):
        shape = [100, 80]
        for ref in ['t', 's']:
            flow = Flow.zero(shape, ref, np.ones(shape, 'bool'))
            flow = flow.pad([10, 20, 30, 40])
            self.assertIsNone(np.testing.assert_equal(flow.shape[:2], [shape[0] + 10 + 20, shape[1] + 30 + 40]))
            self.assertIsNone(np.testing.assert_equal(flow.vecs_numpy, 0))
            self.assertIsNone(np.testing.assert_equal(to_numpy(flow[10:-20, 30:-40].mask), 1))
            flow.mask[10:-20, 30:-40] = 0
            self.assertIsNone(np.testing.assert_equal(to_numpy(flow.mask), 0))
            self.assertIs(flow.ref, ref)

        # 'Replicate' padding
        flow = Flow.from_transforms([['rotation', 30, 50, 30]], shape, ref)
        padded_flow = flow.pad([10, 10, 20, 20], mode='replicate')
        self.assertIsNone(np.testing.assert_equal(padded_flow.vecs_numpy[0, 20:-20], flow.vecs_numpy[0]))
        self.assertIsNone(np.testing.assert_equal(padded_flow.vecs_numpy[10:-10, 0], flow.vecs_numpy[:, 0]))

        # 'Reflect' padding
        padded_flow = flow.pad([10, 10, 20, 20], mode='reflect')
        self.assertIsNone(np.testing.assert_equal(padded_flow.vecs_numpy[0, 20:-20], flow.vecs_numpy[10]))
        self.assertIsNone(np.testing.assert_equal(padded_flow.vecs_numpy[10:-10, 0], flow.vecs_numpy[:, 20]))

        # Invalid padding mode
        with self.assertRaises(ValueError):
            flow.pad([10, 10, 20, 20], mode='test')

    def test_apply(self):
        img_np = np.moveaxis(cv2.imread('smudge.png'), -1, 0)
        img_pt = torch.tensor(img_np)
        # Check flow.apply results in the same as using apply_flow directly
        for ref in ['t', 's']:
            for consider_mask in [True, False]:
                for device in ['cpu', 'cuda']:
                    for img in [img_pt.to('cpu'), img_pt.to('cuda')]:
                        mask = torch.ones(img_pt.shape[1:], dtype=torch.bool)
                        mask[400:] = False
                        flow = Flow.from_transforms([['rotation', 30, 50, 30]], img.shape[1:], ref, mask, device)
                        # Target is a 3D torch tensor
                        warped_img_desired = apply_flow(flow.vecs, img, ref, mask if consider_mask else None)
                        warped_img_actual = flow.apply(img, consider_mask=consider_mask)
                        self.assertEqual(flow.device, warped_img_actual.device)
                        self.assertIsNone(np.testing.assert_equal(to_numpy(warped_img_actual),
                                                                  to_numpy(warped_img_desired)))
                        warped_img_actual, _ = flow.apply(img, mask, True, consider_mask=consider_mask)
                        self.assertIsNone(np.testing.assert_equal(to_numpy(warped_img_actual),
                                                                  to_numpy(warped_img_desired)))
                        # Target is a 2D torch tensor
                        warped_img_desired = apply_flow(flow.vecs, img[0], ref, mask if consider_mask else None)
                        warped_img_actual = flow.apply(img[0], consider_mask=consider_mask)
                        self.assertEqual(flow.device, warped_img_actual.device)
                        self.assertIsNone(np.testing.assert_equal(to_numpy(warped_img_actual),
                                                                  to_numpy(warped_img_desired)))
                        warped_img_actual, _ = flow.apply(img[0], mask, True, consider_mask=consider_mask)
                        self.assertIsNone(np.testing.assert_equal(to_numpy(warped_img_actual),
                                                                  to_numpy(warped_img_desired)))
                    for f_device in ['cpu', 'cuda']:
                        f = flow.to_device(f_device)
                        # Target is a flow object
                        warped_flow_desired = apply_flow(flow.vecs, f.vecs, ref, mask if consider_mask else None)
                        warped_flow_actual = flow.apply(f, consider_mask=consider_mask)
                        self.assertEqual(flow.device, warped_flow_actual.device)
                        self.assertIsNone(np.testing.assert_equal(to_numpy(warped_flow_actual.vecs),
                                                                  to_numpy(warped_flow_desired)))
        # Check using a smaller flow field on a larger target works the same as a full flow field on the same target
        img = img_pt
        ref = 't'
        flow = Flow.from_transforms([['rotation', 30, 50, 30]], img.shape[1:], ref)
        warped_img_desired = apply_flow(flow.vecs, img, ref)
        shape = [img.shape[1] - 90, img.shape[2] - 110]
        padding = [50, 40, 30, 80]
        cut_flow = Flow.from_transforms([['rotation', 0, 0, 30]], shape, ref)
        # # ... not cutting (target torch tensor)
        # warped_img_actual = cut_flow.apply(img, padding=padding, cut=False)
        # self.assertIsNone(np.testing.assert_equal(to_numpy(warped_img_actual[padding[0]:-padding[1],
        #                                                    padding[2]:-padding[3]]),
        #                                           to_numpy(warped_img_desired[padding[0]:-padding[1],
        #                                                    padding[2]:-padding[3]])))
        # ... cutting (target torch tensor)
        warped_img_actual = cut_flow.apply(img, padding=padding, cut=True)
        self.assertIsNone(np.testing.assert_allclose(to_numpy(warped_img_actual).astype('f'),
                                                     to_numpy(warped_img_desired[:, padding[0]:-padding[1],
                                                              padding[2]:-padding[3]]).astype('f'),
                                                     atol=1))  # result rounded (uint8), so errors can be 1
        # ... not cutting (target flow object)
        target_flow = Flow.from_transforms([['rotation', 30, 50, 30]], img.shape[1:], ref)
        warped_flow_desired = apply_flow(flow.vecs, target_flow.vecs, ref)
        warped_flow_actual = cut_flow.apply(target_flow, padding=padding, cut=False)
        self.assertIsNone(np.testing.assert_allclose(to_numpy(warped_flow_actual.vecs[:, padding[0]:-padding[1],
                                                              padding[2]:-padding[3]]),
                                                     to_numpy(warped_flow_desired[:, padding[0]:-padding[1],
                                                              padding[2]:-padding[3]]),
                                                     atol=1e-1))
        # ... cutting (target flow object)
        warped_flow_actual = cut_flow.apply(target_flow, padding=padding, cut=True)
        self.assertIsNone(np.testing.assert_allclose(to_numpy(warped_flow_actual.vecs),
                                                     to_numpy(warped_flow_desired[:, padding[0]:-padding[1],
                                                              padding[2]:-padding[3]]),
                                                     atol=1e-1))

        # Non-valid padding values
        for ref in ['t', 's']:
            flow = Flow.from_transforms([['rotation', 0, 0, 30]], shape, ref)
            with self.assertRaises(TypeError):
                flow.apply(target_flow, return_valid_area='test')
            with self.assertRaises(TypeError):
                flow.apply(target_flow, consider_mask='test')
            with self.assertRaises(TypeError):
                flow.apply(target_flow, padding=100, cut=True)
            with self.assertRaises(ValueError):
                flow.apply(target_flow, padding=[10, 20, 30, 40, 50], cut=True)
            with self.assertRaises(ValueError):
                flow.apply(target_flow, padding=[10., 20, 30, 40], cut=True)
            with self.assertRaises(ValueError):
                flow.apply(target_flow, padding=[-10, 10, 10, 10], cut=True)
            with self.assertRaises(TypeError):
                flow.apply(target_flow, padding=[10, 20, 30, 40, 50], cut=2)
            with self.assertRaises(TypeError):
                flow.apply(target_flow, padding=[10, 20, 30, 40, 50], cut='true')

    def test_switch_ref(self):
        shape = (200, 300)
        # Mode 'invalid'
        for refs in [['t', 's'], ['s', 't']]:
            flow = Flow.from_transforms([['rotation', 30, 50, 30]], shape, refs[0])
            flow = flow.switch_ref(mode='invalid')
            self.assertEqual(flow.ref, refs[1])

        # Mode 'valid'
        transforms = [['rotation', 256, 256, 30]]
        flow_s = Flow.from_transforms(transforms, shape, 's')
        flow_t = Flow.from_transforms(transforms, shape, 't')
        switched_s = flow_t.switch_ref()
        self.assertIsNone(np.testing.assert_allclose(switched_s.vecs_numpy[switched_s.mask_numpy],
                                                     flow_s.vecs_numpy[switched_s.mask_numpy],
                                                     rtol=1e-3, atol=1e-3))
        switched_t = flow_s.switch_ref()
        self.assertIsNone(np.testing.assert_allclose(switched_t.vecs_numpy[switched_t.mask_numpy],
                                                     flow_t.vecs_numpy[switched_t.mask_numpy],
                                                     rtol=1e-3, atol=1e-3))

        # Invalid mode passed
        flow = Flow.from_transforms([['rotation', 30, 50, 30]], shape, 't')
        with self.assertRaises(ValueError):
            flow.switch_ref('test')
        with self.assertRaises(ValueError):
            flow.switch_ref(1)

    def test_invert(self):
        f_s = Flow.from_transforms([['rotation', 256, 256, 30]], (512, 512), 's')   # Forwards
        f_t = Flow.from_transforms([['rotation', 256, 256, 30]], (512, 512), 't')   # Forwards
        b_s = Flow.from_transforms([['rotation', 256, 256, -30]], (512, 512), 's')  # Backwards
        b_t = Flow.from_transforms([['rotation', 256, 256, -30]], (512, 512), 't')  # Backwards

        # Inverting s to s
        b_s_inv = f_s.invert()
        self.assertIsNone(np.testing.assert_allclose(b_s_inv.vecs_numpy[b_s_inv.mask_numpy],
                                                     b_s.vecs_numpy[b_s_inv.mask_numpy],
                                                     rtol=1e-3, atol=1e-3))
        f_s_inv = b_s.invert()
        self.assertIsNone(np.testing.assert_allclose(f_s_inv.vecs_numpy[f_s_inv.mask_numpy],
                                                     f_s.vecs_numpy[f_s_inv.mask_numpy],
                                                     rtol=1e-3, atol=1e-3))

        # Inverting s to t
        b_t_inv = f_s.invert('t')
        self.assertIsNone(np.testing.assert_allclose(b_t_inv.vecs_numpy[b_t_inv.mask_numpy],
                                                     b_t.vecs_numpy[b_t_inv.mask_numpy],
                                                     rtol=1e-3, atol=1e-3))
        f_t_inv = b_s.invert('t')
        self.assertIsNone(np.testing.assert_allclose(f_t_inv.vecs_numpy[f_t_inv.mask_numpy],
                                                     f_t.vecs_numpy[f_t_inv.mask_numpy],
                                                     rtol=1e-3, atol=1e-3))

        # Inverting t to t
        b_t_inv = f_t.invert()
        self.assertIsNone(np.testing.assert_allclose(b_t_inv.vecs_numpy[b_t_inv.mask_numpy],
                                                     b_t.vecs_numpy[b_t_inv.mask_numpy],
                                                     rtol=1e-3, atol=1e-3))
        f_t_inv = b_t.invert()
        self.assertIsNone(np.testing.assert_allclose(f_t_inv.vecs_numpy[f_t_inv.mask_numpy],
                                                     f_t.vecs_numpy[f_t_inv.mask_numpy],
                                                     rtol=1e-3, atol=1e-3))

        # Inverting t to s
        b_s_inv = f_t.invert('s')
        self.assertIsNone(np.testing.assert_allclose(b_s_inv.vecs_numpy[b_s_inv.mask_numpy],
                                                     b_s.vecs_numpy[b_s_inv.mask_numpy],
                                                     rtol=1e-3, atol=1e-3))
        f_s_inv = b_t.invert('s')
        self.assertIsNone(np.testing.assert_allclose(f_s_inv.vecs_numpy[f_s_inv.mask_numpy],
                                                     f_s.vecs_numpy[f_s_inv.mask_numpy],
                                                     rtol=1e-3, atol=1e-3))

    def test_track(self):
        f_s = Flow.from_transforms([['rotation', 0, 0, 30]], (512, 512), 's')
        f_t = Flow.from_transforms([['rotation', 0, 0, 30]], (512, 512), 't')

        # Test valid status for 't' flow
        f_t.mask[:, 200:] = False
        pts = torch.tensor([
            [0, 50],            # Moved out of bounds by a valid flow vector
            [0, 500],           # Moved out of bounds by an invalid flow vector
            [8.3, 7.2],         # Moved normally by valid flow vector
            [120.4, 160.2],     # Moved normally by valid flow vector
            [300, 200]          # Moved normally by invalid flow vector
        ])
        desired_valid_status = [False, False, True, True, False]
        _, tracked = f_t.track(pts, get_valid_status=True)
        self.assertIsNone(np.testing.assert_equal(to_numpy(tracked), desired_valid_status))

        # Test valid status for 's' flow
        f_s.mask[:, 200:] = False
        pts = torch.tensor([
            [0, 50],            # Moved out of bounds by a valid flow vector
            [0, 500],           # Moved out of bounds by an invalid flow vector
            [8.3, 7.2],         # Moved normally by valid flow vector
            [120.4, 160.2],     # Moved normally by valid flow vector
            [300, 200]          # Moved normally by invalid flow vector
        ])
        desired_valid_status = [False, False, True, True, False]
        _, tracked = f_s.track(pts, get_valid_status=True)
        self.assertIsNone(np.testing.assert_equal(to_numpy(tracked), desired_valid_status))

        # Invalid inputs
        with self.assertRaises(TypeError):
            f_s.track(pts, True, get_valid_status='test')

    def test_valid_target(self):
        transforms = [['rotation', 0, 0, 45]]
        shape = (7, 7)
        mask = np.ones(shape, 'bool')
        mask[4:, :3] = False
        f_s_masked = Flow.from_transforms(transforms, shape, 's', mask)
        mask = np.ones(shape, 'bool')
        mask[:3, 4:] = False
        f_t_masked = Flow.from_transforms(transforms, shape, 't', mask)
        f_s = Flow.from_transforms(transforms, shape, 's')
        f_t = Flow.from_transforms(transforms, shape, 't')
        desired_area_s = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]).astype('bool')
        desired_area_t = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]).astype('bool')
        desired_area_s_masked_consider_mask = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]).astype('bool')
        desired_area_s_masked = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]).astype('bool')
        desired_area_t_masked = np.array([
            [1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]).astype('bool')
        self.assertIsNone(np.testing.assert_equal(to_numpy(f_s.valid_target()), desired_area_s))
        self.assertIsNone(np.testing.assert_equal(to_numpy(f_t.valid_target()), desired_area_t))
        self.assertIsNone(np.testing.assert_equal(f_s_masked.valid_target(), desired_area_s_masked_consider_mask))
        self.assertIsNone(np.testing.assert_equal(to_numpy(f_s_masked.valid_target(False)), desired_area_s_masked))
        self.assertIsNone(np.testing.assert_equal(to_numpy(f_t_masked.valid_target()), desired_area_t_masked))

    def test_valid_source(self):
        transforms = [['rotation', 0, 0, 45]]
        shape = (7, 7)
        mask = np.ones(shape, 'bool')
        mask[4:, :3] = False
        f_s_masked = Flow.from_transforms(transforms, shape, 's', mask)
        mask = np.ones(shape, 'bool')
        mask[:3, 4:] = False
        f_t_masked = Flow.from_transforms(transforms, shape, 't', mask)
        f_s = Flow.from_transforms(transforms, shape, 's')
        f_t = Flow.from_transforms(transforms, shape, 't')
        desired_area_s = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0]
        ]).astype('bool')
        desired_area_t = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0]
        ]).astype('bool')
        desired_area_s_masked = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]).astype('bool')
        desired_area_t_masked_consider_mask = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0]
        ]).astype('bool')
        desired_area_t_masked = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0]
        ]).astype('bool')
        self.assertIsNone(np.testing.assert_equal(to_numpy(f_s.valid_source()), desired_area_s))
        self.assertIsNone(np.testing.assert_equal(to_numpy(f_t.valid_source()), desired_area_t))
        self.assertIsNone(np.testing.assert_equal(to_numpy(f_s_masked.valid_source()), desired_area_s_masked))
        self.assertIsNone(np.testing.assert_equal(f_t_masked.valid_source(), desired_area_t_masked_consider_mask))
        self.assertIsNone(np.testing.assert_equal(f_t_masked.valid_source(False), desired_area_t_masked))

    def test_get_padding(self):
        transforms = [['rotation', 0, 0, 45]]
        shape = (7, 7)
        mask = np.ones(shape, 'bool')
        mask[:, 4:] = False
        f_s_masked = Flow.from_transforms(transforms, shape, 's', mask)
        mask = np.ones(shape, 'bool')
        mask[4:] = False
        f_t_masked = Flow.from_transforms(transforms, shape, 't', mask)
        f_s = Flow.from_transforms(transforms, shape, 's')
        f_t = Flow.from_transforms(transforms, shape, 't')
        f_s_desired = [5, 0, 0, 3]
        f_t_desired = [0, 3, 5, 0]
        f_s_masked_desired = [3, 0, 0, 1]
        f_t_masked_desired = [0, 1, 3, 0]
        self.assertIsNone(np.testing.assert_equal(f_s.get_padding(), f_s_desired))
        self.assertIsNone(np.testing.assert_equal(f_t.get_padding(), f_t_desired))
        self.assertIsNone(np.testing.assert_equal(f_s_masked.get_padding(), f_s_masked_desired))
        self.assertIsNone(np.testing.assert_equal(f_t_masked.get_padding(), f_t_masked_desired))

        f = Flow.zero(shape)
        f._vecs[0] = torch.rand(*shape) * 1e-4
        self.assertIsNone(np.testing.assert_equal(f.get_padding(), [0, 0, 0, 0]))

    def test_is_zero(self):
        shape = (10, 10)
        mask = np.ones(shape, 'bool')
        mask[0, 0] = False
        flow = np.zeros(shape + (2,))
        flow[0, 0] = 10
        flow = Flow(flow, mask=mask)
        self.assertEqual(flow.is_zero(), True)
        self.assertEqual(flow.is_zero(masked=True), True)
        self.assertEqual(flow.is_zero(masked=False), False)

        with self.assertRaises(TypeError):  # Masked wrong type
            flow.is_zero(masked='test')

    def test_visualise(self):
        # Correct values for the different modes
        # Horizontal flow towards the right is red
        flow = Flow.from_transforms([['translation', 1, 0]], [200, 300])
        desired_img = np.tile(np.array([0, 0, 255]).reshape((1, 1, 3)), (200, 300, 1))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('bgr', return_tensor=False), desired_img))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('rgb', return_tensor=False), desired_img[..., ::-1]))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', return_tensor=False)[..., 0], 0))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', return_tensor=False)[..., 1], 255))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', return_tensor=False)[..., 2], 255))

        # Flow outwards at the angle of 240 degrees (counter-clockwise) is green
        flow = Flow.from_transforms([['translation', -1, math.sqrt(3)]], [200, 300])
        desired_img = np.tile(np.array([0, 255, 0]).reshape((1, 1, 3)), (200, 300, 1))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('bgr', return_tensor=False), desired_img))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('rgb', return_tensor=False), desired_img))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', return_tensor=False)[..., 0], 60))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', return_tensor=False)[..., 1], 255))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', return_tensor=False)[..., 2], 255))

        # Flow outwards at the angle of 240 degrees (counter-clockwise) is blue
        flow = Flow.from_transforms([['translation', -1, -math.sqrt(3)]], [200, 300])
        desired_img = np.tile(np.array([255, 0, 0]).reshape((1, 1, 3)), (200, 300, 1))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('bgr', return_tensor=False), desired_img))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('rgb', return_tensor=False), desired_img[..., ::-1]))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', return_tensor=False)[..., 0], 120))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', return_tensor=False)[..., 1], 255))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', return_tensor=False)[..., 2], 255))

        # Show the flow mask
        mask = np.zeros((200, 300))
        mask[30:-30, 40:-40] = 1
        flow = Flow.from_transforms([['translation', 1, 0]], (200, 300), 't', mask)
        self.assertIsNone(np.testing.assert_equal(flow.visualise('bgr', True, return_tensor=False)[10, 10],
                                                  [0, 0, 180]))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('rgb', True, return_tensor=False)[10, 10],
                                                  [180, 0, 0]))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', True, return_tensor=False)[..., 0], 0))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', True, return_tensor=False)[..., 1], 255))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', True, return_tensor=False)[10, 10, 2], 180))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', True, return_tensor=False)[100, 100, 2], 255))

        # Show the flow mask border
        mask = np.zeros((200, 300))
        mask[30:-30, 40:-40] = 1
        flow = Flow.from_transforms([['translation', 1, 0]], (200, 300), 't', mask)
        self.assertIsNone(np.testing.assert_equal(flow.visualise('bgr', True, True, return_tensor=False)[30, 40],
                                                  [0, 0, 0]))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('rgb', True, True, return_tensor=False)[30, 40],
                                                  [0, 0, 0]))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', True, True, return_tensor=False)[..., 0], 0))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', True, True, return_tensor=False)[30, 40, 1], 0))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', True, True, return_tensor=False)[30, 40, 2], 0))

        # Output is tensor if required
        mask = np.zeros((200, 300))
        mask[30:-30, 40:-40] = 1
        flow = Flow.from_transforms([['translation', 1, 0]], (200, 300), 't', mask)
        self.assertIsInstance(flow.visualise('bgr', True, True), torch.Tensor)

        # Invalid arguments
        flow = Flow.zero([10, 10])
        with self.assertRaises(ValueError):
            flow.visualise(mode=3)
        with self.assertRaises(ValueError):
            flow.visualise(mode='test')
        with self.assertRaises(TypeError):
            flow.visualise('rgb', show_mask=2)
        with self.assertRaises(TypeError):
            flow.visualise('rgb', show_mask_borders=2)
        with self.assertRaises(TypeError):
            flow.visualise('rgb', return_tensor=2)
        with self.assertRaises(TypeError):
            flow.visualise('rgb', range_max='2')
        with self.assertRaises(ValueError):
            flow.visualise('rgb', range_max=-1)

    def test_visualise_arrows(self):
        img = cv2.imread('smudge.png')
        mask = np.zeros(img.shape[:2])
        mask[50:-50, 20:-20] = 1
        flow = Flow.from_transforms([['rotation', 256, 256, 30]], img.shape[:2], 's', mask)
        for scaling in [0.1, 1, 2]:
            for show_mask in [True, False]:
                for show_mask_border in [True, False]:
                    for return_tensor in [True, False]:
                        img = flow.visualise_arrows(
                            grid_dist=10,
                            scaling=scaling,
                            show_mask=show_mask,
                            show_mask_borders=show_mask_border,
                            return_tensor=return_tensor
                        )
                        if return_tensor:
                            self.assertIsInstance(img, torch.Tensor)
                        else:
                            self.assertIsInstance(img, np.ndarray)
        with self.assertRaises(TypeError):
            flow.visualise_arrows(grid_dist='test')
        with self.assertRaises(ValueError):
            flow.visualise_arrows(grid_dist=-1)
        with self.assertRaises(TypeError):
            flow.visualise_arrows(10, img='test')
        with self.assertRaises(ValueError):
            flow.visualise_arrows(10, img=mask)
        with self.assertRaises(ValueError):
            flow.visualise_arrows(10, img=mask[10:])
        with self.assertRaises(ValueError):
            flow.visualise_arrows(10, img=img[..., :2])
        with self.assertRaises(TypeError):
            flow.visualise_arrows(10, img, scaling='test')
        with self.assertRaises(ValueError):
            flow.visualise_arrows(10, img, scaling=-1)
        with self.assertRaises(TypeError):
            flow.visualise_arrows(10, img, None, show_mask='test')
        with self.assertRaises(TypeError):
            flow.visualise_arrows(10, img, None, True, show_mask_borders='test')
        with self.assertRaises(TypeError):
            flow.visualise_arrows(10, img, None, True, True, colour='test')
        with self.assertRaises(ValueError):
            flow.visualise_arrows(10, img, None, True, True, colour=(0, 0))
        with self.assertRaises(TypeError):
            flow.visualise_arrows(10, img, None, True, True, colour=(0, 0, 0), return_tensor='test')

    def test_show(self):
        flow = Flow.zero([200, 300])
        with self.assertRaises(TypeError):
            flow.show('test')
        with self.assertRaises(ValueError):
            flow.show(-1)

    def test_show_arrows(self):
        flow = Flow.zero([200, 300])
        with self.assertRaises(TypeError):
            flow.show_arrows('test')
        with self.assertRaises(ValueError):
            flow.show_arrows(-1)

    def test_matrix(self):
        # Partial affine transform, test reconstruction with all methods
        transforms = [
            ['translation', 2, 1],
            ['rotation', 20, 20, 30],
            ['scaling', 10, 10, 1.1]
        ]
        matrix = matrix_from_transforms(transforms)
        flow_s = Flow.from_matrix(matrix, (100, 200), 's')
        flow_t = Flow.from_matrix(matrix, (100, 200), 't')
        actual_matrix_s = flow_s.matrix(dof=4, method='ransac')
        actual_matrix_t = flow_t.matrix(dof=4, method='ransac')
        self.assertIsNone(np.testing.assert_allclose(to_numpy(actual_matrix_s), matrix, rtol=1e-6))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(actual_matrix_t), matrix, rtol=1e-3))
        actual_matrix_s = flow_s.matrix(dof=4, method='lmeds')
        actual_matrix_t = flow_t.matrix(dof=4, method='lmeds')
        self.assertIsNone(np.testing.assert_allclose(to_numpy(actual_matrix_s), matrix, rtol=1e-6))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(actual_matrix_t), matrix, rtol=1e-3))
        actual_matrix_s = flow_s.matrix(dof=6, method='ransac')
        actual_matrix_t = flow_t.matrix(dof=6, method='ransac')
        self.assertIsNone(np.testing.assert_allclose(to_numpy(actual_matrix_s), matrix, rtol=1e-6))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(actual_matrix_t), matrix, rtol=1e-3))
        actual_matrix_s = flow_s.matrix(dof=6, method='lmeds')
        actual_matrix_t = flow_t.matrix(dof=6, method='lmeds')
        self.assertIsNone(np.testing.assert_allclose(to_numpy(actual_matrix_s), matrix, rtol=1e-6))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(actual_matrix_t), matrix, rtol=1e-3))
        actual_matrix_s = flow_s.matrix(dof=8, method='lms')
        actual_matrix_t = flow_t.matrix(dof=8, method='lms')
        self.assertIsNone(np.testing.assert_allclose(to_numpy(actual_matrix_s), matrix, rtol=1e-6, atol=1e-4))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(actual_matrix_t), matrix, rtol=1e-6, atol=1e-4))
        actual_matrix_s = flow_s.matrix(dof=8, method='ransac')
        actual_matrix_t = flow_t.matrix(dof=8, method='ransac')
        self.assertIsNone(np.testing.assert_allclose(to_numpy(actual_matrix_s), matrix, rtol=1e-6, atol=1e-4))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(actual_matrix_t), matrix, rtol=1e-6, atol=1e-4))
        actual_matrix_s = flow_s.matrix(dof=8, method='lmeds')
        actual_matrix_t = flow_t.matrix(dof=8, method='lmeds')
        self.assertIsNone(np.testing.assert_allclose(to_numpy(actual_matrix_s), matrix, rtol=1e-6, atol=1e-4))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(actual_matrix_t), matrix, rtol=1e-6, atol=1e-4))

        # Random matrix, check to see how often an approximate 'reconstruction' fails, target is <5% of cases
        failed = 0
        for i in range(1000):
            matrix = (np.random.rand(3, 3) - .5) * 20
            if -1e-4 < matrix[2, 2] < 1e-4:
                matrix[2, 2] = 0
            else:
                matrix /= matrix[2, 2]
            flow_s = Flow.from_matrix(matrix, (50, 100), 's')
            try:
                np.testing.assert_allclose(flow_s.matrix(8, 'lms'), matrix, atol=1e-2, rtol=1e-2)
                np.testing.assert_allclose(flow_s.matrix(8, 'ransac'), matrix, atol=1e-2, rtol=1e-2)
                np.testing.assert_allclose(flow_s.matrix(8, 'lmeds'), matrix, atol=1e-2, rtol=1e-2)
            except AssertionError:
                failed += 1
        self.assertTrue(failed <= 50)

        # Partial affine transform reconstruction in the presence of noise, only check first 4 values
        matrix = matrix_from_transforms(transforms)
        flow_s = Flow.from_matrix(matrix, (100, 200), 's')
        flow_noise = (np.random.rand(100, 200, 2) - .5) * 5
        actual_matrix_4_ransac = (flow_s + flow_noise).matrix(4, 'ransac')
        actual_matrix_4_lmeds = (flow_s + flow_noise).matrix(4, 'lmeds')
        actual_matrix_6_ransac = (flow_s + flow_noise).matrix(6, 'ransac')
        actual_matrix_6_lmeds = (flow_s + flow_noise).matrix(6, 'lmeds')
        actual_matrix_8_lms = (flow_s + flow_noise).matrix(8, 'lms')
        actual_matrix_8_ransac = (flow_s + flow_noise).matrix(8, 'ransac')
        actual_matrix_8_lmeds = (flow_s + flow_noise).matrix(8, 'lmeds')
        for actual_matrix in [actual_matrix_4_ransac, actual_matrix_4_lmeds,
                              actual_matrix_6_ransac, actual_matrix_6_lmeds,
                              actual_matrix_8_lms, actual_matrix_8_ransac, actual_matrix_8_lmeds]:
            self.assertIsNone(np.testing.assert_allclose(actual_matrix[:2, :2], matrix[:2, :2], atol=1e-2, rtol=1e-1))

        # Masked vs non-masked matrix fitting
        matrix = matrix_from_transforms(transforms)
        mask = np.zeros((100, 200), 'bool')
        mask[:50, :50] = 1  # upper left corner will contain the real values
        flow = Flow.from_matrix(matrix, (100, 200), 's', mask)
        random_vecs = (np.random.rand(2, 100, 200) - 0.5) * 200
        random_vecs[:, :50, :50] = flow.vecs[:, :50, :50]
        flow.vecs = random_vecs
        # Make sure this fails with the 'lmeds' method:
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(flow.matrix(4, 'lmeds', False), matrix)
        # Test that it does NOT fail when the invalid flow elements are masked out
        self.assertIsNone(np.testing.assert_allclose(flow.matrix(4, 'lmeds', True), matrix))

        # Fallback of 'lms' to 'ransac' when dof == 4 or dof == 6
        matrix = matrix_from_transforms(transforms)
        flow_s = Flow.from_matrix(matrix, (100, 200), 's')
        actual_matrix_s_lms = flow_s.matrix(dof=4, method='lms')
        actual_matrix_s_ransac = flow_s.matrix(dof=4, method='ransac')
        self.assertIsNone(np.testing.assert_equal(to_numpy(actual_matrix_s_lms), to_numpy(actual_matrix_s_ransac)))

        # Invalid inputs
        matrix = matrix_from_transforms(transforms)
        flow_s = Flow.from_matrix(matrix, (100, 200), 's')
        with self.assertRaises(ValueError):
            flow_s.matrix(dof='test')
        with self.assertRaises(ValueError):
            flow_s.matrix(dof=5)
        with self.assertRaises(ValueError):
            flow_s.matrix(dof=4, method='test')
        with self.assertRaises(TypeError):
            flow_s.matrix(dof=4, method='lms', masked='test')

    def test_combine_with(self):
        img = cv2.imread('smudge.png')
        shape = img.shape[:2]
        transforms = [
            ['rotation', 255.5, 255.5, -30],
            ['scaling', 100, 100, 0.8],
        ]
        for ref in ['s', 't']:
            f1 = Flow.from_transforms(transforms[0:1], shape, ref)
            f2 = Flow.from_transforms(transforms[1:2], shape, ref)
            f3 = Flow.from_transforms(transforms, shape, ref)

            # Mode 1
            f1_actual = f2.combine_with(f3, 1)
            # f1.show(500, show_mask=True, show_mask_borders=True)
            # f1_actual.show(show_mask=True, show_mask_borders=True)
            self.assertIsInstance(f1_actual, Flow)
            self.assertEqual(f1_actual.ref, ref)
            comb_mask = f1_actual.mask_numpy & f1.mask_numpy
            self.assertIsNone(np.testing.assert_allclose(f1_actual.vecs_numpy[comb_mask], f1.vecs_numpy[comb_mask],
                                                         atol=5e-2))

            # Mode 2
            f2_actual = f1.combine_with(f3, 2)
            # f2.show(500, show_mask=True, show_mask_borders=True)
            # f2_actual.show(show_mask=True, show_mask_borders=True)
            self.assertIsInstance(f2_actual, Flow)
            self.assertEqual(f2_actual.ref, ref)
            comb_mask = f2_actual.mask_numpy & f2.mask_numpy
            self.assertIsNone(np.testing.assert_allclose(f2_actual.vecs_numpy[comb_mask], f2.vecs_numpy[comb_mask],
                                                         atol=5e-2))

            # Mode 3
            f3_actual = f1.combine_with(f2, 3)
            # f3.show(500, show_mask=True, show_mask_borders=True)
            # f3_actual.show(show_mask=True, show_mask_borders=True)
            self.assertIsInstance(f3_actual, Flow)
            self.assertEqual(f3_actual.ref, ref)
            comb_mask = f3_actual.mask_numpy & f3.mask_numpy
            self.assertIsNone(np.testing.assert_allclose(f3_actual.vecs_numpy[comb_mask], f3.vecs_numpy[comb_mask],
                                                         atol=5e-2))

        # Invalid inputs
        fs = Flow.from_transforms(transforms[0:1], [20, 20], 's')
        ft = Flow.from_transforms(transforms[1:2], [20, 20], 't')
        fs2 = Flow.from_transforms(transforms[0:1], [20, 30], 's')
        with self.assertRaises(TypeError):  # Flow not a Flow object
            fs.combine_with(fs.vecs, 1)
        with self.assertRaises(ValueError):  # Flow not the same shape
            fs.combine_with(fs2, 1)
        with self.assertRaises(ValueError):  # Flow not the same reference
            fs.combine_with(ft, 1)
        with self.assertRaises(ValueError):  # Mode not 1, 2 or 3
            fs.combine_with(fs, mode=0)
        with self.assertRaises(TypeError):  # Thresholded not boolean
            fs.combine_with(fs, 1, thresholded='test')


if __name__ == '__main__':
    unittest.main()
