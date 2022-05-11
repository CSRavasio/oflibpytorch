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

import unittest
import torch
import cv2
import numpy as np
import math
import sys
sys.path.append('..')
from src.oflibpytorch.flow_class import Flow
from src.oflibpytorch.flow_operations import batch_flows
from src.oflibpytorch.utils import to_numpy, apply_flow, matrix_from_transforms, resize_flow, \
    get_pure_pytorch, set_pure_pytorch, unset_pure_pytorch


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
        pt_2hw = torch.zeros((2, 100, 200), requires_grad=True)
        pt_hw2 = torch.zeros((100, 200, 2), requires_grad=True)
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
                        self.assertEqual(flow.shape, (1, 100, 200))
                        self.assertEqual(flow.ref, ref_expected)
                        self.assertIsNone(np.testing.assert_equal(to_numpy(flow.mask), mask_1np))
                        self.assertEqual(flow.device.type, device_expected)
                        self.assertEqual(flow.vecs.device.type, device_expected)
                        self.assertEqual(flow.mask.device.type, device_expected)
                        if isinstance(vecs, torch.Tensor):
                            self.assertIsNotNone(flow.vecs.grad_fn)

        # 4-dim vec inputs
        np_n2hw = np.zeros((5, 2, 100, 200))
        np_nhw2 = np.zeros((5, 100, 200, 2))
        pt_n2hw = torch.zeros((5, 2, 100, 200), requires_grad=True)
        pt_nhw2 = torch.zeros((5, 100, 200, 2), requires_grad=True)
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
                        self.assertEqual(flow.shape, (5, 100, 200))
                        self.assertEqual(flow.ref, ref_expected)
                        self.assertIsNone(np.testing.assert_equal(to_numpy(flow.mask), mask_np_nhw))
                        self.assertEqual(flow.device.type, device_expected)
                        self.assertEqual(flow.vecs.device.type, device_expected)
                        self.assertEqual(flow.mask.device.type, device_expected)
                        if isinstance(vecs, torch.Tensor):
                            self.assertTrue(flow.vecs.requires_grad)

        # tensor to cuda, test cuda
        if torch.cuda.is_available():
            expected_device_list = ['cpu', 'cuda', 'cuda']
        else:
            expected_device_list = ['cpu', 'cpu', 'cpu']
        vecs_pt_cuda = torch.zeros((2, 100, 200), requires_grad=True).to('cuda')
        for ref, ref_expected in zip(['t', 's', None], ['t', 's', 't']):
            for mask in [mask_empty, mask_np, mask_pt]:
                for device, device_expected in zip(['cpu', 'cuda', None], expected_device_list):
                    flow = Flow(vecs_pt_cuda, ref=ref, mask=mask, device=device)
                    self.assertIsNotNone(flow.vecs.grad_fn)
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
        matrix_pt = torch.tensor(matrix_np, requires_grad=True)
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
                    if isinstance(matrix, torch.Tensor):
                        self.assertIsNotNone(flow.vecs.grad_fn)
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
        v = torch.tensor(vectors, requires_grad=True)
        flow = Flow(v)
        self.assertIsNotNone(flow.vecs.grad_fn)
        flow_copy = flow.copy()
        self.assertIsNotNone(flow_copy.vecs.grad_fn)

    def test_to_device(self):
        if torch.cuda.is_available():
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
            v = torch.tensor(vectors, requires_grad=True)
            flow = Flow(v, device='cpu')
            self.assertIsNotNone(flow.vecs.grad_fn)
            flow_copy = flow.to_device('cuda')
            self.assertIsNotNone(flow_copy.vecs.grad_fn)

    def test_str(self):
        flow = Flow.zero(shape=(100, 200), ref='s', device='cuda')
        self.assertEqual(str(flow)[:68],
                         "Flow object, reference s, batch size 1, shape 100*200, device cuda:0")

    def test_select(self):
        matrix = torch.tensor([[math.sqrt(3) / 2, -.5, 26.3397459622],
                               [.5, math.sqrt(3) / 2, 1.69872981078],
                               [0, 0, 1]])
        matrix = torch.stack((torch.eye(3), matrix), dim=0).requires_grad_()
        shape = [200, 300]
        flow = Flow.from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_equal(flow.select(0).vecs_numpy, 0))
        self.assertIsNone(np.testing.assert_allclose(flow.select(1).vecs_numpy[0, 50, 10], [0, 0], atol=1e-4))
        self.assertIsNone(np.testing.assert_allclose(flow.select(-1).vecs_numpy[0, 50, 10], [0, 0], atol=1e-4))
        self.assertIsNotNone(flow.select(0).vecs.grad_fn)
        self.assertIsNotNone(flow.select(1).vecs.grad_fn)

        with self.assertRaises(TypeError):  # item not an integer
            flow.select(0.5)
        with self.assertRaises(IndexError):  # item out of bounds
            flow.select(2)

    def test_getitem(self):
        vectors = np.random.rand(200, 200, 2)
        flow = Flow(torch.tensor(vectors, requires_grad=True))
        indices = np.random.randint(0, 150, size=(20, 2))
        for i in indices:
            # Cutting a number of elements
            sel = flow[i]
            self.assertIsNone(np.testing.assert_allclose(sel.vecs_numpy[0], vectors[i]))
            self.assertIsNotNone(sel.vecs.grad_fn)
            # Cutting a specific item
            sel = flow[i[0]:i[0] + 1, i[1]:i[1] + 1]
            self.assertIsNone(np.testing.assert_allclose(sel.vecs_numpy[0], vectors[i[0]:i[0] + 1, i[1]:i[1] + 1]))
            self.assertIsNotNone(sel.vecs.grad_fn)
            # Cutting an area
            sel = flow[i[0]:i[0] + 40, i[1]:i[1] + 40]
            self.assertIsNone(np.testing.assert_allclose(sel.vecs_numpy[0], vectors[i[0]:i[0] + 40, i[1]:i[1] + 40]))
            self.assertIsNotNone(sel.vecs.grad_fn)
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
        flow1 = Flow(torch.tensor(vecs1, requires_grad=True), mask=mask1)
        flow2 = Flow(vecs2, mask=mask2)
        flow3 = Flow(vecs3)
        flow4 = Flow(np.random.rand(3, 200, 200, 2))
        flow5 = Flow(np.random.rand(5, 200, 200, 2))
        vecs6 = np.random.rand(5, 100, 200, 2)
        flow6 = Flow(vecs6)

        # Addition
        for vecs in vec_list:
            s = flow1 + vecs
            self.assertIsNotNone(s.vecs.grad_fn)
            if isinstance(vecs, torch.Tensor):
                v = to_numpy(vecs)
            else:
                v = vecs
            if v.shape[0] == 2:
                v = np.moveaxis(v, 0, -1)
            self.assertIsNone(np.testing.assert_allclose(s.vecs_numpy[0], vecs1 + v, rtol=1e-6, atol=1e-6))
            self.assertEqual(s.device, flow1.vecs.device)
            self.assertEqual(s.device, flow1.mask.device)
        s = flow1 + flow2
        self.assertIsNotNone(s.vecs.grad_fn)
        self.assertIsNone(np.testing.assert_allclose(s.vecs_numpy[0], vecs1 + vecs2, rtol=1e-6, atol=1e-6))
        self.assertIsNone(np.testing.assert_equal(np.sum(to_numpy(s.mask)), (60 - 40) * 200))
        s = flow1 + flow6
        self.assertIsNotNone(s.vecs.grad_fn)
        self.assertIsNone(np.testing.assert_allclose(s.vecs_numpy, vecs1 + vecs6, rtol=1e-6, atol=1e-6))

        flow1 = Flow(vecs1, mask=mask1)
        flow2 = Flow(torch.tensor(vecs2, requires_grad=True), mask=mask2)
        flow6 = Flow(torch.tensor(vecs6, requires_grad=True))
        for vecs in vec_list:
            if isinstance(vecs, torch.Tensor):
                v = to_numpy(vecs)
                vecs.requires_grad_()
            else:
                v = vecs
            s = flow1 + vecs
            if isinstance(vecs, torch.Tensor):
                self.assertIsNotNone(s.vecs.grad_fn)
            if v.shape[0] == 2:
                v = np.moveaxis(v, 0, -1)
            self.assertIsNone(np.testing.assert_allclose(s.vecs_numpy[0], vecs1 + v, rtol=1e-6, atol=1e-6))
            self.assertEqual(s.device, flow1.vecs.device)
            self.assertEqual(s.device, flow1.mask.device)
        s = flow1 + flow2
        self.assertIsNotNone(s.vecs.grad_fn)
        self.assertIsNone(np.testing.assert_allclose(s.vecs_numpy[0], vecs1 + vecs2, rtol=1e-6, atol=1e-6))
        self.assertIsNone(np.testing.assert_equal(np.sum(to_numpy(s.mask)), (60 - 40) * 200))
        s = flow1 + flow6
        self.assertIsNotNone(s.vecs.grad_fn)
        self.assertIsNone(np.testing.assert_allclose(s.vecs_numpy, vecs1 + vecs6, rtol=1e-6, atol=1e-6))

        # Errors
        with self.assertRaises(TypeError):
            flow1 + 'test'
        with self.assertRaises(ValueError):
            flow1 + flow3
        with self.assertRaises(ValueError):
            flow1 + vecs3
        with self.assertRaises(ValueError):
            flow4 + flow5

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
        flow1 = Flow(torch.tensor(vecs1, requires_grad=True), mask=mask1)
        flow2 = Flow(vecs2, mask=mask2)
        flow3 = Flow(vecs3)
        flow4 = Flow(np.random.rand(3, 200, 200, 2))
        flow5 = Flow(np.random.rand(5, 200, 200, 2))
        vecs6 = np.random.rand(5, 100, 200, 2)
        flow6 = Flow(vecs6)

        # Addition
        for vecs in vec_list:
            s = flow1 - vecs
            self.assertIsNotNone(s.vecs.grad_fn)
            if isinstance(vecs, torch.Tensor):
                v = to_numpy(vecs)
            else:
                v = vecs
            if v.shape[0] == 2:
                v = np.moveaxis(v, 0, -1)
            self.assertIsNone(np.testing.assert_allclose(s.vecs_numpy[0], vecs1 - v, rtol=1e-6, atol=1e-6))
            self.assertEqual(s.device, flow1.vecs.device)
            self.assertEqual(s.device, flow1.mask.device)
        s = flow1 - flow2
        self.assertIsNotNone(s.vecs.grad_fn)
        self.assertIsNone(np.testing.assert_allclose(s.vecs_numpy[0], vecs1 - vecs2, rtol=1e-6, atol=1e-6))
        self.assertIsNone(np.testing.assert_equal(np.sum(to_numpy(s.mask)), (60 - 40) * 200))
        s = flow1 - flow6
        self.assertIsNotNone(s.vecs.grad_fn)
        self.assertIsNone(np.testing.assert_allclose(s.vecs_numpy, vecs1 - vecs6, rtol=1e-6, atol=1e-6))

        flow1 = Flow(vecs1, mask=mask1)
        flow2 = Flow(torch.tensor(vecs2, requires_grad=True), mask=mask2)
        flow6 = Flow(torch.tensor(vecs6, requires_grad=True))
        for vecs in vec_list:
            if isinstance(vecs, torch.Tensor):
                v = to_numpy(vecs)
                vecs.requires_grad_()
            else:
                v = vecs
            s = flow1 - vecs
            if isinstance(vecs, torch.Tensor):
                self.assertIsNotNone(s.vecs.grad_fn)
            if v.shape[0] == 2:
                v = np.moveaxis(v, 0, -1)
            self.assertIsNone(np.testing.assert_allclose(s.vecs_numpy[0], vecs1 - v, rtol=1e-6, atol=1e-6))
            self.assertEqual(s.device, flow1.vecs.device)
            self.assertEqual(s.device, flow1.mask.device)
        s = flow1 - flow2
        self.assertIsNotNone(s.vecs.grad_fn)
        self.assertIsNone(np.testing.assert_allclose(s.vecs_numpy[0], vecs1 - vecs2, rtol=1e-6, atol=1e-6))
        self.assertIsNone(np.testing.assert_equal(np.sum(to_numpy(s.mask)), (60 - 40) * 200))
        s = flow1 - flow6
        self.assertIsNotNone(s.vecs.grad_fn)
        self.assertIsNone(np.testing.assert_allclose(s.vecs_numpy, vecs1 - vecs6, rtol=1e-6, atol=1e-6))

        # Errors
        with self.assertRaises(TypeError):
            flow1 - 'test'
        with self.assertRaises(ValueError):
            flow1 - flow3
        with self.assertRaises(ValueError):
            flow1 - vecs3
        with self.assertRaises(ValueError):
            flow4 - flow5

    def test_mul(self):
        vecs1 = np.random.rand(100, 200, 2)
        vecs2 = np.random.rand(100, 200, 2)
        flow1 = Flow(vecs1)
        flow1_g = Flow(torch.tensor(vecs1, requires_grad=True))

        # Multiplication
        ints = np.random.randint(-10, 10, 100)
        floats = (np.random.rand(100) - .5) * 20
        # ... using ints and floats
        for i, f in zip(ints, floats):
            m = flow1_g * i
            self.assertIsNotNone(m.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(m.vecs_numpy[0], vecs1 * i, rtol=1e-6, atol=1e-6))
            m = flow1_g * f
            self.assertIsNotNone(m.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(m.vecs_numpy[0], vecs1 * f, rtol=1e-6, atol=1e-6))

        # ... using a list of length 2
        int_list = np.random.randint(-10, 10, (100, 2))
        for li in int_list:
            v = vecs1.astype('f')
            v[..., 0] *= li[0]
            v[..., 1] *= li[1]
            m = flow1_g * list(li)
            self.assertIsNotNone(m.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(m.vecs_numpy[0], v, rtol=1e-6, atol=1e-6))

        # ... using a numpy array of size 2
        int_list = np.random.randint(-10, 10, (100, 2))
        for li in int_list:
            v = vecs1.astype('f')
            v[..., 0] *= li[0]
            v[..., 1] *= li[1]
            m = flow1_g * li
            self.assertIsNotNone(m.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(m.vecs_numpy[0], v, rtol=1e-6, atol=1e-6))

        # ... using a numpy array and torch tensor of the same shape as the flow
        vecs2_pt_hw2 = torch.rand(100, 200, 2)
        vecs_list = [vecs2, vecs2_pt_hw2]
        if torch.cuda.is_available():
            vecs_list.append(torch.rand(100, 200, 2).to('cuda'))
        for vecs in vecs_list:
            if isinstance(vecs, torch.Tensor):
                vecs.requires_grad_()
                v = to_numpy(vecs)
            else:
                v = vecs
            m = flow1 * vecs[..., 0]
            if isinstance(vecs, torch.Tensor):
                self.assertIsNotNone(m.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(m.vecs_numpy[0], vecs1 * v[..., :1], rtol=1e-6, atol=1e-6))
            self.assertEqual(m.device, flow1.vecs.device)
            self.assertEqual(m.device, flow1.mask.device)

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
                vecs.requires_grad_()
                v = to_numpy(vecs)
            else:
                v = vecs
            if v.shape[0] == 2:
                v = np.moveaxis(v, 0, -1)
            m = flow1 * vecs
            if isinstance(vecs, torch.Tensor):
                self.assertIsNotNone(m.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(m.vecs_numpy[0], vecs1 * v, rtol=1e-6, atol=1e-6))
            self.assertEqual(m.device, flow1.vecs.device)
            self.assertEqual(m.device, flow1.mask.device)

        # ... using torch tensors of the same, and different, batch dimension
        v1 = np.random.rand(1, 100, 200, 2)
        v2 = np.random.rand(3, 100, 200, 2)
        v3 = np.random.rand(5, 100, 200, 2)
        f2 = Flow(torch.tensor(v2, requires_grad=True))
        m = f2 * np.moveaxis(v1, -1, 1)
        self.assertIsNotNone(m.vecs.grad_fn)
        self.assertIsNone(np.testing.assert_allclose(m.vecs_numpy, v1 * v2, rtol=1e-6, atol=1e-6))
        m = flow1_g * np.moveaxis(v3, -1, 1)
        self.assertIsNotNone(m.vecs.grad_fn)
        self.assertIsNone(np.testing.assert_allclose(m.vecs_numpy, vecs1 * v3, rtol=1e-6, atol=1e-6))

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
        with self.assertRaises(ValueError):
            f2 * np.random.rand(6, 2, 100, 200)

    def test_div(self):
        vecs1 = np.random.rand(100, 200, 2) + .5
        vecs2 = -np.random.rand(100, 200, 2) - .5
        flow1 = Flow(vecs1)
        flow1_g = Flow(torch.tensor(vecs1, requires_grad=True))

        # Division
        ints = np.random.randint(-10, 10, 100)
        floats = (np.random.rand(100) - .5) * 20
        # ... using ints and floats
        for i, f in zip(ints, floats):
            if i < -1e-5 or i > 1e-5:
                d = flow1_g / i
                self.assertIsNotNone(d.vecs.grad_fn)
                self.assertIsNone(np.testing.assert_allclose(d.vecs_numpy[0], vecs1 / i, rtol=1e-6, atol=1e-6))
            if f < -1e-5 or f > 1e-5:
                d = flow1_g / f
                self.assertIsNotNone(d.vecs.grad_fn)
                self.assertIsNone(np.testing.assert_allclose(d.vecs_numpy[0], vecs1 / f, rtol=1e-6, atol=1e-6))

        # ... using a list of length 2
        int_list = np.random.randint(-10, 10, (100, 2))
        for li in int_list:
            if li[0] != 0 and li[1] != 0:
                v = vecs1.astype('f')
                v[..., 0] /= li[0]
                v[..., 1] /= li[1]
                d = flow1_g / list(li)
                self.assertIsNotNone(d.vecs.grad_fn)
                self.assertIsNone(np.testing.assert_allclose(d.vecs_numpy[0], v, rtol=1e-6, atol=1e-6))

        # ... using a numpy array of size 2
        int_list = np.random.randint(-10, 10, (100, 2))
        for li in int_list:
            if li[0] != 0 and li[1] != 0:
                v = vecs1.astype('f')
                v[..., 0] /= li[0]
                v[..., 1] /= li[1]
                d = flow1_g / li
                self.assertIsNotNone(d.vecs.grad_fn)
                self.assertIsNone(np.testing.assert_allclose(d.vecs_numpy[0], v, rtol=1e-6, atol=1e-6))

        # ... using a numpy array and torch tensor of the same shape as the flow
        vecs2_pt_hw2 = torch.rand(100, 200, 2) + .5
        vecs_list = [vecs2, vecs2_pt_hw2]
        if torch.cuda.is_available():
            vecs_list.append(torch.rand(100, 200, 2).to('cuda') + .5)
        for vecs in vecs_list:
            if isinstance(vecs, torch.Tensor):
                vecs.requires_grad_()
                v = to_numpy(vecs)
            else:
                v = vecs
            d = flow1 / vecs[..., 0]
            if isinstance(vecs, torch.Tensor):
                self.assertIsNotNone(d.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(d.vecs_numpy[0], vecs1 / v[..., :1], rtol=1e-6, atol=1e-6))
            self.assertEqual(d.device, flow1.vecs.device)
            self.assertEqual(d.device, flow1.mask.device)

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
                vecs.requires_grad_()
                v = to_numpy(vecs)
            else:
                v = vecs
            if v.shape[0] == 2:
                v = np.moveaxis(v, 0, -1)
            d = flow1 / vecs
            if isinstance(vecs, torch.Tensor):
                self.assertIsNotNone(d.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(d.vecs_numpy[0], vecs1 / v, rtol=1e-6, atol=1e-6))
            self.assertEqual(d.device, flow1.vecs.device)
            self.assertEqual(d.device, flow1.mask.device)

        # ... using torch tensors of the same, and different, batch dimension
        v1 = np.random.rand(1, 100, 200, 2)
        v2 = np.random.rand(3, 100, 200, 2)
        v3 = np.random.rand(5, 100, 200, 2)
        f2 = Flow(torch.tensor(v2, requires_grad=True))
        d = f2 / np.moveaxis(v1, -1, 1)
        self.assertIsNotNone(d.vecs.grad_fn)
        self.assertIsNone(np.testing.assert_allclose(d.vecs_numpy, v2 / v1, rtol=1e-6, atol=1e-6))
        d = flow1_g / np.moveaxis(v3, -1, 1)
        self.assertIsNotNone(d.vecs.grad_fn)
        self.assertIsNone(np.testing.assert_allclose(d.vecs_numpy, vecs1 / v3, rtol=1e-6, atol=1e-6))

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
        with self.assertRaises(ValueError):
            f2 * np.random.rand(6, 2, 100, 200)

    def test_pow(self):
        vecs1 = np.random.rand(100, 200, 2)
        vecs2 = np.random.rand(100, 200, 2)
        flow1 = Flow(vecs1)
        flow1_g = Flow(torch.tensor(vecs1, requires_grad=True))

        # Exponentiation
        ints = np.random.randint(-2, 2, 100)
        floats = (np.random.rand(100) - .5) * 4
        # ... using ints and floats
        for i, f in zip(ints, floats):
            p = flow1_g ** i
            self.assertIsNotNone(p.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(p.vecs_numpy[0], vecs1 ** i, rtol=1e-6, atol=1e-6))
            p = flow1_g ** f
            self.assertIsNotNone(p.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(p.vecs_numpy[0], vecs1 ** f, rtol=1e-6, atol=1e-6))

        # ... using a list of length 2
        int_list = np.random.randint(-5, 5, (100, 2))
        for li in int_list:
            v = vecs1.astype('f')
            v[..., 0] **= li[0]
            v[..., 1] **= li[1]
            p = flow1_g ** list(li)
            self.assertIsNotNone(p.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(p.vecs_numpy[0], v, rtol=1e-6, atol=1e-6))

        # ... using a numpy array of size 2
        int_list = np.random.randint(-5, 5, (100, 2))
        for li in int_list:
            v = vecs1.astype('f')
            v[..., 0] **= li[0]
            v[..., 1] **= li[1]
            p = flow1_g ** li
            self.assertIsNotNone(p.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(p.vecs_numpy[0], v, rtol=1e-6, atol=1e-6))

        # ... using a numpy array and torch tensor of the same shape as the flow
        vecs2_pt_hw2 = torch.rand(100, 200, 2)
        vecs_list = [vecs2, vecs2_pt_hw2]
        if torch.cuda.is_available():
            vecs_list.append(torch.rand(100, 200, 2).to('cuda'))
        for vecs in vecs_list:
            if isinstance(vecs, torch.Tensor):
                vecs.requires_grad_()
                v = to_numpy(vecs)
            else:
                v = vecs
            p = flow1 ** vecs[..., 0]
            if isinstance(vecs, torch.Tensor):
                self.assertIsNotNone(p.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(p.vecs_numpy[0], vecs1 ** v[..., :1], rtol=1e-6, atol=1e-6))
            self.assertEqual(p.device, flow1.vecs.device)
            self.assertEqual(p.device, flow1.mask.device)

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
                vecs.requires_grad_()
                v = to_numpy(vecs)
            else:
                v = vecs
            if v.shape[0] == 2:
                v = np.moveaxis(v, 0, -1)
            p = flow1 ** vecs
            if isinstance(vecs, torch.Tensor):
                self.assertIsNotNone(p.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(p.vecs_numpy[0], vecs1 ** v, rtol=1e-6, atol=1e-6))
            self.assertEqual(p.device, flow1.vecs.device)
            self.assertEqual(p.device, flow1.mask.device)

        # ... using torch tensors of the same, and different, batch dimension
        v1 = np.random.rand(1, 100, 200, 2)
        v2 = np.random.rand(3, 100, 200, 2)
        v3 = np.random.rand(5, 100, 200, 2)
        f2 = Flow(torch.tensor(v2, requires_grad=True))
        p = f2 ** np.moveaxis(v1, -1, 1)
        self.assertIsNotNone(p.vecs.grad_fn)
        self.assertIsNone(np.testing.assert_allclose(p.vecs_numpy, v2 ** v1, rtol=1e-6, atol=1e-6))
        p = flow1_g ** np.moveaxis(v3, -1, 1)
        self.assertIsNotNone(p.vecs.grad_fn)
        self.assertIsNone(np.testing.assert_allclose(p.vecs_numpy, vecs1 ** v3, rtol=1e-6, atol=1e-6))

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
        with self.assertRaises(ValueError):
            f2 * np.random.rand(6, 2, 100, 200)

    def test_neg(self):
        vecs1 = torch.rand((100, 200, 2), requires_grad=True)
        flow1 = Flow(vecs1)
        self.assertIsNotNone(flow1.vecs.grad_fn)
        self.assertIsNone(np.testing.assert_allclose((-flow1).vecs_numpy[0], -to_numpy(vecs1)))

    def test_resize(self):
        shape = [20, 10]
        ref = 's'
        flow = Flow.from_transforms([['rotation', 30, 50, 30]], shape, ref)
        flow.vecs.requires_grad_()

        # Different scales
        scales = [.2, .5, 1, 1.5, 2, 10]
        for scale in scales:
            r = flow.resize(scale)
            self.assertIsNotNone(r.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_equal(r.vecs_numpy,
                                                      to_numpy(resize_flow(flow.vecs, scale), switch_channels=True)))
        # Scale mask
        shape_small = (20, 40)
        shape_large = (30, 80)
        mask_small = np.ones(shape_small, 'bool')
        mask_small[:6, :20] = 0
        mask_large = np.ones(shape_large, 'bool')
        mask_large[:9, :40] = 0
        flow_small = Flow.from_transforms([['rotation', 0, 0, 30]], shape_small, 't', mask_small)
        flow_small.vecs.requires_grad_()
        flow_large = flow_small.resize((1.5, 2))
        self.assertIsNotNone(flow_large.vecs.grad_fn)
        self.assertIsNone(np.testing.assert_equal(to_numpy(flow_large.mask)[0], mask_large))

        # Check scaling is performed correctly based on the actual flow field
        ref = 't'
        flow_small = Flow.from_transforms([['rotation', 0, 0, 30]], (50, 80), ref)
        flow_large = Flow.from_transforms([['rotation', 0, 0, 30]], (150, 240), ref)
        flow_large.vecs.requires_grad_()
        flow_resized = flow_large.resize(1/3)
        self.assertIsNotNone(flow_resized.vecs.grad_fn)
        self.assertIsNone(np.testing.assert_allclose(flow_resized.vecs_numpy, flow_small.vecs_numpy, atol=1, rtol=.1))

    def test_pad(self):
        shape = [100, 80]
        for ref in ['t', 's']:
            flow = Flow.zero(shape, ref, np.ones(shape, 'bool'))
            flow.vecs.requires_grad_()
            flow = flow.pad([10, 20, 30, 40])
            self.assertIsNotNone(flow.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_equal(flow.shape[1:], [shape[0] + 10 + 20, shape[1] + 30 + 40]))
            self.assertIsNone(np.testing.assert_equal(flow.vecs_numpy, 0))
            self.assertIsNone(np.testing.assert_equal(to_numpy(flow[10:-20, 30:-40].mask), 1))
            flow.mask[..., 10:-20, 30:-40] = 0
            self.assertIsNone(np.testing.assert_equal(to_numpy(flow.mask), 0))
            self.assertIs(flow.ref, ref)

        # 'Replicate' padding
        flow = Flow.from_transforms([['rotation', 30, 50, 30]], shape, ref)
        flow.vecs.requires_grad_()
        padded_flow = flow.pad([10, 10, 20, 20], mode='replicate')
        self.assertIsNotNone(padded_flow.vecs.grad_fn)
        self.assertIsNone(np.testing.assert_equal(padded_flow.vecs_numpy[:, 0, 20:-20], flow.vecs_numpy[:, 0]))
        self.assertIsNone(np.testing.assert_equal(padded_flow.vecs_numpy[:, 10:-10, 0], flow.vecs_numpy[:, :, 0]))

        # 'Reflect' padding
        padded_flow = flow.pad([10, 10, 20, 20], mode='reflect')
        self.assertIsNotNone(padded_flow.vecs.grad_fn)
        self.assertIsNone(np.testing.assert_equal(padded_flow.vecs_numpy[:, 0, 20:-20], flow.vecs_numpy[:, 10]))
        self.assertIsNone(np.testing.assert_equal(padded_flow.vecs_numpy[:, 10:-10, 0], flow.vecs_numpy[:, :, 20]))

        # Invalid padding mode
        with self.assertRaises(ValueError):
            flow.pad([10, 10, 20, 20], mode='test')

    def test_apply(self):
        img_np = np.moveaxis(cv2.resize(cv2.imread('smudge.png'), None, fx=.25, fy=.25), -1, 0)
        img_pt = torch.tensor(img_np)
        # Check flow.apply results in the same as using apply_flow directly
        for f in [set_pure_pytorch, unset_pure_pytorch]:
            f()  # set PURE_PYTORCH to True or False
            for ref in ['t', 's']:
                for consider_mask in [True, False]:
                    for device in ['cpu', 'cuda']:
                        for img in [img_pt.to('cpu'), img_pt.to('cuda')]:
                            img = img.float().requires_grad_()
                            mask = torch.ones(img_pt.shape[1:], dtype=torch.bool)
                            mask[20:] = False
                            flow = Flow.from_transforms([['rotation', 30, 50, 30]], img.shape[1:], ref, mask, device)
                            # Target is a 3D torch tensor
                            warped_img_desired = apply_flow(flow.vecs, img, ref, mask if consider_mask else None)
                            warped_img_actual = flow.apply(img, consider_mask=consider_mask)
                            if ref != 's' or get_pure_pytorch():
                                self.assertIsNotNone(warped_img_actual.grad_fn)
                            self.assertEqual(flow.device, warped_img_actual.device)
                            self.assertIsNone(np.testing.assert_equal(to_numpy(warped_img_actual),
                                                                      to_numpy(warped_img_desired)))
                            warped_img_actual, _ = flow.apply(img, mask, True, consider_mask=consider_mask)
                            if ref != 's' or get_pure_pytorch():
                                self.assertIsNotNone(warped_img_actual.grad_fn)
                            self.assertIsNone(np.testing.assert_equal(to_numpy(warped_img_actual),
                                                                      to_numpy(warped_img_desired)))
                            # Target is a 2D torch tensor
                            warped_img_desired = apply_flow(flow.vecs, img[0], ref, mask if consider_mask else None)
                            warped_img_actual = flow.apply(img[0], consider_mask=consider_mask)
                            if ref != 's' or get_pure_pytorch():
                                self.assertIsNotNone(warped_img_actual.grad_fn)
                            self.assertEqual(flow.device, warped_img_actual.device)
                            self.assertIsNone(np.testing.assert_equal(to_numpy(warped_img_actual),
                                                                      to_numpy(warped_img_desired)))
                            warped_img_actual, _ = flow.apply(img[0], mask, True, consider_mask=consider_mask)
                            if ref != 's' or get_pure_pytorch():
                                self.assertIsNotNone(warped_img_actual.grad_fn)
                            self.assertIsNone(np.testing.assert_equal(to_numpy(warped_img_actual),
                                                                      to_numpy(warped_img_desired)))
                        for f_device in ['cpu', 'cuda']:
                            f = flow.to_device(f_device)
                            f.vecs.requires_grad_()
                            # Target is a flow object
                            warped_flow_desired = apply_flow(flow.vecs, f.vecs, ref, mask if consider_mask else None)
                            warped_flow_actual = flow.apply(f, consider_mask=consider_mask)
                            if ref != 's' or get_pure_pytorch():
                                self.assertIsNotNone(warped_flow_actual.vecs.grad_fn)
                            self.assertEqual(flow.device, warped_flow_actual.device)
                            self.assertIsNone(np.testing.assert_equal(to_numpy(warped_flow_actual.vecs),
                                                                      to_numpy(warped_flow_desired)))
            # Check using a smaller flow field on a larger target works the same as a full flow field on the same target
            img = img_pt.to(torch.float).requires_grad_()
            ref = 't'
            flow = Flow.from_transforms([['rotation', 30, 50, 30]], img.shape[1:], ref)
            warped_img_desired = apply_flow(flow.vecs, img, ref)
            shape = [img.shape[1] - 90, img.shape[2] - 110]
            padding = [50, 40, 30, 80]
            cut_flow = Flow.from_transforms([['rotation', 0, 0, 30]], shape, ref)
            # ... not cutting (target torch tensor)
            warped_img_actual = cut_flow.apply(img, padding=padding, cut=False)
            if ref != 's' or get_pure_pytorch():
                self.assertIsNotNone(warped_img_actual.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(to_numpy(warped_img_actual[:, padding[0]:-padding[1],
                                                                  padding[2]:-padding[3]]),
                                                         to_numpy(warped_img_desired[:, padding[0]:-padding[1],
                                                                  padding[2]:-padding[3]]),
                                                         atol=1e-3))
            # ... cutting (target torch tensor)
            warped_img_actual = cut_flow.apply(img, padding=padding, cut=True)
            if ref != 's' or get_pure_pytorch():
                self.assertIsNotNone(warped_img_actual.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(to_numpy(warped_img_actual).astype('f'),
                                                         to_numpy(warped_img_desired[:, padding[0]:-padding[1],
                                                                  padding[2]:-padding[3]]).astype('f'),
                                                         atol=1e-3))
            # ... not cutting (target flow object)
            target_flow = Flow.from_transforms([['rotation', 30, 50, 30]], img.shape[1:], ref)
            warped_flow_desired = apply_flow(flow.vecs, target_flow.vecs, ref)
            warped_flow_actual = cut_flow.apply(target_flow, padding=padding, cut=False)
            if ref != 's' or get_pure_pytorch():
                self.assertIsNotNone(warped_img_actual.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(to_numpy(warped_flow_actual.vecs[:, :, padding[0]:-padding[1],
                                                                  padding[2]:-padding[3]]),
                                                         to_numpy(warped_flow_desired[:, :, padding[0]:-padding[1],
                                                                  padding[2]:-padding[3]]),
                                                         atol=1e-1))
            # ... cutting (target flow object)
            warped_flow_actual = cut_flow.apply(target_flow, padding=padding, cut=True)
            if ref != 's' or get_pure_pytorch():
                self.assertIsNotNone(warped_img_actual.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(to_numpy(warped_flow_actual.vecs),
                                                         to_numpy(warped_flow_desired[:, :, padding[0]:-padding[1],
                                                                  padding[2]:-padding[3]]),
                                                         atol=1e-1))
            # All combinations of differing batch sizes
            i_shape = img_pt.shape[-2:]
            i_chw = img_pt
            i_hw = img_pt[0]
            i_1chw = i_chw.unsqueeze(0)
            i_11hw = i_1chw[:, 0:1]
            i_nchw = i_1chw.repeat(4, 1, 1, 1)
            i_n1hw = i_11hw.repeat(4, 1, 1, 1)
            for ref in ['s', 't']:
                flows = [
                    Flow.from_transforms([['translation', 10, -20]], i_shape, ref),
                    batch_flows(tuple(Flow.from_transforms([['translation', 10, -20]], i_shape, ref)
                                      for _ in range(4))),
                ]
                for f1 in flows:
                    f1 = f1.copy()
                    f1.vecs.requires_grad_()
                    for i in [i_1chw, i_11hw, i_nchw, i_n1hw]:
                        for consider_mask in [True, False]:
                            warped_i = f1.apply(i, consider_mask=consider_mask)
                            if get_pure_pytorch():
                                self.assertIsNotNone(warped_i.grad_fn)
                            self.assertEqual(warped_i.shape[0], max(f1.shape[0], i.shape[0]))
                            for w_ind, i_ind in zip(warped_i, i):
                                self.assertIsNone(np.testing.assert_equal(to_numpy(w_ind[:-20, 10:]),
                                                                          to_numpy(i_ind[20:, :-10])))
                    for f2 in flows:
                        for consider_mask in [True, False]:
                            warped_f2 = f1.apply(f2, consider_mask=consider_mask)
                            if ref != 's' or get_pure_pytorch():
                                self.assertIsNotNone(warped_f2.vecs.grad_fn)
                            self.assertEqual(warped_f2.shape[0], max(f1.shape[0], f2.shape[0]))
                            v = warped_f2.vecs_numpy
                            for v_ind in v:
                                self.assertIsNone(np.testing.assert_equal(v_ind[:, :-20, 10:],
                                                                          f2.vecs_numpy[0, :, 20:, :-10]))

            f = Flow.from_transforms([['translation', 10, -20]], i_shape, 't').vecs.repeat(4, 1, 1, 1)
            warped_i = apply_flow(f, i_hw, 't')
            self.assertEqual(warped_i.shape, (4, i_hw.shape[0], i_hw.shape[1]))
            warped_i = apply_flow(f, i_chw, 't')
            self.assertEqual(warped_i.shape, (4, 3, i_hw.shape[0], i_hw.shape[1]))

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
        flow_s.vecs.requires_grad_()
        flow_t = Flow.from_transforms(transforms, shape, 't')
        flow_t.vecs.requires_grad_()
        for f in [set_pure_pytorch, unset_pure_pytorch]:
            f()
            if get_pure_pytorch():
                rtol, atol = 1e-3, 6e-1
            else:
                rtol, atol = 1e-3, 1e-3
            switched_s = flow_t.switch_ref()
            if get_pure_pytorch():
                self.assertIsNotNone(switched_s.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(switched_s.vecs_numpy[switched_s.mask_numpy],
                                                         flow_s.vecs_numpy[switched_s.mask_numpy],
                                                         rtol=rtol, atol=atol))
            switched_t = flow_s.switch_ref()
            if get_pure_pytorch():
                self.assertIsNotNone(switched_t.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(switched_t.vecs_numpy[switched_t.mask_numpy],
                                                         flow_t.vecs_numpy[switched_t.mask_numpy],
                                                         rtol=rtol, atol=atol))

            # Mode 'valid', batched flow
            transforms = [[['rotation', 256, 256, 30]], [['translation', 10, -20]]]
            flow_s = batch_flows([Flow.from_transforms(t, shape, 's') for t in transforms])
            flow_s.vecs.requires_grad_()
            flow_t = batch_flows([Flow.from_transforms(t, shape, 't') for t in transforms])
            flow_t.vecs.requires_grad_()
            switched_s = flow_t.switch_ref()
            if get_pure_pytorch():
                self.assertIsNotNone(switched_s.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(switched_s.vecs_numpy[switched_s.mask_numpy],
                                                         flow_s.vecs_numpy[switched_s.mask_numpy],
                                                         rtol=rtol, atol=atol))
            switched_t = flow_s.switch_ref()
            if get_pure_pytorch():
                self.assertIsNotNone(switched_t.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(switched_t.vecs_numpy[switched_t.mask_numpy],
                                                         flow_t.vecs_numpy[switched_t.mask_numpy],
                                                         rtol=rtol, atol=atol))

        # Invalid mode passed
        flow = Flow.from_transforms([['rotation', 30, 50, 30]], shape, 't')
        with self.assertRaises(ValueError):
            flow.switch_ref('test')
        with self.assertRaises(ValueError):
            flow.switch_ref(1)

    def test_invert(self):
        f_s = Flow.from_transforms([['rotation', 50, 40, 30]], (80, 100), 's')   # Forwards
        f_t = Flow.from_transforms([['rotation', 50, 40, 30]], (80, 100), 't')   # Forwards
        b_s = Flow.from_transforms([['rotation', 50, 40, -30]], (80, 100), 's')  # Backwards
        b_t = Flow.from_transforms([['rotation', 50, 40, -30]], (80, 100), 't')  # Backwards
        f_s.vecs.requires_grad_()
        f_t.vecs.requires_grad_()
        b_s.vecs.requires_grad_()
        b_t.vecs.requires_grad_()
        for f in [set_pure_pytorch, unset_pure_pytorch]:
            f()
            if get_pure_pytorch():
                rtol, atol = 1e-3, 6e-1
            else:
                rtol, atol = 1e-3, 1e-3

            # Inverting s to s
            b_s_inv = f_s.invert()
            if get_pure_pytorch():
                self.assertIsNotNone(b_s_inv.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(b_s_inv.vecs_numpy[b_s_inv.mask_numpy],
                                                         b_s.vecs_numpy[b_s_inv.mask_numpy],
                                                         rtol=rtol, atol=atol))
            f_s_inv = b_s.invert()
            if get_pure_pytorch():
                self.assertIsNotNone(f_s_inv.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(f_s_inv.vecs_numpy[f_s_inv.mask_numpy],
                                                         f_s.vecs_numpy[f_s_inv.mask_numpy],
                                                         rtol=rtol, atol=atol))

            # Inverting s to t
            b_t_inv = f_s.invert('t')
            self.assertIsNotNone(b_t_inv.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(b_t_inv.vecs_numpy[b_t_inv.mask_numpy],
                                                         b_t.vecs_numpy[b_t_inv.mask_numpy],
                                                         rtol=rtol, atol=atol))
            f_t_inv = b_s.invert('t')
            self.assertIsNotNone(f_t_inv.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(f_t_inv.vecs_numpy[f_t_inv.mask_numpy],
                                                         f_t.vecs_numpy[f_t_inv.mask_numpy],
                                                         rtol=rtol, atol=atol))

            # Inverting t to t
            b_t_inv = f_t.invert()
            if get_pure_pytorch():
                self.assertIsNotNone(b_t_inv.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(b_t_inv.vecs_numpy[b_t_inv.mask_numpy],
                                                         b_t.vecs_numpy[b_t_inv.mask_numpy],
                                                         rtol=rtol, atol=atol))
            f_t_inv = b_t.invert()
            if get_pure_pytorch():
                self.assertIsNotNone(f_t_inv.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(f_t_inv.vecs_numpy[f_t_inv.mask_numpy],
                                                         f_t.vecs_numpy[f_t_inv.mask_numpy],
                                                         rtol=rtol, atol=atol))

            # Inverting t to s
            b_s_inv = f_t.invert('s')
            self.assertIsNotNone(b_s_inv.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(b_s_inv.vecs_numpy[b_s_inv.mask_numpy],
                                                         b_s.vecs_numpy[b_s_inv.mask_numpy],
                                                         rtol=rtol, atol=atol))
            f_s_inv = b_t.invert('s')
            self.assertIsNotNone(f_s_inv.vecs.grad_fn)
            self.assertIsNone(np.testing.assert_allclose(f_s_inv.vecs_numpy[f_s_inv.mask_numpy],
                                                         f_s.vecs_numpy[f_s_inv.mask_numpy],
                                                         rtol=rtol, atol=atol))

            # All of the above batched
            bf_s = batch_flows((f_s, b_s))
            bf_t = batch_flows((f_t, b_t))
            bb_s = batch_flows((b_s, f_s))
            bb_t = batch_flows((b_t, f_t))
            bf_s.vecs.requires_grad_()
            bf_t.vecs.requires_grad_()
            bb_s.vecs.requires_grad_()
            bb_t.vecs.requires_grad_()
            f_s_inv = bf_s.invert()
            f_s_inv_t = bf_s.invert('t')
            f_t_inv = bf_t.invert()
            f_t_inv_s = bf_t.invert('s')
            if get_pure_pytorch():
                self.assertIsNotNone(f_s_inv.vecs.grad_fn)
                self.assertIsNotNone(f_t_inv.vecs.grad_fn)
            self.assertIsNotNone(f_s_inv_t.vecs.grad_fn)
            self.assertIsNotNone(f_t_inv_s.vecs.grad_fn)
            # Inverting s to s
            self.assertIsNone(np.testing.assert_allclose(f_s_inv.vecs_numpy[f_s_inv.mask_numpy],
                                                         bb_s.vecs_numpy[f_s_inv.mask_numpy],
                                                         rtol=rtol, atol=atol))
            # Inverting s to t
            self.assertIsNone(np.testing.assert_allclose(f_s_inv_t.vecs_numpy[f_s_inv_t.mask_numpy],
                                                         bb_t.vecs_numpy[f_s_inv_t.mask_numpy],
                                                         rtol=rtol, atol=atol))
            # Inverting t to t
            self.assertIsNone(np.testing.assert_allclose(f_t_inv.vecs_numpy[f_t_inv.mask_numpy],
                                                         bb_t.vecs_numpy[f_t_inv.mask_numpy],
                                                         rtol=rtol, atol=atol))
            # Inverting t to s
            self.assertIsNone(np.testing.assert_allclose(f_t_inv_s.vecs_numpy[f_t_inv_s.mask_numpy],
                                                         bb_s.vecs_numpy[f_t_inv_s.mask_numpy],
                                                         rtol=rtol, atol=atol))

    def test_track(self):
        f_s = Flow.from_transforms([['rotation', 0, 0, 30]], (512, 512), 's')
        f_s.mask[:, :, 200:] = False
        f_t = Flow.from_transforms([['rotation', 0, 0, 30]], (512, 512), 't')
        f_t.mask[:, :, 200:] = False

        # Test valid status for 't' flow
        pts = torch.tensor([
            [0, 50],            # Moved out of bounds by a valid flow vector
            [0, 500],           # Moved out of bounds by an invalid flow vector
            [8.3, 7.2],         # Moved normally by valid flow vector
            [120.4, 160.2],     # Moved normally by valid flow vector
            [300, 200]          # Moved normally by invalid flow vector
        ], requires_grad=True)
        desired_valid_status = [False, False, True, True, False]
        set_pure_pytorch()
        t_pts, tracked = f_t.track(pts, get_valid_status=True)
        self.assertIsNotNone(t_pts.grad_fn)
        self.assertIsNone(np.testing.assert_equal(to_numpy(tracked), desired_valid_status))
        unset_pure_pytorch()
        t_pts, tracked = f_t.track(pts, get_valid_status=True)
        self.assertIsNone(np.testing.assert_equal(to_numpy(tracked), desired_valid_status))

        # Batched
        f_s.vecs.requires_grad_()
        f3 = batch_flows([f_s, f_s, f_s])
        d1 = [desired_valid_status]
        d3 = [desired_valid_status, desired_valid_status, desired_valid_status]
        pts.requires_grad = False
        pts1 = pts.unsqueeze(0)
        pts3 = pts1.repeat(3, 1, 1)
        set_pure_pytorch()
        p_11, t_1_1 = f_s.track(pts1, get_valid_status=True)
        p_31, t_3_1 = f3.track(pts1, get_valid_status=True)
        p_33, t_3_3 = f3.track(pts3, get_valid_status=True)
        for p in [p_11, p_31, p_33]:
            self.assertIsNotNone(p.grad_fn)
        self.assertIsNone(np.testing.assert_equal(to_numpy(t_1_1), d1))
        self.assertIsNone(np.testing.assert_equal(to_numpy(t_3_1), d3))
        self.assertIsNone(np.testing.assert_equal(to_numpy(t_3_3), d3))
        unset_pure_pytorch()
        p_11, t_1_1 = f_s.track(pts1, get_valid_status=True)
        p_31, t_3_1 = f3.track(pts1, get_valid_status=True)
        p_33, t_3_3 = f3.track(pts3, get_valid_status=True)
        for p in [p_11, p_31, p_33]:
            self.assertIsNotNone(p.grad_fn)
        self.assertIsNone(np.testing.assert_equal(to_numpy(t_1_1), d1))
        self.assertIsNone(np.testing.assert_equal(to_numpy(t_3_1), d3))
        self.assertIsNone(np.testing.assert_equal(to_numpy(t_3_3), d3))

        # Test valid status for 's' flow
        pts = torch.tensor([
            [0, 50],            # Moved out of bounds by a valid flow vector
            [0, 500],           # Moved out of bounds by an invalid flow vector
            [8.3, 7.2],         # Moved normally by valid flow vector
            [120.4, 160.2],     # Moved normally by valid flow vector
            [300, 200]          # Moved normally by invalid flow vector
        ], requires_grad=True)
        f_s.vecs.requires_grad = False
        desired_valid_status = [False, False, True, True, False]
        set_pure_pytorch()
        t_pts, tracked = f_s.track(pts, get_valid_status=True)
        self.assertIsNotNone(t_pts.grad_fn)
        self.assertIsNone(np.testing.assert_equal(to_numpy(tracked), desired_valid_status))
        unset_pure_pytorch()
        t_pts, tracked = f_s.track(pts, get_valid_status=True)
        self.assertIsNotNone(t_pts.grad_fn)
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
        f_s_b = batch_flows((f_s, f_s_masked))
        f_t_b = batch_flows((f_t, f_t_masked))
        desired_area_s = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]).astype('bool')
        desired_area_s_pp = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0],
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
        desired_area_s_masked_consider_mask_pp = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0, 1, 1],
            [0, 0, 1, 1, 0, 0, 1],
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
        desired_area_s_masked_pp = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 0],
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

        # Test using pure pytorch
        set_pure_pytorch()
        self.assertIsNone(np.testing.assert_equal(f_s.valid_target()[0], desired_area_s_pp))
        self.assertIsNone(np.testing.assert_equal(f_t.valid_target()[0], desired_area_t))
        self.assertIsNone(np.testing.assert_equal(f_s_masked.valid_target()[0], desired_area_s_masked_consider_mask_pp))
        self.assertIsNone(np.testing.assert_equal(f_s_masked.valid_target(False)[0], desired_area_s_masked_pp))
        self.assertIsNone(np.testing.assert_equal(f_t_masked.valid_target()[0], desired_area_t_masked))
        # All of the above batched
        self.assertIsNone(np.testing.assert_equal(f_s_b.valid_target()[0], desired_area_s_pp))
        self.assertIsNone(np.testing.assert_equal(f_s_b.valid_target()[1], desired_area_s_masked_consider_mask_pp))
        self.assertIsNone(np.testing.assert_equal(f_t_b.valid_target()[0], desired_area_t))
        self.assertIsNone(np.testing.assert_equal(f_t_b.valid_target()[1], desired_area_t_masked))

        # Test using griddata
        unset_pure_pytorch()
        self.assertIsNone(np.testing.assert_equal(f_s.valid_target()[0], desired_area_s))
        self.assertIsNone(np.testing.assert_equal(f_t.valid_target()[0], desired_area_t))
        self.assertIsNone(np.testing.assert_equal(f_s_masked.valid_target()[0], desired_area_s_masked_consider_mask))
        self.assertIsNone(np.testing.assert_equal(f_s_masked.valid_target(False)[0], desired_area_s_masked))
        self.assertIsNone(np.testing.assert_equal(f_t_masked.valid_target()[0], desired_area_t_masked))
        # All of the above batched
        self.assertIsNone(np.testing.assert_equal(f_s_b.valid_target()[0], desired_area_s))
        self.assertIsNone(np.testing.assert_equal(f_s_b.valid_target()[1], desired_area_s_masked_consider_mask))
        self.assertIsNone(np.testing.assert_equal(f_t_b.valid_target()[0], desired_area_t))
        self.assertIsNone(np.testing.assert_equal(f_t_b.valid_target()[1], desired_area_t_masked))

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
        f_s_b = batch_flows((f_s, f_s_masked))
        f_t_b = batch_flows((f_t, f_t_masked))
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
        desired_area_t_pp = np.array([
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0, 0]
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
        desired_area_t_masked_consider_mask_pp = np.array([
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0]
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
        desired_area_t_masked_pp = np.array([
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0]
        ]).astype('bool')

        set_pure_pytorch()
        self.assertIsNone(np.testing.assert_equal(f_s.valid_source()[0], desired_area_s))
        self.assertIsNone(np.testing.assert_equal(f_t.valid_source()[0], desired_area_t_pp))
        self.assertIsNone(np.testing.assert_equal(f_s_masked.valid_source()[0], desired_area_s_masked))
        self.assertIsNone(np.testing.assert_equal(f_t_masked.valid_source()[0], desired_area_t_masked_consider_mask_pp))
        self.assertIsNone(np.testing.assert_equal(f_t_masked.valid_source(False)[0], desired_area_t_masked_pp))
        # All of the above batched
        self.assertIsNone(np.testing.assert_equal(f_s_b.valid_source()[0], desired_area_s))
        self.assertIsNone(np.testing.assert_equal(f_s_b.valid_source()[1], desired_area_s_masked))
        self.assertIsNone(np.testing.assert_equal(f_t_b.valid_source()[0], desired_area_t_pp))
        self.assertIsNone(np.testing.assert_equal(f_t_b.valid_source()[1], desired_area_t_masked_consider_mask_pp))

        unset_pure_pytorch()
        self.assertIsNone(np.testing.assert_equal(f_s.valid_source()[0], desired_area_s))
        self.assertIsNone(np.testing.assert_equal(f_t.valid_source()[0], desired_area_t))
        self.assertIsNone(np.testing.assert_equal(f_s_masked.valid_source()[0], desired_area_s_masked))
        self.assertIsNone(np.testing.assert_equal(f_t_masked.valid_source()[0], desired_area_t_masked_consider_mask))
        self.assertIsNone(np.testing.assert_equal(f_t_masked.valid_source(False)[0], desired_area_t_masked))
        # All of the above batched
        self.assertIsNone(np.testing.assert_equal(f_s_b.valid_source()[0], desired_area_s))
        self.assertIsNone(np.testing.assert_equal(f_s_b.valid_source()[1], desired_area_s_masked))
        self.assertIsNone(np.testing.assert_equal(f_t_b.valid_source()[0], desired_area_t))
        self.assertIsNone(np.testing.assert_equal(f_t_b.valid_source()[1], desired_area_t_masked_consider_mask))

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

        set_pure_pytorch()
        self.assertIsNone(np.testing.assert_equal(f_s.get_padding()[0], f_s_desired))
        self.assertIsNone(np.testing.assert_equal(f_t.get_padding()[0], f_t_desired))
        self.assertIsNone(np.testing.assert_equal(f_s_masked.get_padding()[0], f_s_masked_desired))
        self.assertIsNone(np.testing.assert_equal(f_t_masked.get_padding()[0], f_t_masked_desired))
        # batched
        f = batch_flows((f_s, f_s))
        self.assertIsNone(np.testing.assert_equal(f.get_padding(), [f_s_desired, f_s_desired]))
        f = Flow.zero(shape)
        f._vecs[0] = torch.rand(*shape) * 1e-4
        self.assertIsNone(np.testing.assert_equal(f.get_padding(), [[0, 0, 0, 0]]))

        unset_pure_pytorch()
        self.assertIsNone(np.testing.assert_equal(f_s.get_padding()[0], f_s_desired))
        self.assertIsNone(np.testing.assert_equal(f_t.get_padding()[0], f_t_desired))
        self.assertIsNone(np.testing.assert_equal(f_s_masked.get_padding()[0], f_s_masked_desired))
        self.assertIsNone(np.testing.assert_equal(f_t_masked.get_padding()[0], f_t_masked_desired))
        # batched
        f = batch_flows((f_s, f_s))
        self.assertIsNone(np.testing.assert_equal(f.get_padding(), [f_s_desired, f_s_desired]))
        f = Flow.zero(shape)
        f._vecs[0] = torch.rand(*shape) * 1e-4
        self.assertIsNone(np.testing.assert_equal(f.get_padding(), [[0, 0, 0, 0]]))

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
        flow = batch_flows([flow, flow, flow])
        self.assertEqual(all(flow.is_zero()), True)
        self.assertEqual(all(flow.is_zero(masked=True)), True)
        self.assertEqual(all(flow.is_zero(masked=False)), False)

        with self.assertRaises(TypeError):  # Masked wrong type
            flow.is_zero(masked='test')

    def test_visualise(self):
        # Correct values for the different modes
        # Horizontal flow towards the right is red
        flow = Flow.from_transforms([['translation', 1, 0]], [200, 300])
        desired_img = np.tile(np.array([0, 0, 255]).reshape((1, 1, 3)), (200, 300, 1))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('bgr', return_tensor=False)[0], desired_img))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('rgb', return_tensor=False)[0],
                                                  desired_img[..., ::-1]))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', return_tensor=False)[0, ..., 0], 0))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', return_tensor=False)[0, ..., 1], 255))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', return_tensor=False)[0, ..., 2], 255))

        # Flow outwards at the angle of 240 degrees (counter-clockwise) is green
        flow = Flow.from_transforms([['translation', -1, math.sqrt(3)]], [200, 300])
        desired_img = np.tile(np.array([0, 255, 0]).reshape((1, 1, 3)), (200, 300, 1))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('bgr', return_tensor=False)[0], desired_img))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('rgb', return_tensor=False)[0], desired_img))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', return_tensor=False)[0, ..., 0], 60))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', return_tensor=False)[0, ..., 1], 255))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', return_tensor=False)[0, ..., 2], 255))

        # Flow outwards at the angle of 240 degrees (counter-clockwise) is blue
        flow = Flow.from_transforms([['translation', -1, -math.sqrt(3)]], [200, 300])
        desired_img = np.tile(np.array([255, 0, 0]).reshape((1, 1, 3)), (200, 300, 1))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('bgr', return_tensor=False)[0], desired_img))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('rgb', return_tensor=False)[0],
                                                  desired_img[..., ::-1]))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', return_tensor=False)[0, ..., 0], 120))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', return_tensor=False)[0, ..., 1], 255))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', return_tensor=False)[0, ..., 2], 255))

        # Batched flow gives batched output
        flow = batch_flows((
            Flow.from_transforms([['translation', 1, 0]], [200, 300]),
            Flow.from_transforms([['translation', -1, math.sqrt(3)]], [200, 300]),
            Flow.from_transforms([['translation', -1, -math.sqrt(3)]], [200, 300])
        ))
        desired_img = np.stack((
            np.tile(np.array([0, 0, 255]).reshape((1, 1, 3)), (200, 300, 1)),
            np.tile(np.array([0, 255, 0]).reshape((1, 1, 3)), (200, 300, 1)),
            np.tile(np.array([255, 0, 0]).reshape((1, 1, 3)), (200, 300, 1))
        ), axis=0)
        vis = flow.visualise('bgr', return_tensor=False)
        for v, d in zip(vis, desired_img):
            self.assertIsNone(np.testing.assert_equal(v, d))

        # Show the flow mask
        mask = np.zeros((200, 300))
        mask[30:-30, 40:-40] = 1
        flow = Flow.from_transforms([['translation', 1, 0]], (200, 300), 't', mask)
        self.assertIsNone(np.testing.assert_equal(flow.visualise('bgr', True, return_tensor=False)[0, 10, 10],
                                                  [0, 0, 180]))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('rgb', True, return_tensor=False)[0, 10, 10],
                                                  [180, 0, 0]))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', True, return_tensor=False)[0, ..., 0], 0))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', True, return_tensor=False)[0, ..., 1], 255))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', True, return_tensor=False)[0, 10, 10, 2], 180))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', True, return_tensor=False)[0, 100, 100, 2],
                                                  255))

        # Show the flow mask border
        mask = np.zeros((200, 300))
        mask[30:-30, 40:-40] = 1
        flow = Flow.from_transforms([['translation', 1, 0]], (200, 300), 't', mask)
        self.assertIsNone(np.testing.assert_equal(flow.visualise('bgr', True, True, return_tensor=False)[0, 30, 40],
                                                  [0, 0, 0]))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('rgb', True, True, return_tensor=False)[0, 30, 40],
                                                  [0, 0, 0]))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', True, True, return_tensor=False)[0, ..., 0], 0))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', True, True, return_tensor=False)[0, 30, 40, 1],
                                                  0))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', True, True, return_tensor=False)[0, 30, 40, 2],
                                                  0))

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
        with self.assertRaises(TypeError):
            flow.visualise('rgb', range_max=(1, 2))
        with self.assertRaises(ValueError):
            flow.visualise('rgb', range_max=[0])
        with self.assertRaises(ValueError):
            flow.visualise('rgb', range_max=-1)

    def test_visualise_arrows(self):
        img_np = cv2.resize(cv2.imread('smudge.png'), None, fx=.25, fy=.25)
        img_np_1 = img_np[np.newaxis, ...]
        img_np_3 = np.broadcast_to(img_np, (3, *img_np.shape)).copy()
        img_pt = torch.tensor(img_np).permute(2, 0, 1)
        img_pt_1 = img_pt.unsqueeze(0)
        img_pt_3 = img_pt_1.repeat(3, 1, 1, 1)
        mask = np.zeros(img_np.shape[:2])
        mask[50:-50, 20:-20] = 1
        for ref in ['s', 't']:
            flow1 = batch_flows((
                Flow.from_transforms([['translation', 10, -8]], img_np.shape[:2], ref, mask),
                Flow.from_transforms([['translation', -5, 10]], img_np.shape[:2], ref, mask),
                Flow.from_transforms([['rotation', 30, 50, 30]], img_np.shape[:2], ref, mask)
            ))
            flow2 = Flow.from_transforms([['rotation', 10, 30, 20]], img_np.shape[:2], ref, mask)
            for scaling in [0.1, 1, 2]:
                for show_mask in [True, False]:
                    for show_mask_border in [True, False]:
                        for return_tensor in [True, False]:
                            for img in [None, img_np, img_pt, img_np_1, img_pt_1, img_np_3, img_pt_3]:
                                arrow_img = flow1.visualise_arrows(
                                    grid_dist=10,
                                    scaling=scaling,
                                    img=img,
                                    show_mask=show_mask,
                                    show_mask_borders=show_mask_border,
                                    return_tensor=return_tensor
                                )
                                if return_tensor:
                                    self.assertIsInstance(arrow_img, torch.Tensor)
                                else:
                                    # Uncomment the following lines to see test images
                                    # for a in arrow_img:
                                    #     cv2.imshow('test', a)
                                    #     cv2.waitKey(10)
                                    self.assertIsInstance(arrow_img, np.ndarray)
                            for img in [None, img_np, img_pt, img_np_1, img_pt_1]:
                                for colour in [None, (100, 100, 100)]:
                                    arrow_img = flow2.visualise_arrows(
                                        grid_dist=10,
                                        scaling=scaling,
                                        img=img,
                                        show_mask=show_mask,
                                        show_mask_borders=show_mask_border,
                                        return_tensor=return_tensor,
                                        colour=colour
                                    )
                                    if return_tensor:
                                        self.assertIsInstance(arrow_img, torch.Tensor)
                                    else:
                                        # Uncomment the following lines to see test images
                                        # for a in arrow_img:
                                        #     cv2.imshow('test', a)
                                        #     cv2.waitKey(10)
                                        self.assertIsInstance(arrow_img, np.ndarray)
        with self.assertRaises(TypeError):
            flow1.visualise_arrows(grid_dist='test')
        with self.assertRaises(ValueError):
            flow1.visualise_arrows(grid_dist=-1)
        with self.assertRaises(TypeError):
            flow1.visualise_arrows(10, img='test')
        with self.assertRaises(ValueError):
            flow1.visualise_arrows(10, img=mask)
        with self.assertRaises(ValueError):
            flow1.visualise_arrows(10, img=mask[10:])
        with self.assertRaises(ValueError):
            flow1.visualise_arrows(10, img=img_np[..., :2])
        with self.assertRaises(ValueError):
            flow1.visualise_arrows(10, img=img_np_3[:2])
        with self.assertRaises(TypeError):
            flow1.visualise_arrows(10, img_np, scaling='test')
        with self.assertRaises(ValueError):
            flow1.visualise_arrows(10, img_np, scaling=-1)
        with self.assertRaises(TypeError):
            flow1.visualise_arrows(10, img_np, None, show_mask='test')
        with self.assertRaises(TypeError):
            flow1.visualise_arrows(10, img_np, None, True, show_mask_borders='test')
        with self.assertRaises(TypeError):
            flow1.visualise_arrows(10, img_np, None, True, True, colour='test')
        with self.assertRaises(ValueError):
            flow1.visualise_arrows(10, img_np, None, True, True, colour=(0, 0))
        with self.assertRaises(TypeError):
            flow1.visualise_arrows(10, img_np, None, True, True, colour=(0, 0, 0), return_tensor='test')

    def test_show(self):
        flow = Flow.from_transforms([['translation', 10, -8]], (100, 150))
        # Uncomment the following line to see the flow in an OpenCV window
        # flow.show()
        with self.assertRaises(TypeError):
            flow.show(wait='test')
        with self.assertRaises(ValueError):
            flow.show(wait=-1)
        with self.assertRaises(TypeError):
            flow.show(elem=.3)
        with self.assertRaises(ValueError):
            flow.show(elem=1)

    def test_show_arrows(self):
        flow = Flow.from_transforms([['translation', 10, -8]], (100, 150))
        # Uncomment the following line to see the flow in an OpenCV window
        # flow.show_arrows()
        with self.assertRaises(TypeError):
            flow.show_arrows(wait='test')
        with self.assertRaises(ValueError):
            flow.show_arrows(wait=-1)
        with self.assertRaises(TypeError):
            flow.show(elem=.3)
        with self.assertRaises(ValueError):
            flow.show(elem=1)

    def test_matrix(self):
        # Partial affine transform, test reconstruction with all methods
        transforms = [
            ['translation', 2, 1],
            ['rotation', 20, 20, 30],
            ['scaling', 10, 10, 1.1]
        ]
        matrix = matrix_from_transforms(transforms).unsqueeze(0)
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

        # Batched matrices
        transforms = [
            ['translation', 2, 1],
            ['rotation', 20, 20, 30],
            ['scaling', 10, 10, 1.1]
        ]
        matrix1 = matrix_from_transforms(transforms[:2])
        matrix2 = matrix_from_transforms(transforms[1:])
        flow = batch_flows((
            Flow.from_matrix(matrix1, (100, 200), 's'),
            Flow.from_matrix(matrix2, (100, 200), 's'),
        ))
        actual_matrix = flow.matrix(dof=4, method='ransac')
        self.assertIsNone(np.testing.assert_allclose(to_numpy(actual_matrix)[0], matrix1, rtol=1e-6))
        self.assertIsNone(np.testing.assert_allclose(to_numpy(actual_matrix)[1], matrix2, rtol=1e-6))

        # Partial affine transform reconstruction in the presence of noise, only check first 4 values
        matrix = matrix_from_transforms(transforms)
        flow_s = Flow.from_matrix(matrix, (100, 200), 's')
        flow_noise = (np.random.rand(100, 200, 2) - .5) * 5
        actual_matrix_4_ransac = (flow_s + flow_noise).matrix(4, 'ransac')[0]
        actual_matrix_4_lmeds = (flow_s + flow_noise).matrix(4, 'lmeds')[0]
        actual_matrix_6_ransac = (flow_s + flow_noise).matrix(6, 'ransac')[0]
        actual_matrix_6_lmeds = (flow_s + flow_noise).matrix(6, 'lmeds')[0]
        actual_matrix_8_lms = (flow_s + flow_noise).matrix(8, 'lms')[0]
        actual_matrix_8_ransac = (flow_s + flow_noise).matrix(8, 'ransac')[0]
        actual_matrix_8_lmeds = (flow_s + flow_noise).matrix(8, 'lmeds')[0]
        for actual_matrix in [actual_matrix_4_ransac, actual_matrix_4_lmeds,
                              actual_matrix_6_ransac, actual_matrix_6_lmeds,
                              actual_matrix_8_lms, actual_matrix_8_ransac, actual_matrix_8_lmeds]:
            self.assertIsNone(np.testing.assert_allclose(actual_matrix[:2, :2], matrix[:2, :2], atol=1e-2, rtol=1e-1))

        # Masked vs non-masked matrix fitting
        matrix = matrix_from_transforms(transforms)
        mask = np.zeros((100, 200), 'bool')
        mask[:50, :50] = 1  # upper left corner will contain the real values
        flow = Flow.from_matrix(matrix, (100, 200), 's', mask)
        random_vecs = (np.random.rand(1, 2, 100, 200) - 0.5) * 200
        random_vecs[:, :, :50, :50] = flow.vecs[:, :, :50, :50]
        flow.vecs = random_vecs
        # Make sure this fails with the 'lmeds' method:
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(flow.matrix(4, 'lmeds', False), matrix)
        # Test that it does NOT fail when the invalid flow elements are masked out
        self.assertIsNone(np.testing.assert_allclose(flow.matrix(4, 'lmeds', True)[0], matrix))

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
        img = cv2.resize(cv2.imread('smudge.png'), None, fx=.125, fy=.125)
        shape = img.shape[:2]
        transforms = [
            ['rotation', 60, 80, -30],
            ['scaling', 40, 30, 0.8],
        ]
        transforms2 = [
            ['rotation', 50, 70, -20],
            ['scaling', 20, 50, 0.9],
        ]
        for f in [set_pure_pytorch, unset_pure_pytorch]:
            f()
            for ref in ['s', 't']:
                atol = 8e-1 if get_pure_pytorch() else 5e-2
                f1 = Flow.from_transforms(transforms[0:1], shape, ref)
                f2 = Flow.from_transforms(transforms[1:2], shape, ref)
                f3 = Flow.from_transforms(transforms, shape, ref)
                bf1 = batch_flows((f1, Flow.from_transforms(transforms2[0:1], shape, ref)))
                bf2 = batch_flows((f2, Flow.from_transforms(transforms2[1:2], shape, ref)))
                bf3 = batch_flows((f3, Flow.from_transforms(transforms2, shape, ref)))

                # Mode 1
                f2_g = f2.copy()
                f2_g.vecs.requires_grad_()
                f1_actual = f2_g.combine_with(f3, 1)
                if get_pure_pytorch() or ref != 't':
                    self.assertIsNotNone(f1_actual.vecs.grad_fn)
                # Uncomment the following two lines to see / check the flow fields
                # f1.show(wait=500, show_mask=True, show_mask_borders=True)
                # f1_actual.show(show_mask=True, show_mask_borders=True)
                self.assertIsInstance(f1_actual, Flow)
                self.assertEqual(f1_actual.ref, ref)
                comb_mask = f1_actual.mask_numpy & f1.mask_numpy
                self.assertIsNone(np.testing.assert_allclose(f1_actual.vecs_numpy[comb_mask], f1.vecs_numpy[comb_mask],
                                                             atol=atol))
                bf3_g = bf3.copy()
                bf3_g.vecs.requires_grad_()
                bf1_actual = bf2.combine_with(bf3_g, 1)
                if get_pure_pytorch() or ref != 't':
                    self.assertIsNotNone(bf1_actual.vecs.grad_fn)
                # Uncomment the following two lines to see / check the flow fields
                # bf1.show(1, wait=500, show_mask=True, show_mask_borders=True)
                # bf1_actual.show(1, show_mask=True, show_mask_borders=True)
                self.assertIsInstance(bf1_actual, Flow)
                self.assertEqual(bf1_actual.ref, ref)
                comb_mask = bf1_actual.mask_numpy & bf1.mask_numpy
                self.assertIsNone(np.testing.assert_allclose(bf1_actual.vecs_numpy[comb_mask], bf1.vecs_numpy[comb_mask],
                                                             atol=atol))

                # Mode 2
                f1_g = f1.copy()
                f1_g.vecs.requires_grad_()
                f2_actual = f1_g.combine_with(f3, 2)
                if get_pure_pytorch():
                    self.assertIsNotNone(f2_actual.vecs.grad_fn)
                # Uncomment the following two lines to see / check the flow fields
                # f2.show(wait=500, show_mask=True, show_mask_borders=True)
                # f2_actual.show(show_mask=True, show_mask_borders=True)
                self.assertIsInstance(f2_actual, Flow)
                self.assertEqual(f2_actual.ref, ref)
                comb_mask = f2_actual.mask_numpy & f2.mask_numpy
                self.assertIsNone(np.testing.assert_allclose(f2_actual.vecs_numpy[comb_mask], f2.vecs_numpy[comb_mask],
                                                             atol=atol))
                bf3_g = bf3.copy()
                bf3_g.vecs.requires_grad_()
                bf2_actual = bf1.combine_with(bf3_g, 2)
                if get_pure_pytorch():
                    self.assertIsNotNone(bf2_actual.vecs.grad_fn)
                # Uncomment the following two lines to see / check the flow fields
                # bf2.show(1, wait=500, show_mask=True, show_mask_borders=True)
                # bf2_actual.show(1, show_mask=True, show_mask_borders=True)
                self.assertIsInstance(bf2_actual, Flow)
                self.assertEqual(bf2_actual.ref, ref)
                comb_mask = bf2_actual.mask_numpy & bf2.mask_numpy
                self.assertIsNone(np.testing.assert_allclose(bf2_actual.vecs_numpy[comb_mask], bf2.vecs_numpy[comb_mask],
                                                             atol=atol))

                # Mode 3
                f1_g = f1.copy()
                f1_g.vecs.requires_grad_()
                f3_actual = f1_g.combine_with(f2, 3)
                self.assertIsNotNone(f3_actual.vecs.grad_fn)
                # Uncomment the following two lines to see / check the flow fields
                # f3.show(wait=500, show_mask=True, show_mask_borders=True)
                # f3_actual.show(show_mask=True, show_mask_borders=True)
                self.assertIsInstance(f3_actual, Flow)
                self.assertEqual(f3_actual.ref, ref)
                comb_mask = f3_actual.mask_numpy & f3.mask_numpy
                self.assertIsNone(np.testing.assert_allclose(f3_actual.vecs_numpy[comb_mask], f3.vecs_numpy[comb_mask],
                                                             atol=5e-2))
                bf2_g = bf2.copy()
                bf2_g.vecs.requires_grad_()
                bf3_actual = bf1.combine_with(bf2_g, 3)
                self.assertIsNotNone(bf3_actual.vecs.grad_fn)
                # Uncomment the following two lines to see / check the flow fields
                # bf3.show(1, wait=500, show_mask=True, show_mask_borders=True)
                # bf3_actual.show(1, show_mask=True, show_mask_borders=True)
                self.assertIsInstance(bf3_actual, Flow)
                self.assertEqual(bf3_actual.ref, ref)
                comb_mask = bf3_actual.mask_numpy & bf3.mask_numpy
                self.assertIsNone(np.testing.assert_allclose(bf3_actual.vecs_numpy[comb_mask], bf3.vecs_numpy[comb_mask],
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
