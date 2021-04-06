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

import unittest
import torch
import numpy as np
import math
from oflibpytorch.flow_class import Flow
from oflibpytorch.utils import to_numpy


class FlowTest(unittest.TestCase):
    def test_flow(self):
        if torch.cuda.is_available():
            expected_device_list = ['cpu', 'cuda', 'cpu']
        else:
            expected_device_list = ['cpu', 'cpu', 'cpu']
        vecs_np_2hw = np.zeros((2, 100, 200))
        vecs_np_hw2 = np.zeros((100, 200, 2))
        vecs_pt_2hw = torch.zeros((2, 100, 200))
        vecs_pt_hw2 = torch.zeros((100, 200, 2))
        mask_empty = None
        mask_np = np.ones((100, 200), 'bool')
        mask_pt = torch.ones(100, 200).to(torch.bool)
        for vecs in [vecs_np_2hw, vecs_np_hw2, vecs_pt_2hw, vecs_pt_hw2]:
            for ref, ref_expected in zip(['t', 's', None], ['t', 's', 't']):
                for mask in [mask_empty, mask_np, mask_pt]:
                    for device, device_expected in zip(['cpu', 'cuda', None], expected_device_list):
                        flow = Flow(vecs, ref=ref, mask=mask, device=device)
                        self.assertIsNone(np.testing.assert_equal(to_numpy(flow.vecs), vecs_np_2hw))
                        self.assertIsNone(np.testing.assert_equal(flow.vecs_numpy, vecs_np_hw2))
                        self.assertEqual(flow.ref, ref_expected)
                        self.assertIsNone(np.testing.assert_equal(to_numpy(flow.mask), mask_np))
                        self.assertEqual(flow.device, device_expected)
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
                    self.assertIsNone(np.testing.assert_equal(to_numpy(flow.vecs), vecs_np_2hw))
                    self.assertIsNone(np.testing.assert_equal(flow.vecs_numpy, vecs_np_hw2))
                    self.assertEqual(flow.ref, ref_expected)
                    self.assertIsNone(np.testing.assert_equal(to_numpy(flow.mask), mask_np))
                    self.assertEqual(flow.device, device_expected)
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
        shape = [200, 300]
        zero_flow = Flow.zero(shape)
        self.assertIsNone(np.testing.assert_equal(zero_flow.shape, shape))
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
        matrix_pt = torch.tensor([[math.sqrt(3) / 2, -.5, 26.3397459622],
                                  [.5, math.sqrt(3) / 2, 1.69872981078],
                                  [0, 0, 1]])
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
                    self.assertEqual(flow.device, flow_expected_device)
                    self.assertIsNone(np.testing.assert_allclose(flow.vecs_numpy[50, 10], [0, 0], atol=1e-4))
                    self.assertIsNone(np.testing.assert_allclose(flow.vecs_numpy[50, 299], [38.7186583063, 144.5],
                                                                 atol=1e-4, rtol=1e-4))
                    self.assertIsNone(np.testing.assert_allclose(flow.vecs_numpy[199, 10], [-74.5, 19.9622148361],
                                                                 atol=1e-4, rtol=1e-4))
                    self.assertIsNone(np.testing.assert_equal(flow.shape, shape))

        # Invalid input
        with self.assertRaises(TypeError):
            Flow.from_matrix('test', [10, 10])
        with self.assertRaises(ValueError):
            Flow.from_matrix(np.eye(4), [10, 10])
        with self.assertRaises(ValueError):
            Flow.from_matrix(torch.eye(4), [10, 10])


if __name__ == '__main__':
    unittest.main()
