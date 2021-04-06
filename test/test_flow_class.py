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
from oflibpytorch.flow_class import Flow
from oflibpytorch.utils import to_numpy


class FlowTest(unittest.TestCase):
    def test_flow(self):
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
                    for device, device_expected in zip(['cpu', 'cuda', None], ['cpu', 'cuda', 'cpu']):
                        flow = Flow(vecs, ref=ref, mask=mask, device=device)
                        self.assertIsNone(np.testing.assert_equal(to_numpy(flow.vecs), vecs_np_2hw))
                        self.assertEqual(flow.ref, ref_expected)
                        self.assertIsNone(np.testing.assert_equal(to_numpy(flow.mask), mask_np))
                        self.assertEqual(flow.device, device_expected)
                        self.assertEqual(flow.vecs.device.type, device_expected)
                        self.assertEqual(flow.mask.device.type, device_expected)

        # tensor to cuda, test cuda
        vecs_pt_cuda = torch.zeros((2, 100, 200)).to('cuda')
        for ref, ref_expected in zip(['t', 's', None], ['t', 's', 't']):
            for mask in [mask_empty, mask_np, mask_pt]:
                for device, device_expected in zip(['cpu', 'cuda', None], ['cpu', 'cuda', 'cuda']):
                    flow = Flow(vecs_pt_cuda, ref=ref, mask=mask, device=device)
                    self.assertIsNone(np.testing.assert_equal(to_numpy(flow.vecs), vecs_np_2hw))
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


if __name__ == '__main__':
    unittest.main()
