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
import cv2
import numpy as np
import torch
import sys
sys.path.append('..')
from src.oflibpytorch.flow_class import Flow
from src.oflibpytorch.flow_operations import combine_flows, switch_flow_ref, invert_flow, valid_target, valid_source, \
    get_flow_padding, get_flow_matrix, visualise_flow, visualise_flow_arrows
from src.oflibpytorch.utils import to_numpy, matrix_from_transforms


class TestFlowOperations(unittest.TestCase):
    def test_combine_flows(self):
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
            f1_actual_f = combine_flows(f2, f3, 1)
            f1_actual = combine_flows(f2.vecs_numpy, f3.vecs, 1, ref)
            # f1.show(500, show_mask=True, show_mask_borders=True)
            # f1_actual_f.show(show_mask=True, show_mask_borders=True)
            self.assertIsInstance(f1_actual_f, Flow)
            self.assertEqual(f1_actual_f.ref, ref)
            comb_mask = f1_actual_f.mask_numpy & f1.mask_numpy
            self.assertIsNone(np.testing.assert_allclose(f1_actual_f.vecs_numpy[comb_mask], f1.vecs_numpy[comb_mask],
                                                         atol=5e-2))
            self.assertIsInstance(f1_actual, torch.Tensor)
            self.assertIsNone(np.testing.assert_equal(f1_actual_f.vecs_numpy,
                                                      to_numpy(f1_actual, switch_channels=True)))

            # Mode 2
            f2_actual_f = combine_flows(f1, f3, 2)
            f2_actual = combine_flows(f1.vecs, f3.vecs_numpy, 2, ref)
            # f2.show(500, show_mask=True, show_mask_borders=True)
            # f2_actual_f.show(show_mask=True, show_mask_borders=True)
            self.assertIsInstance(f2_actual_f, Flow)
            self.assertEqual(f2_actual_f.ref, ref)
            comb_mask = f2_actual_f.mask_numpy & f2.mask_numpy
            self.assertIsNone(np.testing.assert_allclose(f2_actual_f.vecs_numpy[comb_mask], f2.vecs_numpy[comb_mask],
                                                         atol=5e-2))
            self.assertIsInstance(f2_actual, torch.Tensor)
            self.assertIsNone(np.testing.assert_equal(f2_actual_f.vecs_numpy,
                                                      to_numpy(f2_actual, switch_channels=True)))

            # Mode 3
            f3_actual_f = combine_flows(f1, f2, 3)
            f3_actual = combine_flows(torch.tensor(f1.vecs_numpy), to_numpy(f2.vecs), 3, ref)
            # f3.show(500, show_mask=True, show_mask_borders=True)
            # f3_actual_f.show(show_mask=True, show_mask_borders=True)
            self.assertIsInstance(f3_actual_f, Flow)
            self.assertEqual(f3_actual_f.ref, ref)
            comb_mask = f3_actual_f.mask_numpy & f3.mask_numpy
            self.assertIsNone(np.testing.assert_allclose(f3_actual_f.vecs_numpy[comb_mask], f3.vecs_numpy[comb_mask],
                                                         atol=5e-2))
            self.assertIsInstance(f3_actual, torch.Tensor)
            self.assertIsNone(np.testing.assert_equal(f3_actual_f.vecs_numpy,
                                                      to_numpy(f3_actual, switch_channels=True)))

    def test_switch_flow_ref(self):
        shape = [10, 20]
        transforms = [['rotation', 5, 10, 30]]
        flow_s = Flow.from_transforms(transforms, shape, 's')
        flow_t = Flow.from_transforms(transforms, shape, 't')
        fl_op_switched_s = to_numpy(switch_flow_ref(flow_s.vecs, 's'), switch_channels=True)
        fl_op_switched_t = to_numpy(switch_flow_ref(flow_t.vecs_numpy, 't'), switch_channels=True)
        self.assertIsNone(np.testing.assert_equal(flow_s.switch_ref().vecs_numpy, fl_op_switched_s))
        self.assertIsNone(np.testing.assert_equal(flow_t.switch_ref().vecs_numpy, fl_op_switched_t))

    def test_invert_flow(self):
        shape = [10, 20]
        transforms = [['rotation', 5, 10, 30]]
        flow_s = Flow.from_transforms(transforms, shape, 's')
        flow_t = Flow.from_transforms(transforms, shape, 't')
        s_invert = to_numpy(invert_flow(flow_s.vecs, 's'), switch_channels=True)
        s_invert_t = to_numpy(invert_flow(flow_s.vecs_numpy, 's', 't'), switch_channels=True)
        t_invert = to_numpy(invert_flow(flow_t.vecs_numpy, 't'), switch_channels=True)
        t_invert_s = to_numpy(invert_flow(flow_t.vecs, 't', 's'), switch_channels=True)
        self.assertIsNone(np.testing.assert_equal(flow_s.invert().vecs_numpy, s_invert))
        self.assertIsNone(np.testing.assert_equal(flow_s.invert('t').vecs_numpy, s_invert_t))
        self.assertIsNone(np.testing.assert_equal(flow_t.invert().vecs_numpy, t_invert))
        self.assertIsNone(np.testing.assert_equal(flow_t.invert('s').vecs_numpy, t_invert_s))

    def test_valid_target(self):
        transforms = [['rotation', 0, 0, 45]]
        shape = (7, 7)
        f_s = Flow.from_transforms(transforms, shape, 's')
        f_t = Flow.from_transforms(transforms, shape, 't')
        target_s = to_numpy(valid_target(f_s.vecs_numpy, 's'))
        target_t = to_numpy(valid_target(f_t.vecs, 't'))
        self.assertIsNone(np.testing.assert_equal(to_numpy(f_s.valid_target()), target_s))
        self.assertIsNone(np.testing.assert_equal(to_numpy(f_t.valid_target()), target_t))

    def test_valid_source(self):
        transforms = [['rotation', 0, 0, 45]]
        shape = (7, 7)
        f_s = Flow.from_transforms(transforms, shape, 's')
        f_t = Flow.from_transforms(transforms, shape, 't')
        source_s = to_numpy(valid_source(f_s.vecs, 's'))
        source_t = to_numpy(valid_source(f_t.vecs_numpy, 't'))
        self.assertIsNone(np.testing.assert_equal(to_numpy(f_s.valid_source()), source_s))
        self.assertIsNone(np.testing.assert_equal(to_numpy(f_t.valid_source()), source_t))

    def test_get_flow_padding(self):
        transforms = [['rotation', 0, 0, 45]]
        shape = (7, 7)
        f_s = Flow.from_transforms(transforms, shape, 's')
        f_t = Flow.from_transforms(transforms, shape, 't')
        padding_s = get_flow_padding(to_numpy(f_s.vecs), 's')
        padding_t = get_flow_padding(torch.tensor(f_t.vecs_numpy), 't')
        self.assertIsNone(np.testing.assert_equal(f_s.get_padding(), padding_s))
        self.assertIsNone(np.testing.assert_equal(f_t.get_padding(), padding_t))

    def test_get_flow_matrix(self):
        # Partial affine transform, test reconstruction with all methods
        transforms = [
            ['translation', 20, 10],
            ['rotation', 200, 200, 30],
            ['scaling', 100, 100, 1.1]
        ]
        matrix = matrix_from_transforms(transforms)
        flow_s = Flow.from_matrix(matrix, (1000, 2000), 's')
        flow_t = Flow.from_matrix(matrix, (1000, 2000), 't')
        m_s_f = to_numpy(flow_s.matrix(dof=4, method='ransac'))
        m_t_f = to_numpy(flow_t.matrix(dof=4, method='ransac'))
        m_s = to_numpy(get_flow_matrix(to_numpy(flow_s.vecs), 's', dof=4, method='ransac'))
        m_t = to_numpy(get_flow_matrix(torch.tensor(flow_t.vecs_numpy), 't', dof=4, method='ransac'))
        self.assertIsNone(np.testing.assert_allclose(m_s_f, m_s))
        self.assertIsNone(np.testing.assert_allclose(m_t_f, m_t))

    def test_visualise_flow(self):
        flow = Flow.from_transforms([['translation', 1, 0]], [200, 300])
        self.assertIsNone(np.testing.assert_equal(flow.visualise('bgr', return_tensor=False),
                                                  visualise_flow(flow.vecs, 'bgr', return_tensor=False)))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('rgb', return_tensor=False),
                                                  visualise_flow(flow.vecs_numpy, 'rgb', return_tensor=False)))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', return_tensor=False),
                                                  visualise_flow(to_numpy(flow.vecs), 'hsv', return_tensor=False)))

    def test_visualise_flow_arrows(self):
        for ref in ['s', 't']:
            flow = Flow.from_transforms([['rotation', 10, 10, 30]], [20, 20], ref)
            self.assertIsNone(np.testing.assert_equal(flow.visualise_arrows(return_tensor=False),
                                                      visualise_flow_arrows(flow.vecs, ref, return_tensor=False)))


if __name__ == '__main__':
    unittest.main()
