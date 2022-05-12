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
import cv2
import numpy as np
import torch
import sys
sys.path.append('..')
from src.oflibpytorch.flow_class import Flow
from src.oflibpytorch.flow_operations import combine_flows, switch_flow_ref, invert_flow, valid_target, valid_source, \
    get_flow_padding, get_flow_matrix, visualise_flow, visualise_flow_arrows, batch_flows
from src.oflibpytorch.utils import to_numpy, matrix_from_transforms, \
    set_pure_pytorch, unset_pure_pytorch, get_pure_pytorch


class TestFlowOperations(unittest.TestCase):
    def test_combine_flows(self):
        img = cv2.resize(cv2.imread('smudge.png'), None, fx=.125, fy=.125)
        shape = img.shape[:2]
        transforms = [
            ['rotation', 30, 40, -30],
            ['scaling', 10, 10, 0.8],
        ]
        for f in [set_pure_pytorch, unset_pure_pytorch]:
            f()
            for ref in ['s', 't']:
                atol = 8e-1 if get_pure_pytorch() else 5e-2
                f1 = Flow.from_transforms(transforms[0:1], shape, ref)
                f2 = Flow.from_transforms(transforms[1:2], shape, ref)
                f3 = Flow.from_transforms(transforms, shape, ref)
                f1.vecs.requires_grad_()
                f2.vecs.requires_grad_()
                f3.vecs.requires_grad_()

                # Mode 1
                f1_actual_f = combine_flows(f2, f3, 1)
                if get_pure_pytorch():
                    f1_g = combine_flows(f2.vecs, f3.vecs, 1, ref)
                    self.assertIsNotNone(f1_g.grad_fn)
                f1_actual = combine_flows(f2.vecs_numpy, f3.vecs, 1, ref)
                # Uncomment the following two lines to see / check the flow fields
                # f1.show(500, show_mask=True, show_mask_borders=True)
                # f1_actual_f.show(show_mask=True, show_mask_borders=True)
                self.assertIsInstance(f1_actual_f, Flow)
                self.assertEqual(f1_actual_f.ref, ref)
                comb_mask = f1_actual_f.mask_numpy & f1.mask_numpy
                self.assertIsNone(np.testing.assert_allclose(f1_actual_f.vecs_numpy[comb_mask],
                                                             f1.vecs_numpy[comb_mask],
                                                             atol=atol))
                self.assertIsInstance(f1_actual, torch.Tensor)
                self.assertIsNone(np.testing.assert_equal(f1_actual_f.vecs_numpy,
                                                          to_numpy(f1_actual, switch_channels=True)))

                # Mode 2
                f2_actual_f = combine_flows(f1, f3, 2)
                if get_pure_pytorch():
                    f2_g = combine_flows(f1.vecs, f3.vecs, 2, ref)
                    self.assertIsNotNone(f2_g.grad_fn)
                f2_actual = combine_flows(f1.vecs, f3.vecs_numpy, 2, ref)
                # Uncomment the following two lines to see / check the flow fields
                # f2.show(500, show_mask=True, show_mask_borders=True)
                # f2_actual_f.show(show_mask=True, show_mask_borders=True)
                self.assertIsInstance(f2_actual_f, Flow)
                self.assertEqual(f2_actual_f.ref, ref)
                comb_mask = f2_actual_f.mask_numpy & f2.mask_numpy
                self.assertIsNone(np.testing.assert_allclose(f2_actual_f.vecs_numpy[comb_mask],
                                                             f2.vecs_numpy[comb_mask],
                                                             atol=atol))
                self.assertIsInstance(f2_actual, torch.Tensor)
                self.assertIsNone(np.testing.assert_equal(f2_actual_f.vecs_numpy,
                                                          to_numpy(f2_actual, switch_channels=True)))

                # Mode 3
                f3_actual_f = combine_flows(f1, f2, 3)
                if get_pure_pytorch():
                    f3_g = combine_flows(f1.vecs, f2.vecs, 3, ref)
                    self.assertIsNotNone(f3_g.grad_fn)
                f3_actual = combine_flows(torch.tensor(f1.vecs_numpy), to_numpy(f2.vecs), 3, ref)
                # Uncomment the following two lines to see / check the flow fields
                # f3.show(500, show_mask=True, show_mask_borders=True)
                # f3_actual_f.show(show_mask=True, show_mask_borders=True)
                self.assertIsInstance(f3_actual_f, Flow)
                self.assertEqual(f3_actual_f.ref, ref)
                comb_mask = f3_actual_f.mask_numpy & f3.mask_numpy
                self.assertIsNone(np.testing.assert_allclose(f3_actual_f.vecs_numpy[comb_mask],
                                                             f3.vecs_numpy[comb_mask],
                                                             atol=atol))
                self.assertIsInstance(f3_actual, torch.Tensor)
                self.assertIsNone(np.testing.assert_equal(f3_actual_f.vecs_numpy,
                                                          to_numpy(f3_actual, switch_channels=True)))
        f_3dim = combine_flows(f2.vecs[0], f3.vecs[0], 1, ref)
        self.assertEqual(len(f_3dim.shape), 3)

    def test_switch_flow_ref(self):
        shape = [10, 20]
        transforms = [['rotation', 5, 10, 30]]
        flow_s = Flow.from_transforms(transforms, shape, 's')
        flow_t = Flow.from_transforms(transforms, shape, 't')
        flow_s.vecs.requires_grad_()
        flow_t.vecs.requires_grad_()
        set_pure_pytorch()
        sfr_ss = switch_flow_ref(flow_s.vecs, 's')
        sfr_tt = switch_flow_ref(flow_t.vecs, 't')
        fl_op_switched_s = to_numpy(sfr_ss, switch_channels=True)
        fl_op_switched_t = to_numpy(sfr_tt, switch_channels=True)
        self.assertIsNotNone(sfr_ss.grad_fn)
        self.assertIsNotNone(sfr_tt.grad_fn)
        self.assertIsNone(np.testing.assert_equal(flow_s.switch_ref().vecs_numpy, fl_op_switched_s))
        self.assertIsNone(np.testing.assert_equal(flow_t.switch_ref().vecs_numpy, fl_op_switched_t))
        fl_op_switched_s_3dim = to_numpy(switch_flow_ref(flow_s.vecs.squeeze(0), 's'), switch_channels=True)
        self.assertEqual(len(fl_op_switched_s_3dim.shape), 3)
        unset_pure_pytorch()
        fl_op_switched_s = to_numpy(switch_flow_ref(flow_s.vecs, 's'), switch_channels=True)
        fl_op_switched_t = to_numpy(switch_flow_ref(flow_t.vecs_numpy, 't'), switch_channels=True)
        self.assertIsNone(np.testing.assert_equal(flow_s.switch_ref().vecs_numpy, fl_op_switched_s))
        self.assertIsNone(np.testing.assert_equal(flow_t.switch_ref().vecs_numpy, fl_op_switched_t))
        fl_op_switched_s_3dim = to_numpy(switch_flow_ref(flow_s.vecs.squeeze(0), 's'), switch_channels=True)
        self.assertEqual(len(fl_op_switched_s_3dim.shape), 3)

    def test_invert_flow(self):
        shape = [10, 20]
        transforms = [['rotation', 5, 10, 30]]
        flow_s = Flow.from_transforms(transforms, shape, 's')
        flow_t = Flow.from_transforms(transforms, shape, 't')
        flow_s.vecs.requires_grad_()
        flow_t.vecs.requires_grad_()
        set_pure_pytorch()
        if_ss = invert_flow(flow_s.vecs, 's')
        if_st = invert_flow(flow_s.vecs, 's', 't')
        if_tt = invert_flow(flow_t.vecs, 't')
        if_ts = invert_flow(flow_t.vecs, 't', 's')
        s_invert = to_numpy(if_ss, switch_channels=True)
        s_invert_t = to_numpy(if_st, switch_channels=True)
        t_invert = to_numpy(if_tt, switch_channels=True)
        t_invert_s = to_numpy(if_ts, switch_channels=True)
        self.assertIsNotNone(if_ss.grad_fn)
        self.assertIsNotNone(if_st.grad_fn)
        self.assertIsNotNone(if_tt.grad_fn)
        self.assertIsNotNone(if_ts.grad_fn)
        self.assertIsNone(np.testing.assert_equal(flow_s.invert().vecs_numpy, s_invert))
        self.assertIsNone(np.testing.assert_equal(flow_s.invert('t').vecs_numpy, s_invert_t))
        self.assertIsNone(np.testing.assert_equal(flow_t.invert().vecs_numpy, t_invert))
        self.assertIsNone(np.testing.assert_equal(flow_t.invert('s').vecs_numpy, t_invert_s))
        s_invert_3dim = to_numpy(invert_flow(flow_s.vecs[0], 's'), switch_channels=True)
        self.assertEqual(len(s_invert_3dim.shape), 3)
        unset_pure_pytorch()
        s_invert = to_numpy(invert_flow(flow_s.vecs, 's'), switch_channels=True)
        s_invert_t = to_numpy(invert_flow(flow_s.vecs_numpy, 's', 't'), switch_channels=True)
        t_invert = to_numpy(invert_flow(flow_t.vecs_numpy, 't'), switch_channels=True)
        t_invert_s = to_numpy(invert_flow(flow_t.vecs, 't', 's'), switch_channels=True)
        self.assertIsNone(np.testing.assert_equal(flow_s.invert().vecs_numpy, s_invert))
        self.assertIsNone(np.testing.assert_equal(flow_s.invert('t').vecs_numpy, s_invert_t))
        self.assertIsNone(np.testing.assert_equal(flow_t.invert().vecs_numpy, t_invert))
        self.assertIsNone(np.testing.assert_equal(flow_t.invert('s').vecs_numpy, t_invert_s))
        s_invert_3dim = to_numpy(invert_flow(flow_s.vecs[0], 's'), switch_channels=True)
        self.assertEqual(len(s_invert_3dim.shape), 3)

    def test_valid_target(self):
        transforms = [['rotation', 0, 0, 45]]
        shape = (7, 7)
        f_s = Flow.from_transforms(transforms, shape, 's')
        f_t = Flow.from_transforms(transforms, shape, 't')
        set_pure_pytorch()
        target_s = to_numpy(valid_target(f_s.vecs_numpy, 's'))
        target_t = to_numpy(valid_target(f_t.vecs, 't'))
        self.assertIsNone(np.testing.assert_equal(to_numpy(f_s.valid_target()), target_s))
        self.assertIsNone(np.testing.assert_equal(to_numpy(f_t.valid_target()), target_t))
        target_t_2dim = to_numpy(valid_target(f_t.vecs[0], 't'))
        self.assertEqual(len(target_t_2dim.shape), 2)
        unset_pure_pytorch()
        target_s = to_numpy(valid_target(f_s.vecs_numpy, 's'))
        target_t = to_numpy(valid_target(f_t.vecs, 't'))
        self.assertIsNone(np.testing.assert_equal(to_numpy(f_s.valid_target()), target_s))
        self.assertIsNone(np.testing.assert_equal(to_numpy(f_t.valid_target()), target_t))
        target_t_2dim = to_numpy(valid_target(f_t.vecs[0], 't'))
        self.assertEqual(len(target_t_2dim.shape), 2)

    def test_valid_source(self):
        transforms = [['rotation', 0, 0, 45]]
        shape = (7, 7)
        f_s = Flow.from_transforms(transforms, shape, 's')
        f_t = Flow.from_transforms(transforms, shape, 't')
        source_s = to_numpy(valid_source(f_s.vecs, 's'))
        source_t = to_numpy(valid_source(f_t.vecs_numpy, 't'))
        self.assertIsNone(np.testing.assert_equal(to_numpy(f_s.valid_source()), source_s))
        self.assertIsNone(np.testing.assert_equal(to_numpy(f_t.valid_source()), source_t))
        source_s_2dim = to_numpy(valid_source(f_s.vecs[0], 's'))
        self.assertEqual(len(source_s_2dim.shape), 2)

    def test_get_flow_padding(self):
        transforms = [['rotation', 0, 0, 45]]
        shape = (7, 7)
        f_s = Flow.from_transforms(transforms, shape, 's')
        f_t = Flow.from_transforms(transforms, shape, 't')
        padding_s = get_flow_padding(to_numpy(f_s.vecs), 's')
        padding_t = get_flow_padding(torch.tensor(f_t.vecs_numpy), 't')
        set_pure_pytorch()
        self.assertIsNone(np.testing.assert_equal(f_s.get_padding(), padding_s))
        self.assertIsNone(np.testing.assert_equal(f_t.get_padding(), padding_t))
        padding_s_list = get_flow_padding(to_numpy(f_s.vecs[0]), 's')
        self.assertEqual(len(padding_s_list), 4)
        unset_pure_pytorch()
        self.assertIsNone(np.testing.assert_equal(f_s.get_padding(), padding_s))
        self.assertIsNone(np.testing.assert_equal(f_t.get_padding(), padding_t))
        padding_s_list = get_flow_padding(to_numpy(f_s.vecs[0]), 's')
        self.assertEqual(len(padding_s_list), 4)

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
        m_s_2d = to_numpy(get_flow_matrix(to_numpy(flow_s.vecs[0]), 's', dof=4, method='ransac'))
        self.assertEqual(len(m_s_2d.shape), 2)

    def test_visualise_flow(self):
        flow = Flow.from_transforms([['translation', 1, 0]], [200, 300])
        self.assertIsNone(np.testing.assert_equal(flow.visualise('bgr', return_tensor=False),
                                                  visualise_flow(flow.vecs, 'bgr', return_tensor=False)))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('rgb', return_tensor=False),
                                                  visualise_flow(flow.vecs_numpy, 'rgb', return_tensor=False)))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', return_tensor=False),
                                                  visualise_flow(to_numpy(flow.vecs), 'hsv', return_tensor=False)))
        v = visualise_flow(to_numpy(flow.vecs[0]), 'hsv', return_tensor=False)
        self.assertEqual(len(v.shape), 3)

    def test_visualise_flow_arrows(self):
        for ref in ['s', 't']:
            flow = Flow.from_transforms([['rotation', 10, 10, 30]], [20, 20], ref)
            self.assertIsNone(np.testing.assert_equal(flow.visualise_arrows(return_tensor=False),
                                                      visualise_flow_arrows(flow.vecs, ref, return_tensor=False)))
        a = visualise_flow_arrows(flow.vecs[0], ref, return_tensor=False)
        self.assertEqual(len(a.shape), 3)

    def test_batch_flows(self):
        shape1 = [10, 50]
        shape2 = [20, 50]
        transforms = [
            ['rotation', 255.5, 255.5, -30],
            ['scaling', 100, 100, 0.8],
        ]
        f1 = Flow.from_transforms(transforms[0:1], shape1, 's')
        f2 = Flow.from_transforms(transforms[1:2], shape1, 's')
        f3 = Flow.from_transforms(transforms[1:2], shape1, 's')
        f1.vecs.requires_grad_()
        f2.vecs.requires_grad_()
        f_batched1 = batch_flows((f1, f2))
        self.assertIsNotNone(f_batched1.vecs.grad_fn)
        f_batched2 = batch_flows((f_batched1, f3))
        self.assertIsNotNone(f_batched2.vecs.grad_fn)
        # Check the batched dimensions fit
        self.assertEqual(f_batched1.shape[0], f1.shape[0] + f2.shape[0])
        self.assertEqual(f_batched2.shape[0], f_batched1.shape[0] + f3.shape[0])
        # Check the contents are still the same
        self.assertIsNone(np.testing.assert_equal(f_batched2.vecs_numpy[0], f1.vecs_numpy[0]))
        self.assertIsNone(np.testing.assert_equal(f_batched2.vecs_numpy[2], f3.vecs_numpy[0]))
        self.assertIsNone(np.testing.assert_equal(f_batched2.mask_numpy[1], f2.mask_numpy[0]))
        # Check the validity checks work
        f1s2 = Flow.from_transforms(transforms[0:1], shape2, 's')
        f1t = Flow.from_transforms(transforms[0:1], shape2, 't')
        with self.assertRaises(TypeError):  # flows arg not a list or tuple
            batch_flows('test')
        with self.assertRaises(TypeError):  # flows arg not all flow objects
            batch_flows((f1, 'test'))
        with self.assertRaises(ValueError):  # flows not all of the same shape
            batch_flows((f1, f1s2))
        with self.assertRaises(ValueError):  # flows not all with same reference
            batch_flows((f1, f1t))
        if torch.cuda.is_available():
            with self.assertRaises(ValueError):  # flows not all of the same device
                batch_flows((f1.to_device('cpu'), f1.to_device('cuda')))


if __name__ == '__main__':
    unittest.main()
