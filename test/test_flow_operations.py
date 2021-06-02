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
from oflibpytorch.flow_class import Flow
from oflibpytorch.flow_operations import combine_flows, visualise_definition


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
            f1_actual = combine_flows(f2, f3, 1)
            # f1.show(500, show_mask=True, show_mask_borders=True)
            # f1_actual.show(show_mask=True, show_mask_borders=True)
            self.assertIsInstance(f1_actual, Flow)
            self.assertEqual(f1_actual.ref, ref)
            comb_mask = f1_actual.mask & f1.mask
            self.assertIsNone(np.testing.assert_allclose(f1_actual.vecs_numpy[comb_mask], f1.vecs_numpy[comb_mask],
                                                         atol=5e-2))

            # Mode 2
            f2_actual = combine_flows(f1, f3, 2)
            # f2.show(500, show_mask=True, show_mask_borders=True)
            # f2_actual.show(show_mask=True, show_mask_borders=True)
            self.assertIsInstance(f2_actual, Flow)
            self.assertEqual(f2_actual.ref, ref)
            comb_mask = f2_actual.mask & f2.mask
            self.assertIsNone(np.testing.assert_allclose(f2_actual.vecs_numpy[comb_mask], f2.vecs_numpy[comb_mask],
                                                         atol=5e-2))

            # Mode 3
            f3_actual = combine_flows(f1, f2, 3)
            # f3.show(500, show_mask=True, show_mask_borders=True)
            # f3_actual.show(show_mask=True, show_mask_borders=True)
            self.assertIsInstance(f3_actual, Flow)
            self.assertEqual(f3_actual.ref, ref)
            comb_mask = f3_actual.mask & f3.mask
            self.assertIsNone(np.testing.assert_allclose(f3_actual.vecs_numpy[comb_mask], f3.vecs_numpy[comb_mask],
                                                         atol=5e-2))

        with self.assertRaises(TypeError):  # wrong type
            combine_flows(torch.ones(shape), f2, 3)
        with self.assertRaises(ValueError):  # wrong shape
            combine_flows(f1.resize(.5), f2, 3)
        with self.assertRaises(ValueError):  # wrong ref
            combine_flows(Flow.from_transforms(transforms[0:1], shape, 's'),
                          Flow.from_transforms(transforms[1:2], shape, 't'), 3)
        if torch.cuda.is_available():  # The following test assumes 'cuda' device is available
            with self.assertRaises(ValueError):  # wrong device
                combine_flows(f1.to_device('cuda'), f2, 3)
        with self.assertRaises(ValueError):  # wrong mode
            combine_flows(f1, f2, 0)
        with self.assertRaises(TypeError):  # wrong thresholded
            combine_flows(f1, f2, 1, thresholded='test')


if __name__ == '__main__':
    unittest.main()
