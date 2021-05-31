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

import cv2
import numpy as np
from typing import Union
import torch
from .flow_class import Flow
from .utils import validate_shape, to_numpy
from scipy.interpolate import griddata


FlowAlias = 'Flow'


def visualise_definition(
    mode: str,
    shape: Union[list, tuple] = None,
    insert_text: bool = None,
    return_tensor: bool = None
) -> Union[np.ndarray, torch.Tensor]:
    """Return an image that shows the definition of the flow visualisation.

    :param mode: Desired output colour space: ``rgb``, ``bgr``, or ``hsv``
    :param shape: List or tuple of shape :math:`(2)` containing the desired image shape as values ``(H, W)``. Defaults
        to (601, 601) - do not change if you leave `insert_text` as ``True`` as otherwise the text will appear in the
        wrong location
    :param insert_text: Boolean determining whether explanatory text is put on the image (using :func:`cv2.putText`),
        defaults to ``True``
    :param return_tensor: Boolean determining whether the result is returned as a tensor. Note that the result is
        originally a numpy array. Defaults to ``True``
    :return: Numpy array of shape :math:`(H, W, 3)` or torch tensor of shape :math:`(3, H, W)`, type ``uint8``,
        showing the colour definition of the flow visualisation
    """

    # Default arguments and input validation
    shape = [601, 601] if shape is None else shape
    validate_shape(shape)
    insert_text = True if insert_text is None else insert_text
    if not isinstance(insert_text, bool):
        raise TypeError("Error visualising the flow definition: Insert_text needs to be a boolean")
    return_tensor = True if return_tensor is None else return_tensor
    if not isinstance(return_tensor, bool):
        raise TypeError("Error visualising flow: Return_tensor needs to be boolean")

    # Creating the flow and getting the flow visualisation
    h, w = shape
    flow = Flow.from_transforms([['scaling', w//2, h//2, 1.1]], shape)

    flow.vecs = (torch.abs(flow.vecs) ** 1.2) * torch.sign(flow.vecs)
    img = flow.visualise(mode, return_tensor=False).astype('f')  # dtype 'f' necessary for cv2.arrowedLine

    # Draw on the flow image
    line_colour = (0, 0, 0)
    font_colour = (0, 0, 0)
    cv2.arrowedLine(img, (2, h // 2 + 1), (w - 6, h // 2 + 1), line_colour, 2, tipLength=0.02)
    cv2.arrowedLine(img, (w // 2 + 1, 2), (w // 2 + 1, h - 6), line_colour, 2, tipLength=0.02)

    # Insert explanatory text if required
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.7
    if insert_text:
        cv2.putText(img, 'flow[..., 0]', (450, 285), font, font_scale, font_colour, 1)
        cv2.putText(img, 'flow[..., 1]', (310, 570), font, font_scale, font_colour, 1)
        cv2.putText(img, '[-, -]', (90, 155), font, 1, font_colour)
        cv2.putText(img, '[-, +]', (90, 470), font, 1, font_colour)
        cv2.putText(img, '[+, -]', (400, 155), font, 1, font_colour)
        cv2.putText(img, '[+, +]', (400, 470), font, 1, font_colour)
    if return_tensor:
        return torch.round(torch.tensor(np.moveaxis(img, -1, 0))).to(torch.uint8)
    else:
        return np.round(img).astype('uint8')


# noinspection PyProtectedMember
def combine_flows(input_1: FlowAlias, input_2: FlowAlias, mode: int, thresholded: bool = None) -> FlowAlias:
    """Function that returns the result of the combination of two flow objects of the same shape :attr:`shape` and
    reference :attr:`ref`

    .. tip::
       All of the flow field combinations in this function rely on some combination of the
       :meth:`~oflibpytorch.Flow.apply`, :meth:`~oflibpytorch.Flow.invert`, and :func:`~oflibpytorch.combine_flows`
       methods, and can be very slow (several seconds) due to calling :func:`scipy.interpolate.griddata` multiple
       times. The table below aids decision-making with regards to which reference a flow field should be provided in
       to obtain the fastest result.

        .. list-table:: Calls to :func:`scipy.interpolate.griddata`
           :header-rows: 1
           :stub-columns: 1
           :widths: 10, 15, 15
           :align: center

           * - `mode`
             - ``ref = 's'``
             - ``ref = 't'``
           * - 1
             - 1
             - 3
           * - 2
             - 1
             - 1
           * - 3
             - 0
             - 0

    All formulas used in this function have been derived from first principles. The base formula is
    :math:`flow_1 ⊕ flow_2 = flow_3`, where :math:`⊕` is a non-commutative flow composition operation. This can be
    visualised with the start / end points of the flows as follows:

    .. code-block::

        S = Start point    S1 = S3 ─────── f3 ────────────┐
        E = End point       │                             │
        f = flow           f1                             v
                            └───> E1 = S2 ── f2 ──> E2 = E3

    The main difficulty in combining flow fields is that it would be incorrect to simply add up or subtract flow vectors
    at one location in the flow field area :math:`H \\times W`. This appears to work given e.g. a translation to the
    right, and a translation downwards: the result will be the linear combination of the two vectors, or a translation
    towards the bottom right. However, looking more closely, it becomes evident that this approach isn't actually
    correct: A pixel that has been moved from `S1` to `E1` by the first flow field `f1` is then moved from that location
    by the flow vector of the flow field `f2` that corresponds to the new pixel location `E1`, *not* the original
    location `S1`. If the flow vectors are the same everywhere in the field, the difference will not be noticeable.
    However, if the flow vectors of `f2` vary throughout the field, such as with a rotation around some point, it will!

    In this case (corresponding to calling :func:`combine_flows(f1, f2, mode=3)<combine_flows>`), and if the flow
    reference :attr:`ref` is ``s`` ("source"), the solution is to first apply the inverse of `f1` to `f2`, essentially
    linking up each location `E1` back to `S1`, and *then* to add up the flow vectors. Analogous observations apply for
    the other permutations of flow combinations and reference :attr:`ref` values.

    .. note::
       This is consistent with the observation that two translations are commutative in their application - the order
       does not matter, and the vectors can simply be added up at every pixel location -, while a translation followed
       by a rotation is not the same as a rotation followed by a translation: adding up vectors at each pixel cannot be
       the correct solution as there wouldn't be a difference based on the order of vector addition.

    :param input_1: First input flow object
    :param input_2: Second input flow object
    :param mode: Integer determining how the input flows are combined, where the number corresponds to the position in
        the formula :math:`flow_1 ⊕ flow_2 = flow_3`:

        - Mode ``1``: `input_1` corresponds to :math:`flow_2`, `input_2` corresponds to :math:`flow_3`, the result will
          be :math:`flow_1`
        - Mode ``2``: `input_1` corresponds to :math:`flow_1`, `input_2` corresponds to :math:`flow_3`, the result will
          be :math:`flow_2`
        - Mode ``3``: `input_1` corresponds to :math:`flow_1`, `input_2` corresponds to :math:`flow_2`, the result will
          be :math:`flow_3`
    :param thresholded: Boolean determining whether flows are thresholded during an internal call to
        :meth:`~oflibpytorch.Flow.is_zero`, defaults to ``False``
    :return: New flow object
    """

    # Check input validity
    if not isinstance(input_1, Flow) or not isinstance(input_2, Flow):
        raise ValueError("Error combining flows: Inputs need to be of type 'Flow'")
    if not input_1.shape == input_2.shape:
        raise ValueError("Error combining flows: Flow field inputs need to have the same shape")
    if not input_1.ref == input_2.ref:
        raise ValueError("Error combining flows: Flow fields need to have the same reference")
    if mode not in [1, 2, 3]:
        raise ValueError("Error combining flows: Mode needs to be 1, 2 or 3")
    thresholded = False if thresholded is None else thresholded
    if not isinstance(thresholded, bool):
        raise TypeError("Error combining flows: Thresholded needs to be a boolean")

    # Check if one input is zero, return early if so
    if input_1.is_zero(thresholded=thresholded):
        # if mode == 1:  # Flows are in order (desired_result, input_1=0, input_2)
        #     return input_2
        # elif mode == 2:  # Flows are in order (input_1=0, desired_result, input_2)
        #     return input_2
        # elif mode == 3:  # Flows are in order (input_1=0, input_2, desired_result)
        #     return input_2
        # Above code simplifies to:
        return input_2
    elif input_2.is_zero(thresholded=thresholded):
        if mode == 1:  # Flows are in order (desired_result, input_1, input_2=0)
            return input_1.invert()
        elif mode == 2:  # Flows are in order (input_1, desired_result, input_2=0)
            return input_1.invert()
        elif mode == 3:  # Flows are in order (input_1, input_2=0, desired_result)
            return input_1

    result = None
    if mode == 1:  # Flows are in order (desired_result, input_1, input_2)
        if input_1._ref == input_2._ref == 's':
            # Explanation: f1 is (f3 minus f2), when S2 is moved to S3, achieved by applying f2 to move S2 to E3,
            # then inverted(f3) to move from E3 to S3.
            # F1_s = F2_s - combine(F2_s, F3_s^-1_s, 3){F2_s}
            # result = input_2 - combine_flows(input_1, input_2.invert(), mode=3).apply(input_1)
            #
            # Alternative: this should take an equivalent amount of time (same number of griddata calls), but is
            # slightly faster in tests
            # F1_s = F2_s - combine(F2_s-as-t, F3_s^-1_t, 3){F2_s}
            # result = input_2 - combine_flows(input_1.switch_ref(), input_2.invert('t'), mode=3).apply(input_1)
            # To avoid call to combine_flows and associated overhead, do the corresponding operations directly:
            input_2_inv_t = input_2.invert('t')
            result = input_2 - (input_2_inv_t + input_2_inv_t.apply(input_1.switch_ref())).apply(input_1)

        elif input_1._ref == input_2._ref == 't':
            # Explanation: currently no native implementation to ref 't', so just "translated" from ref 's'
            # F1_t = (F2_t-as-s - combine(F2_t, F3_t^-1_t, 3){F2_t-as-s})_as-t
            # result = input_2.switch_ref() - combine_flows(input_1,
            #                                               input_2.invert(), mode=3).apply(input_1.switch_ref())
            # result = result.switch_ref()
            #
            # Alternative: saves one call to griddata. However, test shows barely a difference - not sure as to reason
            # F1_t = (F2_t-as-s - combine(F2_t-as-s, F3_t^-1_s, 3){F2_t-as-s})_as-t
            # input_1_s = input_1.switch_ref()
            # result = input_2.switch_ref() - combine_flows(input_1_s, input_2.invert('s'), mode=3).apply(input_1_s)
            # result = result.switch_ref()
            # To avoid call to combine_flows and associated overhead, do the corresponding operations directly:
            input_1_s = input_1.switch_ref()
            result = input_2.switch_ref() - \
                (input_1_s + input_1_s.invert(ref='t').apply(input_2.invert('s'))).apply(input_1_s)
            result = result.switch_ref()
    elif mode == 2:  # Flows are in order (input_1, desired_result, input_2)
        if input_1._ref == input_2._ref == 's':
            # Explanation: f2 is (f3 minus f1), when S1 = S3 is moved to S2, achieved by applying f1
            # F2_s = F1_s{F3_s - F1_s}
            result = input_1.apply(input_2 - input_1)
        elif input_1._ref == input_2._ref == 't':
            # Strictly "translated" version from the ref 's' case:
            # F2_t = F1_t{F3_t-as-s - F1_t-as-s}_as-t)
            # result = (input_1.apply(input_2.switch_ref() - input_1.switch_ref())).switch_ref()

            # Improved version cutting down on operational complexity
            # F3 - F1, where F1 has been resampled to the source positions of F3.
            coord_1 = -input_1.vecs_numpy
            coord_1[:, :, 0] += np.arange(coord_1.shape[1])
            coord_1[:, :, 1] += np.arange(coord_1.shape[0])[:, np.newaxis]
            coord_1_flat = np.reshape(coord_1, (-1, 2))
            vecs_with_mask = np.concatenate((input_1.vecs_numpy, input_1.mask_numpy[..., np.newaxis]), axis=-1)
            vals_flat = np.reshape(vecs_with_mask, (-1, 3))
            coord_3 = -input_2.vecs_numpy
            coord_3[:, :, 0] += np.arange(coord_3.shape[1])
            coord_3[:, :, 1] += np.arange(coord_3.shape[0])[:, np.newaxis]
            vals_resampled = griddata(coord_1_flat, vals_flat,
                                      (coord_3[..., 0], coord_3[..., 1]),
                                      method='linear', fill_value=0)
            result = input_2 - Flow(vals_resampled[..., :-1], 't', vals_resampled[..., -1] > .99)
    elif mode == 3:  # Flows are in order (input_1, input_2, desired_result)
        if input_1._ref == input_2._ref == 's':
            # Explanation: f3 is (f1 plus f2), when S2 is moved to S1, achieved by applying inverted(f1)
            # F3_s = F1_s + (F1_s)^-1_t{F2_s}
            # Note: instead of inverting the ref-s flow field to a ref-s flow field (slow) which is then applied to the
            #   other flow field (also slow), it is inverted to a ref-t flow field (fast) which is then also much faster
            #   to apply to the other flow field.
            result = input_1 + input_1.invert(ref='t').apply(input_2)
        elif input_1._ref == input_2._ref == 't':
            # Explanation: f3 is (f2 plus f1), with f1 pulled towards the f2 grid by applying f2 to f1.
            # F3_t = F2_t + F2_t{F1_t}
            result = input_2 + input_2.apply(input_1)

    return result
