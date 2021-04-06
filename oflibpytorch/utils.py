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

import math
import torch
import numpy as np
from typing import Any, Union


DEFAULT_THRESHOLD = 1e-3


def get_valid_ref(ref: Any) -> str:
    """Checks flow reference input for validity

    :param ref: Flow reference to be checked
    :return: Valid flow reference, either 't' or 's'
    """

    if ref is None:
        ref = 't'
    else:
        if not isinstance(ref, str):
            raise TypeError("Error setting flow reference: Input is not a string")
        if ref not in ['s', 't']:
            raise ValueError("Error setting flow reference: Input is not 's' or 't', but {}".format(ref))
    return ref


def get_valid_device(device: Any) -> str:
    """Checks tensor device input for validity

    :param device: Tensor device to be checked
    :return: Valid tensor device, either 'cpu' or 'cuda'
    """
    if device is None:
        device = 'cpu'
    else:
        if device not in ['cpu', 'cuda']:
            raise ValueError("Error setting tensor device: Input is not 'cpu' or 'cuda', but {}".format(device))
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError("Error setting tensor device: Input is 'cuda', but cuda is not available")
    return device


def get_valid_padding(padding: Any, error_string: str = None) -> list:
    """Checks padding input for validity

    :param padding: Padding to be checked, should be a list of length 4 of positive integers
    :param error_string: Optional string to be added before the error message
    :return: valid padding list, if indeed valid
    """

    error_string = '' if error_string is None else error_string
    if not isinstance(padding, list):
        raise TypeError(error_string + "Padding needs to be a list [top, bot, left, right]")
    if len(padding) != 4:
        raise ValueError(error_string + "Padding list needs to be a list of length 4 [top, bot, left, right]")
    if not all(isinstance(item, int) for item in padding):
        raise ValueError(error_string + "Padding list [top, bot, left, right] items need to be integers")
    if not all(item > 0 for item in padding):
        raise ValueError(error_string + "Padding list [top, bot, left, right] items need to be 0 or larger")
    return padding


def validate_shape(shape: Any) -> Union[tuple, list]:
    if not isinstance(shape, (list, tuple)):
        raise TypeError("Error creating flow from matrix: Dims need to be a list or a tuple")
    if len(shape) != 2:
        raise ValueError("Error creating flow from matrix: Dims need to be a list or a tuple of length 2")
    if any((item <= 0 or not isinstance(item, int)) for item in shape):
        raise ValueError("Error creating flow from matrix: Dims need to be a list or a tuple of integers above zero")


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Tensor to numpy, calls .cpu() if necessary"""
    with torch.no_grad():
        if tensor.device.type == 'cuda':
            tensor = tensor.cpu()
        else:
            tensor = tensor.detach()
        return tensor.numpy()


def flow_from_matrix(matrix: torch.Tensor, shape: Union[list, tuple]) -> torch.Tensor:
    """Flow calculated from a transformation matrix

    NOTE: This corresponds to a flow with reference 's': based on meshgrid in image 1, warped to image 2, flow vectors
      at each meshgrid point in image 1 corresponding to (warped end points in image 2 - start points in image 1)

    :param matrix: Transformation matrix, torch tensor of shape 3-3
    :param shape: List or tuple [H, W] containing required size of the flow field
    :return: Flow field according to cv2 standards, torch tensor 2-H-W
    """

    # Make default vector field and populate it with homogeneous coordinates
    h, w = shape
    device = matrix.device
    ones = torch.ones(shape).to(device)
    grid_x, grid_y = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    default_vec_hom = torch.stack((grid_y.to(torch.float).to(device),
                                   grid_x.to(torch.float).to(device),
                                   ones), dim=-1)

    # Calculate the flow from the difference of the transformed default vectors, and the original default vector field
    transformed_vec_hom = torch.matmul(matrix.to(torch.float), default_vec_hom.unsqueeze(-1)).squeeze(-1)
    transformed_vec = transformed_vec_hom[..., 0:2] / transformed_vec_hom[..., 2:3]
    transformed_vec = (transformed_vec - default_vec_hom[..., 0:2]).unsqueeze(0).transpose(0, -1).squeeze(-1)
    return transformed_vec


def matrix_from_transforms(transform_list: list) -> torch.Tensor:
    """Calculates a transformation matrix from a given list of transforms

    :param transform_list: List of transforms to be turned into a flow field, where each transform is expressed as
        a list of [transform name, transform value 1, ... , transform value n]. Supported options:
            ['translation', horizontal shift in px, vertical shift in px]
            ['rotation', horizontal centre in px, vertical centre in px, angle in degrees, counter-clockwise]
            ['scaling', horizontal centre in px, vertical centre in px, scaling fraction]
    :return: Transformation matrix as torch tensor of shape 3-3
    """

    matrix = torch.eye(3)
    for transform in reversed(transform_list):
        matrix = matrix @ matrix_from_transform(transform[0], transform[1:])
    return matrix


def matrix_from_transform(transform: str, values: list) -> torch.Tensor:
    """Calculates a transformation matrix from given transform types and values

    :param transform: Transform type. Options: 'translation', 'rotation', 'scaling'
    :param values: Transform values as list. Options:
        For 'translation':  [<horizontal shift in px>, <vertical shift in px>]
        For 'rotation':     [<horizontal centre in px>, <vertical centre in px>, <angle in degrees, counter-clockwise>]
        For 'scaling':      [<horizontal centre in px>, <vertical centre in px>, <scaling fraction>]
    :return: Transformation matrix as torch tensor of shape 3-3
    """

    matrix = torch.eye(3)
    if transform == 'translation':  # translate: value is a list of [horizontal movement, vertical movement]
        matrix[0, 2] = values[0]
        matrix[1, 2] = values[1]
    if transform == 'scaling':  # zoom: value is a list of [horizontal coord, vertical coord, scaling]
        translation_matrix_1 = matrix_from_transform('translation', [-values[0], -values[1]])
        translation_matrix_2 = matrix_from_transform('translation', values[:2])
        matrix[0, 0] = values[2]
        matrix[1, 1] = values[2]
        matrix = translation_matrix_2 @ matrix @ translation_matrix_1
    if transform == 'rotation':  # rotate: value is a list of [horizontal coord, vertical coord, rotation in degrees]
        rot = math.radians(values[2])
        translation_matrix_1 = matrix_from_transform('translation', [-values[0], -values[1]])
        translation_matrix_2 = matrix_from_transform('translation', values[:2])
        matrix[0:2, 0:2] = torch.tensor([[math.cos(rot), math.sin(rot)], [-math.sin(rot), math.cos(rot)]])
        # NOTE: diff from usual signs in rot matrix [[+, -], [+, +]] results from 'y' axis pointing down instead of up
        matrix = translation_matrix_2 @ matrix @ translation_matrix_1
    return matrix
