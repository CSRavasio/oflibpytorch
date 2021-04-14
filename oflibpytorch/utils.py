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
import torch.nn.functional as f
from scipy.interpolate import griddata
import numpy as np
from typing import Any, Union


DEFAULT_THRESHOLD = 1e-3


def get_valid_vecs(vecs: Any, desired_shape: Union[tuple, list] = None, error_string: str = None) -> torch.Tensor:
    """Checks array or tensor input for validity and returns 2-H-W tensor for use as flow vectors

    :param vecs: Valid if numpy array or torch tensor, either shape 2-H-W (assumed first) or H-W-2
    :param desired_shape: List or tuple of (H, W) the input vecs should be compared about. Optional
    :param error_string: Optional string to be added before the error message if input is invalid. Optional
    :return: Tensor valid for flow vectors, shape 2-H-W
    """

    error_string = '' if error_string is None else error_string

    # Check type and dimensions
    if not isinstance(vecs, (np.ndarray, torch.Tensor)):
        raise TypeError(error_string + "Input is not a numpy array or a torch tensor")
    if vecs.ndim != 3:
        raise ValueError(error_string + "Input is not 3-dimensional")

    # Transform to tensor if necessary
    if isinstance(vecs, np.ndarray):
        vecs = torch.tensor(vecs, dtype=torch.float, device='cpu')

    # Check channels, transpose if necessary
    if vecs.shape[0] != 2:  # Check if input shape can be interpreted as 2-H-W
        if vecs.shape[2] == 2:  # Input shape is H-W-2
            vecs = vecs.unsqueeze(0).transpose(0, -1).squeeze(-1)
        else:  # Input shape is neither H-W-2 nor 2-H-W
            raise ValueError(error_string + "Input needs to be shape H-W-2 or 2-H-W")

    # Check shape if necessary
    if desired_shape is not None:
        if vecs.shape[1] != desired_shape[0] or vecs.shape[2] != desired_shape[1]:
            raise ValueError(error_string + "Input shape H or W does not match the desired shape")

    # Check for invalid values
    if not torch.isfinite(vecs).all():
        raise ValueError(error_string + "Input contains NaN, Inf or -Inf values")

    return vecs


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
    :param error_string: Optional string to be added before the error message, if padding is invalid
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


def reverse_transform_values(transform_list: list) -> list:
    """Changes the values for all transforms in the list so the result is equal to the reverse transform

    :param transform_list: List of transforms to be turned into a flow field, where each transform is expressed as
        a list of [transform name, transform value 1, ... , transform value n]. Supported options:
            ['translation', horizontal shift in px, vertical shift in px]
            ['rotation', horizontal centre in px, vertical centre in px, angle in degrees, counter-clockwise]
            ['scaling', horizontal centre in px, vertical centre in px, scaling fraction]
    :return: List of reversed transforms
    """
    reversed_transform_list = []
    for value_list in transform_list:
        transform, values = value_list[0], value_list[1:]
        if transform == 'translation':  # translate: value is a list of [horizontal movement, vertical movement]
            reversed_transform_list.append([transform, -values[0], -values[1]])
        if transform == 'scaling':  # zoom: value is a list of [horizontal coord, vertical coord, scaling]
            reversed_transform_list.append([transform, values[0], values[1], 1/values[2]])
        if transform == 'rotation':  # rotate: value is a list of [horizontal coord, vertical coord, rotation in deg]
            reversed_transform_list.append([transform, values[0], values[1], -values[2]])
    return reversed_transform_list


def normalise_coords(coords: torch.Tensor, shape: Union[tuple, list]) -> torch.Tensor:
    """Normalise actual coordinates to [-1, 1]

    Coordinate locations start "mid-pixel" and end "mid-pixel" (pixel box model):
        Pixels | 0 | 1 | 2 |
                 |   |   |
          Grid  -1   0   1

    :param coords: tensor of any shape, ending in a dim=2, which is (x, y) = [hor, ver]
    :param shape: list of flow (or image) size [ver, hor]
    :return: Normalised coordinates
    """
    normalised_coords = 2. * coords
    normalised_coords[..., 0] /= (shape[1] - 1)  # points[..., 0] is x, which is horizontal, so shape[1]
    normalised_coords[..., 1] /= (shape[0] - 1)  # points[..., 1] is y, which is vertical, so shape[0]
    normalised_coords -= 1
    return normalised_coords


def apply_flow(flow: torch.Tensor, target: torch.Tensor, ref: str = None) -> torch.Tensor:
    """Warps target according to flow of given reference

    :param flow: Torch tensor 2-H-W containing the flow vectors in cv2 convention (1st channel hor, 2nd channel ver)
    :param target: Torch tensor H-W, C-H-W, or N-C-H-W containing the content to be warped
    :param ref: Reference of the flow, 't' or 's'. Defaults to 't'
    :return: Torch tensor of the same shape as the target, with the content warped by the flow
    """

    # Check if all flow vectors are almost zero
    if torch.all(torch.norm(flow, dim=0) <= DEFAULT_THRESHOLD):  # If the flow field is actually 0 or very close
        return target

    # Set up
    ref = get_valid_ref(ref)
    device = flow.device.type
    h, w = flow.shape[1:]

    # Prepare target dtype, device, and shape
    target_dtype = target.dtype
    target = target.to(torch.float)
    if target.device != flow.device:
        target = target.to(flow.device)
    target_dims = target.ndim
    for _ in range(4 - target_dims):
        # Get to ndim = 4
        target = target.unsqueeze(0)

    # Warp target
    if ref == 't':
        # Prepare grid
        grid_x, grid_y = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid = torch.stack((grid_y, grid_x), dim=-1).to(torch.float).to(device)
        field = normalise_coords(grid.unsqueeze(0) - flow.unsqueeze(-1).transpose(-1, 0), (h, w))
        if target.shape[0] > 1:  # target wasn't just unsqueezed, but has a true N dimension
            field = field.repeat(target.shape[0], 1, 1, 1)
        # noinspection PyArgumentList
        result = f.grid_sample(target, field, align_corners=True).squeeze(0)
        # Comment on grid_sample: given grid_sample(input, grid), the input is sampled at grid points.
        #   For this to work:
        #   - input is shape NCHW (containing data vals in C)
        #   - grid is shape NHW2, where 2 is [x, y], each in limits [-1, 1]
        #   - grid locations by default start "mid-pixel", end "mid-pixel" (box model): Pixels | 0 | 1 | 2 |
        #                                                                                        |   |   |
        #                                                                                 Grid  -1   0   1
        #   - in practice, this box model leads to artefacts around the corners (might be fixable), setting align_corner
        #     to True fixes this.
        #   - x and y are spatially defined as follows, same as the cv2 convention (e.g. Farnebäck flow)
        #       -1    0    1
        #     -1 +---------+--> x
        #        |         |
        #      0 |  image  |
        #        |         |
        #      1 +---------+
        #        v
        #        y
    else:  # ref == 's'
        field = np.moveaxis(to_numpy(flow), 0, -1).astype('float32')
        x, y = np.mgrid[:field.shape[0], :field.shape[1]]
        positions = np.swapaxes(np.vstack([x.ravel(), y.ravel()]), 0, 1)
        flow_flat = np.reshape(field[..., ::-1], (-1, 2))
        target_np = np.moveaxis(to_numpy(target), 1, -1)
        target_flat = np.reshape(target_np, (target.shape[0], -1, target.shape[1]))
        results = np.copy(target_np)
        for i in range(target_flat.shape[0]):
            pos = positions + flow_flat
            result = griddata(pos, target_flat[i], (x, y), method='linear')
            results[i] = np.nan_to_num(result)
        # Make sure the output is returned with the same dtype as the input, if necessary rounded
        result = torch.tensor(np.moveaxis(results, -1, 1))

    # Reduce target to original shape
    for _ in range(4 - target_dims):
        result = result.squeeze(0)

    # Return target with original dtype, rounding if necessary
    # noinspection PyUnresolvedReferences
    if not target_dtype.is_floating_point:
        result = torch.round(result)
    result = result.to(target_dtype)

    return result
