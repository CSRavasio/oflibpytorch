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
# This file is part of oflibpytorch. It contains functions needed by the methods of the custom flow class in flow_class.

import math
import torch
import torch.nn.functional as f
from scipy.interpolate import griddata
import numpy as np
import cv2
from typing import Any, Union, List


DEFAULT_THRESHOLD = 1e-3


def get_valid_vecs(vecs: Any, desired_shape: Union[tuple, list] = None, error_string: str = None) -> torch.Tensor:
    """Checks array or tensor input for validity and returns N-2-H-W tensor of dtype float for use as flow vectors

    :param vecs: Valid if numpy array or torch tensor, either shape (N-)2-H-W (assumed first) or (N-)H-W-2
    :param desired_shape: List or tuple of ((N, )H, W) the input vecs should be compared about. If no batch dimension
        is given, N is assumed to be 1. Optional
    :param error_string: Optional string to be added before the error message if input is invalid. Optional
    :return: Tensor valid for flow vectors, shape N-2-H-W, dtype float
    """

    error_string = '' if error_string is None else error_string

    # Check type and dimensions
    if not isinstance(vecs, (np.ndarray, torch.Tensor)):
        raise TypeError(error_string + "Input is not a numpy array or a torch tensor")
    ndim = len(vecs.shape)
    if ndim != 3 and ndim != 4:
        raise ValueError(error_string + "Input has {} dimensions, should be 3 or 4".format(ndim))

    # Transform to tensor if necessary
    if isinstance(vecs, np.ndarray):
        vecs = torch.tensor(vecs, dtype=torch.float, device='cpu')

    # Check for invalid values
    if not torch.isfinite(vecs).all():
        raise ValueError(error_string + "Input contains NaN, Inf or -Inf values")

    # Add dimension if necessary
    if ndim == 3:
        vecs = vecs.unsqueeze(0)

    # Check channels, transpose if necessary
    if vecs.shape[1] != 2:  # Check if input shape can be interpreted as N-2-H-W
        if vecs.shape[3] == 2:  # Input shape is N-H-W-2
            vecs = move_axis(vecs, -1, 1)
        else:  # Input shape is neither N-H-W-2 nor N-2-H-W
            raise ValueError(error_string + "Input needs to be shape (N-)H-W-2 or (N-)2-H-W")

    # Check shape if necessary
    if desired_shape is not None:
        d = get_valid_shape(desired_shape)
        if vecs.shape[0] != d[0] or vecs.shape[2] != d[1] or vecs.shape[3] != d[2]:
            raise ValueError(error_string + "Input shape does not match the desired shape")

    return vecs.float()


def get_valid_shape(shape: Any) -> tuple:
    if not isinstance(shape, (list, tuple)):
        raise TypeError("Error creating flow from matrix: Dims need to be a list or a tuple")
    if len(shape) != 2 and len(shape) != 3:
        raise ValueError("Error creating flow from matrix: Dims need to be a list or a tuple of length 2 or 3")
    if any((item <= 0 or not isinstance(item, int)) for item in shape):
        raise ValueError("Error creating flow from matrix: Dims need to be a list or a tuple of integers above zero")
    if len(shape) == 2:
        return (1,) + tuple(shape)
    else:
        return tuple(shape)


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


def get_valid_mask(mask: Any, desired_shape: Union[tuple, list] = None, error_string: str = None) -> torch.Tensor:
    """Checks array or tensor input for validity and returns N-H-W tensor for use as flow mask

    :param mask: Valid if numpy array or torch tensor of shape (N-)H-W
    :param desired_shape: List or tuple of ((N, )H, W) the input vecs should be compared about. Optional
    :param error_string: Optional string to be added before the error message if input is invalid. Optional
    :return: Tensor valid for flow mask, shape N-H-W, dtype float
    """

    error_string = '' if error_string is None else error_string

    # Check type, dimensions, shape
    if not isinstance(mask, (np.ndarray, torch.Tensor)):
        raise TypeError(error_string + "Input is not a numpy array or a torch tensor")
    ndim = len(mask.shape)
    if ndim != 2 and ndim != 3:
        raise ValueError(error_string + "Input has {} dimensions, should be 2 or 3".format(ndim))

    # Transform to tensor if necessary
    if isinstance(mask, np.ndarray):
        mask = torch.tensor(mask)

    # Check for invalid values
    if ((mask != 0) & (mask != 1)).any():
        raise ValueError(error_string + "Values must be 0 or 1")

    # Add dimension if necessary
    if ndim == 2:
        mask = mask.unsqueeze(0)

    # Check shape if necessary
    if desired_shape is not None:
        if mask.shape != get_valid_shape(desired_shape):
            raise ValueError(error_string + "Input shape does not match the desired shape")

    return mask.to(torch.bool)


def get_valid_device(device: Any) -> str:
    """Checks tensor device input for validity, defaults to torch.device('cpu') for input 'None'. 'cuda' inputs without
    an explicit device index default to cuda.current_device()

    :param device: Tensor device to be checked
    :return: Valid torch.device
    """

    if device is None:
        device = torch.device('cpu')
    elif isinstance(device, torch.device):
        pass
    else:
        try:
            device = torch.device(device)
        except RuntimeError:
            raise ValueError("Error setting tensor device: Input needs to be a torch.device, or valid input to "
                             "torch.device(). Instead found {}".format(device))
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError("Error setting tensor device: Input is 'cuda', but cuda is not available")
        if device.index is None:
            device = torch.device(torch.cuda.current_device())
    return device


def get_valid_padding(padding: Any, error_string: str = None) -> list:
    """Checks padding input for validity

    :param padding: Padding to be checked, should be a list of length 4 of positive integers
    :param error_string: Optional string to be added before the error message, if padding is invalid
    :return: valid padding list, if indeed valid
    """

    error_string = '' if error_string is None else error_string
    if not isinstance(padding, (list, tuple)):
        raise TypeError(error_string + "Padding needs to be a list [top, bot, left, right]")
    if len(padding) != 4:
        raise ValueError(error_string + "Padding list needs to be a list of length 4 [top, bot, left, right]")
    if not all(isinstance(item, int) for item in padding):
        raise ValueError(error_string + "Padding list [top, bot, left, right] items need to be integers")
    if not all(item > 0 for item in padding):
        raise ValueError(error_string + "Padding list [top, bot, left, right] items need to be 0 or larger")
    return padding


def move_axis(input_tensor: torch.Tensor, source: int, destination: int) -> torch.Tensor:
    """Helper function to imitate np.moveaxis

    :param input_tensor: Input torch tensor, e.g. N-H-W-C
    :param source: Source position of the dimension to be moved, e.g. -1
    :param destination: Target position of the dimension to be moved, e.g. 1
    :return: Output torch tensor, e.g. N-C-H-W
    """

    source %= input_tensor.dim()
    destination %= input_tensor.dim()
    if source < destination:
        destination += 1  # Otherwise e.g. source = 0, destination = 1 won't give correct result
    elif source > destination:
        source += 1  # Otherwise e.g. source = 1, destination = 0 won't give correct result
    return input_tensor.unsqueeze(destination).transpose(source, destination).squeeze(source)


def to_numpy(tensor: torch.Tensor, switch_channels: bool = None) -> np.ndarray:
    """Tensor to numpy, calls .cpu() if necessary

    :param tensor: Input tensor
    :param switch_channels: Boolean determining whether the channels are moved from the second to the last dimension,
        assuming the input is of shape :math:`(N, C, H, W)`, changing it to :math:`(N, H, W, C)`. defaults to ``False``
    :return: Numpy array, with channels switched if required
    """

    switch_channels = False if switch_channels is None else switch_channels
    with torch.no_grad():
        if tensor.device.type == 'cuda':
            tensor = tensor.cpu()
        else:
            tensor = tensor.detach()
        arr = tensor.numpy()
        if switch_channels:
            arr = np.moveaxis(arr, 1, -1)
        return arr


def to_tensor(
    array: np.ndarray,
    switch_channels: str = None,
    device: Union[torch.device, int, str] = None
) -> torch.Tensor:
    """Numpy to tensor

    :param array: Input array
    :param switch_channels: String determining whether the channels are moved from the last to the first dimension
        (if 'single'), or from last to second dimension (if 'batched'). Defaults to ``None`` (no channels moved)
    :param device: Tensor device, either a :class:`torch.device` or a valid input to ``torch.device()``,
            such as a string (``cpu`` or ``cuda``). For a device of type ``cuda``, the device index defaults to
            ``torch.cuda.current_device()``. If the input is ``None``, it defaults to ``torch.device('cpu')``
    :return: Torch tensor, with channels switched if required
    """

    device = get_valid_device(device)
    if switch_channels is not None:
        if switch_channels == 'single':
            array = np.moveaxis(array, -1, 0)
        elif switch_channels == 'batched':
            array = np.moveaxis(array, -1, 1)
    tens = torch.tensor(array).to(device)
    return tens


def show_masked_image(img: Union[torch.Tensor, np.ndarray], mask: Union[torch.Tensor, np.ndarray] = None) -> np.ndarray:
    """Mimics flow.show(), for an input image and a mask

    :param img: Torch tensor of shape :math:`(3, H, W)` or numpy array of shape :math:`(H, W, 3)`, BGR input image
    :param mask: Torch tensor or numpy array of shape :math:`(H, W)`, boolean mask showing the valid area
    :return: Masked image, in BGR colour space
    """

    if isinstance(img, torch.Tensor):
        if len(img.shape) == 4:
            img = to_numpy(img, switch_channels=True)[0]
        else:  # len(img.shape) == 3
            img = to_numpy(img)
            img = np.moveaxis(img, 0, -1)
    if img.shape[-1] == 1:
        img = img[..., 0]
    if mask is None:
        mask = np.ones(img.shape[:2], 'bool')
    elif isinstance(mask, torch.Tensor):
        mask = to_numpy(mask)
    img = np.clip(np.round(img), 0, 255).astype('uint8')
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[np.invert(mask), 2] = hsv[np.invert(mask), 2] / 2
    contours, hierarchy = cv2.findContours((255 * mask).astype('uint8'),
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(hsv, contours, -1, (0, 0, 0), 1)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("Image masked by valid area", bgr)
    cv2.waitKey()
    return bgr


def flow_from_matrix(matrix: torch.Tensor, shape: list) -> torch.Tensor:
    """Flow calculated from a transformation matrix

    NOTE: This corresponds to a flow with reference 's': based on meshgrid in image 1, warped to image 2, flow vectors
      at each meshgrid point in image 1 corresponding to (warped end points in image 2 - start points in image 1)

    :param matrix: Transformation matrix, torch tensor of shape N-3-3
    :param shape: List [N, H, W] containing required size of the flow field
    :return: Flow field according to cv2 standards, torch tensor N-2-H-W
    """

    # Make default vector field and populate it with homogeneous coordinates
    n, h, w = shape
    device = matrix.device
    ones = torch.ones((h, w)).to(device)
    torch_version = globals()['torch'].__version__
    if int(torch_version[0]) == 1 and float(torch_version[2:4]) <= 9:
        grid_x, grid_y = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    else:
        grid_x, grid_y = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')
    default_vec_hom = torch.stack((grid_y.to(torch.float).to(device),
                                   grid_x.to(torch.float).to(device),
                                   ones), dim=-1)
    # default_vec_hom = default_vec_hom.unsqueeze(0).repeat(n, 1, 1, 1)

    # Calculate the flow from the difference of the transformed default vectors, and the original default vector field
    transformed_vec_hom = torch.matmul(matrix.to(torch.float).unsqueeze(1).unsqueeze(1),    # [N, 1, 1, 3, 3]
                                       default_vec_hom.unsqueeze(-1))                       # [   H, W, 3, 1]
    transformed_vec_hom = transformed_vec_hom.squeeze(-1)                                   # [N, H, W, 3]
    transformed_vec = transformed_vec_hom[..., 0:2] / transformed_vec_hom[..., 2:3]         # [N, H, W, 2]
    transformed_vec = move_axis(transformed_vec - default_vec_hom[..., 0:2], -1, 1)         # [N, 2, H, W]
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

    if len(shape) != 2:
        raise ValueError("Error normalising coords: Given shape needs to be list or tuple of length 2")

    normalised_coords = coords.float() * 2
    normalised_coords[..., 0] /= (shape[1] - 1)  # points[..., 0] is x, which is horizontal, so shape[1]
    normalised_coords[..., 1] /= (shape[0] - 1)  # points[..., 1] is y, which is vertical, so shape[0]
    normalised_coords -= 1
    return normalised_coords


def apply_flow(
    flow: Union[np.ndarray, torch.Tensor],
    target: torch.Tensor,
    ref: str,
    mask: Union[np.ndarray, torch.Tensor] = None
) -> torch.Tensor:
    """Uses a given flow to warp a target. The flow reference, if not given, is assumed to be ``t``. Optionally, a mask
    can be passed which (only for flows in ``s`` reference) masks undesired (e.g. undefined or invalid) flow vectors.

    :param flow: Flow field as a numpy array or torch tensor, shape :math:`(2, H, W)`, :math:`(H, W, 2)`,
        :math:`(N, 2, H, W)`, or :math:`(N, H, W, 2)`
    :param target: Torch tensor containing the content to be warped, with shape :math:`(H, W)`, :math:`(C, H, W)`, or
        :math:`(N, C, H, W)`
    :param ref: Reference of the flow, ``t`` or ``s``
    :param mask: Flow mask as numpy array or torch tensor, with shape :math:`(H, W)` or :math:`(N, H, W)`, matching
        the flow field. Only relevant for ``s`` flows. Defaults to ``True`` everywhere
    :return: Torch tensor of the same shape as the target, with the content warped by the flow
    """

    # Input validity check
    ref = get_valid_ref(ref)
    flow = get_valid_vecs(flow, error_string="Error applying flow to a target: ")
    if all(is_zero_flow(flow, thresholded=True)):  # If the flow field is actually 0 or very close
        return target
    if not isinstance(target, torch.Tensor):
        raise TypeError("Error applying flow to a target: Target needs to be a torch tensor")
    if len(target.shape) not in [2, 3, 4]:
        raise ValueError("Error applying flow to a target: Target tensor needs to have shape H-W, C-H-W, or N-C-H-W")
    if target.shape[-2:] != flow.shape[-2:]:
        raise ValueError("Error applying flow to a target: Target height and width needs to match flow field array")
    if mask is not None:
        mask = get_valid_mask(mask, desired_shape=(flow.shape[0],) + flow.shape[2:])

    # Set up
    device = flow.device
    h, w = flow.shape[-2:]

    # Prepare target dtype, device, and shape
    target_dtype = target.dtype
    target = target.to(torch.float)
    if target.device != flow.device:
        target = target.to(flow.device)
    target_dims = target.dim()
    if target_dims == 2:  # shape H-W to 1-1-H-W
        target = target.unsqueeze(0).unsqueeze(0)
    elif target_dims == 3:  # shape C-H-W to 1-C-H-W
        target = target.unsqueeze(0)

    # Determine and check batch dimensions
    if target.shape[0] != flow.shape[0] and target.shape[0] != 1 and flow.shape[0] != 1:
        # batch sizes don't match and are not broadcastable
        raise ValueError("Error applying flow to target: Batch dimensions for flow ({}) and target ({}) don't match"
                         .format(flow.shape[0], target.shape[0]))
    else:
        if target.shape[0] < flow.shape[0]:  # target batch dim smaller than flow, means it is 1, needs to be repeated
            target = target.repeat(flow.shape[0], 1, 1, 1)
        elif flow.shape[0] < target.shape[0]:  # flow batch dim smaller than target, means it is 1, needs to be repeated
            flow = flow.repeat(target.shape[0], 1, 1, 1)
            if mask is not None:
                mask = mask.repeat(target.shape[0], 1, 1)
        # Now batch dims either N and 1->N, 1->N and N, 1 and 1, or N and N

    # Get flow in shape needed
    flow = flow.permute(0, 2, 3, 1)  # N-2-H-W to N-H-W-2

    # Warp target
    if ref == 't':
        # Prepare grid
        torch_version = globals()['torch'].__version__
        if int(torch_version[0]) == 1 and float(torch_version[2:4]) <= 9:
            grid_x, grid_y = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        else:
            grid_x, grid_y = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')
        grid = torch.stack((grid_y, grid_x), dim=-1).to(torch.float).to(device)
        field = normalise_coords(grid.unsqueeze(0) - flow, (h, w))
        torch_version = globals()['torch'].__version__
        if int(torch_version[0]) == 1 and float(torch_version[2:4]) <= 3:
            result = f.grid_sample(target, field)
        else:
            result = f.grid_sample(target, field, align_corners=True)
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
        # Get the positions of the unstructured points with known values
        field = to_numpy(flow).astype('float32')
        flow_flat = np.reshape(field[..., ::-1], (field.shape[0], -1, 2))               # N-H*W-2
        x, y = np.mgrid[:h, :w]                                                         # H-W
        positions = np.swapaxes(np.vstack([x.ravel(), y.ravel()]), 0, 1)                # H*W-2
        pos = positions + flow_flat                                                     # N-H*W-2
        # Get the known values themselves
        target_np = np.moveaxis(to_numpy(target), 1, -1)                                # from N-C-H-W to N-H-W-C
        target_flat = np.reshape(target_np, (target.shape[0], -1, target.shape[1]))     # from N-H-W-C to N-H*W-C
        # Perform interpolation of regular grid from unstructured data
        results = np.copy(target_np)                                                    # N-H-W-C
        if mask is not None:
            mask = to_numpy(mask.view(mask.shape[0], -1))
        for i in range(target_flat.shape[0]):  # Perform griddata for each batch member
            # Mask points, if required
            if mask is not None:
                result = griddata(pos[i][mask[i]], target_flat[i][mask[i]], (x, y), method='linear')
                results[i] = np.nan_to_num(result)
            else:
                result = griddata(pos[i], target_flat[i], (x, y), method='linear')
                results[i] = np.nan_to_num(result)
        result = torch.tensor(np.moveaxis(results, -1, 1)).to(flow.device)              # N-H-W-C to N-C-H-W

    # Reduce target to original shape, as far as possible
    if result.shape[0] == 1:
        if target_dims == 2:  # shape 1-1-H-W to H-W
            result = result.squeeze(0).squeeze(0)
        elif target_dims == 3:  # shape 1-C-H-W to C-H-W
            result = result.squeeze(0)
    else:
        if target_dims == 2:  # shape N-1-H-W to N-H-W
            result = result.squeeze(1)

    # Return target with original dtype, rounding if necessary
    # noinspection PyUnresolvedReferences
    if not target_dtype.is_floating_point:
        result = torch.round(result)
        if target_dtype == torch.uint8:
            result = torch.clamp(result, 0, 255)
    result = result.to(target_dtype)

    return result


def threshold_vectors(vecs: torch.Tensor, threshold: Union[float, int] = None, use_mag: bool = None) -> torch.Tensor:
    """Sets all flow vectors with a magnitude below threshold to zero

    :param vecs: Input flow torch tensor, shape N-2-H-W
    :param threshold: Threshold value as float or int, defaults to DEFAULT_THRESHOLD (top of file)
    :param use_mag: Thresholding uses the vector magnitude instead of simply x and y values. Defaults to False
    :return: Flow tensor with vector magnitudes below the threshold set to 0
    """

    threshold = DEFAULT_THRESHOLD if threshold is None else threshold
    use_mag = False if use_mag is None else use_mag

    f = vecs.clone()
    if use_mag:
        mags = torch.norm(vecs, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        f[mags < threshold] = 0
    else:
        f[(vecs < threshold) & (vecs > -threshold)] = 0
    return f


def from_matrix(
    matrix: Union[np.ndarray, torch.Tensor],
    shape: Union[list, tuple],
    ref: str = None,
    matrix_is_inverse: bool = None
) -> torch.Tensor:
    """Flow vectors calculated from a transformation matrix input

    :param matrix: Transformation matrix to be turned into a flow field, as numpy array or torch tensor of
        shape :math:`(3, 3)` or :math:`(N, 3, 3)`
    :param shape: List or tuple of the shape :math:`(H, W)` of the flow field
    :param ref: Flow reference, string of value ``t`` ("target") or ``s`` ("source"). Defaults to ``t``
    :param matrix_is_inverse: Boolean determining whether the given matrix is already the inverse of the desired
        transformation. Is useful for flow with reference ``t`` to avoid calculation of the pseudo-inverse, but
        will throw a ``ValueError`` if used for flow with reference ``s`` to avoid accidental usage.
        Defaults to ``False``
    :return: Flow vectors of shape :math:`(N, 2, H, W)`
    """

    # Check shape validity
    shape = get_valid_shape(shape)
    if shape[0] != 1:
        raise ValueError("Error creating flow from matrix: Given shape has batch dimension larger than 1")
    # Check matrix validity
    if not isinstance(matrix, (np.ndarray, torch.Tensor)):
        raise TypeError("Error creating flow from matrix: Matrix needs to be a numpy array or a torch tensor")
    if isinstance(matrix, np.ndarray):
        matrix = torch.tensor(matrix)
    ndim = len(matrix.shape)
    if ndim != 2 and ndim != 3:
        raise ValueError("Error creating flow from matrix: Matrix has {} dimensions, should be 2 or 3".format(ndim))
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("Error creating flow from matrix: Matrix needs to be of shape (3, 3)")
    if len(matrix.shape) == 2:
        matrix = matrix.unsqueeze(0)
    matrix = matrix.to(torch.float)
    # Get valid ref
    ref = get_valid_ref(ref)
    # Get valid inverse flag
    matrix_is_inverse = False if matrix_is_inverse is None else matrix_is_inverse
    if not isinstance(matrix_is_inverse, bool):
        raise TypeError("Error creating flow from matrix: Matrix_is_inverse needs to be None or a Boolean")

    if ref == 's':
        # Coordinates correspond to the meshgrid of the original ('s'ource) image. They are transformed according
        # to the transformation matrix. The start points are subtracted from the end points to yield flow vectors.
        if matrix_is_inverse:
            raise ValueError("Error creating flow from matrix: Matrix_is_inverse cannot be True when ref is 's'")
        flow_vectors = flow_from_matrix(matrix, shape)
    else:  # ref == 't':
        # Coordinates correspond to the meshgrid of the warped ('t'arget) image. They are inversely transformed
        # according to the transformation matrix. The end points, which correspond to the flow origin for the
        # meshgrid in the warped image, are subtracted from the start points to yield flow vectors.
        if not matrix_is_inverse:
            matrix = torch.pinverse(matrix)
        flow_vectors = -flow_from_matrix(matrix, shape)
    return flow_vectors


def from_transforms(
    transform_list: list,
    shape: Union[list, tuple],
    ref: str = None,
) -> torch.Tensor:
    """Flow vectors calculated from a list of transforms

    :param transform_list: List of transforms to be turned into a flow field, where each transform is expressed as
        a list of [``transform name``, ``transform value 1``, ... , ``transform value n``]. Supported options:

        - Transform ``translation``, with values ``horizontal shift in px``, ``vertical shift in px``
        - Transform ``rotation``, with values ``horizontal centre in px``, ``vertical centre in px``,
          ``angle in degrees, counter-clockwise``
        - Transform ``scaling``, with values ``horizontal centre in px``, ``vertical centre in px``,
          ``scaling fraction``
    :param shape: List or tuple of the shape :math:`(H, W)` of the flow field
    :param ref: Flow reference, string of value ``t`` ("target") or ``s`` ("source"). Defaults to ``t``
    :return: Flow vectors of shape :math:`(N, 2, H, W)`
    """

    # Get valid reference
    ref = get_valid_ref(ref)
    # Check transform_list validity
    if not isinstance(transform_list, list):
        raise TypeError("Error creating flow from transforms: Transform_list needs to be a list")
    if not all(isinstance(item, list) for item in transform_list):
        raise TypeError("Error creating flow from transforms: Transform_list needs to be a list of lists")
    if not all(len(item) > 1 for item in transform_list):
        raise ValueError("Error creating flow from transforms: Invalid transforms passed")
    for t in transform_list:
        if t[0] == 'translation':
            if not len(t) == 3:
                raise ValueError("Error creating flow from transforms: Not enough transform values passed for "
                                 "'translation' - expected 2, got {}".format(len(t) - 1))
        elif t[0] == 'rotation':
            if not len(t) == 4:
                raise ValueError("Error creating flow from transforms: Not enough transform values passed for "
                                 "'rotation' - expected 3, got {}".format(len(t) - 1))
        elif t[0] == 'scaling':
            if not len(t) == 4:
                raise ValueError("Error creating flow from transforms: Not enough transform values passed for "
                                 "'scaling' - expected 3, got {}".format(len(t) - 1))
        else:
            raise ValueError("Error creating flow from transforms: Transform '{}' not recognised".format(t[0]))
        if not all(isinstance(item, (float, int)) for item in t[1:]):
            raise ValueError("Error creating flow from transforms: "
                             "Transform values for '{}' need to be integers or floats".format(t[0]))

    # Process for flow reference 's' is straightforward: get the transformation matrix for each given transform in
    #   the transform_list, and get the final transformation matrix by multiplying the transformation matrices for
    #   each individual transform sequentially. Finally, call flow_from_matrix to get the corresponding flow field,
    #   which works by applying that final transformation matrix to a meshgrid of vector locations, and subtracting
    #   the start points from the end points.
    #   flow_s = transformed_coords - coords
    #          = final_transform * coords - coords
    #          = t_1 * ... * t_n * coords - coords
    #
    # Process for flow reference 't' can be done in two ways:
    #   1) get the transformation matrix for each given transform in the transform_list, and get the final
    #     transformation matrix by multiplying the transformation matrices for each individual transform in inverse
    #     order. Then, call flow_from_matrix on the *inverse* of this final transformation matrix to get the
    #     negative of the corresponding flow field, which means applying the inverse of that final transformation
    #     matrix to a meshgrid of vector locations, and subtracting the end points from the start points.
    #     flow_t = coords - transformed_coords
    #            = coords - inv(final_transform) * coords
    #            = coords - inv(t_1 * ... * t_n) * coords
    #   2) get the transformation matrix for the reverse of each given transform in the "inverse inverse order",
    #     i.e. in the given order of the transform_list, and get the final transformation matrix by multiplying the
    #     results sequentially. Then, call flow_from_matrix on this final transformation matrix (already
    #     corresponding to the inverse as in method 1)) to get the negative of the corresponding flow field as
    #     before. This method is more complicated, but avoids any numerical issues potentially arising from
    #     calculating the inverse of a matrix.
    #     flow_t = coords - transformed_coords
    #            = coords - final_transform * coords
    #            = coords - inv(t_n) * ... * inv(t_1) * coords
    #     ... because: inv(t_n) * ... * inv(t_1) = inv(t_1 * ... * t_n)

    # Following lines, commented out, correspond to method 1
    # matrix = matrix_from_transforms(transform_list)
    # return cls.from_matrix(matrix, shape, ref, mask, device)

    # Here implemented: method 2, via calling from_matrix where the inverse of the matrix is used if reference 't'
    if ref == 's':
        matrix = matrix_from_transforms(transform_list)
        return from_matrix(matrix, shape, ref, matrix_is_inverse=False)
    else:  # ref == 't'
        transform_list = reverse_transform_values(transform_list)
        matrix = matrix_from_transforms(list(reversed(transform_list)))
        return from_matrix(matrix, shape, ref, matrix_is_inverse=True)


def load_kitti(path: str) -> Union[List[torch.Tensor], torch.Tensor]:
    """Loads the flow field contained in KITTI ``uint16`` png images files, including the valid pixels.
    Follows the official instructions on how to read the provided .png files on the
    `KITTI optical flow dataset website`_.

    .. _KITTI optical flow dataset website: http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow

    :param path: String containing the path to the KITTI flow data (``uint16``, .png file)
    :return: A torch tensor of shape :math:`(3, H, W)` with the KITTI flow data (with valid pixels in the 3rd channel)
    """

    inp = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_UNCHANGED necessary to read uint16 correctly
    if inp is None:
        raise ValueError("Error loading flow from KITTI data: Flow data could not be loaded")
    if len(inp.shape) != 3 or inp.shape[-1] != 3:
        raise ValueError("Error loading flow from KITTI data: Loaded flow data has the wrong shape")
    inp = inp[..., ::-1].astype('float64')  # Invert channels as cv2 loads as BGR instead of RGB
    inp[..., :2] = (inp[..., :2] - 2 ** 15) / 64
    inp[inp[..., 2] > 0, 2] = 1
    return to_tensor(inp, switch_channels='single')


def load_sintel(path: str) -> torch.Tensor:
    """Loads the flow field contained in Sintel .flo byte files. Follows the official instructions provided
    alongside the .flo data on the `Sintel optical flow dataset website`_.

    .. _Sintel optical flow dataset website: http://sintel.is.tue.mpg.de/

    :param path: String containing the path to the Sintel flow data (.flo byte file, little Endian)
    :return: A torch tensor of shape :math:`(2, H, W)` containing the Sintel flow data
    """

    if not isinstance(path, str):
        raise TypeError("Error loading flow from Sintel data: Path needs to be a string")
    with open(path, 'rb') as file:
        if file.read(4).decode('ascii') != 'PIEH':
            raise ValueError("Error loading flow from Sintel data: Path not a valid .flo file")
        w, h = int.from_bytes(file.read(4), 'little'), int.from_bytes(file.read(4), 'little')
        if 99999 < w < 1:
            raise ValueError("Error loading flow from Sintel data: Invalid width read from file ('{}')".format(w))
        if 99999 < h < 1:
            raise ValueError("Error loading flow from Sintel data: Invalid height read from file ('{}')".format(h))
        dt = np.dtype('float32')
        dt = dt.newbyteorder('<')
        flow = np.fromfile(file, dtype=dt).reshape(h, w, 2)
    return to_tensor(flow, switch_channels='single')


def load_sintel_mask(path: str) -> torch.Tensor:
    """Loads the invalid pixels contained in Sintel .png mask files, as a boolean mask marking valid pixels with
    ``True``. Follows the official instructions provided alongside the .flo data on the
    `Sintel optical flow dataset website`_.

    .. _Sintel optical flow dataset website: http://sintel.is.tue.mpg.de/

    :param path: String containing the path to the Sintel invalid pixel data (.png, black and white)
    :return: A torch tensor containing the Sintel valid pixels (mask) data
    """

    if not isinstance(path, str):
        raise TypeError("Error loading flow from Sintel data: Path needs to be a string")
    mask = cv2.imread(path, 0)
    if mask is None:
        raise ValueError("Error loading flow from Sintel data: Invalid mask could not be loaded from path")
    mask = ~(mask.astype('bool'))
    return to_tensor(mask)


def resize_flow(flow: Union[np.ndarray, torch.Tensor], scale: Union[float, int, list, tuple]) -> torch.Tensor:
    """Resize a flow field numpy array or torch tensor, scaling the flow vectors values accordingly

    :param flow: Flow field as a numpy array or torch tensor, shape :math:`(2, H, W)`, :math:`(H, W, 2)`,
        :math:`(N, 2, H, W)`, or :math:`(N, H, W, 2)`
    :param scale: Scale used for resizing, options:

        - Integer or float of value ``scaling`` applied both vertically and horizontally
        - List or tuple of shape :math:`(2)` with values ``[vertical scaling, horizontal scaling]``
    :return: Scaled flow field as a torch tensor, shape :math:`(2, H, W)` or :math:`(N, 2, H, W)`, depending on input
    """

    # Check validity
    valid_flow = get_valid_vecs(flow, error_string="Error resizing flow: ")
    if isinstance(scale, (float, int)):
        scale = [scale, scale]
    elif isinstance(scale, (tuple, list)):
        if len(scale) != 2:
            raise ValueError("Error resizing flow: Scale {} must have a length of 2".format(type(scale)))
        if not all(isinstance(item, (float, int)) for item in scale):
            raise ValueError("Error resizing flow: Scale {} items must be integers or floats".format(type(scale)))
    else:
        raise TypeError("Error resizing flow: "
                        "Scale must be an integer, float, or list or tuple of integers or floats")
    if any(s <= 0 for s in scale):
        raise ValueError("Error resizing flow: Scale values must be larger than 0")

    # Resize and adjust values
    resized = f.interpolate(valid_flow, scale_factor=scale, mode='bilinear')
    resized[:, 0] *= scale[1]
    resized[:, 1] *= scale[0]

    # Get rid of first dim if input was only 3-dimensional
    if len(flow.shape) == 3:
        resized = resized.squeeze(0)

    return resized


def is_zero_flow(flow: Union[np.ndarray, torch.Tensor], thresholded: bool = None) -> torch.Tensor:
    """Check whether all flow vectors are zero. Optionally, a threshold flow magnitude value of ``1e-3`` is used.
    This can be useful to filter out motions that are equal to very small fractions of a pixel, which might just be
    a computational artefact to begin with.

    :param flow: Flow field as a numpy array or torch tensor, shape :math:`(2, H, W)`, :math:`(H, W, 2)`,
        :math:`(N, 2, H, W)`, or :math:`(N, H, W, 2)`
    :param thresholded: Boolean determining whether the flow is thresholded, defaults to ``True``
    :return: Tensor of (batch) shape :math:`(N)` which is ``True`` if the flow field is zero everywhere,
        otherwise ``False``
    """

    # Check input validity
    flow = get_valid_vecs(flow, error_string="Error checking whether flow is zero: ")
    thresholded = True if thresholded is None else thresholded
    if not isinstance(thresholded, bool):
        raise TypeError("Error checking whether flow is zero: Thresholded needs to be a boolean")

    f = threshold_vectors(flow) if thresholded else flow
    return torch.sum(f == 0, (1, 2, 3)) == f[0].numel()


def track_pts(
    flow: Union[np.ndarray, torch.Tensor],
    ref: str,
    pts: torch.Tensor,
    int_out: bool = None
) -> torch.Tensor:
    """Warp input points with a flow field, returning the warped point coordinates as integers if required

    .. tip::
        Calling :meth:`~oflibpytorch.track_pts` on a flow field with reference ``s`` ("source") is
        significantly faster, as this does not require a call to :func:`scipy.interpolate.griddata`.

    :param flow: Flow field as a numpy array or torch tensor, shape :math:`(2, H, W)`, :math:`(H, W, 2)`,
        :math:`(N, 2, H, W)`, or :math:`(N, H, W, 2)`
    :param ref: Flow field reference, either ``s`` or ``t``
    :param pts: Torch tensor of shape :math:`(M, 2)` or :math:`(N, M, 2)` containing the point coordinates. If a
        batch dimension is given, it must be 1 or correspond to the flow batch dimension. If the flow is batched but
        the points are not, the same points are warped by each flow field individually. ``pts[:, 0]`` corresponds to
        the vertical coordinate, ``pts[:, 1]`` to the horizontal coordinate
    :param int_out: Boolean determining whether output points are returned as rounded integers, defaults to
        ``False``
    :return: Torch tensor of warped ('tracked') points, tensor device same as the input flow field
    """

    # Validate inputs
    flow = get_valid_vecs(flow, error_string="Error tracking points: ")
    ref = get_valid_ref(ref)
    if not isinstance(pts, torch.Tensor):
        raise TypeError("Error tracking points: Pts needs to be a numpy array or a torch tensor")
    return_2d = False
    if pts.dim() == 2:
        return_2d = True
        pts = pts.unsqueeze(0).repeat(flow.shape[0], 1, 1)  # N, M, 2
    elif pts.dim() == 3:
        if pts.shape[0] != flow.shape[0]:
            if pts.shape[0] == 1:
                pts = pts.repeat(flow.shape[0], 1, 1)
            else:
                raise ValueError("Error tracking points: "
                                 "If used, pts batch size needs to be equal to the flow batch size")
    else:
        raise ValueError("Error tracking points: Pts needs to have shape M-2 or N-M-2")
    if pts.shape[-1] != 2:
        raise ValueError("Error tracking points: Pts needs to have shape M-2 or N-M-2")
    int_out = False if int_out is None else int_out
    if not isinstance(int_out, bool):
        raise TypeError("Error tracking points: Int_out needs to be a boolean")

    pts = pts.to(flow.device)

    if all(is_zero_flow(flow, thresholded=True)):
        warped_pts = pts
    else:
        if ref == 's':
            if not pts.dtype.is_floating_point:
                flow_vecs = flow.permute(0, 2, 3, 1)                                # from N2HW to NHW2
                pts2 = pts[..., 0] * flow.shape[-1] + pts[..., 1]                   # NM, with coords "unravelled"
                pts2 = pts2.unsqueeze(-1).expand(-1, -1, 2)
                flow_vecs = torch.gather(flow_vecs.view(flow_vecs.shape[0], -1, 2), 1, pts2.long())  # NM2
                flow_vecs = flow_vecs.flip(-1)                                      # NM2, from (x, y) to (y, x)
            else:
                pts_4d = pts.unsqueeze(1).to(torch.float).flip(-1)  # from NM2 to N1M2, and from (y, x) to (x, y)
                pts_4d = normalise_coords(pts_4d, flow.shape[-2:])
                torch_version = globals()['torch'].__version__
                if int(torch_version[0]) == 1 and float(torch_version[2:4]) <= 3:
                    flow_vecs = f.grid_sample(flow, pts_4d).flip(1)  # flow is N2HW, pts_4d N1M2, output N21M
                else:
                    flow_vecs = f.grid_sample(flow, pts_4d, align_corners=True).flip(1)
                # flow is N2HW, pts_4d N1M2, output flow_vecs N21M, after flip (x, y) changed to (y, x)
                flow_vecs = flow_vecs.squeeze(2).permute(0, 2, 1)  # N21M to N2M to NM2
            warped_pts = pts.float() + flow_vecs
        else:  # self._ref == 't'
            flow = flow.permute(0, 2, 3, 1)                                     # N-2-H-W to N-H-W-2
            flow = to_numpy(flow).astype('float32')
            flow_flat = np.reshape(flow[..., ::-1], (flow.shape[0], -1, 2))     # N-H*W-2
            x, y = np.mgrid[:flow.shape[1], :flow.shape[2]]                     # H-W
            grid = np.swapaxes(np.vstack([x.ravel(), y.ravel()]), 0, 1)         # 2-H*W to H*W-2
            origin_points = grid - flow_flat                                    # N-H*W-2
            warped_pts = pts.clone()                                            # N-M-2
            for i in range(warped_pts.shape[0]):  # Perform griddata for each batch member
                flow_vecs = griddata(origin_points[i], flow_flat[i], (pts[i, :, 0], pts[i, :, 1]), method='linear')
                warped_pts[i] += torch.tensor(flow_vecs, device=pts.device)
        nan_vals = torch.isnan(warped_pts)
        nan_vals = nan_vals[:, :, 0] | nan_vals[:, :, 1]
        warped_pts[nan_vals] = 0

    if int_out:
        warped_pts = torch.round(warped_pts).long()
    if return_2d:
        warped_pts = warped_pts.squeeze(0)

    return warped_pts


def get_flow_endpoints(flow: torch.Tensor, ref: str) -> tuple:
    """Calculates the endpoint (or strictly speaking start points if ref 't') coordinate grids x, y for a given
    flow field

    :param flow: Flow field of shape N2HW
    :param ref: Flow reference, 's' or 't'
    :return: Tuple of end point (ref 's') or start point (ref 't') coordinate grids x (hor) and y (ver) of shape NHW
    """

    n, _, h, w = flow.shape
    s = +1 if ref == 's' else -1  # if ref == 't'
    x = s * flow[:, 0] + torch.arange(w, device=flow.device)[None, None, :]  # hor
    y = s * flow[:, 1] + torch.arange(h, device=flow.device)[None, :, None]  # ver
    return x, y


def grid_from_unstructured_data(
    x: torch.Tensor,
    y: torch.Tensor,
    data: torch.Tensor,
    mask: torch.Tensor = None,
) -> tuple:
    """Returns unstructured input data on a (sparse) regular grid. Credit:
        - This is based on the algorithm suggested in: Sánchez, J., Salgado de la Nuez, A. J., & Monzón, N., "Direct
        estimation of the backward flow", 2013
        - The code implementation is a heavily reworked version of code suggested by Markus Hofinger in private
        correspondence, a version of which he first used in: Hofinger, M., Bulò, S. R., Porzi, L., Knapitsch, A.,
        Pock, T., & Kontschieder, P., "Improving optical flow on a pyramid level". ECCV 2020
        - Markus Hofinger in turn credits the function _flow2distribution from the HD3 code base, used in: Yin, Z.,
        Darrell, T., & Yu, F., "Hierarchical discrete distribution decomposition for match density estimation",
        CAPER 2019

    :param x: Horizontal data position grid, shape N1HW
    :param y: Vertical data position grid, shape N1HW
    :param data: Data value grid, shape NCHW
    :param mask: Tensor masking data points, shape NCHW
    :return: NCHW tensor of data interpolated on regular grid, N1HW tensor of interpolation density
    """

    # Input grids:          [N, H, W]
    # Input data:           [N, C, H, W]
    # Input mask:           [N, C, H, W]
    # Output data:          [N, C, H, W]
    # Uniqueness density:   [N, 1, H, W]

    n, c, h, w = data.shape

    x0 = torch.floor(x)                                             # NHW
    y0 = torch.floor(y)                                             # NHW

    # The integer points surrounding the flow endpoints
    xx = torch.stack((x0, x0+1), dim=-1)                            # NHW2: x0, x1
    yy = torch.stack((y0, y0+1), dim=-1)                            # NHW2: y0, y1

    # Ensure the integer points are within bounds
    xx_safe = torch.clamp(xx, min=0, max=w-1)                       # NHW2: x0_safe, x1_safe
    yy_safe = torch.clamp(yy, min=0, max=h-1)                       # NHW2: y0_safe, y1_safe

    # Hor / ver weights = offsets of corners from actual point, set to 0 if the safe points are out of bounds
    wt_xx = torch.stack((xx[..., 1] - x, x - xx[..., 0]), dim=-1) * torch.eq(xx, xx_safe).float()  # NHW2: wt_x0, wt_y0
    wt_yy = torch.stack((yy[..., 1] - y, y - yy[..., 0]), dim=-1) * torch.eq(yy, yy_safe).float()  # NHW2: wt_x0, wt_y0

    # Corner weights are multiplied hor / ver weights
    wgt = torch.matmul(wt_yy.unsqueeze(-1), wt_xx.unsqueeze(-2))  # matmul(NHW21, NHW12): N-H-W-2-2
    wgt = wgt.permute(0, 3, 4, 1, 2).reshape(n*4, h*w)                                  # N*4-H*W

    # Convert the corner points coordinates into a running position index
    pos = (w * yy_safe).unsqueeze(-1) + xx_safe.unsqueeze(-2)            # NHW21 + NHW12: N-H-W-2-2
    pos = pos.permute(0, 3, 4, 1, 2).reshape(n*4, h*w)                                  # N*4-H*W

    # Mask the weights where mask is False (if mask given)
    if mask is not None:
        wgt *= mask.repeat_interleave(4, dim=0).view(n*4, h*w).to(torch.uint8)

    # General note for density and grid_data: the 4 corner points that need to be scatter_add_ed individually are
    # present in the shape of a channel dimension. They are "mixed" into the batch dimension in order to be able to
    # use a single .scatter_add_() operation, then "unmixed". Adding up along the "unmixed" channel dim yields the
    # same result as using .scatter_add_ for each corner point individually

    # Calculate the summed up weight for each point by distributing the corner weights down dim=1 according to the
    # corner point coordinates. Then reshape so a summed weight is available for each image coordinate
    density = torch.zeros((n*4, h*w), device=data.device)  # N*4-H*W
    density.scatter_add_(dim=1, index=pos.long(), src=wgt)
    density = torch.sum(density.view(n, 4, h, w), dim=1, keepdim=True)

    # Calculate the summed up data for each point by distributing the corner data down dim=1 according to the corner
    # corner point coordinates
    grid_data = torch.zeros((n*4*c, h*w), device=data.device)  # N*4*C-H*W
    grid_data.scatter_add_(dim=1, index=pos.repeat_interleave(c, dim=0).long(),
                           src=wgt.repeat_interleave(c, dim=0) * data.repeat_interleave(4, dim=0).view(n*4*c, h*w))

    # Normalise by the sum of weights. To avoid division by zero, a minimum of 1e-3 is used. Where the sum of weights
    # is 0, no data should have been allocated, so those divisions will be 0/1e-3 = 0
    grid_data = torch.sum(grid_data.view(n, 4, c, h, w), dim=1) / torch.clamp_min(density, 1e-3)

    # The following commented-out lines are a per-channel version of the above "grid_data" code
    # grid_data_list = []
    # for i in range(c):
    #     grid_data = torch.zeros_like(wgt, device=data.device)
    #     grid_data.scatter_add_(1, pos.long(), wgt * data[:, i, :, :].view(n, h*w).expand(n*4, -1))
    #     grid_data_list.append(torch.sum(grid_data.view(n, 4, h, w), dim=1, keepdim=True))
    # grid_data = torch.cat(grid_data_list, dim=1) / torch.clamp_min(density, 1e-3)

    return grid_data, density


def apply_s_flow(
    flow: torch.Tensor,
    data: torch.Tensor,
    mask: torch.Tensor = None,
    occlude_zero_flow: bool = None
) -> torch.Tensor:
    """Warp data with 's' reference flow

    :param flow: Float tensor of shape N2HW, the input flow with reference 's'
    :param data: Float tensor of shape NCHW, the data to be warped
    :param mask: Boolean tensor of shape NHW, the mask belonging to the flow (optional)
    :param occlude_zero_flow: Boolean determining whether data locations with corresponding flow of zero are occluded
        (overwritten) by other data moved to the same location. Rationale: if the order of objects were reversed, i.e.
        the zero flow points occlude the non-zero flow points, the latter wouldn't be known in the first place. This
        logic breaks down when the flow points concerned have been inferred, e.g. from surrounding non-occluded known
        points. Defaults to False
    :return: Warped data as float tensor of shape NCHW, mask of where data points where warped to as bool tensor of
        shape NHW (if occlude_zero_flow is True, this mask does not include zero flow points)
    """

    occlude_zero_flow = False if occlude_zero_flow is None else occlude_zero_flow

    n, c, h, w = data.shape
    x, y = get_flow_endpoints(flow, 's')                                         # NHW, NHW

    if occlude_zero_flow:
        override_mask = torch.sum(threshold_vectors(flow) == 0, dim=1) != 2
        if mask is not None:
            mask = mask & override_mask
        else:
            mask = override_mask
    else:
        if mask is not None:
            mask = mask.unsqueeze(1)

    warped_data, warped_density = grid_from_unstructured_data(x, y, data, mask)  # NCHW, N1HW
    warped_mask = (warped_density > .25).expand(-1, c, -1, -1)                   # NCHW
    warped_data[~warped_mask] = data[~warped_mask].float()                       # NCHW

    return warped_data, warped_mask[:, 0]
