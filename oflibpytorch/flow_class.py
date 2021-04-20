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

from __future__ import annotations
import torch
import torch.nn.functional as f
from scipy.interpolate import griddata
import numpy as np
import cv2
from typing import Union
from .utils import get_valid_vecs, get_valid_ref, get_valid_device, get_valid_padding, validate_shape, to_numpy, \
    move_axis, flow_from_matrix, matrix_from_transforms, reverse_transform_values, apply_flow, threshold_vectors, \
    normalise_coords


class Flow(object):
    def __init__(
            self,
            flow_vectors: Union[np.ndarray, torch.Tensor],
            ref: str = None,
            mask: Union[np.ndarray, torch.Tensor] = None,
            device: str = None
    ):
        """Flow object constructor

        :param flow_vectors: Numpy array or pytorch tensor. Shape interpreted as H-W-C with C=2 if the last dimension
            is 2, otherwise as C-H-W with C=2, with ValueError thrown if neither fits. The channel dimension contains
            the flow vector in OpenCV convention: C[0] is the horizontal component, C[1] is the vertical component
        :param ref: Flow reference, 't'arget or 's'ource. Defaults to 't'
        :param mask: Numpy array or pytorch tensor of shape H-W, containing a boolean mask indicating where the flow
            vectors are valid. Defaults to True everywhere
        :param device: Tensor device, 'cpu' or 'cuda' (if available). Defaults to the device of the given flow_vectors
            if that is a tensor, otherwise to 'cpu'
        """

        # Prepare attributes and type hinting
        self._vecs: torch.Tensor = None
        self._mask: torch.Tensor = None
        self._ref: str = None
        self._device: str = None

        # Fill attributes
        self.vecs = flow_vectors
        self.ref = ref
        self.mask = mask
        self.device = device

    @property
    def vecs(self) -> torch.Tensor:
        """Gets flow vectors

        :return: Flow vectors as torch tensor of shape 2-H-W
        """

        return self._vecs

    @vecs.setter
    def vecs(self, input_vecs: Union[np.ndarray, torch.Tensor]):
        """Sets flow vectors, after checking validity

        :param input_vecs: Numpy array or pytorch tensor. Shape interpreted as H-W-C with C=2 if the last dimension
            is 2, otherwise as C-H-W with C=2, with ValueError thrown if neither fits. The channel dimension contains
            the flow vector in OpenCV convention: C[0] is the horizontal component, C[1] is the vertical component
        """

        self._vecs = get_valid_vecs(input_vecs, error_string="Error setting flow vectors: ")

    @property
    def vecs_numpy(self) -> np.ndarray:
        """Gets the flow vectors as a numpy array of shape H-W-2, rather than the internal representation of a torch
        tensor of shape 2-H-W

        :return: flow vectors as a numpy array of shape H-W-2
        """

        with torch.no_grad():
            if self._device == 'cuda':
                vecs = self._vecs.cpu().numpy()
            else:  # self._device == 'cpu'
                vecs = self._vecs.detach().numpy()
        return np.moveaxis(vecs, 0, -1)

    @property
    def ref(self) -> str:
        """Gets flow reference

        :return: Flow reference 't' or 's'
        """

        return self._ref

    @ref.setter
    def ref(self, input_ref: str = None):
        """Sets flow reference, after checking validity

        :param input_ref: Flow reference 't' or 's'. Defaults to 't'
        """

        self._ref = get_valid_ref(input_ref)

    @property
    def mask(self) -> torch.Tensor:
        """Gets flow mask

        :return: Flow mask as torch tensor of shape H-W and type 'bool'
        """

        return self._mask

    @mask.setter
    def mask(self, input_mask: Union[np.ndarray, torch.Tensor] = None):
        """Sets flow mask, after checking validity

        :param input_mask: bool torch tensor of shape H-W (self.shape), matching flow vectors with shape 2-H-W
        """

        if input_mask is None:
            self._mask = torch.ones(*self.shape).to(torch.bool)
        else:
            # Check type, dimensions, shape
            if not isinstance(input_mask, (np.ndarray, torch.Tensor)):
                raise TypeError("Error setting flow mask: Input is not a numpy array or a torch tensor")
            if input_mask.ndim != 2:
                raise ValueError("Error setting flow mask: Input is not 2-dimensional")
            if input_mask.shape != self.shape:
                raise ValueError("Error setting flow mask: Input has a different shape than the flow vectors")

            # Transform to tensor if necessary
            if isinstance(input_mask, np.ndarray):
                input_mask = torch.tensor(input_mask, device=self._vecs.device)

            # Check for invalid values
            if ((input_mask != 0) & (input_mask != 1)).any():
                raise ValueError("Error setting flow mask: Values must be 0 or 1")

            self._mask = input_mask.to(torch.bool)

    @property
    def mask_numpy(self) -> np.ndarray:
        """Gets the mask as a numpy array of shape H-W, rather than the internal torch tensor

        :return: mask as a numpy array of shape H-W, dtype 'bool'
        """

        with torch.no_grad():
            if self._device == 'cuda':
                mask = self._mask.cpu().numpy()
            else:  # self._device == 'cpu'
                mask = self._mask.detach().numpy()
        return mask

    @property
    def device(self) -> str:
        """Gets the tensor device

        :return: Tensor device as a string
        """
        return self._device

    @device.setter
    def device(self, input_device: str = None):
        """Sets the tensor device, after checking validity

        :param input_device: Tensor device, 'cpu' or 'cuda'. Defaults to 'cpu'
        """
        if input_device is None:
            device = self._vecs.device.type
        else:
            device = get_valid_device(input_device)
        self._device = device
        self._vecs = self._vecs.to(device)
        self._mask = self._mask.to(device)

    @property
    def shape(self) -> tuple:
        """Gets shape (resolution) of the flow

        :return: Shape (resolution) of the flow field as a tuple
        """

        return tuple(self._vecs.shape[1:])

    @classmethod
    def zero(
            cls,
            shape: Union[list, tuple],
            ref: str = None,
            mask: Union[np.ndarray, torch.Tensor] = None,
            device: str = None,
    ) -> Flow:
        """Flow object constructor, zero everywhere

        :param shape: List or tuple [H, W] of flow field shape
        :param ref: Flow referencce, 't'arget or 's'ource. Defaults to 't'
        :param mask: Numpy array or torch tensor H-W containing a boolean mask indicating where the flow vectors are
            valid. Defaults to True everywhere
        :param device: Tensor device, 'cpu' or 'cuda' (if available). Defaults to 'cpu'
        :return: Flow object
        """

        # Check shape validity
        validate_shape(shape)
        return cls(torch.zeros(2, *shape), ref, mask, device)

    @classmethod
    def from_matrix(
        cls,
        matrix: Union[np.ndarray, torch.Tensor],
        shape: Union[list, tuple],
        ref: str = None,
        mask: Union[np.ndarray, torch.Tensor] = None,
        device: str = None,
        matrix_is_inverse: bool = None
    ) -> Flow:
        """Flow object constructor, based on transformation matrix input

        :param matrix: Transformation matrix to be turned into a flow field, as numpy array or torch tensor of shape 3-3
        :param shape: List or tuple [H, W] of flow field shape
        :param ref: Flow referencce, 't'arget or 's'ource. Defaults to 't'
        :param mask: Numpy array or torch tensor H-W containing a boolean mask indicating where the flow vectors are
            valid. Defaults to True everywhere
        :param device: Tensor device, 'cpu' or 'cuda' (if available). Defaults to 'cpu'
        :param matrix_is_inverse: Boolean determining whether the given matrix is already the inverse of the desired
            transformation. Is useful for flow with reference 't' to avoid calculation of the pseudo-inverse, but will
            throw an error if used for flow with reference 's' to avoid accidental usage. Defaults to False
        :return: Flow object
        """

        # Check shape validity
        validate_shape(shape)
        # Get valid device
        if device is not None:
            device = get_valid_device(device)
        # Check matrix validity
        if not isinstance(matrix, (np.ndarray, torch.Tensor)):
            raise TypeError("Error creating flow from matrix: Matrix needs to be a numpy array or a torch tensor")
        if matrix.shape != (3, 3):
            raise ValueError("Error creating flow from matrix: Matrix needs to be of shape (3, 3)")
        if isinstance(matrix, np.ndarray):
            matrix = torch.tensor(matrix)
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
            return cls(flow_vectors, ref, mask, device)
        elif ref == 't':
            # Coordinates correspond to the meshgrid of the warped ('t'arget) image. They are inversely transformed
            # according to the transformation matrix. The end points, which correspond to the flow origin for the
            # meshgrid in the warped image, are subtracted from the start points to yield flow vectors.
            if not matrix_is_inverse:
                matrix = torch.pinverse(matrix)
            flow_vectors = -flow_from_matrix(matrix, shape)
            return cls(flow_vectors, ref, mask, device)

    @classmethod
    def from_transforms(
        cls,
        transform_list: list,
        shape: Union[list, tuple],
        ref: str = None,
        mask: Union[np.ndarray, torch.Tensor] = None,
        device: str = None,
    ) -> Flow:
        """Flow object constructor, based on list of transforms

        :param transform_list: List of transforms to be turned into a flow field, where each transform is expressed as
            a list of [transform name, transform value 1, ... , transform value n]. Supported options:
                ['translation', horizontal shift in px, vertical shift in px]
                ['rotation', horizontal centre in px, vertical centre in px, angle in degrees, counter-clockwise]
                ['scaling', horizontal centre in px, vertical centre in px, scaling fraction]
        :param shape: List or tuple [H, W] of flow field shape
        :param ref: Flow referencce, 't'arget or 's'ource. Defaults to 't'
        :param mask: Numpy array or torch tensor H-W containing a boolean mask indicating where the flow vectors are
            valid. Defaults to True everywhere
        :param device: Tensor device, 'cpu' or 'cuda' (if available). Defaults to 'cpu'
        :return: Flow object
        """

        # Check shape validity
        validate_shape(shape)
        # Get valid device
        device = get_valid_device(device)
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
            return cls.from_matrix(matrix, shape, ref, mask, device)
        else:  # ref == 't'
            transform_list = reverse_transform_values(transform_list)
            matrix = matrix_from_transforms(list(reversed(transform_list)))
            return cls.from_matrix(matrix, shape, ref, mask, device, matrix_is_inverse=True)

    def __str__(self):
        """Enhanced string representation of the flow object"""
        info_string = "Flow object, reference {}, shape {}*{}, device {}; ".format(self._ref, *self.shape, self._device)
        info_string += self.__repr__()
        return info_string

    def __getitem__(self, item: Union[int, list, slice]) -> Flow:
        """Mimics __getitem__ of a torch tensor, returning a flow object cut accordingly

        Will throw an error if mask.__getitem__(item) or vecs.__getitem__(item) throw an error. Also throws an error if
        sliced vecs (or masks) don't fulfil the conditions to construct a flow object (e.g. have shape H-W-2 for vecs)

        :param item: Slice used to select a part of the flow
        :return: New flow cut as a corresponding torch tensor would be cut
        """

        vecs = move_axis(move_axis(self._vecs, 0, -1).__getitem__(item), -1, 0)
        # Above line is to avoid having to parse item properly to deal with first dim of 2: move this dim to the back
        return Flow(vecs, self._ref, self._mask.__getitem__(item), self._device)

    def __copy__(self) -> Flow:
        """Returns a copy of the flow object

        :return: Copy of the flow object
        """

        return Flow(self._vecs, self._ref, self._mask, self._device)

    def __add__(self, other: Union[np.ndarray, torch.Tensor, Flow]) -> Flow:
        """Adds a flow object, a numpy array or a torch tensor to a flow object

        Note: this is NOT equal to applying the two flows sequentially. For that, use combine_flows(flow1, flow2, None).
        The function also does not check whether the two flow objects have the same reference.

        DO NOT USE if you're not certain about what you're aiming to achieve.

        :param other: Flow object, numpy array, or torch tensor corresponding to the addend. Arrays and tensors need to
            have the shape 2-H-W or H-W-2, where H-W matches the shape of the flow object. Adding a flow object will
            adjust the mask of the resulting flow object to correspond to the logical union of the augend / addend masks
        :return: Flow object corresponding to the sum
        """

        if isinstance(other, Flow):
            if self.shape != other.shape:
                raise ValueError("Error adding to flow: Augend and addend flow objects are not the same shape")
            else:
                vecs = self._vecs + other._vecs
                mask = self._mask & other._mask
                return Flow(vecs, self._ref, mask)
        if isinstance(other, (np.ndarray, torch.Tensor)):
            other = get_valid_vecs(other, desired_shape=self.shape, error_string="Error adding to flow: ")
            vecs = self._vecs + other.to(self._vecs.device)
            return Flow(vecs, self._ref, self._mask, self._device)
        else:
            raise TypeError("Error adding to flow: Addend is not a flow object, numpy array, or torch tensor")

    def __sub__(self, other: Union[np.ndarray, torch.Tensor, Flow]) -> Flow:
        """Subtracts a flow objects, numpy array, or torch tensor from a flow object

        Note: this is NOT equal to subtracting the effects of applying flow fields to an image. For that, used
        combine_flows(flow1, None, flow2) or combine_flows(None, flow1, flow2). The function also does not check whether
        the two flow objects have the same reference.

        DO NOT USE if you're not certain about what you're aiming to achieve.

        :param other: Flow object, numpy array, or torch tensor corresponding to the subtrahend. Arrays and tensors
            need to have the shape 2-H-W or H-W-2, where H-W matches the shape of the flow object. Subtracting a flow
            object will adjust the mask of the resulting flow object to correspond to the logical union of the
            minuend / subtrahend masks
        :return: Flow object corresponding to the difference
        """

        if isinstance(other, Flow):
            if self.shape != other.shape:
                raise ValueError("Error subtracting from flow: "
                                 "Minuend and subtrahend flow objects are not the same shape")
            else:
                vecs = self._vecs - other._vecs
                mask = self._mask & other._mask
                return Flow(vecs, self._ref, mask, self._device)
        if isinstance(other, (np.ndarray, torch.Tensor)):
            other = get_valid_vecs(other, desired_shape=self.shape, error_string="Error subtracting to flow: ")
            vecs = self._vecs - other.to(self._vecs.device)
            return Flow(vecs, self._ref, self._mask, self._device)
        else:
            raise TypeError("Error subtracting from flow: "
                            "Subtrahend is not a flow object, numpy array, or torch tensor")

    def __mul__(self, other: Union[float, int, bool, list, np.ndarray, torch.Tensor]) -> Flow:
        """Multiplies a flow object

        :param other: Multiplier which either can be converted to float or is a list of length 2, or is a numpy array
            or a torch tensor of either the same shape as the flow object (H-W), or either 2-H-W or H-W-2
        :return: Flow object corresponding to the product
        """

        try:  # other is int, float, or can be converted to it
            return Flow(self._vecs * float(other), self._ref, self._mask, self._device)
        except (TypeError, ValueError):
            if isinstance(other, list):
                if len(other) != 2:
                    raise ValueError("Error multiplying flow: Multiplier list not length 2")
                other = torch.tensor(other)
            elif isinstance(other, np.ndarray):
                other = torch.tensor(other)
            if isinstance(other, torch.Tensor):
                if other.ndim == 1 and other.shape[0] == 2:  # shape 2 to 2-1-1
                    other = other.unsqueeze(-1).unsqueeze(-1)
                elif other.ndim == 2 and other.shape == self.shape:  # shape H-W to 2-H-W
                    other = other.unsqueeze(0)
                elif other.ndim == 3 and other.shape == (2,) + self.shape:  # shape 2-H-W: all OK
                    pass
                elif other.ndim == 3 and other.shape == self.shape + (2,):  # shape H-W-2 to 2-H-W
                    other = move_axis(other, -1, 0)
                else:
                    raise ValueError("Error multiplying flow: Multiplier array or tensor needs to be of size 2, of the "
                                     "shape of the flow object (H-W), or either 2-H-W or H-W-2")
                other = other.to(self._vecs.device)
                return Flow(self._vecs * other, self._ref, self._mask, self._device)
            else:
                raise TypeError("Error multiplying flow: Multiplier cannot be converted to float, "
                                "or isn't a list, numpy array, or torch tensor")

    def __truediv__(self, other: Union[float, int, bool, list, np.ndarray, torch.Tensor]) -> Flow:
        """Divides a flow object

        :param other: Divisor which either can be converted to float or is a list of length 2, or is a numpy array
            or a torch tensor of either the same shape as the flow object (H-W), or either 2-H-W or H-W-2
        :return: Flow object corresponding to the quotient
        """

        try:  # other is int, float, or can be converted to it
            return Flow(self._vecs / float(other), self._ref, self._mask, self._device)
        except (TypeError, ValueError):
            if isinstance(other, list):
                if len(other) != 2:
                    raise ValueError("Error dividing flow: Divisor list not length 2")
                other = torch.tensor(other)
            elif isinstance(other, np.ndarray):
                other = torch.tensor(other)
            if isinstance(other, torch.Tensor):
                if other.ndim == 1 and other.shape[0] == 2:  # shape 2 to 2-1-1
                    other = other.unsqueeze(-1).unsqueeze(-1)
                elif other.ndim == 2 and other.shape == self.shape:  # shape H-W to 2-H-W
                    other = other.unsqueeze(0)
                elif other.ndim == 3 and other.shape == (2,) + self.shape:  # shape 2-H-W: all OK
                    pass
                elif other.ndim == 3 and other.shape == self.shape + (2,):  # shape H-W-2 to 2-H-W
                    other = move_axis(other, -1, 0)
                else:
                    raise ValueError("Error dividing flow: Divisor array or tensor needs to be of size 2, of the "
                                     "shape of the flow object (H-W), or either 2-H-W or H-W-2")
                other = other.to(self._vecs.device)
                return Flow(self._vecs / other, self._ref, self._mask, self._device)
            else:
                raise TypeError("Error dividing flow: Divisor cannot be converted to float, "
                                "or isn't a list, numpy array, or torch tensor")

    def __pow__(self, other: Union[float, int, bool, list, np.ndarray, torch.Tensor]) -> Flow:
        """Exponentiates a flow object

        :param other: Exponent which either can be converted to float or is a list of length 2, or is a numpy array
            or a torch tensor of either the same shape as the flow object (H-W), or either 2-H-W or H-W-2
        :return: Flow object corresponding to the power
        """

        try:  # other is int, float, or can be converted to it
            return Flow(self._vecs ** float(other), self._ref, self._mask, self._device)
        except (TypeError, ValueError):
            if isinstance(other, list):
                if len(other) != 2:
                    raise ValueError("Error exponentiating flow: Exponent list not length 2")
                other = torch.tensor(other)
            elif isinstance(other, np.ndarray):
                other = torch.tensor(other)
            if isinstance(other, torch.Tensor):
                if other.ndim == 1 and other.shape[0] == 2:  # shape 2 to 2-1-1
                    other = other.unsqueeze(-1).unsqueeze(-1)
                elif other.ndim == 2 and other.shape == self.shape:  # shape H-W to 2-H-W
                    other = other.unsqueeze(0)
                elif other.ndim == 3 and other.shape == (2,) + self.shape:  # shape 2-H-W: all OK
                    pass
                elif other.ndim == 3 and other.shape == self.shape + (2,):  # shape H-W-2 to 2-H-W
                    other = move_axis(other, -1, 0)
                else:
                    raise ValueError("Error exponentiating flow: Exponent array or tensor needs to be of size 2, of "
                                     "the shape of the flow object (H-W), or either 2-H-W or H-W-2")
                other = other.to(self._vecs.device)
                return Flow(self._vecs ** other, self._ref, self._mask, self._device)
            else:
                raise TypeError("Error exponentiating flow: Exponent cannot be converted to float, "
                                "or isn't a list, numpy array, or torch tensor")

    def __neg__(self) -> Flow:
        """Returns the negative of a flow object

        CAREFUL: this is NOT equal to correctly inverting a flow! For that, use invert().

        DO NOT USE if you're not certain about what you're aiming to achieve.

        :return: Negative flow
        """

        return self * -1

    def resize(self, scale: Union[float, int, list, tuple]) -> Flow:
        """Resizes flow object, also scaling the flow vectors themselves

        :param scale: Scale used for resizing. Integer or float, or a list or tuple [vertical scale, horizontal scale]
        :return: Scaled flow
        """

        # Check validity
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

        # Resize vectors and mask
        to_resize = torch.cat((self._vecs, self._mask.to(torch.float).unsqueeze(0)), dim=0).unsqueeze(0)
        resized = f.interpolate(to_resize, scale_factor=scale, mode='bilinear').squeeze(0)

        # Adjust values
        resized[0] *= scale[1]
        resized[1] *= scale[0]

        return Flow(resized[:2], self._ref, torch.round(resized[2]))

    def pad(self, padding: list = None, mode: str = None) -> Flow:
        """Pads the flow with the given padding. Sets padded mask values to False, and inserts 0 flow values if padding
        mode is 'constant'

        :param padding: [top, bot, left, right] list of padding values
        :param mode: Numpy padding mode for the flow vectors, defaults to 'constant'. Options:
            'constant', 'reflect', 'replicate' (see torch.nn.functional.pad documentation). 'Constant' value is 0.
        :return: Padded flow
        """

        mode = 'constant' if mode is None else mode
        if mode not in ['constant', 'reflect', 'replicate']:
            raise ValueError("Error padding flow: Mode should be one of "
                             "'constant', 'reflect', or 'replicate', but instead got '{}'".format(mode))
        padding = get_valid_padding(padding, "Error padding flow: ")
        padded_vecs = f.pad(self._vecs.unsqueeze(0), (*padding[2:], *padding[:2]), mode=mode).squeeze(0)
        padded_mask = f.pad(self._mask.unsqueeze(0).unsqueeze(0), (*padding[2:], *padding[:2])).squeeze(0).squeeze(0)
        return Flow(padded_vecs, self._ref, padded_mask)

    def apply(
        self,
        target: Union[np.ndarray, torch.Tensor, Flow],
        return_valid_area: bool = None,
        padding: list = None,
        cut: bool = None
    ) -> Union[np.ndarray, torch.Tensor, Flow]:
        """Applies the flow to the target, which can be a numpy array or a Flow object.

        :param target: Numpy array or torch tensor of shape C-H-W, or flow object the flow should be applied to
        :param return_valid_area: Boolean determining whether a boolean numpy array of shape H-W containing the valid
            image area is returned (only relevant if target is a numpy array). This array is true where the image
            values in the function output:
                1) have been affected by flow vectors: always true if the flow has reference 't' as the target image by
                    default has a corresponding flow vector in each position, but only true for some parts of the image
                    if the flow has reference 's': some target image positions would only be reachable by flow vectors
                    originating outside of the source image area, which is obviously impossible
                2) have been affected by flow vectors that were themselves valid, as determined by the flow mask
        :param padding: If flow applied only covers part of the target; [top, bot, left, right]; default None
        :param cut: If padding is given, whether the input is returned as cut to shape of flow; default True
        :return: An object of the same type as the input (numpy array, torch tensor, or flow)
        """

        cut = False if cut is None else cut
        if not isinstance(cut, bool):
            raise TypeError("Error applying flow: Cut needs to be a boolean")
        if padding is not None:
            padding = get_valid_padding(padding, "Error applying flow: ")
            if self.shape[0] + padding[0] + padding[1] != target.shape[-2] or \
                    self.shape[1] + padding[2] + padding[3] != target.shape[-1]:
                raise ValueError("Error applying flow: Padding values do not match flow and target shape difference")

        # Type check, prepare arrays
        return_device = 'cpu'
        return_array = False
        if isinstance(target, Flow):
            return_flow = True
            return_device = target._vecs.device.type
            t = target._vecs.to(self._vecs.device)
            mask = target._mask.unsqueeze(0).to(self._vecs.device)
        else:
            return_flow = False
            if isinstance(target, np.ndarray):
                return_array = True
                t = torch.tensor(target).to(self._vecs.device)
            elif isinstance(target, torch.Tensor):
                t = target
            else:
                raise ValueError("Error applying flow: Target needs to be either a flow object, a numpy ndarray, or a "
                                 "torch tensor")
            mask = torch.ones(1, *t.shape[1:]).to(torch.bool).to(self._vecs.device)

        # Concatenate the flow vectors with the mask if required, so they are warped in one step
        if return_flow or return_valid_area:
            # if self.ref == 't': Just warp the mask, which self.vecs are valid taken into account after warping
            if self._ref == 's':
                # Warp the target mask after ANDing with flow mask to take into account which self.vecs are valid
                if mask.shape[1:] != self.shape:
                    # If padding in use, mask can be smaller than self.mask
                    tmp = mask[0, padding[0]:padding[0] + self.shape[0], padding[2]:padding[2] + self.shape[1]].clone()
                    mask[...] = False
                    mask[0, padding[0]:padding[0] + self.shape[0], padding[2]:padding[2] + self.shape[1]] = \
                        tmp & self._mask
                else:
                    mask = (mask & self._mask)
            t = torch.cat((t, mask.to(torch.float)), dim=0)

        # Determine flow to use for warping, and warp
        if padding is None:
            if not target.shape[-2:] == self.shape:
                raise ValueError("Error applying flow: Flow and target have to have the same shape")
            warped_t = apply_flow(self._vecs, t, self._ref)
        else:
            mode = 'constant' if self._ref == 't' else 'replicate'
            # Note: this mode is very important: irrelevant for flow with reference 't' as this by definition covers
            # the area of the target image, so 'constant' (defaulting to filling everything with 0) is fine. However,
            # for flows with reference 's', if locations in the source image with some flow vector border padded
            # locations with flow zero, very strange interpolation artefacts will result, both in terms of the image
            # being warped, and the mask being warped. By padding with the 'edge' mode, large gradients in flow vector
            # values at the edge of the original flow area are avoided, as are interpolation artefacts.
            flow = self.pad(padding, mode=mode)
            warped_t = apply_flow(flow._vecs, t, flow._ref)

        # Cut if necessary
        if padding is not None and cut:
            warped_t = warped_t[..., padding[0]:padding[0] + self.shape[0], padding[2]:padding[2] + self.shape[1]]

        # Extract and finalise mask if required
        if return_flow or return_valid_area:
            mask = torch.round(warped_t[-1]).to(torch.bool)
            # if self.ref == 's': Valid self.vecs already taken into account by ANDing with self.mask before warping
            if self._ref == 't':
                # Still need to take into account which self.vecs are actually valid by ANDing with self.mask
                if mask.shape != self._mask.shape:
                    # If padding is in use, but warped_t has not been cut: AND with self.mask inside the flow area, and
                    # set everything else to False as not warped by the flow
                    t = mask[padding[0]:padding[0] + self.shape[0], padding[2]:padding[2] + self.shape[1]].clone()
                    mask[...] = False
                    mask[padding[0]:padding[0] + self.shape[0], padding[2]:padding[2] + self.shape[1]] = t & self._mask
                else:
                    mask = mask & self._mask

        # Return as correct type
        if return_flow:
            return Flow(warped_t[:2, ...], target._ref, mask, device=return_device)
        else:
            if return_array:
                warped_t = to_numpy(warped_t)
            if return_valid_area:
                return warped_t[:-1, ...], mask
            else:
                return warped_t

    def track(
        self,
        pts: Union[np.ndarray, torch.Tensor],
        int_out: bool = None,
        get_valid_status: bool = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """Warps input points according to the flow field, can be returned as integers if required

        :param pts: Numpy array or torch tensor of points shape N-2, 1st coordinate vertical (height), 2nd coordinate
            horizontal (width).
        :param int_out: Boolean determining whether output points are returned as rounded integers, defaults to False
        :param get_valid_status: Boolean determining whether an array of shape N-2 containing the status of each point
            is returned. This will corresponds to self.valid_source() as applied to the point positions, and will show
            True for the points that are tracked by valid flow vectors, and end up inside the target image area.
        :return: Numpy array or torch tensor of warped ('tracked') points, and optionally an array or tensor (following
            the warped point type) of the point tracking status. The tensor device (if applicable) will be the same as
            the flow field device.
        """

        # Validate inputs
        pts_type = 'tensor'
        if isinstance(pts, np.ndarray):
            pts = torch.tensor(pts)
            pts_type = 'array'
        elif not isinstance(pts, torch.Tensor):
            raise TypeError("Error tracking points: Pts needs to be a numpy array or a torch tensor")
        if pts.ndim != 2:
            raise ValueError("Error tracking points: Pts needs to have shape N-2")
        if pts.shape[1] != 2:
            raise ValueError("Error tracking points: Pts needs to have shape N-2")
        int_out = False if int_out is None else int_out
        get_valid_status = False if get_valid_status is None else get_valid_status
        if not isinstance(int_out, bool):
            raise TypeError("Error tracking points: Int_out needs to be a boolean")
        if not isinstance(get_valid_status, bool):
            raise TypeError("Error tracking points: Get_tracked needs to be a boolean")

        pts = pts.to(self._device)

        if self.is_zero(thresholded=True):
            warped_pts = pts
        else:
            if self._ref == 's':
                # noinspection PyUnresolvedReferences
                if not pts.dtype.is_floating_point:
                    flow_vecs = self._vecs[:, pts[:, 0].long(), pts[:, 1].long()]  # flow_vecs shape 2-N
                    flow_vecs = flow_vecs.transpose(0, 1).flip(-1)  # flow_vecs shape N-2 and (y, x) instead of (x, y)
                else:
                    pts_4d = pts.unsqueeze(0).unsqueeze(0).to(torch.float).flip(-1)  # pts_4d now 1-1-N-2 and (x, y)
                    pts_4d = normalise_coords(pts_4d, self.shape)
                    # noinspection PyArgumentList
                    flow_vecs = f.grid_sample(self._vecs.unsqueeze(0), pts_4d, align_corners=True).flip(1)
                    #  vecs are 1-2-H-W, pts_4d is 1-1-N-2, output will be 1-2-1-N
                    flow_vecs = flow_vecs.transpose(0, -1).squeeze(-1).squeeze(-1)  # flow_vecs now N-2
                warped_pts = pts + flow_vecs
            else:  # self._ref == 't'
                x, y = np.mgrid[:self.shape[0], :self.shape[1]]
                grid = np.swapaxes(np.vstack([x.ravel(), y.ravel()]), 0, 1)
                flow_flat = np.reshape(self.vecs_numpy[..., ::-1], (-1, 2))
                origin_points = grid - flow_flat
                flow_vecs = griddata(origin_points, flow_flat, (pts[:, 0], pts[:, 1]), method='linear')
                warped_pts = pts + torch.tensor(flow_vecs, device=pts.device)
            nan_vals = torch.isnan(warped_pts)
            nan_vals = nan_vals[:, 0] | nan_vals[:, 1]
            warped_pts[nan_vals] = 0
        if int_out:
            warped_pts = torch.round(warped_pts).long()
        if pts_type == 'array':
            warped_pts = to_numpy(warped_pts)

        if get_valid_status:
            status_array = self.valid_source()[torch.round(pts[..., 0]).long(),
                                               torch.round(pts[..., 1]).long()]
            if pts_type == 'array':
                status_array = to_numpy(status_array)
            return warped_pts, status_array
        else:
            return warped_pts

    def switch_ref(self, mode: str = None) -> Flow:
        """Switches the reference coordinates from 's'ource to 't'arget, or vice versa

        :param mode: 'valid' or 'invalid':
            'invalid' means just the flow reference attribute is switched without any flow values being changed. This
                is functionally equivalent to simply using flow.ref = 't' for a flow of ref 's', and the flow vectors
                aren't changed.
            'valid' means actually switching the flow field to the other coordinate reference, with flow vectors being
                recalculated to correspond to this other reference.
        :return: Flow with switched coordinate reference.
        """

        mode = 'valid' if mode is None else mode
        if mode == 'valid':
            if self.is_zero(thresholded=False):  # In case the flow is 0, no further calculations are necessary
                return self.switch_ref(mode='invalid')
            else:
                if self._ref == 's':
                    switched_ref_flow = self.apply(self)  # apply_to is done s-based; see window pic 08/04/19
                    switched_ref_flow._ref = 't'
                    return switched_ref_flow
                elif self._ref == 't':
                    flow_copy_s = self.switch_ref(mode='invalid')  # so apply_to is ref-s; see window pic 08/04/19
                    return (-flow_copy_s).apply(flow_copy_s)
        elif mode == 'invalid':
            if self._ref == 's':
                return Flow(self._vecs, 't', self._mask)
            elif self._ref == 't':
                return Flow(self._vecs, 's', self._mask)
        else:
            raise ValueError("Error switching flow reference: Mode not recognised, should be 'valid' or 'invalid'")

    def invert(self, ref: str = None) -> Flow:
        """Inverting a flow: img1 -- f --> img2 becomes img1 <-- f -- img2

        The smaller the input flow, the closer the inverse is to simply multiplying the flow by -1.

        :param ref: Desired reference of the output field, defaults to reference of original flow field
        :return: Inverse flow field
        """

        ref = self._ref if ref is None else get_valid_ref(ref)
        if self._ref == 's':
            if ref == 's':
                return self.apply(-self)
            elif ref == 't':
                return Flow(-self._vecs, 't', self._mask)
        elif self._ref == 't':
            if ref == 's':
                return Flow(-self._vecs, 's', self._mask)
            elif ref == 't':
                return self.invert('s').switch_ref()

    def valid_target(self) -> torch.Tensor:
        """Finds the valid area in the target image

        Given source image, flow, and target image created by warping the source with the flow, the valid area is a
        boolean mask that is True wherever the value in the target stems from warping a value from the source, and
        False where no valid information is known. Pixels that are False in this valid area will often be black (or
        'empty') in the warped target image, but not necessarily, due to warping artefacts etc. Even when they are all
        empty, the valid area allows a distinction between pixels that are black due to no actual information being
        available at this position, and pixels that are black due to black pixel values having been warped to that
        location by the flow.

        :return: Boolean tensor of the valid area in the target image
        """

        if self._ref == 's':
            # Flow mask in 's' flow refers to valid flow vecs in the source image. Warping this mask to the target image
            # gives a boolean mask of which positions in the target image are valid, i.e. have been filled by values
            # warped there from the source by flow vectors that were themselves valid:
            # area = F{source & mask}, where: source & mask = mask, because: source = True everywhere
            area = apply_flow(self._vecs, self._mask.to(torch.float), self._ref)
            area = torch.round(area).to(torch.bool)
        else:  # ref is 't'
            # Flow mask in 't' flow refers to valid flow vecs in the target image. Therefore, warping a test array that
            # is true everywhere, ANDed with the flow mask, will yield a boolean mask of valid positions in the target
            # image, i.e. positions that have been filled by values warped there from the source by flow vectors that
            # were themselves valid:
            # area = F{source} & mask, where: source = True everywhere
            area = apply_flow(self._vecs, torch.ones(self.shape), self._ref)
            area = torch.round(area).to(torch.bool)
            area = area & self._mask
        return area

    def valid_source(self) -> torch.Tensor:
        """Finds the area in the source image that will end up being valid in the target image after warping

        Given source image, flow, and target image created by warping the source with the flow, the 'source area' is a
        boolean mask that is True wherever the value in the source will end up somewhere in the valid target area, and
        False where the value in the source will either be warped outside of the target image, or not be warped at all
        due to a lack of valid flow vectors connecting to this position.

        :return: Boolean tensor of the area in the source image valid in target image after warping
        """

        if self._ref == 's':
            # Flow mask in 's' flow refers to valid flow vecs in the source image. Therefore, to find the area in the
            # source image that will end up being valid in the target image after warping, equal to self.valid_target(),
            # warping a test array that is True everywhere from target to source with the inverse of the flow, ANDed
            # with the flow mask, will yield a boolean mask of valid positions in the source image:
            # area = F.inv{target} & mask, where target = True everywhere
            area = apply_flow(-self._vecs, torch.ones(self.shape), 't')
            # Note: this is equal to: area = self.invert('t').apply(np.ones(self.shape)), but more efficient as there
            # is no unnecessary warping of the mask
            area = torch.round(area).to(torch.bool)
            area = area & self._mask
        else:  # ref is 't'
            # Flow mask in 't' flow refers to valid flow vecs in the target image. Therefore, to find the area in the
            # source image that will end up being valid in the target image after warping, equal to self.valid_target(),
            # warping the flow mask from target to source with the inverse of the flow will yield a boolean mask of
            # valid positions in the source image:
            # area = F.inv{target & mask}, where target & mask = mask, because target = True everywhere
            area = apply_flow(-self._vecs, self._mask.to(torch.float), 's')
            # Note: this is equal to: area = self.invert('s').apply(self.mask.astype('f')), but more efficient as there
            # is no unnecessary warping of the mask
            area = torch.round(area).to(torch.bool)
        # Note: alternative way of seeing this: self.valid_source() = self.invert(<other ref>).valid_target()
        return area

    def get_padding(self) -> list:
        """Determine necessary padding from the flow field.

        When the flow reference is 't', this corresponds to the padding needed for an input image, so that the output
        when warped with the flow field contains no undefined areas inside defined flow areas.

        When the flow reference is 's', this corresponds to the padding needed for the flow so that applying it to an
        input image will result in no input image information being lost in the warped output, i.e each input image
        pixel will come to lie inside the padded area.

        :return: Padding as a list [top, bottom, left, right]
        """

        # Threshold to avoid very small flow values (possible artefacts) triggering a need for padding
        v = threshold_vectors(self._vecs)
        if self._ref == 's':
            v *= -1

        # Prepare grid
        grid_x, grid_y = torch.meshgrid(torch.arange(0, self.shape[0]), torch.arange(0, self.shape[1]))
        v[0] -= grid_y.to(self._device)
        v[1] -= grid_x.to(self._device)
        v *= -1

        # Calculate padding
        padding = [
            max(-torch.min(v[1, self._mask]), 0),
            max(torch.max(v[1, self._mask]) - (self.shape[0] - 1), 0),
            max(-torch.min(v[0, self._mask]), 0),
            max(torch.max(v[0, self._mask]) - (self.shape[1] - 1), 0)
        ]
        padding = [int(np.ceil(p)) for p in padding]
        return padding

        # x, y = np.mgrid[:self.shape[0], :self.shape[1]]
        # v[..., 0] -= y
        # v[..., 1] -= x
        # v *= -1
        # padding = [
        #     np.maximum(-np.min(v[self._mask, 1]), 0),
        #     np.maximum(np.max(v[self._mask, 1]) - (self.shape[0] - 1), 0),
        #     np.maximum(-np.min(v[self._mask, 0]), 0),
        #     np.maximum(np.max(v[self._mask, 0]) - (self.shape[1] - 1), 0)
        # ]
        # padding = [int(np.ceil(p)) for p in padding]
        # return padding

    def is_zero(self, thresholded: bool = None) -> bool:
        """Checks whether all flow vectors (where mask is True) are zero, thresholding if necessary.

        Flow vector magnitude threshold used is DEFAULT_THRESHOLD, defined at top of the utils file

        :param thresholded: Boolean determining whether the flow is thresholded, defaults to True
        :return: True if flow is zero, False if not
        """

        thresholded = True if thresholded is None else thresholded
        if not isinstance(thresholded, bool):
            raise TypeError("Error checking whether flow is zero: Thresholded needs to be a boolean")

        if thresholded:
            vecs = threshold_vectors(self._vecs)
        else:
            vecs = self._vecs
        return torch.all(vecs == 0)

    def visualise(
        self,
        mode: str,
        show_mask: bool = None,
        show_mask_borders: bool = None,
        range_max: float = None,
        return_tensor: bool = None
    ) -> torch.Tensor:
        """Returns a flow visualisation as a tensor containing an rgb/bgr/hsv img of the same shape as the flow

        NOTE: this currently runs internally based on NumPy & OpenCV, due to a lack of easily accessible function

        :param mode: Output mode, options: 'rgb', 'bgr', 'hsv'
        :param show_mask: Boolean determining whether the flow mask is visualised, defaults to False
        :param show_mask_borders: Boolean determining whether the flow mask border is visualised, defaults to False
        :param range_max: Maximum vector magnitude expected, corresponding to the HSV maximum Value of 255 when scaling
            the flow magnitudes. Defaults to the 99th percentile of the current flow field
        :param return_tensor: Boolean determining whether the result is returned as a tensor. Note that the result is
            originally a numpy array. Defaults to True
        :return: Tensor or array containing the flow visualisation as an rgb / bgr / hsv image of the same shape as the
            flow
        """

        show_mask = False if show_mask is None else show_mask
        show_mask_borders = False if show_mask_borders is None else show_mask_borders
        return_tensor = True if return_tensor is None else return_tensor
        if not isinstance(show_mask, bool):
            raise TypeError("Error visualising flow: Show_mask needs to be boolean")
        if not isinstance(show_mask_borders, bool):
            raise TypeError("Error visualising flow: Show_mask_borders needs to be boolean")
        if not isinstance(return_tensor, bool):
            raise TypeError("Error visualising flow: Return_tensor needs to be boolean")

        f = np.moveaxis(to_numpy(threshold_vectors(self._vecs)), 0, -1)
        # Threshold the flow: very small numbers can otherwise lead to issues when calculating mag / angle

        # Colourise the flow
        hsv = np.zeros((f.shape[0], f.shape[1], 3), 'f')
        mag, ang = cv2.cartToPolar(f[..., 0], f[..., 1], angleInDegrees=True)
        hsv[..., 0] = np.mod(ang, 360) / 2
        hsv[..., 2] = 255

        # Add mask if required
        if show_mask:
            hsv[np.invert(self.mask_numpy), 2] = 180

        # Scale flow
        if range_max is None:
            if np.percentile(mag, 99) > 0:  # Use 99th percentile to avoid extreme outliers skewing the scaling
                range_max = np.percentile(mag, 99)
            elif np.max(mag):  # If the 99th percentile is 0, use the actual maximum instead
                range_max = np.max(mag)
            else:  # If the maximum is 0 too (i.e. the flow field is entirely 0)
                range_max = 1
        if not isinstance(range_max, (float, int)):
            raise TypeError("Error visualising flow: Range_max needs to be an integer or a float")
        if range_max <= 0:
            raise ValueError("Error visualising flow: Range_max needs to be larger than zero")
        hsv[..., 1] = np.clip(mag * 255 / range_max, 0, 255)

        # Add mask borders if required
        if show_mask_borders:
            contours, hierarchy = cv2.findContours((255 * self.mask_numpy).astype('uint8'),
                                                   cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(hsv, contours, -1, (0, 0, 0), 1)

        # Process and return the flow visualisation
        if mode == 'hsv':
            return np.round(hsv).astype('uint8')
        elif mode == 'rgb' or mode == 'bgr':
            h = hsv[..., 0] / 180
            s = hsv[..., 1] / 255
            v = hsv[..., 2] / 255
            # Credit to stackoverflow.com/questions/24852345/hsv-to-rgb-color-conversion
            i = np.int_(h * 6.)
            f = h * 6. - i
            i = np.ravel(i)
            t = np.ravel(1. - f)
            f = np.ravel(f)
            i %= 6
            c_list = (1 - np.ravel(s) * np.vstack([np.zeros_like(f), np.ones_like(f), f, t])) * np.ravel(v)
            # 0:v 1:p 2:q 3:t
            order = np.array([[0, 3, 1], [2, 0, 1], [1, 0, 3], [1, 2, 0], [3, 1, 0], [0, 1, 2]])
            rgb = c_list[order[i], np.arange(np.prod(h.shape))[:, None]].reshape(*h.shape, 3)
            return_arr = np.round(rgb * 255).astype('uint8')
            if mode == 'bgr':
                return_arr = return_arr[..., ::-1]
            if return_tensor:
                return torch.tensor(return_arr.copy(), device=self._device)
            else:
                return return_arr
        else:
            raise ValueError("Error visualising flow: Mode needs to be either 'bgr', 'rgb', or 'hsv'")
