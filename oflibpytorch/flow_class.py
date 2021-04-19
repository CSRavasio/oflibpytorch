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
import numpy as np
from typing import Union
from .utils import get_valid_vecs, get_valid_ref, get_valid_device, get_valid_padding, validate_shape, to_numpy, \
    move_axis, flow_from_matrix, matrix_from_transforms, reverse_transform_values


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
