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
import numpy as np
from typing import Union
from .utils import get_valid_vecs, get_valid_ref, get_valid_device, validate_shape, to_numpy, flow_from_matrix, \
    matrix_from_transforms


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
        np_vecs = np.moveaxis(to_numpy(self._vecs), 0, -1)
        return np_vecs

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
        return cls(torch.zeros(2, shape[0], shape[1]), ref, mask, device)

    @classmethod
    def from_matrix(
        cls,
        matrix: Union[np.ndarray, torch.Tensor],
        shape: Union[list, tuple],
        ref: str = None,
        mask: Union[np.ndarray, torch.Tensor] = None,
        device: str = None,
    ) -> Flow:
        """Flow object constructor, based on transformation matrix input

        :param matrix: Transformation matrix to be turned into a flow field, as numpy array or torch tensor of shape 3-3
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
        if device is not None:
            device = get_valid_device(device)
        # Check matrix validity
        if not isinstance(matrix, (np.ndarray, torch.Tensor)):
            raise TypeError("Error creating flow from matrix: Matrix needs to be a numpy array or a torch tensor")
        if matrix.shape != (3, 3):
            raise ValueError("Error creating flow from matrix: Matrix needs to be of shape (3, 3)")
        if isinstance(matrix, np.ndarray):
            matrix = torch.tensor(matrix)
        # Get valid ref
        ref = get_valid_ref(ref)

        if ref == 's':
            # Coordinates correspond to the meshgrid of the original ('s'ource) image. They are transformed according
            # to the transformation matrix. The start points are subtracted from the end points to yield flow vectors.
            flow_vectors = flow_from_matrix(matrix, shape)
            return cls(flow_vectors, ref, mask, device)
        elif ref == 't':
            # Coordinates correspond to the meshgrid of the warped ('t'arget) image. They are inversely transformed
            # according to the transformation matrix. The end points, which correspond to the flow origin for the
            # meshgrid in the warped image, are subtracted from the start points to yield flow vectors.
            flow_vectors = -flow_from_matrix(torch.pinverse(matrix), shape)
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

        # Here implemented: method 1, via calling from_matrix where the inverse of the matrix is used if reference 't'
        ref = get_valid_ref(ref)
        matrix = matrix_from_transforms(transform_list)
        return cls.from_matrix(matrix, shape, ref, mask, device)

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

        vecs = self._vecs.unsqueeze(-1).transpose(-1, 0).squeeze(0).__getitem__(item)
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
                mask = np.logical_and(self._mask, other._mask)
                return Flow(vecs, self._ref, mask)
        if isinstance(other, (np.ndarray, torch.Tensor)):
            vecs = self._vecs + get_valid_vecs(other, desired_shape=self.shape, error_string="Error adding to flow: ")
            return Flow(vecs, self._ref, self._mask, self._device)
        else:
            raise TypeError("Error adding to flow: Addend is not a flow object, numpy array, or torch tensor")
