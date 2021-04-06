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
from .utils import get_valid_ref, get_valid_device, validate_shape, to_numpy, flow_from_matrix


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

        # Check type and dimensions
        if not isinstance(input_vecs, (np.ndarray, torch.Tensor)):
            raise TypeError("Error setting flow vectors: Input is not a numpy array or a torch tensor")
        if input_vecs.ndim != 3:
            raise ValueError("Error setting flow vectors: Input is not 3-dimensional")

        # Check channels
        transpose_necessary = False
        if input_vecs.shape[2] == 2:  # Input shape is H-W-2
            transpose_necessary = True
        elif input_vecs.shape[0] != 2:  # Input shape is neither H-W-2 nor 2-H-W
            raise ValueError("Error setting flow vectors: Input needs to be shape H-W-2 or 2-H-W")

        # Transform to tensor if necessary, transpose if necessary
        if isinstance(input_vecs, np.ndarray):
            input_vecs = torch.tensor(input_vecs, dtype=torch.float, device='cpu')
        if transpose_necessary:
            input_vecs = input_vecs.unsqueeze(0).transpose(0, -1).squeeze(-1)

        # Check for invalid values
        if not torch.isfinite(input_vecs).all():
            raise ValueError("Error setting flow vectors: Input contains NaN, Inf or -Inf values")
        self._vecs = input_vecs

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
            device = self.vecs.device.type
        else:
            device = get_valid_device(input_device)
        self._device = device
        self.vecs = self.vecs.to(device)
        self.mask = self.mask.to(device)

    @property
    def shape(self) -> tuple:
        """Gets shape (resolution) of the flow

        :return: Shape (resolution) of the flow field as a tuple
        """

        return tuple(self.vecs.shape[1:])

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
