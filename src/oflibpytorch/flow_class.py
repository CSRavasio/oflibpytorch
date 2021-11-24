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
# This file is part of oflibpytorch. It contains the custom flow class and makes use of functions defined in utils.

from __future__ import annotations
import torch
import torch.nn.functional as f
from scipy.interpolate import griddata
import numpy as np
import cv2
import math
from typing import Union, Tuple
import warnings
from .utils import get_valid_vecs, get_valid_ref, get_valid_device, get_valid_padding, validate_shape, to_numpy, \
    move_axis, flow_from_matrix, matrix_from_transforms, reverse_transform_values, apply_flow, threshold_vectors, \
    normalise_coords, load_kitti, load_sintel, load_sintel_mask


FlowAlias = 'Flow'


class Flow(object):
    _vecs: torch.Tensor
    _mask: torch.Tensor
    _ref: str
    _device: str

    def __init__(
        self,
        flow_vectors: Union[np.ndarray, torch.Tensor],
        ref: str = None,
        mask: Union[np.ndarray, torch.Tensor] = None,
        device: str = None
    ):
        """Flow object constructor. For a more detailed explanation of the arguments, see the class attributes
        :attr:`vecs`, :attr:`ref`, :attr:`mask`, and :attr:`device`.

        :param flow_vectors: Numpy array or pytorch tensor with 3 dimensions. The shape is interpreted as
            :math:`(2, H, W)` if the first dimension is 2, otherwise as :math:`(H, W, 2)` if the last dimension is 2.
            Throws a ``ValueError`` if neither the first nor the last dimensions is 2. The dimension that is 2 (the
            channel dimension) contains the flow vector in OpenCV convention: ``flow_vectors[..., 0]`` are the
            horizontal, ``flow_vectors[..., 1]`` are the vertical vector components, defined as positive when pointing
            to the right / down.
        :param ref: Flow reference, either ``t`` for "target", or ``s`` for "source". Defaults to ``t``
        :param mask: Numpy array or pytorch tensor of shape :math:`(H, W)` containing a boolean mask indicating where
            the flow vectors are valid. Defaults to ``True`` everywhere
        :param device: Tensor device, either ``cpu`` or ``cuda`` (if available). Defaults to the device of the given
            flow_vectors, or to ``cpu`` if they were given as a numpy array
        """

        # Fill attributes
        self.vecs = flow_vectors
        self.ref = ref
        self.mask = mask
        self.device = device

    @property
    def vecs(self) -> torch.Tensor:
        """Flow vectors, a torch tensor of shape :math:`(2, H, W)`. The first dimension contains the flow vectors.
        These are in the order horizontal component first, vertical component second (OpenCV convention). They
        are defined as positive towards the right and the bottom, meaning the origin is located in the left upper
        corner of the :math:`H \\times W` flow field area.

        :return: Flow vectors as torch tensor of shape :math:`(2, H, W)`, dtype ``float``, device ``self.device``
        """

        return self._vecs

    @vecs.setter
    def vecs(self, input_vecs: Union[np.ndarray, torch.Tensor]):
        """Sets flow vectors, after checking validity

        :param input_vecs: Numpy array or pytorch tensor with 3 dimensions. The shape is interpreted as
            :math:`(2, H, W)` if the first dimension is 2, otherwise as :math:`(H, W, 2)` if the last dimension is 2.
            Throws a ``ValueError`` if neither the first nor the last dimensions is 2. The dimension that is 2 (the
            channel dimension) contains the flow vector in OpenCV convention: ``flow_vectors[..., 0]`` are the
            horizontal, ``flow_vectors[..., 1]`` are the vertical vector components, defined as positive when pointing
            to the right / down.
        """

        self._vecs = get_valid_vecs(input_vecs, error_string="Error setting flow vectors: ")

    @property
    def vecs_numpy(self) -> np.ndarray:
        """Convenience function to get the flow vectors as a numpy array of shape :math:`(H, W, 2)`. Otherwise same
        as :attr:`vecs`: The last dimension contains the flow vectors, in the order of horizontal component first,
        vertical component second (OpenCV convention). They are defined as positive towards the right and the bottom,
        meaning the origin is located in the left upper corner of the :math:`H \\times W` flow field area.

        :return: Flow vectors as a numpy array of shape :math:`(H, W, 2)`, dtype ``float32``
        """

        with torch.no_grad():
            if self._device == 'cuda':
                vecs = self._vecs.cpu().numpy()
            else:  # self._device == 'cpu'
                vecs = self._vecs.detach().numpy()
        return np.moveaxis(vecs, 0, -1)

    @property
    def ref(self) -> str:
        """Flow reference, a string: either ``s`` for "source" or ``t`` for "target". This determines whether the
        regular grid of shape :math:`(H, W)` associated with the flow vectors should be understood as the source of the
        vectors (which then point to any other position), or the target of the vectors (whose start point can then be
        any other position). The flow reference ``t`` is the default, meaning the regular grid refers to the
        coordinates the pixels whose motion is being recorded by the vectors end up at.

        Applying a flow with reference ``s`` is known as "forward" warping, while reference ``t`` corresponds to what
        is termed "backward" or "reverse" warping.

        .. caution::

            The :meth:`~oflibpytorch.Flow.apply` method for warping an image is significantly faster with a flow in
            ``t`` reference. The reason is that this requires interpolating unstructured points from a regular grid,
            while reference ``s`` requires interpolating a regular grid from unstructured points. The former uses the
            fast PyTorch :func:`nn.functional.grid_sample` function, the latter is much more operationally complex and
            relies on the SciPy :func:`griddata` function.

        .. caution::

            The :meth:`~oflibpytorch.Flow.track` method for tracking points is significantly faster with a flow in ``s``
            reference, again due to not requiring a call to SciPy's :func:`griddata` function.

        .. tip::

            If some algorithm :func:`get_flow` is set up to calculate a flow field with reference ``t`` (or ``s``) as in
            ``flow_one_ref = get_flow(img1, img2)``, it is very simple to obtain the flow in reference ``s`` (or ``t``)
            instead: simply call the algorithm with the images in the reversed order, and multiply the resulting flow
            vectors by -1: ``flow_other_ref = -1 * get_flow(img2, img1)``

        :return: Flow reference, as string of value ``t`` or ``s``
        """

        return self._ref

    @ref.setter
    def ref(self, input_ref: str = None):
        """Sets flow reference, after checking validity

        :param input_ref: Flow reference as string of value ``t`` or ``s``. Defaults to ``t``
        """

        self._ref = get_valid_ref(input_ref)

    @property
    def mask(self) -> torch.Tensor:
        """Flow mask as a torch tensor of shape :math:`(H, W)` and type ``bool``. This array indicates, for each
        flow vector, whether it is considered "valid". As an example, this allows for masking of the flow based on
        object segmentations. It is also necessary to keep track of which flow vectors are valid when different flow
        fields are combined, as those operations often lead to undefined (partially or fully unknown) points in the
        given :math:`H \\times W` area where the flow vectors are either completely unknown, or will not have valid
        values.

        :return: Flow mask as a torch tensor of shape :math:`(H, W)` and type ``bool``
        """

        return self._mask

    @mask.setter
    def mask(self, input_mask: Union[np.ndarray, torch.Tensor] = None):
        """Sets flow mask, after checking validity

        :param input_mask: numpy array or torch tensor of shape :math:`(H, W)` and type ``bool``, matching flow vectors
            of shape :math:`(H, W, 2)`
        """

        if input_mask is None:
            self._mask = torch.ones(*self.shape).to(torch.bool)
        else:
            # Check type, dimensions, shape
            if not isinstance(input_mask, (np.ndarray, torch.Tensor)):
                raise TypeError("Error setting flow mask: Input is not a numpy array or a torch tensor")
            if len(input_mask.shape) != 2:
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
        """Convenience function to get the mask as a numpy array of shape :math:`(H, W)`. Otherwise same as
        :attr:`mask`: this array indicates, for each flow vector, whether it is considered "valid". As an example,
        this allows for masking of the flow based on object segmentations. It is also necessary to keep track of
        which flow vectors are valid when different flow fields are combined, as those operations often lead to
        undefined (partially or fully unknown) points in the given :math:`H \\times W` area where the flow vectors
        are either completely unknown, or will not have valid values.

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
        """The device of all flow object tensors, either ``cpu`` or ``cuda``

        :return: Tensor device as a string
        """

        return self._device

    @device.setter
    def device(self, input_device: str = None):
        """Sets the tensor device, after checking validity

        :param input_device: Tensor device, either ``cpu`` or ``cuda`` (if available). Defaults to ``cpu``
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
        """Shape (resolution) :math:`(H, W)` of the flow, corresponding to the last two dimensions of the flow
        vector tensor of shape :math:`(2, H, W)`

        :return: Tuple of the shape (resolution) :math:`(H, W)` of the flow object
        """

        return tuple(self._vecs.shape[1:])

    @classmethod
    def zero(
        cls,
        shape: Union[list, tuple],
        ref: str = None,
        mask: Union[np.ndarray, torch.Tensor] = None,
        device: str = None,
    ) -> FlowAlias:
        """Flow object constructor, zero everywhere

        :param shape: List or tuple of the shape :math:`(H, W)` of the flow field
        :param ref: Flow reference, string of value ``t`` ("target") or ``s`` ("source"). Defaults to ``t``
        :param mask: Numpy array or torch tensor of shape :math:`(H, W)` and type ``bool`` indicating where the flow
            vectors are valid. Defaults to ``True`` everywhere
        :param device: Tensor device, either ``cpu`` or ``cuda`` (if available). Defaults to ``cpu``
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
    ) -> FlowAlias:
        """Flow object constructor, based on transformation matrix input

        :param matrix: Transformation matrix to be turned into a flow field, as numpy array or torch tensor of
            shape :math:`(3, 3)`
        :param shape: List or tuple of the shape :math:`(H, W)` of the flow field
        :param ref: Flow reference, string of value ``t`` ("target") or ``s`` ("source"). Defaults to ``t``
        :param mask: Numpy array or torch tensor of shape :math:`(H, W)` and type ``bool`` indicating where the flow
            vectors are valid. Defaults to ``True`` everywhere
        :param device: Tensor device, either ``cpu`` or ``cuda`` (if available). Defaults to ``cpu``
        :param matrix_is_inverse: Boolean determining whether the given matrix is already the inverse of the desired
            transformation. Is useful for flow with reference ``t`` to avoid calculation of the pseudo-inverse, but
            will throw a ``ValueError`` if used for flow with reference ``s`` to avoid accidental usage.
            Defaults to ``False``
        :return: Flow object
        """

        # Check shape validity
        validate_shape(shape)
        # Get valid device
        # device = get_valid_device(device)  # Actually not needed, as done in Flow device setter anyway
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
    ) -> FlowAlias:
        """Flow object constructor, based on list of transforms

        :param transform_list: List of transforms to be turned into a flow field, where each transform is expressed as
            a list of [``transform name``, ``transform value 1``, ... , ``transform value n``]. Supported options:

            - Transform ``translation``, with values ``horizontal shift in px``, ``vertical shift in px``
            - Transform ``rotation``, with values ``horizontal centre in px``, ``vertical centre in px``,
              ``angle in degrees, counter-clockwise``
            - Transform ``scaling``, with values ``horizontal centre in px``, ``vertical centre in px``,
              ``scaling fraction``
        :param shape: List or tuple of the shape :math:`(H, W)` of the flow field
        :param ref: Flow reference, string of value ``t`` ("target") or ``s`` ("source"). Defaults to ``t``
        :param mask: Numpy array or torch tensor of shape :math:`(H, W)` and type ``bool`` indicating where the flow
            vectors are valid. Defaults to ``True`` everywhere
        :param device: Tensor device, either ``cpu`` or ``cuda`` (if available). Defaults to ``cpu``
        :return: Flow object
        """

        # Check shape validity
        validate_shape(shape)
        # Get valid device
        # device = get_valid_device(device)  # Actually not needed, as done in Flow device setter anyway
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

    @classmethod
    def from_kitti(cls, path: str, load_valid: bool = None, device: str = None) -> FlowAlias:
        """Loads the flow field contained in KITTI ``uint16`` png images files, optionally including the valid pixels.
        Follows the official instructions on how to read the provided .png files

        :param path: String containing the path to the KITTI flow data (``uint16``, .png file)
        :param load_valid: Boolean determining whether the valid pixels are loaded as the flow :attr:`mask`. Defaults
            to ``True``
        :param device: Tensor device, either ``cpu`` or ``cuda`` (if available). Defaults to ``cpu``
        :return: A flow object corresponding to the KITTI flow data, with flow reference :attr:`ref` ``s``.
        """

        load_valid = True if load_valid is None else load_valid
        if not isinstance(load_valid, bool):
            raise TypeError("Error loading flow from KITTI data: Load_valid needs to be boolean")

        data = load_kitti(path)
        if load_valid:
            return cls(data[:2, ...], 's', data[2, ...], device=device)
        else:
            return cls(data[:2, ...], 's', device=device)

    @classmethod
    def from_sintel(cls, path: str, inv_path: str = None, device: str = None) -> FlowAlias:
        """Loads the flow field contained in Sintel .flo byte files, including the invalid pixels if required. Follows
        the official instructions provided alongside the .flo data.

        :param path: String containing the path to the Sintel flow data (.flo byte file, little Endian)
        :param inv_path: String containing the path to the Sintel invalid pixel data (.png, black and white)
        :param device: Tensor device, either ``cpu`` or ``cuda`` (if available). Defaults to ``cpu``
        :return: A flow object corresponding to the Sintel flow data, with flow reference :attr:`ref` ``s``
        """

        flow = load_sintel(path)
        mask = None if inv_path is None else load_sintel_mask(inv_path)
        return cls(flow, 's', mask, device=device)

    def copy(self) -> FlowAlias:
        """Copy a flow object by constructing a new one with the same vectors :attr:`vecs`, reference :attr:`ref`,
        mask :attr:`mask`, and device :attr:`device`

        :return: Copy of the flow object
        """

        return Flow(self._vecs, self._ref, self._mask, self._device)

    def to_device(self, device) -> FlowAlias:
        """Returns a new flow object on the desired torch device

        :param device: Device the flow object is to be moved to, either ``cpu`` or ``cuda`` (if available)
        :return: New flow object on the desired torch device
        """

        device = get_valid_device(device)
        return Flow(self._vecs.to(device), self._ref, self._mask.to(device), device)

    def __str__(self) -> str:
        """Enhanced string representation of the flow object, containing the flow reference :attr:`ref`, shape
        :attr:`shape`, and device :attr:`device`

        :return: String representation
        """

        info_string = "Flow object, reference {}, shape {}*{}, device {}; ".format(self._ref, *self.shape, self._device)
        info_string += self.__repr__()
        return info_string

    def __getitem__(self, item: Union[int, list, slice]) -> FlowAlias:
        """Mimics ``__getitem__`` of a torch tensor, returning a new flow object cut accordingly

        Will throw an error if ``mask.__getitem__(item)`` or ``vecs.__getitem__(item)`` (corresponding to
        ``mask[item]`` and ``vecs[item]``) throw an error. Also throws an error if sliced :attr:`vecs` or :attr:`mask`
        are not suitable to construct a new flow object with, e.g. if the number of dimensions is too low.

        :param item: Slice used to select a part of the flow
        :return: New flow object cut as a corresponding torch tensor would be cut
        """

        vecs = move_axis(move_axis(self._vecs, 0, -1).__getitem__(item), -1, 0)
        # Above line is to avoid having to parse item properly to deal with first dim of 2: move this dim to the back
        return Flow(vecs, self._ref, self._mask.__getitem__(item), self._device)

    def __add__(self, other: Union[np.ndarray, torch.Tensor, FlowAlias]) -> FlowAlias:
        """Adds a flow object, a numpy array, or a torch tensor to a flow object

        .. caution::
            This is **not** equal to applying the two flows sequentially. For that, use
            :func:`~oflibpytorch.combine_flows` with ``mode`` set to ``3``.

        .. caution::
            If this method is used to add two flow objects, there is no check on whether they have the same reference
            :attr:`ref`.

        :param other: Flow object, numpy array, or torch tensor corresponding to the addend. Adding a flow object will
            adjust the mask of the resulting flow object to correspond to the logical union of the augend / addend masks
        :return: New flow object corresponding to the sum
        """

        if isinstance(other, Flow):
            if self.shape != other.shape:
                raise ValueError("Error adding to flow: Augend and addend flow objects are not the same shape")
            else:
                vecs = self._vecs + other._vecs.to(self._vecs.device)
                mask = self._mask & other._mask.to(self._vecs.device)
                return Flow(vecs, self._ref, mask)
        if isinstance(other, (np.ndarray, torch.Tensor)):
            other = get_valid_vecs(other, desired_shape=self.shape, error_string="Error adding to flow: ")
            vecs = self._vecs + other.to(self._vecs.device)
            return Flow(vecs, self._ref, self._mask, self._device)
        else:
            raise TypeError("Error adding to flow: Addend is not a flow object, numpy array, or torch tensor")

    def __sub__(self, other: Union[np.ndarray, torch.Tensor, FlowAlias]) -> FlowAlias:
        """Subtracts a flow object, a numpy array, or a torch tensor from a flow object

        .. caution::
            This is **not** equal to subtracting the effects of applying flow fields to an image. For that, use
            :func:`~oflibpytorch.combine_flows` with ``mode`` set to ``1`` or ``2``.

        .. caution::
            If this method is used to subtract two flow objects, there is no check on whether they have the same
            reference :attr:`ref`.

        :param other: Flow object, numpy array, or torch tensor corresponding to the subtrahend. Subtracting a flow
            object will adjust the mask of the resulting flow object to correspond to the logical union of the
            minuend / subtrahend masks
        :return: New flow object corresponding to the difference
        """

        if isinstance(other, Flow):
            if self.shape != other.shape:
                raise ValueError("Error subtracting from flow: "
                                 "Minuend and subtrahend flow objects are not the same shape")
            else:
                vecs = self._vecs - other._vecs.to(self._vecs.device)
                mask = self._mask & other._mask.to(self._vecs.device)
                return Flow(vecs, self._ref, mask, self._device)
        if isinstance(other, (np.ndarray, torch.Tensor)):
            other = get_valid_vecs(other, desired_shape=self.shape, error_string="Error subtracting to flow: ")
            vecs = self._vecs - other.to(self._vecs.device)
            return Flow(vecs, self._ref, self._mask, self._device)
        else:
            raise TypeError("Error subtracting from flow: "
                            "Subtrahend is not a flow object, numpy array, or torch tensor")

    def __mul__(self, other: Union[float, int, bool, list, np.ndarray, torch.Tensor]) -> FlowAlias:
        """Multiplies a flow object with a single number, a list, a numpy array, or a torch tensor

        :param other: Multiplier, options:

            - can be converted to a float
            - a list of shape :math:`(2)`
            - a numpy array or torch tensor of the same shape :math:`(H, W)` as the flow object
            - a numpy array or torch tensor of the same shape :math:`(H, W, 2)` as the flow vectors
        :return: New flow object corresponding to the product
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
                if other.dim() == 1 and other.shape[0] == 2:  # shape 2 to 2-1-1
                    other = other.unsqueeze(-1).unsqueeze(-1)
                elif other.dim() == 2 and other.shape == self.shape:  # shape H-W to 2-H-W
                    other = other.unsqueeze(0)
                elif other.dim() == 3 and other.shape == (2,) + self.shape:  # shape 2-H-W: all OK
                    pass
                elif other.dim() == 3 and other.shape == self.shape + (2,):  # shape H-W-2 to 2-H-W
                    other = move_axis(other, -1, 0)
                else:
                    raise ValueError("Error multiplying flow: Multiplier array or tensor needs to be of size 2, of the "
                                     "shape of the flow object (H-W), or either 2-H-W or H-W-2")
                other = other.to(self._vecs.device)
                return Flow(self._vecs * other, self._ref, self._mask, self._device)
            else:
                raise TypeError("Error multiplying flow: Multiplier cannot be converted to float, "
                                "or isn't a list, numpy array, or torch tensor")

    def __truediv__(self, other: Union[float, int, bool, list, np.ndarray, torch.Tensor]) -> FlowAlias:
        """Divides a flow object by a single number, a list, a numpy array, or a torch tensor

        :param other: Divisor, options:

            - can be converted to a float
            - a list of shape :math:`(2)`
            - a numpy array or torch tensor of the same shape :math:`(H, W)` as the flow object
            - a numpy array or torch tensor of the same shape :math:`(H, W, 2)` as the flow vectors
        :return: New flow object corresponding to the quotient
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
                if other.dim() == 1 and other.shape[0] == 2:  # shape 2 to 2-1-1
                    other = other.unsqueeze(-1).unsqueeze(-1)
                elif other.dim() == 2 and other.shape == self.shape:  # shape H-W to 2-H-W
                    other = other.unsqueeze(0)
                elif other.dim() == 3 and other.shape == (2,) + self.shape:  # shape 2-H-W: all OK
                    pass
                elif other.dim() == 3 and other.shape == self.shape + (2,):  # shape H-W-2 to 2-H-W
                    other = move_axis(other, -1, 0)
                else:
                    raise ValueError("Error dividing flow: Divisor array or tensor needs to be of size 2, of the "
                                     "shape of the flow object (H-W), or either 2-H-W or H-W-2")
                other = other.to(self._vecs.device)
                return Flow(self._vecs / other, self._ref, self._mask, self._device)
            else:
                raise TypeError("Error dividing flow: Divisor cannot be converted to float, "
                                "or isn't a list, numpy array, or torch tensor")

    def __pow__(self, other: Union[float, int, bool, list, np.ndarray, torch.Tensor]) -> FlowAlias:
        """Exponentiates a flow object by a single number, a list, a numpy array, or a torch tensor

        :param other: Exponent, options:

            - can be converted to a float
            - a list of shape :math:`(2)`
            - a numpy array or torch tensor of the same shape :math:`(H, W)` as the flow object
            - a numpy array or torch tensor of the same shape :math:`(H, W, 2)` as the flow vectors
        :return: New flow object corresponding to the power
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
                if other.dim() == 1 and other.shape[0] == 2:  # shape 2 to 2-1-1
                    other = other.unsqueeze(-1).unsqueeze(-1)
                elif other.dim() == 2 and other.shape == self.shape:  # shape H-W to 2-H-W
                    other = other.unsqueeze(0)
                elif other.dim() == 3 and other.shape == (2,) + self.shape:  # shape 2-H-W: all OK
                    pass
                elif other.dim() == 3 and other.shape == self.shape + (2,):  # shape H-W-2 to 2-H-W
                    other = move_axis(other, -1, 0)
                else:
                    raise ValueError("Error exponentiating flow: Exponent array or tensor needs to be of size 2, of "
                                     "the shape of the flow object (H-W), or either 2-H-W or H-W-2")
                other = other.to(self._vecs.device)
                return Flow(self._vecs ** other, self._ref, self._mask, self._device)
            else:
                raise TypeError("Error exponentiating flow: Exponent cannot be converted to float, "
                                "or isn't a list, numpy array, or torch tensor")

    def __neg__(self) -> FlowAlias:
        """Returns a new flow object with all the flow vectors inverted

        .. caution::
            This is **not** equal to inverting the transformation a flow field corresponds to! For that, use
            :meth:`~oflibpytorch.Flow.invert`.

        :return: New flow object with inverted flow vectors
        """

        return self * -1

    def resize(self, scale: Union[float, int, list, tuple]) -> FlowAlias:
        """Resize a flow field, scaling the flow vectors values :attr:`vecs` accordingly.

        :param scale: Scale used for resizing, options:

            - Integer or float of value ``scaling`` applied both vertically and horizontally
            - List or tuple of shape :math:`(2)` with values ``[vertical scaling, horizontal scaling]``
        :return: New flow object scaled as desired
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

    def pad(self, padding: list = None, mode: str = None) -> FlowAlias:
        """Pad the flow with the given padding. Padded flow :attr:`vecs` values are either constant (set to ``0``),
        reflect the existing flow values along the edges, or replicate those edge values.
        Padded :attr:`mask` values are set to ``False``.

        :param padding: List or tuple of shape :math:`(4)` with padding values ``[top, bot, left, right]``
        :param mode: String of the numpy padding mode for the flow vectors, with options ``constant`` (fill value
            ``0``), ``reflect``, ``replicate`` (see documentation for :func:`torch.nn.functional.pad`).
            Defaults to ``constant``
        :return: New flow object with the padded flow field
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
        target: Union[torch.Tensor, FlowAlias],
        target_mask: torch.Tensor = None,
        return_valid_area: bool = None,
        consider_mask: bool = None,
        padding: list = None,
        cut: bool = None
    ) -> Union[Union[torch.Tensor, FlowAlias], Tuple[Union[torch.Tensor, FlowAlias], torch.Tensor]]:
        """Apply the flow to a target, which can be a torch tensor or a Flow object itself. If the flow shape
        :math:`(H_{flow}, W_{flow})` is smaller than the target shape :math:`(H_{target}, W_{target})`, a list of
        padding values needs to be passed to localise the flow in the larger :math:`H_{target} \\times W_{target}` area.

        The valid image area that can optionally be returned is ``True`` where the image values in the function output:

        1) have been affected by flow vectors. If the flow has a reference :attr:`ref` value of ``t`` ("target"),
           this is always ``True`` as the target image by default has a corresponding flow vector at each pixel
           location in :math:`H \\times W`. If the flow has a reference :attr:`ref` value of ``s`` ("source"), this
           is only ``True`` for some parts of the image: some target image pixel locations in :math:`H \\times W`
           would only be reachable by flow vectors originating outside of the source image area, which is impossible
           by definition
        2) have been affected by flow vectors that were themselves valid, as determined by the flow mask

        .. caution::

            The parameter `consider_mask` relates to whether the invalid flow vectors in a flow field with reference
            ``s`` are removed before application (default behaviour) or not. Doing so results in a smoother flow field,
            but can cause artefacts to arise where the outline of the area returned by
            :meth:`~oflibpytorch.Flow.valid_target` is not a convex hull. For a more detailed explanation with an
            illustrative example, see the section ":ref:`Applying a Flow`" in the usage documentation.

        :param target: Torch tensor of shape :math:`(H, W)` or :math:`(H, W, C)`, or a flow object of shape
            :math:`(H, W)` to which the flow should be applied, where :math:`H` and :math:`W` are equal or larger
            than the corresponding dimensions of the flow itself
        :param target_mask: Optional torch tensor of shape :math:`(H, W)` and type ``bool`` that indicates which part
            of the target is valid (only relevant if `target` is a numpy array). Defaults to ``True`` everywhere
        :param return_valid_area: Boolean determining whether the valid image area is returned (only if the target is a
            numpy array), defaults to ``False``. The valid image area is returned as a boolean torch tensor of shape
            :math:`(H, W)`.
        :param consider_mask: Boolean determining whether the flow vectors are masked before application (only relevant
            for flows with reference ``ref = 's'``). Results in smoother outputs, but more artefacts. Defaults to
            ``True``
        :param padding: List or tuple of shape :math:`(4)` with padding values ``[top, bottom, left, right]``. Required
            if the flow and the target don't have the same shape. Defaults to ``None``, which means no padding needed
        :param cut: Boolean determining whether the warped target is returned cut from :math:`(H_{target}, W_{target})`
            to :math:`(H_{flow}, W_{flow})`, in the case that the shapes are not the same. Defaults to ``True``
        :return: The warped target of the same shape :math:`(C, H, W)` and type as the input (rounded if necessary),
            and optionally the valid area of the flow as a boolean torch tensor of shape :math:`(H, W)`.
        """

        return_valid_area = False if return_valid_area is None else return_valid_area
        if not isinstance(return_valid_area, bool):
            raise TypeError("Error applying flow: Return_valid_area needs to be a boolean")
        consider_mask = True if consider_mask is None else consider_mask
        if not isinstance(consider_mask, bool):
            raise TypeError("Error applying flow: Consider_mask needs to be a boolean")
        cut = True if cut is None else cut
        if not isinstance(cut, bool):
            raise TypeError("Error applying flow: Cut needs to be a boolean")
        if padding is not None:
            padding = get_valid_padding(padding, "Error applying flow: ")
            if self.shape[0] + padding[0] + padding[1] != target.shape[-2] or \
                    self.shape[1] + padding[2] + padding[3] != target.shape[-1]:
                raise ValueError("Error applying flow: Padding values do not match flow and target shape difference")

        # Type check, prepare arrays
        return_dtype = torch.float
        return_2d = False
        if isinstance(target, Flow):
            return_flow = True
            t = target._vecs.to(self._vecs.device)
            mask = target._mask.unsqueeze(0).to(self._vecs.device)
        elif isinstance(target, torch.Tensor):
            return_flow = False
            if target.dim() == 3:
                t = target.to(self._vecs.device)
            elif target.dim() == 2:
                return_2d = True
                t = target.unsqueeze(0).to(self._vecs.device)
            else:
                raise ValueError("Error applying flow: Target needs to have the shape H-W (2 dimensions) "
                                 "or H-W-C (3 dimensions)")
            if target_mask is None:
                mask = torch.ones(1, *t.shape[1:]).to(torch.bool).to(self._vecs.device)
            else:
                if not isinstance(target_mask, torch.Tensor):
                    raise TypeError("Error applying flow: Target_mask needs to be a torch tensor")
                if target_mask.shape != t.shape[1:]:
                    raise ValueError("Error applying flow: Target_mask needs to match the target shape")
                if target_mask.dtype != torch.bool:
                    raise TypeError("Error applying flow: Target_mask needs to have dtype 'bool'")
                if not return_valid_area:
                    warnings.warn("Warning applying flow: a mask is passed, but return_valid_area is False - so the "
                                  "mask passed will not affect the output, but possibly make the function slower.")
                mask = target_mask.unsqueeze(0).to(self._vecs.device)
            return_dtype = target.dtype
        else:
            raise ValueError("Error applying flow: Target needs to be either a flow object or a torch tensor")

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
            t = torch.cat((t.float(), mask.float()), dim=0)

        # Determine flow to use for warping, and warp
        if padding is None:
            if not target.shape[-2:] == self.shape:
                raise ValueError("Error applying flow: Flow and target have to have the same shape")
            warped_t = apply_flow(self._vecs, t, self._ref, self._mask if consider_mask else None)
        else:
            mode = 'constant' if self._ref == 't' else 'replicate'
            # Note: this mode is very important: irrelevant for flow with reference 't' as this by definition covers
            # the area of the target image, so 'constant' (defaulting to filling everything with 0) is fine. However,
            # for flows with reference 's', if locations in the source image with some flow vector border padded
            # locations with flow zero, very strange interpolation artefacts will result, both in terms of the image
            # being warped, and the mask being warped. By padding with the 'edge' mode, large gradients in flow vector
            # values at the edge of the original flow area are avoided, as are interpolation artefacts.
            flow = self.pad(padding, mode=mode)
            warped_t = apply_flow(flow._vecs, t, flow._ref, flow._mask if consider_mask else None)

        # Cut if necessary
        if padding is not None and cut:
            warped_t = warped_t[..., padding[0]:padding[0] + self.shape[0], padding[2]:padding[2] + self.shape[1]]

        # Extract and finalise mask if required
        if return_flow or return_valid_area:
            mask = warped_t[-1] > 0.99999
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
            return Flow(warped_t[:2, :, :], target._ref, mask)
        else:
            if return_valid_area:
                warped_t = warped_t[:-1, :, :]
            # noinspection PyUnresolvedReferences
            if not return_dtype.is_floating_point:
                warped_t = torch.round(warped_t.float())
            if return_2d:
                warped_t = warped_t[0, :, :]
            if return_valid_area:
                return warped_t.to(return_dtype), mask
            else:
                return warped_t.to(return_dtype)

    def track(
        self,
        pts: torch.Tensor,
        int_out: bool = None,
        get_valid_status: bool = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Warp input points with the flow field, returning the warped point coordinates as integers if required

        .. tip::
            Calling :meth:`~oflibpytorch.Flow.track` on a flow field with reference :attr:`ref` ``s`` ("source") is
            significantly faster (as long as `s_exact_mode` is not set to ``True``), as this does not require a call to
            :func:`scipy.interpolate.griddata`.

        :param pts: Torch tensor of shape :math:`(N, 2)` containing the point coordinates. ``pts[:, 0]`` corresponds
            to the vertical coordinate, ``pts[:, 1]`` to the horizontal coordinate
        :param int_out: Boolean determining whether output points are returned as rounded integers, defaults to
            ``False``
        :param get_valid_status: Boolean determining whether a tensor of shape :math:`(N, 2)` is returned, which
            contains the status of each point. This corresponds to applying :meth:`~oflibpytorch.Flow.valid_source` to
            the point positions, and returns ``True`` for the points that 1) tracked by valid flow vectors, and 2) end
            up inside the flow area of :math:`H \\times W`. Defaults to ``False``
        :return: Torch tensor of warped ('tracked') points, and optionally a torch tensor of the point tracking status.
            The tensor device (if applicable) will be the same as the flow field device.
        """

        # Validate inputs
        if not isinstance(pts, torch.Tensor):
            raise TypeError("Error tracking points: Pts needs to be a numpy array or a torch tensor")
        if pts.dim() != 2:
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
                    torch_version = globals()['torch'].__version__
                    if int(torch_version[0]) == 1 and float(torch_version[2:4]) <= 3:
                        flow_vecs = f.grid_sample(self._vecs.unsqueeze(0), pts_4d).flip(1)
                    else:
                        # noinspection PyArgumentList
                        flow_vecs = f.grid_sample(self._vecs.unsqueeze(0), pts_4d, align_corners=True).flip(1)
                    #  vecs are 1-2-H-W, pts_4d is 1-1-N-2, output will be 1-2-1-N
                    flow_vecs = flow_vecs.transpose(0, -1).squeeze(-1).squeeze(-1)  # flow_vecs now N-2
                warped_pts = pts.float() + flow_vecs
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

        if get_valid_status:
            # noinspection PyUnresolvedReferences
            if pts.dtype.is_floating_point:
                pts = torch.round(pts)
            status_array = self.valid_source()[pts[..., 0].long(), pts[..., 1].long()]
            return warped_pts, status_array
        else:
            return warped_pts

    def switch_ref(self, mode: str = None) -> FlowAlias:
        """Switch the reference :attr:`ref` between ``s`` ("source") and ``t`` ("target")

        .. caution::

            Do not use ``mode=invalid`` if avoidable: it does not actually change any flow values, and the resulting
            flow object, when applied to an image, will no longer yield the correct result.

        :param mode: Mode used for switching, available options:

            - ``invalid``: just the flow reference attribute is switched without any flow values being changed. This
              is functionally equivalent to simply assigning ``flow.ref = 't'`` for a "source" flow or
              ``flow.ref = 's'`` for a "target" flow
            - ``valid``: the flow field is switched to the other coordinate reference, with flow vectors recalculated
              accordingly
        :return: New flow object with switched coordinate reference
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

    def invert(self, ref: str = None) -> FlowAlias:
        """Inverting a flow: `img`\\ :sub:`1` -- `f` --> `img`\\ :sub:`2` becomes `img`\\ :sub:`1` <-- `f` --
        `img`\\ :sub:`2`. The smaller the input flow, the closer the inverse is to simply multiplying the flow by -1.

        :param ref: Desired reference of the output field, defaults to the reference of original flow field
        :return: New flow object, inverse of the original
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

    def valid_target(self, consider_mask: bool = None) -> torch.Tensor:
        """Find the valid area in the target domain

        Given a source image and a flow, both of shape :math:`(H, W)`, the target image is created by warping the source
        with the flow. The valid area is then a boolean numpy array of shape :math:`(H, W)` that is ``True`` wherever
        the value in the target img stems from warping a value from the source, and ``False`` where no valid information
        is known.

        Pixels that are ``False`` will often be black (or 'empty') in the warped target image - but not necessarily, due
        to warping artefacts etc. The valid area also allows a distinction between pixels that are black due to no
        actual information being available at this position (validity ``False``), and pixels that are black due to black
        pixel values having been warped to that (valid) location by the flow.

        :param consider_mask: Boolean determining whether the flow vectors are masked before application (only relevant
            for flows with reference ``ref = 's'``, analogous to :meth:`~oflibpytorch.Flow.apply`). Results in smoother
            outputs, but more artefacts. Defaults to ``True``
        :return: Boolean torch tensor of the same shape :math:`(H, W)` as the flow
        """

        consider_mask = True if consider_mask is None else consider_mask
        if not isinstance(consider_mask, bool):
            raise TypeError("Error applying flow: Consider_mask needs to be a boolean")
        if self._ref == 's':
            # Flow mask in 's' flow refers to valid flow vecs in the source image. Warping this mask to the target image
            # gives a boolean mask of which positions in the target image are valid, i.e. have been filled by values
            # warped there from the source by flow vectors that were themselves valid:
            # area = F{source & mask}, where: source & mask = mask, because: source = True everywhere
            area = apply_flow(self._vecs, self._mask.to(torch.float), 's', self._mask if consider_mask else None)
            area = area == 1
        else:  # ref is 't'
            # Flow mask in 't' flow refers to valid flow vecs in the target image. Therefore, warping a test array that
            # is true everywhere, ANDed with the flow mask, will yield a boolean mask of valid positions in the target
            # image, i.e. positions that have been filled by values warped there from the source by flow vectors that
            # were themselves valid:
            # area = F{source} & mask, where: source = True everywhere
            area = apply_flow(self._vecs, torch.ones(self.shape), 't')
            area = area > 0.9999
            area = area & self._mask
        return area

    def valid_source(self, consider_mask: bool = None) -> torch.Tensor:
        """Finds the area in the source domain that will end up being valid in the target domain (see
        :meth:`~oflibpytorch.Flow.valid_target`) after warping

        Given a source image and a flow, both of shape :math:`(H, W)`, the target image is created by warping the source
        with the flow. The source area is then a boolean numpy array of shape :math:`(H, W)` that is ``True`` wherever
        the value in the source will end up somewhere inside the valid target area, and ``False`` where the value in the
        source will either be warped outside of the target image, or not be warped at all due to a lack of valid flow
        vectors connecting to this position.

        :param consider_mask: Boolean determining whether the flow vectors are masked before application (only relevant
            for flows with reference ``ref = 't'`` as their inverse flow will be applied, using the reference ``s``;
            analogous to :meth:`~oflibpytorch.Flow.apply`). Results in smoother outputs, but more artefacts. Defaults
            to ``True``
        :return: Boolean torch tensor of the same shape :math:`(H, W)` as the flow
        """

        consider_mask = True if consider_mask is None else consider_mask
        if not isinstance(consider_mask, bool):
            raise TypeError("Error applying flow: Consider_mask needs to be a boolean")
        if self._ref == 's':
            # Flow mask in 's' flow refers to valid flow vecs in the source image. Therefore, to find the area in the
            # source image that will end up being valid in the target image after warping, equal to self.valid_target(),
            # warping a test array that is True everywhere from target to source with the inverse of the flow, ANDed
            # with the flow mask, will yield a boolean mask of valid positions in the source image:
            # area = F.inv{target} & mask, where target = True everywhere
            area = apply_flow(-self._vecs, torch.ones(self.shape), 't')
            # Note: this is equal to: area = self.invert('t').apply(np.ones(self.shape)), but more efficient as there
            # is no unnecessary warping of the mask
            area = area > 0.9999
            area = area & self._mask
        else:  # ref is 't'
            # Flow mask in 't' flow refers to valid flow vecs in the target image. Therefore, to find the area in the
            # source image that will end up being valid in the target image after warping, equal to self.valid_target(),
            # warping the flow mask from target to source with the inverse of the flow will yield a boolean mask of
            # valid positions in the source image:
            # area = F.inv{target & mask}, where target & mask = mask, because target = True everywhere
            area = apply_flow(-self._vecs, self._mask.to(torch.float), 's', self._mask if consider_mask else None)
            # Note: this is equal to: area = self.invert('s').apply(self.mask.astype('f')), but more efficient as there
            # is no unnecessary warping of the mask
            area = area == 1
        # Note: alternative way of seeing this: self.valid_source() = self.invert(<other ref>).valid_target()
        return area

    def get_padding(self) -> list:
        """Determine necessary padding from the flow field:

        - When the flow reference :attr:`ref` has the value ``t`` ("target"), this corresponds to the padding needed in
          a source image which ensures that every flow vector in :attr:`vecs` marked as valid by the
          mask :attr:`mask` will find a value in the source domain to warp towards the target domain. I.e. any invalid
          locations in the area :math:`H \\times W` of the target domain (see :meth:`~oflibpytorch.Flow.valid_target`)
          are purely due to no valid flow vector being available to pull a source value to this target location, rather
          than no source value being available in the first place.
        - When the flow reference :attr:`ref` has the value ``s`` ("source"), this corresponds to the padding needed for
          the flow itself, so that applying it to a source image will result in no input image information being lost in
          the warped output, i.e each input image pixel will come to lie inside the padded area.

        :return: A list of shape :math:`(4)` with the values ``[top, bottom, left, right]``
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
        """Check whether all flow vectors (where :attr:`mask` is ``True``) are zero. Optionally, a threshold flow
        magnitude value of ``1e-3`` is used. This can be useful to filter out motions that are equal to very small
        fractions of a pixel, which might just be a computational artefact to begin with.

        :param thresholded: Boolean determining whether the flow is thresholded, defaults to ``True``
        :return: ``True`` if the flow field is zero everywhere, otherwise ``False``
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
    ) -> Union[np.ndarray, torch.Tensor]:
        """Visualises the flow as an rgb / bgr / hsv image, optionally showing the outline of the flow mask :attr:`mask`
        as a black line, and the invalid areas greyed out.

        .. note::
           This currently runs internally based on NumPy & OpenCV, due to a lack of easily accessible equivalent
            functions for coordinate and colour space conversions

        :param mode: Output mode, options: ``rgb``, ``bgr``, ``hsv``
        :param show_mask: Boolean determining whether the flow mask is visualised, defaults to ``False``
        :param show_mask_borders: Boolean determining whether the flow mask border is visualised, defaults to ``False``
        :param range_max: Maximum vector magnitude expected, corresponding to the HSV maximum Value of 255 when scaling
            the flow magnitudes. Defaults to the 99th percentile of the flow field magnitudes
        :param return_tensor: Boolean determining whether the result is returned as a tensor. Note that the result is
            originally a numpy array. Defaults to ``True``
        :return: Numpy array of shape :math:`(H, W, 3)` or torch tensor of shape :math:`(3, H, W)` containing the
            flow visualisation
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

        f = to_numpy(threshold_vectors(self._vecs), True)
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
                range_max = float(np.percentile(mag, 99))
            elif np.max(mag):  # If the 99th percentile is 0, use the actual maximum instead
                range_max = float(np.max(mag))
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
                return torch.tensor(np.moveaxis(return_arr, -1, 0).copy(), device=self._device)
                # Note: .copy() necessary to avoid negative strides in numpy array
            else:
                return return_arr
        else:
            raise ValueError("Error visualising flow: Mode needs to be either 'bgr', 'rgb', or 'hsv'")

    def visualise_arrows(
        self,
        grid_dist: int = None,
        img: Union[np.ndarray, torch.Tensor] = None,
        scaling: Union[float, int] = None,
        show_mask: bool = None,
        show_mask_borders: bool = None,
        colour: tuple = None,
        thickness: int = None,
        return_tensor: bool = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """Visualises the flow as arrowed lines, optionally showing the outline of the flow mask :attr:`mask` as a black
        line, and the invalid areas greyed out.

        :param grid_dist: Integer of the distance of the flow points to be used for the visualisation, defaults to
            ``20``
        :param img: Torch tensor or numpy array with the background image to use (in BGR mode), defaults to white
        :param scaling: Float or int of the flow line scaling, defaults to scaling the 99th percentile of arrowed line
            lengths to be equal to twice the grid distance (empirical value)
        :param show_mask: Boolean determining whether the flow mask is visualised, defaults to ``False``
        :param show_mask_borders: Boolean determining whether the flow mask border is visualised, defaults to ``False``
        :param colour: Tuple of the flow arrow colour, defaults to hue based on flow direction as in
            :meth:`~oflibpytorch.Flow.visualise`
        :param thickness: Integer of the flow arrow thickness, larger than zero. Defaults to ``1``
        :param return_tensor: Boolean determining whether the result is returned as a tensor. Note that the result is
            originally a numpy array. Defaults to ``True``
        :return: Numpy array of shape :math:`(H, W, 3)` or torch tensor of shape :math:`(3, H, W)` containing the
                flow visualisation, in ``bgr`` colour space
        """

        # Validate arguments
        grid_dist = 20 if grid_dist is None else grid_dist
        if not isinstance(grid_dist, int):
            raise TypeError("Error visualising flow arrows: Grid_dist needs to be an integer value")
        if grid_dist > min(self.shape) // 2:
            print("Warning: grid_dist in visualise_arrows is '{}', which is too large for a flow field of shape "
                  "({}, {}). grid_dist will be reset to '{}'."
                  .format(grid_dist, *self.shape, min(self.shape) // 2))
            grid_dist = min(self.shape) // 2
        if not grid_dist > 0:
            raise ValueError("Error visualising flow arrows: Grid_dist needs to be an integer larger than zero")
        if img is None:
            img = np.full(self.shape[:2] + (3,), 255, 'uint8')
        if isinstance(img, torch.Tensor):
            img = to_numpy(img, True)
        if not isinstance(img, np.ndarray):
            raise TypeError("Error visualising flow arrows: Img needs to be a numpy array or a torch tensor")
        if not len(img.shape) == 3 or img.shape[:2] != self.shape or img.shape[2] != 3:
            raise ValueError("Error visualising flow arrows: "
                             "Img needs to have 3 channels and the same shape as the flow")
        if scaling is not None:
            if not isinstance(scaling, (float, int)):
                raise TypeError("Error visualising flow arrows: Scaling needs to be a float or an integer")
            if scaling <= 0:
                raise ValueError("Error visualising flow arrows: Scaling needs to be larger than zero")
        show_mask = False if show_mask is None else show_mask
        show_mask_borders = False if show_mask_borders is None else show_mask_borders
        return_tensor = True if return_tensor is None else return_tensor
        if not isinstance(show_mask, bool):
            raise TypeError("Error visualising flow: Show_mask needs to be boolean")
        if not isinstance(show_mask_borders, bool):
            raise TypeError("Error visualising flow: Show_mask_borders needs to be boolean")
        if not isinstance(return_tensor, bool):
            raise TypeError("Error visualising flow: Return_tensor needs to be boolean")
        if colour is not None:
            if not isinstance(colour, tuple):
                raise TypeError("Error visualising flow: Colour needs to be a tuple")
            if len(colour) != 3:
                raise ValueError("Error visualising flow arrows: Colour list or tuple needs to have length 3")
        thickness = 1 if thickness is None else thickness
        if not isinstance(thickness, int):
            raise TypeError("Error visualising flow: Thickness needs to be an integer")
        if thickness <= 0:
            raise ValueError("Error visualising flow: Thickness needs to be a integer larger than zero")

        # Thresholding
        f = to_numpy(threshold_vectors(self._vecs), True)

        # Make points
        x, y = np.mgrid[grid_dist//2:f.shape[0] - 1:grid_dist, grid_dist//2:f.shape[1] - 1:grid_dist]
        i_pts = np.dstack((x, y))
        i_pts_flat = np.reshape(i_pts, (-1, 2)).astype('i')
        f_at_pts = f[i_pts_flat[..., 0], i_pts_flat[..., 1]]
        flow_mags, ang = cv2.cartToPolar(f_at_pts[..., 0], f_at_pts[..., 1], angleInDegrees=True)
        if scaling is None:
            scaling = grid_dist / np.percentile(flow_mags, 99)
        flow_mags *= scaling
        f *= scaling
        colours = None
        tip_size = math.sqrt(thickness) * 3.5  # Empirical value
        if colour is None:
            hsv = np.full((1, ang.shape[0], 3), 255, 'uint8')
            hsv[0, :, 0] = np.round(np.mod(ang[:, 0], 360) / 2)
            colours = np.squeeze(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        for i_num, i_pt in enumerate(i_pts_flat):
            if flow_mags[i_num] > 0.5:  # Only draw if the flow length rounds to at least one pixel
                c = tuple(int(item) for item in colours[i_num]) if colour is None else colour
                tip_length = float(tip_size / flow_mags[i_num])
                if self.ref == 's':
                    e_pt = np.round(i_pt + f[i_pt[0], i_pt[1]][::-1]).astype('i')
                    cv2.arrowedLine(img, (i_pt[1], i_pt[0]), (e_pt[1], e_pt[0]), c,
                                    thickness=1, tipLength=tip_length, line_type=cv2.LINE_AA)
                else:  # self.ref == 't'
                    e_pt = np.round(i_pt - f[i_pt[0], i_pt[1]][::-1]).astype('i')
                    cv2.arrowedLine(img, (e_pt[1], e_pt[0]), (i_pt[1], i_pt[0]), c,
                                    thickness=thickness, tipLength=tip_length, line_type=cv2.LINE_AA)
            img[i_pt[0], i_pt[1]] = [0, 0, 255]

        # Show mask and mask borders if required
        if show_mask:
            img[~self.mask_numpy] = np.round(0.5 * img[~self.mask_numpy]).astype('uint8')
        if show_mask_borders:
            mask_as_img = np.array(255 * self.mask_numpy, 'uint8')
            contours, hierarchy = cv2.findContours(mask_as_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0, 0, 0), 1)
        if return_tensor:
            return torch.tensor(np.moveaxis(img, -1, 0), device=self._device)
        else:
            return img

    def show(self, wait: int = None, show_mask: bool = None, show_mask_borders: bool = None):
        """Shows the flow in an OpenCV window using :meth:`~oflibpytorch.Flow.visualise`

        :param wait: Integer determining how long to show the flow for, in milliseconds. Defaults to ``0``, which means
            it will be shown until the window is closed, or the process is terminated
        :param show_mask: Boolean determining whether the flow mask is visualised, defaults to ``False``
        :param show_mask_borders: Boolean determining whether flow mask border is visualised, defaults to ``False``
        """

        wait = 0 if wait is None else wait
        if not isinstance(wait, int):
            raise TypeError("Error showing flow: Wait needs to be an integer")
        if wait < 0:
            raise ValueError("Error showing flow: Wait needs to be an integer larger than zero")
        img = self.visualise('bgr', show_mask, show_mask_borders, return_tensor=False)
        cv2.imshow('Visualise and show flow', img)
        cv2.waitKey(wait)

    def show_arrows(
        self,
        wait: int = None,
        grid_dist: int = None,
        img: np.ndarray = None,
        scaling: Union[float, int] = None,
        show_mask: bool = None,
        show_mask_borders: bool = None,
        colour: tuple = None
    ):
        """Shows the flow in an OpenCV window using :meth:`~oflibpytorch.Flow.visualise_arrows`

        :param wait: Integer determining how long to show the flow for, in milliseconds. Defaults to ``0``, which means
            it will be shown until the window is closed, or the process is terminated
        :param grid_dist: Integer of the distance of the flow points to be used for the visualisation, defaults to
            ``20``
        :param img: Numpy array with the background image to use (in BGR colour space), defaults to black
        :param scaling: Float or int of the flow line scaling, defaults to scaling the 99th percentile of arrowed line
            lengths to be equal to twice the grid distance (empirical value)
        :param show_mask: Boolean determining whether the flow mask is visualised, defaults to ``False``
        :param show_mask_borders: Boolean determining whether the flow mask border is visualised, defaults to ``False``
        :param colour: Tuple of the flow arrow colour, defaults to hue based on flow direction as in
            :meth:`~oflibpytorch.Flow.visualise`
        """

        wait = 0 if wait is None else wait
        if not isinstance(wait, int):
            raise TypeError("Error showing flow: Wait needs to be an integer")
        if wait < 0:
            raise ValueError("Error showing flow: Wait needs to be an integer larger than zero")
        img = self.visualise_arrows(grid_dist, img, scaling, show_mask, show_mask_borders, colour, return_tensor=False)
        cv2.imshow('Visualise and show flow', img)
        cv2.waitKey(wait)

    def matrix(self, dof: int = None, method: str = None, masked: bool = None) -> np.ndarray:
        """Fit a transformation matrix to the flow field using OpenCV functions

        :param dof: Integer describing the degrees of freedom in the transformation matrix to be fitted, defaults to
            ``8``. Options are:

            - ``4``: Partial affine transform with rotation, translation, scaling
            - ``6``: Affine transform with rotation, translation, scaling, shearing
            - ``8``: Projective transform, i.e estimation of a homography
        :param method: String describing the method used to fit the transformations matrix by OpenCV, defaults to
            ``ransac``. Options are:

            - ``lms``: Least mean squares
            - ``ransac``: RANSAC-based robust method
            - ``lmeds``: Least-Median robust method
        :param masked: Boolean determining whether the flow mask is used to ignore flow locations where the mask
            :attr:`mask` is ``False``. Defaults to ``True``
        :return: Torch tensor of shape :math:`(3, 3)` and the same device as the flow object, containing the
            transformation matrix
        """

        # Input validation
        dof = 8 if dof is None else dof
        if dof not in [4, 6, 8]:
            raise ValueError("Error fitting transformation matrix to flow: Dof needs to be 4, 6 or 8")
        method = 'ransac' if method is None else method
        if method not in ['lms', 'ransac', 'lmeds']:
            raise ValueError("Error fitting transformation matrix to flow: "
                             "Method needs to be 'lms', 'ransac', or 'lmeds'")
        masked = True if masked is None else masked
        if not isinstance(masked, bool):
            raise TypeError("Error fitting transformation matrix to flow: Masked needs to be boolean")

        # Get the two point arrays
        vecs = to_numpy(self._vecs, True)
        if self._ref == 't':
            dst_pts = np.stack(np.mgrid[:self.shape[0], :self.shape[1]], axis=-1)[..., ::-1]
            src_pts = dst_pts - vecs
        else:  # ref is 's'
            src_pts = np.stack(np.mgrid[:self.shape[0], :self.shape[1]], axis=-1)[..., ::-1]
            dst_pts = src_pts + vecs
        src_pts = src_pts.reshape(-1, 2)
        dst_pts = dst_pts.reshape(-1, 2)

        # Mask if required
        if masked:
            mask = to_numpy(self._mask)
            src_pts = src_pts[mask.ravel()]
            dst_pts = dst_pts[mask.ravel()]

        if dof in [4, 6] and method == 'lms':
            method = 'ransac'
            warnings.warn("Method 'lms' (least mean squares) not supported for fitting a transformation matrix with 4 "
                          "or 6 degrees of freedom to the flow - defaulting to 'ransac'")

        dof_lookup = {
            4: cv2.estimateAffinePartial2D,
            6: cv2.estimateAffine2D,
            8: cv2.findHomography
        }

        method_lookup = {
            'lms': 0,
            'ransac': cv2.RANSAC,
            'lmeds': cv2.LMEDS
        }

        # Fit matrix
        if dof in [4, 6]:
            matrix = np.eye(3)
            matrix[:2] = dof_lookup[dof](src_pts, dst_pts, method=method_lookup[method])[0]
        else:
            matrix = dof_lookup[dof](src_pts, dst_pts, method=method_lookup[method])[0]
        return torch.tensor(matrix).to(self._device)
