Documentation
=============
While this documentation aims to go beyond a simple listing of parameters and instead attempts to explain some of the
principles behind the functions, please see the section ":ref:`Usage`" for more details and usage examples including
code and flow field visualisations.

Using the Flow Class
~~~~~~~~~~~~~~~~~~~~
This section documents the custom flow class and all its class methods. It is the recommended way of using
``oflibpytorch`` and makes the full range of functionality available to the user.

Flow Constructors and Operators
-------------------------------
.. autoclass:: oflibpytorch.Flow
    :members: zero, from_matrix, from_transforms, from_kitti, from_sintel, vecs, vecs_numpy, ref, mask, mask_numpy,
        device, shape, copy, to_device
    :special-members: __str__, __getitem__, __add__, __sub__, __mul__, __truediv__, __pow__, __neg__

    .. automethod:: __init__

Manipulating the Flow
---------------------
.. currentmodule:: oflibpytorch
.. automethod:: Flow.resize
.. automethod:: Flow.pad
.. automethod:: Flow.invert
.. automethod:: Flow.switch_ref
.. automethod:: Flow.combine_with

Applying the Flow
-----------------
.. currentmodule:: oflibpytorch
.. automethod:: Flow.apply
.. automethod:: Flow.track

Evaluating the Flow
-------------------
.. currentmodule:: oflibpytorch
.. automethod:: Flow.is_zero
.. automethod:: Flow.matrix
.. automethod:: Flow.valid_target
.. automethod:: Flow.valid_source
.. automethod:: Flow.get_padding

Visualising the Flow
--------------------
.. currentmodule:: oflibpytorch
.. automethod:: Flow.visualise
.. automethod:: Flow.visualise_arrows
.. automethod:: Flow.show
.. automethod:: Flow.show_arrows
.. autofunction:: oflibpytorch.visualise_definition

Using NumPy Arrays
~~~~~~~~~~~~~~~~~~
This section contains functions that take Torch tensors as well as NumPy arrays as inputs, instead of making use of
the custom flow class. On the one hand, this avoids having to define flow objects. On the other hand, it requires
keeping track of flow attributes manually, and it does not avail itself of the full scope of functionality
``oflibpytorch`` has to offer: most importantly, flow masks are not considered or tracked.

Flow Loading
------------
.. autofunction:: oflibpytorch.from_matrix
.. autofunction:: oflibpytorch.from_transforms
.. autofunction:: oflibpytorch.load_kitti
.. autofunction:: oflibpytorch.load_sintel
.. autofunction:: oflibpytorch.load_sintel_mask

Flow Manipulation
-----------------
.. autofunction:: oflibpytorch.resize_flow
.. autofunction:: oflibpytorch.invert_flow
.. autofunction:: oflibpytorch.switch_flow_ref
.. autofunction:: oflibpytorch.combine_flows

Flow Application
----------------
.. autofunction:: oflibpytorch.apply_flow
.. autofunction:: oflibpytorch.track_pts

Flow Evaluation
---------------
.. autofunction:: oflibpytorch.is_zero_flow
.. autofunction:: oflibpytorch.get_flow_matrix
.. autofunction:: oflibpytorch.valid_target
.. autofunction:: oflibpytorch.valid_source
.. autofunction:: oflibpytorch.get_flow_padding

Flow Visualisation
------------------
.. autofunction:: oflibpytorch.visualise_flow
.. autofunction:: oflibpytorch.visualise_flow_arrows
.. autofunction:: oflibpytorch.show_flow
.. autofunction:: oflibpytorch.show_flow_arrows