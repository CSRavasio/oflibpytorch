Documentation
=============
While this documentation aims to go beyond a simple listing of parameters and instead attempts to explain some of the
principles behind the functions, please see the section ":ref:`Usage`" for more details and usage examples including
code and flow field visualisations.


Flow Constructors and Operators
-------------------------------
.. autoclass:: oflibpytorch.Flow
    :members: zero, from_matrix, from_transforms, vecs, vecs_numpy, ref, mask, mask_numpy, device, shape,
        copy, is_zero, to_device
    :special-members: __str__, __getitem__, __add__, __sub__, __mul__, __truediv__, __pow__, __neg__

    .. automethod:: __init__

Manipulating the Flow
---------------------
.. currentmodule:: oflibpytorch
.. automethod:: Flow.resize
.. automethod:: Flow.pad
.. automethod:: Flow.invert
.. automethod:: Flow.switch_ref

Applying the Flow
-----------------
.. currentmodule:: oflibpytorch
.. automethod:: Flow.apply
.. automethod:: Flow.track

Evaluating the Flow
-------------------
.. currentmodule:: oflibpytorch
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

Flow Operations
---------------
.. autofunction:: oflibpytorch.visualise_definition
.. autofunction:: oflibpytorch.combine_flows
