Introduction
============
**Oflibpytorch:** a handy python **o**\ ptical **f**\ low **lib**\ rary, based on **PyTorch** tensors, that enables
the manipulation and combination of flow fields while keeping track of valid areas (see "Usage") in the context of
machine learning algorithms implemented in PyTorch. It is mostly code written from scratch, but also contains useful
wrappers for specific functions from libraries such as PyTorch's ``grid_sample``, to integrate them with the custom
flow field class introduced by oflibpytorch. Importantly, the **main methods are differentiable** with respect to
their flow field tensor inputs, allowing for a seamless integration with machine learning algorithms. Features:

- All main methods that return a float tensor are differentiable with respect to tensor inputs
- A custom flow field class for both backward and forward ('target' / 'source' based) flow fields, handled as tensors
  with a batch dimension, allowing for efficient batch-wise processing that can be performed on GPU if desired
- A number of class methods to create flow fields from lists of affine transforms, or a transformation matrix, as
  well as methods to resize the flow field, visualise it, warp images, or find necessary image padding, all while
  keeping track of valid flow field areas
- A class method to process three different types of flow field combination operations
- Utility functions including a PyTorch-based approximate interpolation to a grid from unstructured data as a
  replacement for the SciPy ``griddata`` method

Oflibpytorch is based on oflibnumpy (`code on GitHub`_, `documentation on ReadTheDocs`_) and is aimed at allowing the
same operations to be performed with torch tensors instead of numpy arrays as far as currently feasible.

.. _code on GitHub: https://github.com/RViMLab/oflibnumpy
.. _documentation on ReadTheDocs: https://oflibnumpy.rtfd.io

Usage & Documentation
---------------------
A user's guide as well as full documentation of the library is available at ReadTheDocs_. Some quick examples:

.. _ReadTheDocs: https://oflibpytorch.rtfd.io

.. code-block:: python

    import oflibpytorch as of

    shape = (300, 400)
    transform = [['rotation', 200, 150, -30]]

    # Make a flow field and display it
    flow = of.Flow.from_transforms(transform, shape)
    flow.show()

    # Alternative option without using the custom flow class
    flow = of.from_transforms(transform, shape, 't')
    of.show_flow(flow)

.. image:: https://raw.githubusercontent.com/RViMLab/oflibpytorch/main/docs/_static/index_flow_1.png
    :width: 50%

**Above:** Visualisation of optical flow representing a rotation

.. code-block:: python

    # Combine sequentially with another flow field, display the result
    flow_2 = of.Flow.from_transforms([['translation', 40, 0]], shape)
    result = of.combine_flows(flow, flow_2, mode=3)
    result.show(show_mask=True, show_mask_borders=True)

    # Alternative option without using the custom flow class
    flow_2 = of.from_transforms([['translation', 40, 0]], shape, 't')
    result = of.combine_flows(flow, flow_2, mode=3, ref='t')
    of.show_flow(result)  # Note: no way to show the valid flow area (see documentation)

.. image:: https://raw.githubusercontent.com/RViMLab/oflibpytorch/main/docs/_static/index_result.png
    :width: 50%

**Above:** Visualisation of optical flow representing a rotation, translated to the right, using the custom flow class

.. code-block:: python

    result.show_arrows(show_mask=True, show_mask_borders=True)

    # Alternative option without using the custom flow class
    of.show_flow_arrows(result, 't')  # Note: again no way to show the valid flow area

.. image:: https://raw.githubusercontent.com/RViMLab/oflibpytorch/main/docs/_static/index_result_arrows.png
    :width: 50%

**Above:** Visualisation of optical flow representing a rotation, translated to the right, using the custom flow class


Installation
------------
In order for oflibpytorch to work, the python environment needs to contain a PyTorch installation. To enable GPU usage,
the CUDA Toolkit is required as well. As it is difficult to guarantee an automatic installation via pip will use the
correct versions and work on all operating systems, it is left to the user to install PyTorch and Cudatoolkit
independently. The easiest route is a virtual conda environment and the recommended install command
from the `PyTorch website`_, configured for the user's specific system. To install oflibpytorch itself, use the
following command:

.. _PyTorch website: https://pytorch.org

.. code-block::

    pip install oflibpytorch


Testing
------------
Oflibpytorch contains a large number of tests to verify it is working as intended. Use the command line to navigate
to ``oflibpytorch/test`` and run the following code:

.. code-block::

    python -m unittest discover .

The tests will take several minutes to run. Successful completion will be marked with ``OK``.


Contribution & Support
----------------------
- Source Code: https://github.com/RViMLab/oflibpytorch
- Issue Tracker: https://github.com/RViMLab/oflibpytorch/issues


License
-------
Copyright (c) 2022 Claudio S. Ravasio, PhD student at University College London (UCL), research assistant at King's
College London (KCL), supervised by:

- Dr Christos Bergeles, PI of the Robotics and Vision in Medicine (RViM) lab in the School of Biomedical Engineering &
  Imaging Sciences (BMEIS) at King's College London (KCL)
- Prof Lyndon Da Cruz, consultant ophthalmic surgeon, Moorfields Eye Hospital, London UK

This code is licensed under the `MIT License`_.

.. _MIT License: https://opensource.org/licenses/MIT

If you use this code, please acknowledge us with the following citation:

.. code-block:: plaintext

    @article{ravasio_oflib,
      title     = {oflibnumpy {\&} oflibpytorch: Optical Flow Handling and Manipulation in Python},
      author    = {Ravasio, Claudio S. and Da Cruz, Lyndon and Bergeles, Christos},
      journal   = {Journal of Open Research Software (JORS)},
      year      = {2021},
      volume    = {9},
      publisher = {Ubiquity Press, Ltd.},
      doi       = {10.5334/jors.380}
    }