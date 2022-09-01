Changelog
=========

2.1.1 [2022-09-01]
------------------

- :meth:`~oflibpytorch.Flow.unpad` introduced: effectively a helper function to undo :meth:`~oflibpytorch.Flow.pad`
- :meth:`~oflibpytorch.Flow.from_transforms` signature extended by a ``padding'' argument: convenience when creating
  a padded flow field, automatically adjusting the shape and relevant transform parameters
- :meth:`~oflibpytorch.Flow.select` parameter ``item'' can be ``None'', returning ``self''
- :meth:`~oflibpytorch.Flow.get_padding` signature extended by an ``item'' argument, used to select an item in the
  batched flow. Returns a simple list of padding values, rather than a list of lists.
- Minor performance improvement in :meth:`~oflibpytorch.Flow.combine`


2.1.0 [2022-06-21]
------------------

- :meth:`~oflibpytorch.Flow.combine` introduced: efficient, generalised combination of flows with any frame of
  reference :attr:`ref`
- :meth:`~oflibpytorch.Flow.combine_with` improvements, but will become deprecated in a future release in favour of
  :meth:`~oflibpytorch.Flow.combine`
- Test coverage improved
- Documentation updated and extended


2.0.0 [2022-05-13]
------------------

Major update, enhancing usability for deep learning applications.

- Flow vectors and masks are now batched, meaning the shape is :math:`(N, H, W)` instead of :math:`(H, W)`. This
  enables easy integration with any deep learning application or network, harnessing the efficiencies of batch-wise
  processing.
- A differentiable PyTorch function to approximately replace :func:`scipy.interpolate.griddata` was implemented
- A toolbox-wide boolean setting called ``PURE_PYTORCH`` has been introduced. If it is set to ``True``, non-Torch
  operations are avoided as far as possible. Specifically, this means avoiding the slow Scipy-based function
  :func:`scipy.interpolate.griddata` in favour of a more approximate, but significantly faster PyTorch-only method
  that interpolates unstructured data on a defined regular grid.
- If ``PURE_PYTORCH`` is set to ``True``, all oflibpytorch methods that output a float torch tensor are
  differentiable, again allowing for easy integration with deep learning algorithms.
- Some utility functions made available
- Documentation and unit test updates
- Minor bugfixes


1.1.1 [2022-01-28]
------------------

- Type of the flow attribute :attr:`device` changed from string to the :class:`torch.device` class
- If the CUDA device index is left undefined, it defaults to ``torch.cuda.current_device()``. This avoids ambiguities
  and possible CUDA device mismatches when working with multiple GPUs.


1.1.0 [2021-11-30]
------------------

- Introduced functions that largely replicate functionality of flow class methods, but for Torch tensor and
  NumPy array flow inputs
- Documentation updated with above functions, some older errors corrected
- Minor bugfixes
- Bibtex citation to use to acknowledge the authors added


1.0.1 [2021-07-09]
------------------

- Fixed bug in visualise (range_max calculation)
- Removed all usages of the torch tensor attribute :attr:`ndim` for improved backwards compatibility with older torch
  versions.
- Removed print statement in test_utils
- Minor documentation corrections, addition of this changelog


1.0.0 [2021-06-09]
------------------

First full release