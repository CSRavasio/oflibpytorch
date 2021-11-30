Changelog
=========

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