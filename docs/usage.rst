Usage
=====
This section aims to illustrate the benefits of :mod:`oflibpytorch` with examples. In all sample code, it is assumed
that the library was imported using the command ``import oflibpytorch as of``, and therefore the flow class can be
accessed using ``of.Flow`` and the functions using ``of.<function>``. All the main examples use class methods instead
of the alternative tensor-based functions. These work much the same way, but are more limited in their capabilities and
therefore do not illustrate the full scope of :mod:`oflibpytorch`. For some examples, see the section
":ref:`Tensor-Based Functions`".

Note that :mod:`oflibpytorch` is an adaption of :mod:`oflibnumpy` (see `code`_ and `documentation`_) to the use of
torch tensors instead of numpy arrays as far as currently feasible, functionally largely equivalent. Using torch
tensors is advantageous e.g. for work with deep learning models for multiple reasons. Operations can be performed
batched and on GPU, both affording significant speedups. Most importantly, the main functions that output a float
tensor are differentiable with respect to their tensor inputs (see the section ":ref:`Pure PyTorch Setting`").
This allows for a direct integration in machine learning algorithms requiring the option of back-propagation through
all operations.

.. _code:  https://github.com/RViMLab/oflibnumpy
.. _documentation: https://oflibnumpy.rtfd.io

Pure PyTorch Setting
--------------------
PyTorch currently does not offer a built-in function for the interpolation of unstructured data on to a grid, which
is necessary for a number of this module's functions. Therefore, an approximate solution was implemented: inverse
bilinear interpolation, see :meth:`~oflibpytorch.grid_from_unstructured_data`. As a fallback option, the slower and
CPU-only SciPy method :func:`griddata` can be used.

The module-wide variable ``PURE_PYTORCH``, retrieved by calling :meth:`~oflibpytorch.get_pure_pytorch`, determines
whether the former, PyTorch-only, differentiable method is used (:meth:`~oflibpytorch.set_pure_pytorch`), or the
latter, slower, but more accurate SciPy method is called (:meth:`~oflibpytorch.unset_pure_pytorch`).

By default, the ``PURE_PYTORCH`` variable is set to ``True``, to facilitate the out-of-the-box use of
:mod:`oflibpytorch` in the context of machine learning algorithms.

The Flow Object
---------------
The custom flow object introduced here has four attributes: vectors :attr:`~oflibpytorch.Flow.vecs`, reference
:attr:`~oflibpytorch.Flow.ref` (see the section ":ref:`The Flow Reference`"), mask :attr:`~oflibpytorch.Flow.mask`
(see the section ":ref:`The Flow Mask`"), and device :attr:`~oflibpytorch.Flow.device`. It can be initialised using
just a torch tensor or a numpy array containing the flow vectors, or with one of several special constructors:

- :meth:`~oflibpytorch.Flow.zero` requires a desired shape :math:`((N, )H, W)`, and optionally the flow reference,
  a mask, or the desired torch device. As the name indicates, the vectors are zero everywhere.
- :meth:`~oflibpytorch.Flow.from_matrix` requires a :math:`3 \times 3` transformation matrix, a desired shape
  :math:`((N, )H, W)`, and optionally the flow reference, a mask, or the desired torch device. The flow vectors at
  each location in :math:`H \times W` are calculated to correspond to the given matrix.
- :meth:`~oflibpytorch.Flow.from_transforms` requires a list of transforms, a desired shape :math:`((N, )H, W)`, and
  optionally the flow reference, a mask, or the desired torch device. The given transforms are converted into a
  transformation matrix, from which a flow field is constructed as in :meth:`~oflibpytorch.Flow.from_matrix`.
- :meth:`~oflibpytorch.Flow.from_kitti` loads the flow field (and optionally the valid pixels) from ``uint16`` ``png``
  image files, as provided in the `KITTI optical flow dataset`_.
- :meth:`~oflibpytorch.Flow.from_sintel` loads the flow field (and optionally the valid pixels) from ``flo`` files,
  as provided in the `Sintel optical flow dataset`_.

.. _KITTI optical flow dataset: http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow
.. _Sintel optical flow dataset: http://sintel.is.tue.mpg.de/

Tensors are generally expected to follow the channel-first PyTorch convention (shape :math:`((N, )C, H, W)`), and
are the standard input the functions are meant to interact with. If NumPy arrays are a valid input, they are
generally expected to follow the channel-last OpenCV convention (shape :math:`((N, )H, W, C)`). All tensors
belonging to a flow object are kept on the same torch device, and inputs from a different device are automatically
moved to the flow device if necessary. Outputs are on the same device as the flow object as a standard. The
attributes :attr:`~oflibpytorch.Flow.vecs` and :attr:`~oflibpytorch.Flow.mask` can be accessed in PyTorch tensor
form by calling ``flow.vecs`` (shape :math:`(N, 2, H, W)`) or ``flow.mask``, or in NumPy array form by calling
``flow.vecs_numpy`` (shape :math:`(N, H, W, 2)`) or ``flow.mask_numpy``.

Flow objects can be copied with :meth:`~oflibpytorch.Flow.copy`, resized with :meth:`~oflibpytorch.Flow.resize`, padded
with :meth:`~oflibpytorch.Flow.pad`, and sliced using square brackets ``[]`` analogous to numpy slicing, which calls
:meth:`~oflibpytorch.Flow.__get_item__` internally. They can also be added with ``+``, subtracted with ``-``,
multiplied with ``*``, divided with ``/``, exponentiated with ``**``, and negated by prepending ``-``. However, note
that using the standard operator ``+`` is **not** the same as sequentially combining flow fields, and the same goes
for a subtraction or a negation with ``-``. To do this correctly, use :meth:`~oflibpytorch.Flow.combine` (see the
section ":ref:`Combining Flows`").

Additionally, single elements from the batch contained in a Flow object can be extracted as a new Flow object using
:meth:`~oflibpytorch.Flow.select`. Similarly, different Flow objects of the same shape
:attr:`~oflibpytorch.Flow.shape` and flow reference :attr:`~oflibpytorch.Flow.ref` can be batched using
:meth:`~oflibpytorch.batch_flows`.

Visualisation
-------------
The method :meth:`~oflibpytorch.Flow.visualise` returns a common visualisation mode for flow fields: the hue encodes
the flow vector direction, while the saturation encodes the magnitude. Unless a different value is passed, the maximum
saturation will correspond to the maximum magnitude present in the flow field. :meth:`~oflibpytorch.Flow.show` is a
convenience function that will display this visualisation in an OpenCV window using :func:`cv2.imshow`, useful e.g.
for debugging purposes. Note that the flow vectors, i.e. the attribute :attr:`~oflibpytorch.Flow.vecs`, are encoded in
"OpenCV convention": ``vecs[0]`` is the horizontal component of the flow, ``vecs[1]`` the vertical.

.. code-block:: python

    # Get an image of the flow visualisation definition in BGR colour space
    flow_def = of.visualise_definition('bgr')

    # Define a flow as a clockwise rotation and visualise it in BGR colour space
    shape = (601, 601)
    flow = of.Flow.from_transforms([['rotation', 601, 601, -30]], shape)
    flow_img = flow.visualise('bgr')

.. image:: ../docs/_static/usage_vis_flow_definition.png
    :width: 49%
    :alt: Flow visualisation definition

.. image:: ../docs/_static/usage_vis_flow.png
    :width: 49%
    :alt: Sample flow visualisation

**Above:** *Left:* The definition of the flow visualisation, as output by :meth:`~oflibpytorch.visualise_definition`.
*Right:* the visualisation of a clockwise rotation around the lower right corner.

The :meth:`~oflibpytorch.Flow.visualise` method also offers two parameters, `show_mask` and `show_mask_borders`. This
will display the boolean mask :attr:`~oflibpytorch.Flow.mask` attribute of the flow object in the visualisation, by
reducing the image intensity where the mask is ``False``, and drawing a black border around all valid (``True``)
areas, respectively. For an explanation of the usefulness of this mask, see the section ":ref:`The Flow Mask`".

.. code-block:: python

    # Define a flow that is invalid in the upper left corner, and visualise it in BGR colour space
    shape = (601, 601)
    mask = np.ones((601, 601), 'bool')
    mask[:301, :301] = False
    flow = of.Flow.from_transforms([['rotation', 601, 601, -30]], shape, mask=mask)
    flow_img = flow.visualise('bgr', show_mask=True, show_mask_borders=True)

.. image:: ../docs/_static/usage_vis_flow_masked.png
    :width: 49%
    :alt: Sample flow visualisation with mask and border

**Above:** The same clockwise rotation as before, but with a mask that defines the upper left quarter of the flow
field as "invalid". When ``show_mask = True``, this area has a reduced intensity. ``show_mask_borders = True`` adds
a black border around the valid area, i.e. the area where the :attr:`~oflibpytorch.Flow.mask` attribute of the flow
is ``True``.

A second, more intuitive visualisation mode is offered in the :meth:`~oflibpytorch.Flow.visualise_arrows` method. Here,
the flow is drawn out as arrows with either their start or end points on a regular grid (see the documentation for the
reference :attr:`~oflibpytorch.Flow.ref` flow attribute). The colour of the arrows is calculated the same way as in
:meth:`~oflibpytorch.Flow.visualise` by default, but can be set to a different colour if needed. As with
:meth:`~oflibpytorch.Flow.visualise`, the `show_mask` and `show_mask_borders` parameters will visualise the flow mask
:attr:`~oflibpytorch.Flow.mask` attribute. And as before, the :meth:`~oflibpytorch.Flow.show_arrows` method is a
convenience function that will display this visualisation in an OpenCV window using :func:`cv2.imshow`.

.. code-block:: python

    # Define a flow as a clockwise rotation and visualise it in BGR colour space as arrows
    shape = (601, 601)
    flow = of.Flow.from_transforms([['rotation', 601, 601, -30]], shape)
    flow_img = flow.visualise_arrows(80)

    # Define the same flow, but invalid in the upper left corner, and visualise in BGR colour space as arrows
    mask = np.ones((601, 601), 'bool')
    mask[:301, :301] = False
    flow = of.Flow.from_transforms([['rotation', 601, 601, -30]], shape, mask=mask)
    flow_img_masked = flow.visualise_arrows(80, show_mask=True, show_mask_borders=True)

.. image:: ../docs/_static/usage_vis_flow_arrows.png
    :width: 49%
    :alt: Sample flow arrow visualisation

.. image:: ../docs/_static/usage_vis_flow_arrows_masked.png
    :width: 49%
    :alt: Sample flow arrow visualisation with mask and border

**Above:** *Left:* The same flow field as before, a clockwise rotation around the lower right corner, visualised as
arrows. *Right:* The flow field with the upper left corner defined as "invalid": this area is visualised with a lower
intensity, and the border of the valid area, where the flow mask attribute :attr:`~oflibpytorch.Flow.mask` is ``True``,
is drawn in black

The Flow Reference
------------------
The :attr:`~oflibpytorch.Flow.ref` attribute determines whether the regular grid of shape :math:`(H, W)` associated
with the flow vectors should be understood as the source of the vectors, or the target. So given `img`\ :sub:`1` in
the "source" domain, `img`\ :sub:`2` in the "target" domain, and an associated flow field between the two, there are
two possible definitions or frames of reference for flow vectors:

- "Source" reference: The flow vectors originate from a regular grid corresponding to pixels in the area
  :math:`H \times W` in `img`\ :sub:`1`, the source domain. They therefore encode the motion that moves image
  values from this regular grid in `img`\ :sub:`1` to any location in `img`\ :sub:`2`, the target domain.
- "Target" reference: The flow vectors point to a regular grid corresponding to pixels in the area
  :math:`H \times W` in `img`\ :sub:`2`, the target domain. They therefore encode the motion that moves image
  values from any location in `img`\ :sub:`1`, the source domain, to this regular grid in `img`\ :sub:`2`.

The flow reference ``t`` is the default, and it is easier and more accurate to warp an image with a flow in that
reference. The reason is that reference ``t`` requires interpolating unstructured points from a regular
grid (also known as "backward" or "reverse" warping), while reference ``s`` requires interpolating a regular grid
from unstructured points ("forward" warping). Conversely, the :meth:`~oflibpytorch.Flow.track` method for tracking
points (see the section ":ref:`Tracking Points`") is more accurate for a flow in ``s`` reference, as a flow in ``t``
reference would again require interpolating from unstructured points.

In both cases, the issue with interpolating from unstructured points on to a regular grid is the inherent difficulty
of the operation. By default, :mod:`oflibpytorch` uses an inverse bilinear interpolation PyTorch-based function (see
:meth:`~oflibpytorch.grid_from_unstructured_data`) as a good approximation. Alternatively, it is possible to call the
more accurate, but at least an order of magnitude slower SciPy function :func:`griddata`. The user of
:mod:`oflibpytorch: can make this choice via a module-wide variable called ``PURE_PYTORCH``. For more details, see
the section :ref:`Pure PyTorch Setting`.

The images below show how the same motion, in this case a rotation, will result in slightly different flow vectors
values, depending on the reference chosen. This illustrates that the reference attribute :attr:`~oflibpytorch.Flow.ref`
cannot simply be set to a different value if it needs to be changed. For this purpose, the method
:meth:`~oflibpytorch.Flow.switch_ref` should be called. Again, this requires an interpolation from unstructured data,
once more giving the user the choice between a fast, differentiable, but less accurate PyTorch-only method and the
much slower, non-differentiable, but more accurate SciPy method :func:`scipy.interpolate.griddata`.

.. image:: ../docs/_static/ref_s_vectors_gridded.png
   :width: 49%
   :alt: Reference ``s`` (source)
.. image:: ../docs/_static/ref_t_vectors_gridded.png
   :width: 49%
   :alt: Reference ``t`` (target)

**Above:** The same rotation with vectors of reference ``s`` (*left*) and ``t`` (*right*). Note that on the left, the
source of the arrows lies on the regular grid drawn in grey, while on the right, the tip of the arrows lies on the
same regular grid.

If the problem is that a specific algorithm that calculates the flow from a pair of images :func:`get_flow` is set up
to return a flow field in one reference, but the flow field in the other reference is required, there is a simpler
solution than using the method :meth:`~oflibpytorch.Flow.switch_ref`. Instead of calling
``flow_one_ref = get_flow(img1, img2)``, simply call the algorithm with the images in the reversed order, and multiply
the resulting flow vectors by -1: ``flow_other_ref = -1 * get_flow(img2, img1)``. If the flow is needed in both
references with the best-possible accuracy, meaning ``PURE_PYTORCH`` will be set to ``False``, it may even be faster
to use the flow estimation twice in the way explained above, rather than once followed by a use of
:meth:`~oflibpytorch.Flow.switch_ref`. However, this of course depends on the size of the flow field, as well as
the operational complexity of the algorithm used to estimate the flow.

From the previous observations, it also follows that inverting a flow is not a matter of simply inverting the flow
vectors. In flows with reference ``t``, this would mean the target location remains the same while the source switches
to the opposite side, while in flows with reference ``s``, this would mean the source location remains the same while
the target switches to the opposite side. Neither is correct: in actual fact, inverting the flow switches the
source and the target around. This means inverting the flow vectors *and* changing the reference:
:math:`F(vecs, t)^{-1} = F(-vecs, s)` and :math:`F(vecs, s)^{-1} = F(-vecs, t)`. If the flow is needed with the
original reference, :meth:`~oflibpytorch.Flow.switch_ref` would have to be called. The method
:meth:`~oflibpytorch.Flow.invert` does all this internally, and returns the mathematically correct inverse flow in
whichever reference needed.

.. code-block:: python

    # Define a flow
    flow = of.Flow.from_transforms([['rotation', 200, 150, -30]], (300, 300), 't')

    # Get the flow inverse: in the wrong way, and correctly in either reference
    flow_invalid_inverse = -flow
    flow_valid_inverse_t = flow.invert('t')
    flow_valid_inverse_s = flow.invert('s')

.. image:: ../docs/_static/usage_ref_flow.png
   :width: 49%
   :alt: A clockwise rotation as a flow field
.. image:: ../docs/_static/usage_ref_flow_inverse_wrong.png
   :width: 49%
   :alt: The incorrect inverse of the flow field
.. image:: ../docs/_static/usage_ref_flow_inverse_s.png
   :width: 49%
   :alt: Correct inverse of the flow field, reference s
.. image:: ../docs/_static/usage_ref_flow_inverse_t.png
   :width: 49%
   :alt: Correct inverse of the flow field, reference t

**Above:** *Top:* A flow field corresponding to a clockwise rotation in reference ``t``, and the incorrect "inverse"
obtained by simply inverting the flow vectors, also in reference ``t``. *Bottom:* The correct inverse in reference
``s``, and the correct inverse in reference ``t``. Note the difference in the flow vectors between the correct and
incorrect inverse - the former describes a pure rotation, while the latter resembles a spiral.

In the images above, the inverse in reference ``s`` retains the entire area :math:`H \times W` as valid, while the
inverse in reference ``t`` has undefined areas. As with the example in the section ":ref:`The Flow Mask`", this is
not a limitation of the algorithm, but simply a consequence of the operations necessary to invert the flow.

The Flow Mask
-------------
The :attr:`~oflibpytorch.Flow.mask` attribute is necessary to keep track of which flow vectors in the
:attr:`~oflibpytorch.Flow.vecs` attribute are valid. This is useful e.g. when two flow fields are combined (see the
section ":ref:`Combining Flows`"):

.. code-block:: python

    # Define two flows, one rotation, one scaling motion
    shape = (300, 400)
    flow_1 = of.Flow.from_transforms([['rotation', 200, 150, -30]], shape)
    flow_2 = of.Flow.from_transforms([['scaling', 100, 50, 0.7]], shape)

    # Combine the flow fields
    result = flow_1.combine(flow_2, mode=3)

.. image:: ../docs/_static/usage_mask_flow1.png
    :width: 49%
    :alt: Flow 1 visualisation (rotation)

.. image:: ../docs/_static/usage_mask_flow2.png
    :width: 49%
    :alt: Flow 2 visualisation (scaling)

.. image:: ../docs/_static/usage_mask_result.png
    :width: 49%
    :alt: Flow combination visualisation

.. image:: ../docs/_static/usage_mask_result_masked.png
    :width: 49%
    :alt: Flow combination visualisation, masked

**Above:** *Top:* Flow 1 (rotation), Flow 2 (scaling). *Bottom:* Flow combination, plain and masked

The flow visualisations above illustrate how not the entire flow field area :math:`H \times W` will actually contain
valid or useful flow vectors after a flow combination operation, despite both flow fields used being entirely valid.
This is not a limitation of the algorithm, but unavoidable: the scaling operation can be pictured as a "zooming out"
motion, which obviously means there will be a "frame" of values that would have had to come from outside of
:math:`H \times W`, and are therefore undefined.

Applying a Flow
---------------
The :meth:`~oflibpytorch.Flow.apply` method is used to apply a flow field to an image (or any other torch tensor, or
indeed another flow field). Optionally, the ``valid_area`` can be returned, which will be ``True`` where the warped
image is valid, i.e. contains actual content. For an illustration, see the example below.

.. code-block:: python

    # Load image, and define a flow as a combination of a rotation and scaling motion
    img = cv2.imread('thames.jpg')  # 300x400 pixels
    transforms = [['rotation', 200, 150, -30], ['scaling', 100, 50, 0.7]]
    flow = of.Flow.from_transforms(transforms, img.shape[:2])

    # Apply the flow to the image, getting the "valid area"
    img = torch.tensor(np.moveaxis(img, -1, 0))
    warped_img, valid_area = flow.apply(img, return_valid_area=True)

.. image:: ../docs/_static/usage_apply_thames_warped1.png
    :width: 49%
    :alt: Warped image with mask

.. image:: ../docs/_static/usage_apply_thames_warped2.png
    :width: 49%
    :alt: Warped image with mask

**Above:** The result of applying a rotation and scaling motion to an image, with the black border showing the
outline of the returned ``valid_area``. As can be seen, the valid area matches the true image content exactly.
*Left:* the flow field used was the one from the code example above, valid everywhere. *Right:* the flow field used
was the one from the section ":ref:`The Flow Mask`", where the valid area is further reduced by the flow field itself
having a reduced valid area.

It is also possible to pass an image mask, e.g. a segmentation mask, into the :meth:`~oflibpytorch.Flow.apply` method,
which will be combined with the flow mask to eventually result in the ``valid_area``. This can be useful as in the
example below.

.. code-block:: python

    # Make a circular mask
    shape = (300, 350)
    mask = np.mgrid[-shape[0]//2:shape[0]//2, -shape[1]//2:shape[1]//2]
    radius = shape[0] // 2 - 20
    mask = np.linalg.norm(mask, axis=0)
    mask = mask < radius

    # Load image, make two images that simulate a moving telescope
    img = cv2.imread('thames.jpg')  # 300x400 pixels
    img1 = np.copy(img[:, :-50])
    img2 = np.copy(img[:, 50:])
    img1[~mask] = 0
    img2[~mask] = 0

    # Make a flow field that could have been obtained from the above images
    flow = of.Flow.from_transforms([['translation', -50, 0]], shape, 't', mask)
    flow.vecs[:, ~mask] = 0

    # Apply the flow to the image, setting consider_mask to True and False
    img1 = torch.tensor(np.moveaxis(img1, -1, 0))
    mask = torch.tensor(mask)
    warped_img, valid_area = flow.apply(img1, mask, return_valid_area=True)

.. image:: ../docs/_static/usage_apply_masked_img1.png
    :width: 49%
    :alt: Image 1, the Thames through a telescope
.. image:: ../docs/_static/usage_apply_masked_img2.png
    :width: 49%
    :alt: Image 2, the Thames through a telescope
.. image:: ../docs/_static/usage_apply_masked_flow_arrows.png
    :width: 49%
    :alt: The flow corresponding to the motion from Image 1 to 2
.. image:: ../docs/_static/usage_apply_masked_img_warped.png
    :width: 49%
    :alt: Image 1 warped by the flow, masked with the valid_area

**Above:** *Top:* Image 1 and image 2, as they could be seen when looking at the river Thames through a telescope.
*Bottom left:* The flow field corresponding to the motion from image 1 and image 2, a translation of 50px to the left.
The arrows show clearly that some of the pixels being moved originate outside of the field of view of the telescope,
which means the right-hand-side border of this field of view will be shifted towards the left, reducing the "useful"
image area. This cannot be avoided, as the parts of the image moving into view in image 2 are occluded in image 1.
*Bottom right:* the result of warping image 1 with the flow field, passing in the telescope field of view segmentation
from image 1 as a mask. The returned valid_area is shown as an overlay, and perfectly matches the location of the true
image content. So while the loss of "true content" area cannot be avoided, it can be tracked by passing the initial
segmentation into the function, and using ``return_valid_area = True`` to obtain an updated segmentation.


The examples above use a flow field with reference ``t``. This is the recommended standard for various reasons:

- Using :meth:`~oflibpytorch.Flow.apply` with flow fields of reference ``s`` is either less accurate if
  ``PURE_PYTORCH`` is set to ``True``, or else comparatively slow as it will call SciPy's :func:`griddata` function.
- Flow fields of reference ``s`` can contain ambiguities, as vectors from two different locations can point to the same
  target location. This could happen if there are several independently moving objects in a scene which end up
  occluding each other. The only way of resolving this is to assign priorities to the flow vectors. With the exception
  of pixels containing zero flow, which are already de-prioritised with respect to all other flows present when using
  the PyTorch-only interpolation method instead of SciPy's :func:`griddata`, this is left to a possible future
  version of :mod:`oflibpytorch`.
- Furthermore, flow fields of reference ``s`` do not deal well with undefined / invalid flow areas when using SciPy's
  :func:`griddata` function, as the example below shows. One option (the default) considers the flow mask, i.e.
  ignores invalid flow vectors, which leads to a smoother result inside the convex hull of the flow target area but
  risks artefacts appearing. The other option, accessible by setting ``consider_mask = False``, is to use the
  invalid vectors anyway. In this example it inserts a lot of black image values in-between the desired image
  values which are to be interpolated onto the regular grid of the new image: this gets rid of the large artefact
  visible in the concave area, but does not allow the flow field to expand the image properly. In a future version
  of :mod:`oflibpytorch`, this could be at least partially solved by implementing a second step in which the image
  pixels not belonging to the concave hull are set to zero. However, determining the convex hull of unstructured point
  clouds brings its own difficulties.

.. code-block:: python

    # Make a circular mask with the lower left corner missing
    shape = (300, 400)
    mask = np.mgrid[-shape[0]//2:shape[0]//2, -shape[1]//2:shape[1]//2]
    radius = shape[0] // 2 - 20
    mask = np.linalg.norm(mask, axis=0)
    mask = mask < radius

    # Load image, make a flow field, apply masks
    img = cv2.imread('_static/thames_300x400.jpg')
    img[~mask] = 0
    flow_mask = mask.copy()
    mask[150:, :200] = False
    flow_mask[150:, :200] = False
    flow_mask[150:, 260:] = False
    flow = of.Flow.from_transforms([['scaling', 200, 150, 1.3]], shape, 's', flow_mask)
    flow.vecs[:, :, ~mask] = 0

    # Apply the flow to the image, setting consider_mask to True and False
    unset_pure_pytorch()
    img_true = flow.apply(to_tensor(img, 'single'), consider_mask=True)
    img_false = flow.apply(to_tensor(img, 'single'), consider_mask=False)
    set_pure_pytorch()
    img_true_pt = flow.apply(to_tensor(img, 'single'), consider_mask=True)
    img_false_pt = flow.apply(to_tensor(img, 'single'), consider_mask=False)

.. image:: ../docs/_static/usage_apply_consider_mask_img.png
    :width: 49%
    :alt: Masked image
.. image:: ../docs/_static/usage_apply_consider_mask_flow_arrows.png
    :width: 49%
    :alt: Masked flow
.. image:: ../docs/_static/usage_apply_consider_mask_true.png
    :width: 49%
    :alt: Flow applied to the image considering the flow mask (default option), using scipy.interpolate.griddata
.. image:: ../docs/_static/usage_apply_consider_mask_false.png
    :width: 49%
    :alt: Flow applied to the image not considering the flow mask, using scipy.interpolate.griddata
.. image:: ../docs/_static/usage_apply_consider_mask_true_pytorch.png
    :width: 49%
    :alt: Flow applied to the image considering the flow mask (default option), using the PyTorch-only alternative
.. image:: ../docs/_static/usage_apply_consider_mask_false_pytorch.png
    :width: 49%
    :alt: Flow applied to the image not considering the flow mask, using the PyTorch-only alternative

**Above:** *Top:* The image (imagine a monocular) and a masked flow (mask shown as white area) with reference ``s``,
corresponding to a partial scaling motion from the image centre.
*Middle:* The result of using SciPy's :func:`griddata` to apply the flow to the image, with / without considering
the mask, i.e. not using / using all flow vector values. In the former case, large artefacts become visible in
concave areas. In the latter case, the image content that should be superimposed on the black outside areas is
only visible as single pixels here and there, while the black area that has not moved dominates.
*Bottom:* Same as the middle row, but using the faster and differentiable PyTorch-only method
:meth:`~oflibpytorch.grid_from_unstructured_data`. It becomes apparent that this implementation suffers much less
from artefacts, though a detailed comparison of the resulting image values would show it to be less accurate than
the result of :func:`scipy.interpolate.griddata`.


Flow Padding
------------
Given that applying a flow with reference ``t`` to an image can lead to undefined areas (as seen in the section
":ref:`Applying a Flow`"), it can be useful to know how much this image would have to be padded on each side with
respect to the given flow field in order for no undefined areas to show up anymore. A possible application for this
would be the creation of synthetic data for a deep learning optical flow estimation algorithm, with the goal of
obtaining two images and an associated flow field that corresponds to the motion visible between the two images.

The padding can be determined using the :meth:`~oflibpytorch.Flow.get_padding` method, and will be returned as a
list of values ``[top, bottom, left, right]``. If an image padded accordingly is passed to the
:meth:`~oflibpytorch.Flow.apply` method along with the padding values, the image will be warped according to the
flow field and automatically cut down to the size of the flow field, unless the parameter `cut` is set to ``False``.

.. code-block:: python

    # Load an image
    full_img = cv2.imread('thames.jpg')  # original resolution 600x800

    # Define a flow field
    shape = (300, 300)
    transforms = [['rotation', 200, 150, -30], ['scaling', 100, 50, 0.7]]
    flow = of.Flow.from_transforms(transforms, shape)

    # Get the necessary padding
    padding = flow.get_padding()

    # Select an image patch that is equal in size to the flow resolution plus the padding
    padded_patch = full_img[:shape[0] + sum(padding[:2]), :shape[1] + sum(padding[2:])]

    # Apply the flow field to the image patch, passing in the padding
    padded_patch = torch.tensor(np.moveaxis(padded_patch, -1, 0))
    warped_padded_patch = flow.apply(padded_patch, padding=padding)

    # As a comparison: cut an unpadded patch out of the image and warp it with the same flow
    patch = full_img[padding[0]:padding[0] + shape[0], padding[2]:padding[2] + shape[1]]
    patch = torch.tensor(np.moveaxis(patch, -1, 0))
    warped_patch = flow.apply(patch)

.. image:: ../docs/_static/usage_padding_patch.png
    :width: 32%
    :alt: Original unpadded image patch
.. image:: ../docs/_static/usage_padding_warped.png
    :width: 32%
    :alt: Unpadded patch warped with the flow
.. image:: ../docs/_static/usage_padding_padded_warped.png
    :width: 32%
    :alt: Padded patch warped with the flow, cut back to size

**Above:** *Left:* The original unpadded image patch. *Middle:* The unpadded image patch when warped with the same
flow field as the one used in the section ":ref:`Applying a Flow`". Note the similar amount of undefined areas
visible in the result. *Right:* The result of applying the flow to the image patch padded with the necessary amount
of padding, and then cut back to size. The padding was just large enough to avoid any undefined areas becoming
visible.

For flows with reference ``s``, the above calculation of padding is not possible: after all, the flow vectors express
where pixels in the original image are "pushed" to, rather than where pixels in the warped image are "pulled" from.
Instead, the :meth:`~oflibpytorch.Flow.get_padding` method calculates the padding necessary to ensure no content
is being pushed outside of the image.

.. code-block:: python

    # Load an image, define a flow field
    img = cv2.imread('thames.jpg')  # 300x400 pixels
    transforms = [['rotation', 200, 150, -30], ['scaling', 100, 50, 0.9]]
    flow = of.Flow.from_transforms(transforms, img.shape[:2], 's')  # 300x400 pixels

    # Find the padding and pad the image
    padding = flow.get_padding()
    padded_img = np.pad(img, (tuple(padding[:2]), tuple(padding[2:]), (0, 0)))

    # Apply the flow field to the image patch, with and without the padding
    img = torch.tensor(np.moveaxis(img, -1, 0))
    padded_img = torch.tensor(np.moveaxis(padded_img, -1, 0))
    warped_img = flow.apply(img)
    warped_padded_img = flow.apply(padded_img, padding=padding, cut=False)

.. image:: ../docs/_static/usage_padding_s_warped.png
    :width: 49%
    :alt: Image warped with the flow
.. image:: ../docs/_static/usage_padding_s_warped_padded.png
    :width: 49%
    :alt: Padded image warped with the flow

**Above:** *Left:* The original image warped with the flow - note the corners that have been moved outside of
the image, leading to loss of information. *Right:* The padded image warped with the flow: the image has been
padded the exact amount needed not to lose any image content.


Source and Target Areas
-----------------------
The :meth:`~oflibpytorch.Flow.valid_source` and :meth:`~oflibpytorch.Flow.valid_target` methods both serve to
investigate the flow field. Given an image with the area :math:`H \times W` in the source domain and a flow field
of the same shape, applying this flow to the image will give us a warped image in the target domain. Some of the
original image content will no longer be visible after applying the flow: :meth:`~oflibpytorch.Flow.valid_source`
returns a boolean tensor of shape :math:`(N, H, W)` which is ``False`` where content "disappears" after warping.
The warped image, in turn, will contain some areas which are undefined, i.e. not filled by any content from the
original image: :meth:`~oflibpytorch.Flow.valid_target` returns a boolean tensor of shape :math:`(N, H, W)` which
is ``False`` where the warped image does not contain valid content.

.. code-block:: python

    # Define a flow field
    shape = (300, 400)
    transforms = [['rotation', 200, 150, -30], ['scaling', 100, 50, 1.2]]
    flow = of.Flow.from_transforms(transforms, shape)

    # Get the valid source and target areas
    valid_source = flow.valid_source()
    valid_target = flow.valid_target()

    # Load an image and warp it with the flow
    img = cv2.imread('thames.jpg')  # 300x400 pixels
    img = torch.tensor(np.moveaxis(img, -1, 0))
    warped_img = flow.apply(img)

.. image:: ../docs/_static/usage_source_target_img.png
    :width: 49%
    :alt: Original image
.. image:: ../docs/_static/usage_source_target_warped_img.png
    :width: 49%
    :alt: Warped image
.. image:: ../docs/_static/usage_source_target_source.png
    :width: 49%
    :alt: Valid source area
.. image:: ../docs/_static/usage_source_target_target.png
    :width: 49%
    :alt: Valid target area

**Above:** *Top:* Original image, and the image warped by the flow field. *Bottom left:* The valid source area - the
white area covers the parts of the original image ("source" domain) which are still visible after warping.
*Bottom right:* The valid target area - the white area covers the parts of the warped image ("target" domain) with
real image content.


Tracking Points
---------------
The :meth:`~oflibpytorch.Flow.track` method is useful to apply the flow field to a number of points rather than an
entire image. In the following example, the `int_out` parameter is set to ``True`` so the new point locations are
returned as (rounded) integers - this is a useful convenience feature if these points should then be plotted on an
image. By default, the method will return accurate float values. Note that integer outputs are not differentiable.

If ``PURE_PYTORCH`` is ``False`` (see the section ":ref:`Pure PyTorch Setting`"), using
:meth:`~oflibpytorch.Flow.track` for flows with a "target" reference (``ref = 't'``) requires a call to
:func:`scipy.interpolate.griddata`. This is avoided with the PyTorch-only interpolation function, but at the cost
of decreased accuracy.

.. code-block:: python

    # Define a background image, sample points, and a sample flow field
    background = np.zeros((40, 60, 3), 'uint8')
    pts = np.array([[5, 15], [20, 15], [5, 50], [20, 50]])
    flow = of.Flow.from_transforms([['rotation', 0, 0, -15]], background.shape[:2], 's')

    # Track the points with the flow field, and plot original positions in white, new positions in red
    tracked_pts = flow.track(torch.tensor(pts), int_out=True)
    background[pts[:, 0], pts[:, 1]] = 255
    background[tracked_pts[:, 0], tracked_pts[:, 1], 2] = 255

.. image:: ../docs/_static/usage_track_flow.png
    :width: 49%
    :alt: Flow to track points

.. image:: ../docs/_static/usage_track_pts.png
    :width: 49%
    :alt: Tracking points

**Above:** Flow field, and point positions: original points in white, points after applying the flow in red

If the points are rotated more, some will come to lie outside of the image area. In this case, setting the parameter
`get_valid_status` to ``True`` will cause the :meth:`~oflibpytorch.Flow.track` method to return a boolean tensor which
lists the "status" of each output point. It will be ``True`` for any point that was moved by a valid flow vector (see
section ":ref:`The Flow Mask`") *and* remains inside the image area.

.. code-block:: python

    # Define a background image, sample points, and a sample flow field
    background = np.zeros((40, 60, 3), 'uint8')
    pts = np.array([[5, 15], [20, 15], [5, 50], [20, 50]])
    mask = np.ones((40, 60), 'bool')  # Make a flow mask
    mask[:15, :30] = False  # Set the left upper corner of the flow mask to False
    flow = of.Flow.from_transforms([['rotation', 0, 0, -25]], background.shape[:2], 's', mask)

    # Track the points with the flow field, and plot original positions in white, new positions in red
    tracked_pts, valid_status = flow.track(torch.tensor(pts), int_out=True, get_valid_status=True)
    background[pts[:, 0], pts[:, 1]] = 255
    background[tracked_pts[valid_status][:, 0], tracked_pts[valid_status][:, 1], 2] = 255

.. image:: ../docs/_static/usage_track_flow_with_validity.png
    :width: 49%
    :alt: Flow to track points

.. image:: ../docs/_static/usage_track_pts_with_validity.png
    :width: 49%
    :alt: Tracking points

**Above:** Flow field, and point positions: original points in white, points after applying the flow in red. Note the
upper left and lower right points are missing, as they both have a `valid_status` of ``False``. For the upper left
point, this is due to the flow vector at that location having been defined as invalid (see the black border in the
flow field visualisation), as the mask used when creating the flow was set to ``False`` there. For the lower right
point, this is due to the new location of the point being outside of the image area.

Combining Flows
---------------
The :meth:`~oflibpytorch.Flow.combine` function was already used in the section ":ref:`The Flow Mask`" with
``mode = 3`` to sequentially combine two different flow fields. In the formula :math:`flow_1 ⊕ flow_2 = flow_3`,
where :math:`⊕` corresponds to a flow combination operation, this is equivalent to inputting :math:`flow_1` and
:math:`flow_2`, and obtaining :math:`flow_3`. However, it is also possible to obtain either :math:`flow_1` or
:math:`flow_2` when the other flows in the equation are known, by setting ``mode = 1`` or ``mode = 2``, respectively.
The calculation will often lead to a flow field with some invalid areas, similar to the example in the section
":ref:`The Flow Mask`".

This method makes extensive use of :meth:`~oflibpytorch.Flow.apply`, and the same observations with regards to speed
and accuracy apply. It is worth mentioning that if both input flows have reference ``s``, or both have reference
``t``, and ``mode = 3`` is used, the operation will always be fast and accurate regardless of the ``PURE_PYTORCH``
setting used.

.. code-block:: python

    shape = (300, 400)
    flow_1 = of.Flow.from_transforms([['rotation', 200, 150, -30]], shape)
    flow_2 = of.Flow.from_transforms([['scaling', 100, 50, 1.2]], shape)
    flow_3 = of.Flow.from_transforms([['rotation', 200, 150, -30], ['scaling', 100, 50, 1.2]], shape)

    flow_1_result = flow_2.combine(flow_3, mode=1)
    flow_2_result = flow_1.combine(flow_3, mode=2)
    flow_3_result = flow_1.combine(flow_2, mode=3)

.. image:: ../docs/_static/usage_combining_1.png
    :width: 32%
    :alt: Flow 1
.. image:: ../docs/_static/usage_combining_2.png
    :width: 32%
    :alt: Flow 2
.. image:: ../docs/_static/usage_combining_3.png
    :width: 32%
    :alt: Flow 3
.. image:: ../docs/_static/usage_combining_1_result.png
    :width: 32%
    :alt: Calculated flow 1
.. image:: ../docs/_static/usage_combining_2_result.png
    :width: 32%
    :alt: Calculated flow 2
.. image:: ../docs/_static/usage_combining_3_result.png
    :width: 32%
    :alt: Calculated flow 3

**Above:** *Top:* Flows 1 through 3. *Bottom:* Flows 1 through 3, as calculated using
:func:`~oflibpytorch.Flow.combine`, matching the original flow fields. Note that some results may show some
invalid areas.

.. note::
    There is also a previous version of this function called :meth:`~oflibpytorch.Flow.combine_with`. It offers
    identical core functionality, but is limited to combining flow fields of the same flow references :attr:`ref`.
    It will become deprecated in a future version, but continue to work as expected until then.


Tensor-Based Functions
----------------------
Almost all the class methods discussed above are also available as functions that take torch tensors or numpy arrays
representing flow fields as inputs directly. This can appear more straight-forward to use, but they are generally
more limited in their scope, and the user has to keep track of potentially changing flow attributes such as the
reference frame manually. Valid areas are also not tracked. It is recommended to make use of the custom flow class
for anything but the simplest flow operations.

.. code-block:: python

    # Define Torch tensor flow fields
    shape = (100, 100)
    flow = of.from_transforms([['rotation', 50, 100, -30]], shape, 's')
    flow_2 = of.from_transforms([['scaling', 100, 50, 1.2]], shape, 't')

    # Visualise Torch tensor flow field as arrows
    flow_vis = of.show_flow(flow, wait=2000)

    # Combine two Torch tensor flow fields
    flow_t = of.switch_flow_ref(flow, 's')
    flow_3 = of.combine_flows(flow_t, flow_2, 3, 't')

    # Visualise Torch tensor flow field
    flow_3_vis = of.show_flow_arrows(flow_3, 't')

Working with Batched Flows
--------------------------
The :class:`oflibpytorch.Flow` class stores flow vectors in the batched shape :math:`(N, 2, H, W)`. This is the case
even if the optical flow passed to the constructor is just a single vector field, i.e. :math:`N = 1`. The shape is
then :math:`(1, 2, H, W)`.

Often, it is significantly more efficient to process flows in batches. :mod:`oflibpytorch` supports this in all
operations on or with flow fields. If several existing flow objects are to be combined, this can be achieved with
the method :meth:`~oflibpytorch.batch_flows`. For obvious reasons, this is limited to flow fields of the same
flow reference :attr:`~oflibpytorch.Flow.ref` and spatial resolution :math:`(H, W)`, though input flow objects can
have different batch sizes.

Conversely, if single elements of a batched flow are required, they can be extracted using the
:meth:`~oflibpytorch.Flow.select` method. Some functions such as :meth:`~oflibpytorch.Flow.show` allow for the
direct selection of a specific batch element, though they also use :meth:`~oflibpytorch.Flow.select` internally.


.. code-block:: python

    # Define three flow objects
    shape = (300, 400)
    flow_1 = of.Flow.from_transforms([['rotation', 200, 150, -30]], shape)
    flow_2 = of.Flow.from_transforms([['scaling', 100, 50, 1.2]], shape)
    flow_3 = of.Flow.from_transforms([['translation', 10, 10]], shape)

    # Batch two flows of batch size 1
    flow_batched = of.batch_flows((flow_1, flow_2))

    # Batch two flows of batch sizes 2 and 1
    flow_batched = of.batch_flows((flow_batched, flow_3))

    # Using the show method without the elem argument automatically selects the first batch element
    flow_batched.show()

    # Other batch elements can be indicated as an argument
    flow_batched.show(elem=1)

    # Alternatively, a batch element can be selected first and then shown
    flow_batched.select(2).show()

.. image:: ../docs/_static/usage_combining_1.png
    :width: 32%
    :alt: Flow 1
.. image:: ../docs/_static/usage_combining_2.png
    :width: 32%
    :alt: Flow 2
.. image:: ../docs/_static/batched_flows.png
    :width: 32%
    :alt: Flow 3

**Above:** Flows 1 through 3, visualised from a single batched flow object
