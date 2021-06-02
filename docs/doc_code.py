import numpy as np
import cv2
import torch
import oflibpytorch as of
from oflibpytorch.utils import to_numpy, to_tensor, show_masked_image


# # # # Usage / Visualisation
# shape = (601, 601)
# flow = of.Flow.from_transforms([['rotation', 601, 601, -30]], shape)
# flow_def = of.visualise_definition('bgr', return_tensor=False)
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_vis_flow.png', flow.visualise('bgr', return_tensor=False))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_vis_flow_arrows.png',
#             flow.visualise_arrows(80, return_tensor=False))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_vis_flow_definition.png', flow_def)
#
# mask = np.ones((601, 601), 'bool')
# mask[:301, :301] = False
# flow = of.Flow.from_transforms([['rotation', 601, 601, -30]], shape, mask=mask)
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_vis_flow_masked.png',
#             flow.visualise('bgr', True, True, return_tensor=False))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_vis_flow_arrows_masked.png',
#             flow.visualise_arrows(80, show_mask=True, show_mask_borders=True, return_tensor=False))


# # # # Usage / Ref
# # Define a flow
# flow = of.Flow.from_transforms([['rotation', 200, 150, -30]], (300, 300), 't')
#
# # Get the flow inverse: in the wrong way, and correctly in either reference
# flow_invalid_inverse = -flow
# flow_valid_inverse_t = flow.invert('t')
# flow_valid_inverse_s = flow.invert('s')
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_ref_flow.png',
#             flow.visualise_arrows(30, show_mask=True, show_mask_borders=True, return_tensor=False))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_ref_flow_inverse_wrong.png',
#             flow_invalid_inverse.visualise_arrows(30, show_mask=True, show_mask_borders=True, return_tensor=False))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_ref_flow_inverse_t.png',
#             flow_valid_inverse_t.visualise_arrows(30, show_mask=True, show_mask_borders=True, return_tensor=False))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_ref_flow_inverse_s.png',
#             flow_valid_inverse_s.visualise_arrows(30, show_mask=True, show_mask_borders=True, return_tensor=False))


# # # # Usage / Mask
# shape = (300, 400)
# flow_1 = of.Flow.from_transforms([['rotation', 200, 150, -30]], shape)
# flow_2 = of.Flow.from_transforms([['scaling', 100, 50, 0.7]], shape)
# result = of.combine_flows(flow_1, flow_2, mode=3)
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_mask_flow1.png',
#             flow_1.visualise('bgr', return_tensor=False))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_mask_flow2.png',
#             flow_2.visualise('bgr', return_tensor=False))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_mask_result.png',
#             result.visualise('bgr', return_tensor=False))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_mask_result_masked.png',
#             result.visualise('bgr', True, True, return_tensor=False))


# # # # Usage / Apply
# img = cv2.imread('_static/thames_300x400.jpg')
# transforms = [['rotation', 200, 150, -30], ['scaling', 100, 50, 0.7]]
# flow = of.Flow.from_transforms(transforms, img.shape[:2])
# warped_img, valid_area = flow.apply(to_tensor(img, True), return_valid_area=True)
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_apply_thames_warped1.png',
#             show_masked_image(warped_img, valid_area))
# flow_1 = of.Flow.from_transforms([['rotation', 200, 150, -30]], img.shape[:2])
# flow_2 = of.Flow.from_transforms([['scaling', 100, 50, 0.7]], img.shape[:2])
# result = of.combine_flows(flow_1, flow_2, mode=3)
# warped_img, valid_area = result.apply(to_tensor(img, True), return_valid_area=True)
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_apply_thames_warped2.png',
#             show_masked_image(warped_img, valid_area))
#
# # Make a circular mask
# shape = (300, 350)
# mask = np.mgrid[-shape[0]//2:shape[0]//2, -shape[1]//2:shape[1]//2]
# radius = shape[0] // 2 - 20
# mask = np.linalg.norm(mask, axis=0)
# mask = mask < radius
#
# # Load image, make two images that simulate a moving telescope
# img = cv2.imread('_static/thames_300x400.jpg')
# img1 = np.copy(img[:, :-50])
# img2 = np.copy(img[:, 50:])
# img1[~mask] = 0
# img2[~mask] = 0
#
# # Make a flow field that could have been obtained from the above images
# flow = of.Flow.from_transforms([['translation', -50, 0]], shape, 't', mask)
# flow.vecs[:, ~mask] = 0
#
# # Apply the flow to the image, setting consider_mask to True and False
# warped_img, valid_area = flow.apply(to_tensor(img1, True), to_tensor(mask), return_valid_area=True)
#
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_apply_masked_img1.png', img1)
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_apply_masked_img2.png', img2)
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_apply_masked_flow.png',
#             flow.visualise('bgr', True, True, return_tensor=False))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_apply_masked_flow_arrows.png',
#             flow.visualise_arrows(60, None, 1, True, True, return_tensor=False))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_apply_masked_img_warped.png',
#             show_masked_image(warped_img, valid_area))
#
# # Make a circular mask with the lower left corner missing
# shape = (300, 400)
# mask = np.mgrid[-shape[0]//2:shape[0]//2, -shape[1]//2:shape[1]//2]
# radius = shape[0] // 2 - 20
# mask = np.linalg.norm(mask, axis=0)
# mask = mask < radius
# mask[150:, :200] = False
#
# # Load image, make a flow field, mask both
# img = cv2.imread('_static/thames_300x400.jpg')
# flow = of.Flow.from_transforms([['scaling', 200, 150, 1.3]], shape, 's', mask)
# img[~mask] = 0
# flow.vecs[:, ~mask] = 0
#
# # Apply the flow to the image, setting consider_mask to True and False
# img_true = flow.apply(to_tensor(img, True), consider_mask=True)
# img_false = flow.apply(to_tensor(img, True), consider_mask=False)
#
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_apply_consider_mask_img.png', img)
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_apply_consider_mask_flow.png',
#             flow.visualise('bgr', True, True, return_tensor=False))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_apply_consider_mask_flow_arrows.png',
#             flow.visualise_arrows(50, show_mask=True, show_mask_borders=True, return_tensor=False))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_apply_consider_mask_true.png', to_numpy(img_true, True))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_apply_consider_mask_false.png', to_numpy(img_false, True))
#
# # Load image and pad with a black border
# img = cv2.imread('_static/thames_300x400.jpg')
# pad_img = np.pad(img, ((100, 100), (100, 100), (0, 0)))
#
# # Create a flow field with an undefined area in the lower left corner
# mask = np.ones(img.shape[:2], 'bool')
# mask[100:, :200] = False
# transforms = [['scaling', 100, 50, 1.1]]
# flow = of.Flow.from_transforms(transforms, img.shape[:2], 's', mask)
# flow.vecs[100:, :200] = 0
#
# # Apply the flow field to the padded image considering the flow mask (consider_mask=True), default behaviour
# warped_consider_true = flow.apply(to_tensor(pad_img, True),
#                                   padding=(100, 100, 100, 100), cut=False, consider_mask=True)
#
# # Apply the flow field to the padded image without considering the flow mask (consider_mask=False)
# warped_consider_false = flow.apply(to_tensor(pad_img, True),
#                                    padding=(100, 100, 100, 100), cut=False, consider_mask=False)
#
# show_masked_image(warped_consider_true)
# show_masked_image(warped_consider_false)


# # # # Usage / Padding
# # Load an image
# full_img = cv2.imread('_static/thames.jpg')  # original resolution 600x800
#
# # Define a flow field
# shape = (300, 300)
# transforms = [['rotation', 200, 150, -30], ['scaling', 100, 50, 0.7]]
# flow = of.Flow.from_transforms(transforms, shape)
#
# # Get the necessary padding
# padding = flow.get_padding()
#
# # Select an image patch that is equal in size to the flow resolution plus the padding
# padded_patch = full_img[:shape[0] + sum(padding[:2]), :shape[1] + sum(padding[2:])]
#
# # Apply the flow field to the image patch, passing in the padding
# warped_padded_patch = flow.apply(to_tensor(padded_patch, True), padding=padding)
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_padding_padded_warped.png', to_numpy(warped_padded_patch, True))
#
# # As a comparison: cut an unpadded patch out of the image and warp it with the same flow
# patch = full_img[padding[0]:padding[0] + shape[0], padding[2]:padding[2] + shape[1]]
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_padding_patch.png', patch)
# warped_patch = flow.apply(to_tensor(patch, True))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_padding_warped.png', to_numpy(warped_patch, True))
#
# # Load an image, define a flow field
# img = cv2.imread('_static/thames_300x400.jpg')
# transforms = [['rotation', 200, 150, -30], ['scaling', 100, 50, 0.9]]
# flow = of.Flow.from_transforms(transforms, img.shape[:2], 's')  # 300x400 pixels
#
# # Find the padding and pad the image
# padding = flow.get_padding()
# padded_img = np.pad(img, (tuple(padding[:2]), tuple(padding[2:]), (0, 0)))
#
# # Apply the flow field to the image patch, with and without the padding
# warped_img = flow.apply(to_tensor(img, True))
# warped_padded_img = flow.apply(to_tensor(padded_img, True), padding=padding, cut=False)
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_padding_s_warped.png', to_numpy(warped_img, True))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_padding_s_warped_padded.png', to_numpy(warped_padded_img, True))


# # # # Usage / Source & Target
# # Define a flow field
# shape = (300, 400)
# transforms = [['rotation', 200, 150, -30], ['scaling', 100, 50, 1.2]]
# flow = of.Flow.from_transforms(transforms, shape)
#
# # Get the valid source and target areas
# valid_source = flow.valid_source()
# valid_target = flow.valid_target()
#
# # Load an image and warp it with the flow
# img = cv2.imread('_static/thames_300x400.jpg')
# warped_img = flow.apply(to_tensor(img, True))
#
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_source_target_img.png', img)
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_source_target_warped_img.png', to_numpy(warped_img, True))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_source_target_source.png',
#             255 * to_numpy(valid_source).astype('uint8'))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_source_target_target.png',
#             255 * to_numpy(valid_target).astype('uint8'))


# # # # Usage / Track
# background = np.zeros((40, 60, 3), 'uint8')
# pts = np.array([[5, 15], [20, 15], [5, 50], [20, 50]])
# flow = of.Flow.from_transforms([['rotation', 0, 0, -15]], background.shape[:2], 's')
# tracked_pts = flow.track(torch.tensor(pts), int_out=True)
# background[pts[:, 0], pts[:, 1]] = 255
# background[tracked_pts[:, 0], tracked_pts[:, 1], 2] = 255
# background = np.repeat(np.repeat(background, 5, axis=0), 5, axis=1)
# bgr = flow.resize(2.5).visualise_arrows(grid_dist=30, show_mask=True, show_mask_borders=True, return_tensor=False)
# bgr = np.repeat(np.repeat(bgr, 2, axis=0), 2, axis=1)
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_track_flow.png', bgr)
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_track_pts.png', background)
#
# background = np.zeros((40, 60, 3), 'uint8')
# pts = np.array([[5, 15], [20, 15], [5, 50], [20, 50]])
# mask = np.ones((40, 60), 'bool')
# mask[:15, :30] = False
# flow = of.Flow.from_transforms([['rotation', 0, 0, -25]], background.shape[:2], 's', mask)
# tracked_pts, valid_status = flow.track(torch.tensor(pts), int_out=True, get_valid_status=True)
# background[pts[:, 0], pts[:, 1]] = 255
# background[tracked_pts[valid_status][:, 0], tracked_pts[valid_status][:, 1], 2] = 255
# background = np.repeat(np.repeat(background, 5, axis=0), 5, axis=1)
# bgr = flow.resize(2.5).visualise_arrows(grid_dist=30, show_mask=True, show_mask_borders=True, return_tensor=False)
# bgr = np.repeat(np.repeat(bgr, 2, axis=0), 2, axis=1)
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_track_flow_with_validity.png', bgr)
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_track_pts_with_validity.png', background)


# # # # Usage / Combining
# # Define a flow field
# shape = (300, 400)
# flow_1 = of.Flow.from_transforms([['rotation', 200, 150, -30]], shape)
# flow_2 = of.Flow.from_transforms([['scaling', 100, 50, 1.2]], shape)
# flow_3 = of.Flow.from_transforms([['rotation', 200, 150, -30], ['scaling', 100, 50, 1.2]], shape)
#
# flow_1_result = of.combine_flows(flow_2, flow_3, mode=1)
# flow_2_result = of.combine_flows(flow_1, flow_3, mode=2)
# flow_3_result = of.combine_flows(flow_1, flow_2, mode=3)
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_combining_1.png',
#             flow_1.visualise('bgr', True, True, return_tensor=False))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_combining_2.png',
#             flow_2.visualise('bgr', True, True, return_tensor=False))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_combining_3.png',
#             flow_3.visualise('bgr', True, True, return_tensor=False))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_combining_1_result.png',
#             flow_1_result.visualise('bgr', True, True, return_tensor=False))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_combining_2_result.png',
#             flow_2_result.visualise('bgr', True, True, return_tensor=False))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/usage_combining_3_result.png',
#             flow_3_result.visualise('bgr', True, True, return_tensor=False))


# # # # Flow field for flow doc
# shape = (200, 200)
# flow = of.Flow.from_transforms([['rotation', 300, -50, -30]], shape, 't')
# flow.show_arrows(grid_dist=50)
# flow = of.Flow.from_transforms([['rotation', 300, -50, -30]], shape, 's')
# flow.show_arrows(grid_dist=50)


# # # # Flow field for README
# # Make a flow field and display it
# shape = (300, 400)
# flow = of.Flow.from_transforms([['rotation', 200, 150, -30]], shape)
# # flow.show()
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/index_flow_1.png', to_numpy(flow.visualise('bgr'), True))
#
# flow_2 = of.Flow.from_transforms([['translation', 40, 0]], shape)
# result = of.combine_flows(flow, flow_2, mode=3)
# # result.show(show_mask=True, show_mask_borders=True)
# # result.show_arrows(show_mask=True, show_mask_borders=True)
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/index_result.png', to_numpy(result.visualise('bgr', True, True), True))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/index_result_arrows.png',
#             to_numpy(result.visualise_arrows(show_mask=True, show_mask_borders=True), True))

# # # # Images for the repo "social preview"
# img = cv2.imread('_static/thames_300x400.jpg')[60:-40]
# shape = (200, 400)
# flow = of.Flow.from_transforms([['rotation', 200, 100, -30]], shape)
# warped = flow.apply(img)
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/img.png', img)
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/warped.png', warped)
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/flow.png', flow.visualise('bgr'))
# cv2.imwrite('C:/Users/RVIM_Claudio/Downloads/arrows.png', flow.visualise_arrows(50, None, .5, thickness=6))
