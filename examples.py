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
# This file is part of oflibpytorch

import oflibpytorch as of
import torch
import numpy as np
import cv2


# This file contains some quick examples of oflibnumpy functions, and can be used to demonstrate it works as intended.
# When running the code, OpenCV windows with images will be shown. Closing them will cause the code to continue to run.


# Make a flow field and display it
shape = (300, 400)
flow = of.Flow.from_transforms([['rotation', 200, 150, -30]], shape)
flow.show()

# Combine sequentially with another flow field, display the result
flow_2 = of.Flow.from_transforms([['translation', 40, 0]], shape)
result = flow.combine_with(flow_2, mode=3)
result.show(show_mask=True, show_mask_borders=True)

# Display the result with arrows
result.show_arrows(show_mask=True, show_mask_borders=True)

# Get valid source and target areas
source_area = result.valid_source()
target_area = result.valid_target()

# Create an interesting sample image and warp it with the result flow
img = (torch.arange(300)[:, np.newaxis] * torch.arange(400)).to(torch.uint8)
img_warped = result.apply(img)

# Display the image, warped image, and valid source and target areas in one
top_row = torch.cat((img, img_warped), dim=1)
bot_row = 255 * torch.cat((source_area, target_area), dim=1).to(torch.uint8)
combined_img = torch.cat((top_row, bot_row), dim=0)
cv2.imshow("Image", np.array(combined_img))
cv2.waitKey()

# NOTE: the source area contains the parts of the image still visible after warping. The valid area contains the parts
# of the warped image that show content from the original image.
