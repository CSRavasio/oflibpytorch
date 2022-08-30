import random
import math
import numpy as np
import src.oflibpytorch as of


def one_it(mode):
    trans = random.random() * max_length ** 2
    trans_1 = trans * random.random()
    trans_2 = trans - trans_1
    sign_1 = 1 if random.random() < 0.5 else -1
    sign_2 = 1 if random.random() < 0.5 else -1
    transform_list = [
        ['translation', math.sqrt(trans_1) * sign_1, math.sqrt(trans_2) * sign_2],
        ['rotation', w * (-offset + (1 + 2*offset) * random.random()),
         h * (-offset + (1 + 2*offset) * random.random()),
         -max_angle + 2 * max_angle * random.random()],
        ['scaling', w * (-offset + (1 + 2*offset) * random.random()),
         h * (-offset + (1 + 2*offset) * random.random()),
         1 - max_scaling_factor + 2 * max_scaling_factor * random.random()],
    ]
    transforms = random.choices(transform_list, k=2)
    f1 = of.Flow.from_transforms(transforms[0:1], (h, w), random.choice(ref_list), device=device)
    f2 = of.Flow.from_transforms(transforms[1:2], (h, w), random.choice(ref_list), device=device)
    f3 = of.Flow.from_transforms(transforms, (h, w), random.choice(ref_list), device=device)
    result_calc, result_real = 0, 0
    if mode == 1:
        res = f2.combine(f3, mode=1, ref=f1.ref)
        mask = res.mask_numpy & f1.mask_numpy
        result_real = f1.vecs_numpy[mask]
        result_calc = res.vecs_numpy[mask]
    elif mode == 2:
        res = f1.combine(f3, mode=2, ref=f2.ref)
        mask = res.mask_numpy & f2.mask_numpy
        result_real = f2.vecs_numpy[mask]
        result_calc = res.vecs_numpy[mask]
    elif mode == 3:
        res = f1.combine(f2, mode=3, ref=f3.ref)
        mask = res.mask_numpy & f3.mask_numpy
        result_real = f3.vecs_numpy[mask]
        result_calc = res.vecs_numpy[mask]
    err_vecs = result_calc - result_real
    err_abs = np.linalg.norm(err_vecs, axis=-1)
    err_rel = err_abs / np.linalg.norm(result_real, axis=-1)
    return err_abs, err_rel


device = 'cpu'                                  # Device to run on
offset = 0                                      # h or w fraction by which transform centre can be outside img area
h, w = 150, 250                                 # Flow field shape
abs_limit = [0.1, 0.05, 0.01, 0.005]            # Absolute error values (in px)
rel_limit = [0.01, 0.005, 0.001, 0.0005]        # Relative error values
num_its = 100                                  # Number of iterations to run
mode = 1                                        # Flow composition mode
max_length = 50                                 # Maximum flow vector magnitude

max_dist = np.sqrt((w + w*offset)**2 + (h + h*offset)**2)
max_scaling_factor = max_length / max_dist
max_angle = np.rad2deg(np.arctan(max_length / max_dist))
ref_list = ['s', 't']

err_abs_list, err_rel_list = [], []
for i in range(num_its):
    err_abs, err_rel = one_it(mode)
    err_abs_list.extend(err_abs.flatten().tolist())
    err_rel_list.extend(err_rel.flatten().tolist())


print("Mode {}".format(mode))
print("Abs below {}: {}, {}: {}, {}: {}, {}: {}".format(
    abs_limit[0], np.sum(np.array(err_abs_list) < abs_limit[0]) / len(err_abs_list),
    abs_limit[1], np.sum(np.array(err_abs_list) < abs_limit[1]) / len(err_abs_list),
    abs_limit[2], np.sum(np.array(err_abs_list) < abs_limit[2]) / len(err_abs_list),
    abs_limit[3], np.sum(np.array(err_abs_list) < abs_limit[3]) / len(err_abs_list)
))
print("Rel below {}: {}, {}: {}, {}: {}, {}: {}".format(
    rel_limit[0], np.sum(np.array(err_rel_list) < rel_limit[0]) / len(err_rel_list),
    rel_limit[1], np.sum(np.array(err_rel_list) < rel_limit[1]) / len(err_rel_list),
    rel_limit[2], np.sum(np.array(err_rel_list) < rel_limit[2]) / len(err_rel_list),
    rel_limit[3], np.sum(np.array(err_rel_list) < rel_limit[3]) / len(err_rel_list)
))
print("Mean abs: {}, mean rel: {}".format(sum(err_abs_list) / len(err_abs_list), sum(err_rel_list) / len(err_rel_list)))
print("Total points: {}. Max abs: {}".format(len(err_abs_list), max(err_abs_list)))
