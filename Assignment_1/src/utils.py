import numpy as np


def convergence_criteria(list_of_criteria, epsilon):

    for criteria in list_of_criteria:
        if any(abs(abs(criteria[-1] / criteria[-2]) - 1) > epsilon):
            return False
        print("K has converged")

    return True


def get_mask(depth, r_range):
    mask = []
    for idx, d in enumerate(depth):
        if idx:
            mask.append(np.bitwise_and(depth[idx - 1] < r_range, r_range <= d))
        else:
            mask.append(r_range <= d)
    return mask


def apply_mask(parameter, mask, input):
    for idx, m in enumerate(mask):
        parameter[m] = input[idx]
    return parameter
