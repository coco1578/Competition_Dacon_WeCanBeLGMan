import os
import math

import cv2
import numpy as np


def recover_image(path, image_size=204):

    top_lefts = np.load(os.path.join(path, "top_lefts.npy"))
    result_image = np.zeros(
        (top_lefts[-1][0] + image_size, top_lefts[-1][1] + image_size, 3)
    )
    result_masks = np.zeros(
        (top_lefts[-1][0] + image_size, top_lefts[-1][1] + image_size, 3)
    )

    for i in range(len(top_lefts)):
        npy_path = os.path.join(path, f"{i}.npy")
        result_image[
            top_lefts[i][0] : top_lefts[i][0] + image_size,
            top_lefts[i][1] : top_lefts[i][1] + image_size,
            :,
        ] += np.load(npy_path)
        result_masks[
            top_lefts[i][0] : top_lefts[i][0] + image_size,
            top_lefts[i][1] : top_lefts[i][1] + image_size,
            :,
        ] += 1

    result_image = result_image / result_masks
    result_image = np.uint8(result_image)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    return result_image


def rmse_score(y_true, y_pred):
    score = math.sqrt(np.mean((y_true - y_pred) ** 2))

    return score


def psnr_score(y_true, y_pred, pixel_max=255.0):
    score = 20 * np.log10(pixel_max / rmse_score(y_true, y_pred))

    return score


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
