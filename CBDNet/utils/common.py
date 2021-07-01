import math
import numpy as np


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
