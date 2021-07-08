import os
import math

import cv2
import torch
import numpy as np


def recover_image(path, model, device, image_size=256):

    top_lefts = np.load(os.path.join(path, "top_lefts.npy"))
    result_image = np.zeros((2448, 3264, 3))
    result_mask = np.zeros((2448, 3264, 3))

    for i in range(len(top_lefts)):

        piece = np.load(os.path.join(path, f"{i}.npy"))
        piece = torch.from_numpy(piece).permute(2, 0, 1)
        piece = piece.unsqueeze(dim=0)
        piece = piece.to(device)

        output = model(piece)
        output = output.cpu().detach().squeeze().permute(1, 2, 0).numpy()
        output = output * 255.0

        h, w, c = result_image[
            top_lefts[i][0] : top_lefts[i][0] + 256,
            top_lefts[i][1] : top_lefts[i][1] + 256,
            :,
        ].shape

        result_image[
            top_lefts[i][0] : top_lefts[i][0] + 256,
            top_lefts[i][1] : top_lefts[i][1] + 256,
            :,
        ] += output
        result_mask[
            top_lefts[i][0] : top_lefts[i][0] + 256,
            top_lefts[i][1] : top_lefts[i][1] + 256,
            :,
        ] += 1

    result_image = result_image / result_mask
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
