import os
import glob
import argparse

import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from model.gmsnet import GMSNet
from utils.common import AverageMeter, psnr_score
from dataset.loader import BaseDataset

from torchvision.utils import save_image


def submit():

    transformer = transforms.Compose([transforms.ToTensor()])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = GMSNet()
    model_info = torch.load(
        "/home/salmon21/LG/WeCanBeLGMan/save_model/gmsnet/best_model.pth.tar",
        map_location="cpu",
    )
    model.load_state_dict(model_info["state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    results = []

    base_path = "/home/salmon21/Desktop/submit/"
    images = glob.glob("/home/salmon21/LG/dataset/test/*.png")
    # image_path = "/home/salmon21/LG/dataset/test/test_input_20000.png"
    with torch.no_grad():
        for image_path in images:
            image = cv2.imread(image_path)
            image = image.astype(np.float32) / 255.0
            crop = []
            position = []
            batch_count = 0

            result_image = np.zeros_like(image)
            for top in range(0, image.shape[0], 64):
                for left in range(0, image.shape[1], 64):
                    piece = np.zeros([128, 128, 3], np.float32)
                    temp = image[top : top + 128, left : left + 128, :]
                    piece[: temp.shape[0], : temp.shape[1], :] = temp
                    crop.append(piece)
                    position.append([top, left])
                    batch_count += 1
                    if batch_count == 128:
                        pred = []
                        # crop = np.array(crop)
                        for c in crop:
                            c = transformer(c)
                            output = model(c)
                            output = output.cpu().squeeze().detach().numpy() * 255.0
                            output = np.transpose(output, (1, 2, 0))
                            pred.append(output)
                        crop = []
                        for num, (t, l) in enumerate(position):
                            piece = pred[num]
                            h, w, c = result_image[t : t + 128, l : l + 128, :].shape
                            result_image[t : t + 128, l : l + 128, :] += piece

                        position = []

            result_image = result_image.astype(np.uint8)
            results.append(result_image)


if __name__ == "__main__":

    submit()
