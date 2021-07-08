import os
import glob
import argparse

import tqdm
import cv2
import numpy as np


def split_image(image_list, base_path, size=256, stride=128):

    for image in tqdm.tqdm(image_list):
        top_lefts = []
        num = 0
        folder_name = os.path.basename(image).split("_")[-1].split(".")[0] + "_256"
        folder_path = os.path.join(base_path, folder_name)
        top_lefts_folder_path = os.path.join(folder_path, "top_lefts")
        os.makedirs(folder_path)
        os.makedirs(top_lefts_folder_path)
        top_lefts_path = os.path.join(top_lefts_folder_path, "top_lefts.npy")
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for top in range(0, image.shape[0], stride):
            for left in range(0, image.shape[1], stride):
                piece = np.zeros([size, size, 3], np.uint8)
                temp = image[top : top + size, left : left + size, :]
                piece[: temp.shape[0], : temp.shape[1], :] = temp
                save_path = os.path.join(folder_path, f"{num}.npy")
                np.save(save_path, piece)
                num += 1
                top_lefts.append([top, left])
        np.save(top_lefts_path, top_lefts)


input_image_list = sorted(glob.glob("./train/input/*.png"))
label_image_list = sorted(glob.glob("./train/label/*.png"))
test_image_list = sorted(glob.glob("./test/*.png"))

for input_, label_ in zip(input_image_list, label_image_list):
    print(input_, label_)

for test_ in test_image_list:
    print(test_)

split_image(input_image_list, "./train/input")
split_image(label_image_list, "./train/label")
split_image(label_image_list, "./test", stride=32)
