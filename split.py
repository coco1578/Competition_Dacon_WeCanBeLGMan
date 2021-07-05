import os
import glob
import argparse

import tqdm
import cv2
import numpy as np


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=204)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--phase", type=str, required=True)

    args = parser.parse_args()

    return args


def split_image(image_list, base_path, size=204, stride=102):

    for image in tqdm.tqdm(image_list):
        top_lefts = []
        num = 0
        folder_name = os.path.basename(image).split("_")[-1].split(".")[0]
        folder_path = os.path.join(base_path, folder_name)
        top_lefts_path = os.path.join(folder_path, "top_lefts.npy")
        os.makedirs(folder_path)
        image = cv2.imread(image)
        for top in range(0, image.shape[0], stride):
            for left in range(0, image.shape[1], stride):
                piece = np.zeros([size, size, 3], np.uint8)
                temp = image[top : top + size, left : left + size, :]
                if temp.shape[0] != 204 or temp.shape[1] != 204:
                    continue
                else:
                    top_lefts.append([top, left])
                    piece[: temp.shape[0], : temp.shape[1], :] = temp
                    save_path = os.path.join(folder_path, f"{num}.npy")
                    print(save_path)
                    np.save(save_path, piece)
                    num += 1
        if not os.path.exists(top_lefts_path):
            np.save(top_lefts_path, top_lefts)


def main():

    args = parse_args()

    if args.phase.lower() == "train":
        image_list = sorted(glob.glob(os.path.join(args.img_path, "input/*.png")))
        split_image(
            image_list,
            os.path.join(args.img_path, "input", args.save_path),
            size=args.img_size,
            stride=args.stride,
        )
    elif args.phase.lower == "valid":
        image_list = sorted(glob.glob(os.path.join(args.img_path, "label/*.png")))
        split_image(
            image_list,
            os.path.join(args.img_path, "label", args.save_path),
            size=args.img_size,
            stride=args.stride,
        )
    elif args.phas.lower == "test":
        image_list = sorted(glob.glob(os.path.join(args.img_path, "*.png")))
        split_image(
            image_list,
            os.path.join(args.img_path, args.save_path),
            size=args.img_size,
            stride=args.stride,
        )


if __name__ == "__main__":

    main()
