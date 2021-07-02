import os

import numpy as np
import albumentations as A
import torchvision.transforms as transforms

from torch.utils.data import Dataset


def get_npy(image):

    image = np.load(image)

    return image


class Transformer:
    def __init__(self):

        self._compose = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        self._aug_transformer = A.Compose(
            [
                A.OneOf(
                    [
                        A.VerticalFlip(),
                        A.HorizontalFlip(),
                        A.RandomRotate90(),
                        A.Transpose(),
                        A.ShiftScaleRotate(),
                    ]
                ),
                A.OneOf([A.Blur(), A.GaussianBlur()]),
                A.OneOf(
                    [
                        A.ISONoise(),
                        A.GaussNoise(),
                        A.MultiplicativeNoise(
                            multiplier=[0.5, 1.5], elementwise=True, p=1
                        ),
                        A.IAAAdditiveGaussianNoise(),
                    ]
                ),
            ]
        )

    def augmentation(self, image):

        return self._aug_transformer(image=image)["image"]

    def vectorize(self, image):

        return self._compose(image)

    def transform(self, image, train=True):

        if train:
            image = self._augmentation(image)
        image = self._vectorize(image)

        return image


class RealDataset(Dataset):
    def __init__(self, inputs, labels, train=True):

        self.inputs = inputs
        self.labels = labels
        self._train = train
        self.transformer = Transformer()

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, index):

        image = self.inputs[index]
        label = self.labels[index]

        image, label = get_npy(image), get_npy(label)

        if self._train:
            image = self.transformer.augmentation(image)
        image = self.transformer.vectorize(image)
        label = self.transformer.vectorize(label)

        return image, label, np.zeros((3, 204, 204)), np.zeros((3, 204, 204))


class SynDataset(Dataset):
    def __init__(self, noise, sigma):

        self.noise = noise
        self.sigman = sigma
        self.transformer = Transformer()

    def __len__(self):

        return len(self.noise)

    def __getitem__(self, index):

        noise = self.noise[index]
        sigma = self.sigma[index]

        clean = os.path.join(
            "/home/salmon21/LG/dataset/train/input/204_102", os.path.basename(noise)
        )

        clean, noise, sigma = get_npy(clean), get_npy(noise), get_npy(sigma)

        # else:
        clean = self.transformer.vectorize(clean)
        noise = self.transformer.vectorize(noise)
        sigma = self.transformer.vectorize(sigma) / 15.0  # inverse scaling

        return noise, clean, sigma, np.ones((3, 204, 204))
