import os
import glob

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
            ]
        )
        self._pixel_augmentor = A.Compose(
            [
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

    def _horizontal_flip(self, image, label):
        f = np.random.randint(2)
        if f == 0:
            image = np.fliplr(image)
            label = np.fliplr(label)

        return image, label

    def _vertical_flip(self, image, label):
        f = np.random.randint(2)
        if f == 0:
            image = np.flipud(image)
            label = np.flipud(label)

        return image, label

    def _random_rotate90(self, image, label):
        k = np.random.randint(4)

        image = np.rot90(image, k)
        label = np.rot90(label, k)

        return image, label

    def _augmentation(self, image, label):

        image, label = self._horizontal_flip(image, label)
        image, label = self._vertical_flip(image, label)
        image, label = self._random_rotate90(image, label)
        image = self._pixel_augmentor(image=image)["image"]

        return image, label

    def _vectorize(self, image, label):

        return self._compose(image), self._compose(label)

    def transform(self, image, label, train=True):

        if train:
            image, label = self._augmentation(image, label)
        image, label = self._vectorize(image, label)

        return image, label


class ImageDataset(Dataset):
    def __init__(self, inputs, labels, train=True):

        self._train = train
        self.inputs = []
        self.labels = []
        # TODO: is this the best?
        for input_dir, label_dir in zip(inputs, labels):
            self.inputs.extend(glob.glob(os.path.join(input_dir, "*.npy")))
            self.labels.extend(glob.glob(os.path.join(label_dir, "*.npy")))
        self.transformer = Transformer()

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, index):

        image = self.inputs[index]
        label = self.labels[index]

        image, label = get_npy(image), get_npy(label)

        image, label = self.transformer.transform(image, label, self._train)

        return image, label
