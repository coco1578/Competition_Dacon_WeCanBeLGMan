import numpy as np

from torch.utils.data import Dataset


def get_npy(image):

    image = np.load(image)
    image = image.astype(np.float32) / 255.0

    return image


class BaseDataset(Dataset):
    def __init__(self, inputs, labels):

        self.inputs = inputs
        self.labels = labels

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, index):

        image = self.inputs[index]
        label = self.labels[index]

        image, label = get_npy(image), get_npy(label)

        return image, label
