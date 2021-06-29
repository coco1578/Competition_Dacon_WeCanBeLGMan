import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import Dataset


def get_npy(image):

    image = np.load(image)
    image = image.astype(np.float32) / 255.0

    return image


class Transformer:
    def __init__(self):

        self._compose = transforms.Compose([transforms.ToTensor()])

    def transform(self, image):

        image = self._compose(image)

        return image


class BaseDataset(Dataset):
    def __init__(self, inputs, labels):

        self.inputs = inputs
        self.labels = labels
        self.transformer = Transformer()

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, index):

        image = self.inputs[index]
        label = self.labels[index]

        image, label = get_npy(image), get_npy(label)
        image, label = self.transformer.transform(image), self.transformer.transform(
            label
        )

        return image, label
