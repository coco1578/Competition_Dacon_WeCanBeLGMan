import torch.nn as nn
import segmentation_models_pytorch as smp


class Dacon(nn.Module):
    def __init__(self, in_channels=3, classes=3):
        super(Dacon, self).__init__()

        self.model_first = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=classes,
            activation="sigmoid",
        )
        self.model_second = smp.UnetPlusPlus(
            encoder_name="efficientnet-b1",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=classes,
            activation="sigmoid",
        )

    def forward(self, x):

        output = self._model_first(x)
        skip_conn = output + x
        output = self._model_second(skip_conn)

        return output
