from tensorflow.keras import Model
from typing import Tuple

class BaseModel(Model):
    def __init__(self, img_size: Tuple[int, int], channels: int, num_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_size = img_size
        self.channels = channels
        self.num_classes = num_classes
        self.model = None
        self.model_name = ""


if __name__ == "__main__":
    from src.models.classification.unet import UnetModel
    from tensorflow.keras import Input

    inputs = Input(shape=(128, 128) + (1,))
    unet = UnetModel((128, 128))
    unet.build(input_shape=(None, 128, 128) + (1,))
    unet.call(inputs)
    unet.summary()

