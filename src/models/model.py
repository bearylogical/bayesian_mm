from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Softmax
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
    from src.models.unet import UnetModel
    from tensorflow.keras import Input
    inputs = Input(shape=(128, 128) + (1,))
    output = Softmax()
    mdl = UnetModel((128, 128))
    mdl.build(input_shape=(None, 128, 128) + (1,))
    mdl.call(inputs)
    mdl.summary()


