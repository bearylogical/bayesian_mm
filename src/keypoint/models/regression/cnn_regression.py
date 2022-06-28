from keras import Model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Resizing

from src.utils.constants import NUM_TARGETS


class BaseKeypointModel(Model):
    def __init__(self, num_target=NUM_TARGETS, img_size: tuple = (224, 224)):
        """
        Base Model for image regression

        :param num_target: Number of target features as an output 1D vector
        :param img_size: Height x Width of input image.
        """
        super().__init__()
        self.preprocess_resize = Resizing(*img_size, crop_to_aspect_ratio=True)
        self.conv_1 = Conv2D(32, 3, activation="relu", input_shape=(*img_size, 1))
        self.conv_2 = Conv2D(32, 3, activation="relu")
        self.pool_1 = MaxPooling2D(3)
        self.conv_3 = Conv2D(16, 5, activation="relu")
        self.conv_4 = Conv2D(16, 5, activation="relu")
        self.pool_3 = MaxPooling2D(3)
        self.flatten_1 = Flatten()
        self.dense3 = Dense(num_target)
        self.num_target = num_target

    def call(self, inputs, training=None, mask=None):
        x = self.preprocess_resize(inputs)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.pool_1(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.pool_3(x)
        x = self.flatten_1(x)

        return self.dense3(x)

