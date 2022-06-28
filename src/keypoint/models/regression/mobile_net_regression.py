from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Concatenate, Dense, Resizing, Dropout
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import Model, Input

# transfer learning demo

class MobleNetRegresssion(Model):
    def __init__(self, num_target=4, img_size:tuple=(128,128), dropout=0.2):
        """
        Base Model for image regression

        :param num_target: Number of target features as an output 1D vector
        :param img_size: Height x Width of input image.
        """
        super(MobleNetRegresssion, self).__init__()
        self.concat = Concatenate()
        self.preprocess_resize = Resizing(*img_size, crop_to_aspect_ratio=True)
        self.mobile_net = MobileNetV2(input_shape=img_size + (3,), alpha=1.0, include_top=False, weights="imagenet")
        self.mobile_net.trainable = False
        self.avgpool1 = GlobalAveragePooling2D()
        self.dense1 = Dense(500)
        self.dropout1 = Dropout(dropout)
        self.dense2 = Dense(128)
        self.dropout2 = Dropout(dropout)
        self.dense3 = Dense(num_target)
        self.num_target = num_target

    def call(self, inputs, training=None, mask=None):
        x = self.concat([inputs, inputs, inputs])
        x = self.preprocess_resize(x)
        x = preprocess_input(x)
        x = self.mobile_net(x)
        x = self.avgpool1(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return self.dense3(x)


if __name__ == "__main__":
    aa = MobleNetRegresssion()
    model_input = Input((128, 128, 1))
    aa(model_input)
    aa.build(input_shape=(None, 128, 128, 1))
    aa.call(model_input)
    aa.summary()