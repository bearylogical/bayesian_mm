from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, UpSampling2D
from tensorflow.keras import Model
from typing import Tuple


class UnetModel(Model):
    def __init__(self, img_size: Tuple[int, int],
                       channels: int = 1,
                       num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes
        # downsampling blocks
        self.conv1_1 = Conv2D(64, 3, activation='relu', padding='same')
        self.conv1_2 = Conv2D(64, 3, activation='relu', padding='same')
        self.pool1 = MaxPooling2D(pool_size=(2, 2))

        self.conv2_1 = Conv2D(128, 3, activation='relu', padding='same')
        self.conv2_2 = Conv2D(128, 3, activation='relu', padding='same')
        self.pool2 = MaxPooling2D(pool_size=(2, 2))

        self.conv3_1 = Conv2D(256, 3, activation='relu', padding='same')
        self.conv3_2 = Conv2D(256, 3, activation='relu', padding='same')
        self.pool3 = MaxPooling2D(pool_size=(2, 2))

        self.conv4_1 = Conv2D(512, 3, activation='relu', padding='same')
        self.conv4_2 = Conv2D(512, 3, activation='relu', padding='same')
        self.pool4 = MaxPooling2D(pool_size=(2, 2))

        self.conv5_1 = Conv2D(1024, 3, activation='relu', padding='same')
        self.conv5_2 = Conv2D(1024, 3, activation='relu', padding='same')

        # upsampling blocks
        self.up6 = UpSampling2D(size=(2, 2))
        self.conv6_1 = Conv2D(512, 2, activation='relu', padding='same')
        self.conv6_2 = Conv2D(512, 3, activation='relu', padding='same')
        self.conv6_3 = Conv2D(512, 3, activation='relu', padding='same')

        self.up7 = UpSampling2D(size=(2, 2))
        self.conv7_1 = Conv2D(256, 2, activation='relu', padding='same')
        self.conv7_2 = Conv2D(256, 3, activation='relu', padding='same')
        self.conv7_3 = Conv2D(256, 3, activation='relu', padding='same')

        self.up8 = UpSampling2D(size=(2, 2))
        self.conv8_1 = Conv2D(128, 2, activation='relu', padding='same')
        self.conv8_2 = Conv2D(128, 3, activation='relu', padding='same')
        self.conv8_3 = Conv2D(128, 3, activation='relu', padding='same')

        self.up9 = UpSampling2D(size=(2, 2))
        self.conv9_1 = Conv2D(64, 2, activation='relu', padding='same')
        self.conv9_2 = Conv2D(64, 3, activation='relu', padding='same')
        self.conv9_3 = Conv2D(64, 3, activation='relu', padding='same')
        self.conv9_4 = Conv2D(2, 3, activation='relu', padding='same')
        self.conv10 = Conv2D(self.num_classes, 1, activation="sigmoid")

    def call(self, inputs, **kwargs):
        training = kwargs.get('training', False)

        x1 = self.conv1_1(inputs)
        x1 = self.conv1_2(x1)
        p1 = self.pool1(x1)

        x2 = self.conv2_1(p1)
        x2 = self.conv2_2(x2)
        p2 = self.pool2(x2)

        x3 = self.conv3_1(p2)
        x3 = self.conv3_2(x3)
        p3 = self.pool3(x3)

        x4 = self.conv4_1(p3)
        x4 = self.conv4_2(x4)
        p4 = self.pool4(x4)

        x5 = self.conv5_1(p4)
        x5 = self.conv5_2(x5)

        x6 = self.up6(x5)
        x6 = self.conv6_1(x6)
        x6 = concatenate([x4, x6], axis=3)
        x6 = self.conv6_2(x6)
        x6 = self.conv6_3(x6)

        x7 = self.up7(x6)
        x7 = self.conv7_1(x7)
        x7 = concatenate([x3, x7], axis=3)
        x7 = self.conv7_2(x7)
        x7 = self.conv7_3(x7)

        x8 = self.up8(x7)
        x8 = self.conv8_1(x8)
        x8 = concatenate([x2, x8], axis=3)
        x8 = self.conv8_2(x8)
        x8 = self.conv8_3(x8)

        x9 = self.up9(x8)
        x9 = self.conv9_1(x9)
        x9 = concatenate([x1, x9], axis=3)
        x9 = self.conv9_2(x9)
        x9 = self.conv9_3(x9)
        x9 = self.conv9_4(x9)

        return self.conv10(x9)





