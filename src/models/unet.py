from src.models.model import BaseModel
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D, Dropout
from tensorflow.keras import Model


class UnetModel(BaseModel):
    def __init__(self, img_size: tuple[int, int], channels: int = 1, num_classes: int = 1):
        super().__init__(img_size, channels, num_classes)
        self._get_model()

    def _get_model(self):
        inputs = Input(shape=self.img_size + (self.num_classes,))

        # downsampling blocks
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
        drop5 = Dropout(0.5)(conv5)

        # upsampling blocks
        upconv6 = Conv2D(512, 2, activation='relu', padding='same')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, upconv6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

        upconv7 = Conv2D(256, 2, activation='relu', padding='same')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, upconv7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

        upconv8 = Conv2D(128, 2, activation='relu', padding='same')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, upconv8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

        upconv9 = Conv2D(64, 2, activation='relu', padding='same')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, upconv9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        self.model = Model(inputs, conv10)
