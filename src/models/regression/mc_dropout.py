from keras import Model, Input
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Resizing, Dropout
from keras.regularizers import L2
import numpy as np


class MCDropoutRegression(Model):
    def __init__(self,
                 num_target=4,
                 img_size: tuple = (128, 128),
                 dropout_rate: float = 0.5,
                 decay: float = 1.0,
                 batch_size: int = 20,
                 length_scale: int = 100):
        super().__init__()

        self.length_scale = length_scale
        self.decay = decay
        self.tau = np.divide((1 - dropout_rate) * (self.length_scale ** 2), 2 * batch_size * self.decay)

        self.preprocess_resize = Resizing(*img_size, crop_to_aspect_ratio=True)
        self.conv_1 = Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1), kernel_regularizer=L2(self.tau))
        self.dropout_1 = Dropout(dropout_rate)
        self.pool_1 = MaxPooling2D((5, 5))
        self.conv_2 = Conv2D(16, 3, activation='relu', kernel_regularizer=L2(self.tau))
        self.dropout_2 = Dropout(dropout_rate)
        self.pool_2 = MaxPooling2D((3, 3))
        self.flatten_1 = Flatten()
        self.dense3 = Dense(num_target, kernel_regularizer=L2(self.tau))
        self.num_target = num_target

    def call(self, inputs, training=None, mask=None):
        x = self.preprocess_resize(inputs)
        x = self.conv_1(x)
        x = self.dropout_1(x, training=True)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.dropout_2(x, training=True)
        x = self.pool_2(x)
        x = self.flatten_1(x)

        return self.dense3(x)


def predict_mean_var(model: MCDropoutRegression, num_passes: int, input_data):
    """
    Return predictions with mean and variance
    :return:
    """
    T_pred_vals = np.array([model.predict(input_data, verbose=0) for _ in range(num_passes)])
    pred_mean = np.mean(T_pred_vals, axis=0)
    pred_var = np.var(T_pred_vals, axis=0)
    pred_var += model.tau ** -1

    return pred_mean, pred_var
