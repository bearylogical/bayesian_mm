from tqdm import tqdm
from keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    MaxPooling2D,
    Resizing,
    Dropout,
    Concatenate,
)
from keras.models import Model
from keras.regularizers import L2
import numpy as np

from src.utils.constants import NUM_TARGETS
from keras import backend as K


def heteroscedastic_loss(y_true, y_pred):
    mean = y_pred[:, :NUM_TARGETS]
    log_var = y_pred[:, NUM_TARGETS:]
    precision = K.exp(-log_var)
    return K.sum(precision * (y_true - mean) ** 2.0 + log_var, axis=-1)


class MCHomoscedasticDropoutRegression(Model):
    def __init__(
        self,
        num_target=NUM_TARGETS,
        img_size: tuple = (224, 224),
        dropout_rate: float = 0.5,
        decay: float = 1.0,
        batch_size: int = 80,
        length_scale: int = 100,
    ):
        super().__init__()

        self.length_scale = length_scale
        self.decay = decay
        self.tau = np.divide(
            (1 - dropout_rate) * (self.length_scale ** 2), 2 * batch_size * self.decay
        )

        self.preprocess_resize = Resizing(*img_size, crop_to_aspect_ratio=True)
        self.conv_1 = Conv2D(
            32,
            7,
            activation="relu",
            input_shape=(*img_size, 1),
            kernel_regularizer=L2(1e-4),
            padding="same",
        )
        self.conv_2 = Conv2D(
            32, 7, activation="relu", kernel_regularizer=L2(1e-4), padding="same"
        )
        self.dropout_1 = Dropout(dropout_rate)
        self.pool_1 = MaxPooling2D(3, padding="same")
        self.conv_3 = Conv2D(
            16, 3, activation="relu", kernel_regularizer=L2(1e-4), padding="same"
        )
        self.conv_4 = Conv2D(
            16, 3, activation="relu", kernel_regularizer=L2(1e-4), padding="same"
        )
        self.dropout_2 = Dropout(dropout_rate)
        self.pool_2 = MaxPooling2D(3, padding="same")
        self.flatten_1 = Flatten()
        self.dense3 = Dense(num_target, kernel_regularizer=L2(1e-4))
        self.num_target = num_target

    def call(self, inputs, training=None, mask=None):
        x = self.preprocess_resize(inputs)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.dropout_1(x, training=True)
        x = self.pool_1(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.dropout_2(x, training=True)
        x = self.pool_2(x)
        x = self.flatten_1(x)

        return self.dense3(x)


class MCHeteroscedasticDropoutRegression(Model):
    def __init__(
        self,
        num_target=NUM_TARGETS,
        img_size: tuple = (224, 224),
        dropout_rate: float = 0.5,
        decay: float = 1.0,
        batch_size: int = 20,
        length_scale: int = 100,
    ):
        super().__init__()

        self.length_scale = length_scale
        self.decay = decay
        self.tau = np.divide(
            (1 - dropout_rate) * (self.length_scale ** 2), 2 * batch_size * self.decay
        )

        self.preprocess_resize = Resizing(*img_size, crop_to_aspect_ratio=True)
        self.conv_1 = Conv2D(
            32,
            7,
            activation="relu",
            input_shape=(*img_size, 1),
            kernel_regularizer=L2(1e-4),
            padding="same",
        )
        self.conv_2 = Conv2D(
            32, 7, activation="relu", kernel_regularizer=L2(1e-4), padding="same"
        )
        self.dropout_1 = Dropout(dropout_rate)
        self.pool_1 = MaxPooling2D(3, padding="same")
        self.conv_3 = Conv2D(
            16, 3, activation="relu", kernel_regularizer=L2(1e-4), padding="same"
        )
        self.conv_4 = Conv2D(
            16, 3, activation="relu", kernel_regularizer=L2(1e-4), padding="same"
        )
        self.dropout_2 = Dropout(dropout_rate)
        self.pool_2 = MaxPooling2D(3, padding="same")
        self.flatten_1 = Flatten()
        self.mean = Dense(num_target, name="mean")
        self.log_var = Dense(num_target, name="log_var")
        self.concat = Concatenate(name="output")
        self.num_target = num_target

    def call(self, inputs, training=None, mask=None):
        x = self.preprocess_resize(inputs)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.dropout_1(x, training=True)
        x = self.pool_1(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.dropout_2(x, training=True)
        x = self.pool_2(x)
        x = self.flatten_1(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        output = self.concat([mean, log_var])
        return output


def get_epistemic_uncertainty(
    model: MCHomoscedasticDropoutRegression, input_data, num_passes: int = 50,
):
    """
    Return predictions with mean and variance
    :return:
    """
    T_pred_vals = np.array(
        [model.predict(input_data, verbose=0) for _ in range(num_passes)]
    )
    pred_mean = np.mean(T_pred_vals, axis=0)
    pred_var = np.var(T_pred_vals, axis=0)
    pred_var += model.tau ** -1

    return pred_mean, pred_var


def get_uncertainties(
    model: MCHeteroscedasticDropoutRegression,
    input_data,
    num_passes: int = 50,
    num_targets: int = NUM_TARGETS,
):
    """
    Return predictions with mean and variance
    :return:
    """
    MC_samples = np.array(
        [model.predict(input_data, verbose=0) for _ in tqdm(num_passes)]
    )
    means = MC_samples[:, :, :num_targets]  # K x N
    epistemic_uncertainty = np.mean(np.var(means, axis=0), axis=-1)
    logvar = np.mean(MC_samples[:, :, num_targets:], axis=-1)
    aleatoric_uncertainty = np.exp(logvar).mean(axis=-1)  # log variance
    pred_mean = np.mean(means, axis=0)
    pred_var = epistemic_uncertainty + aleatoric_uncertainty

    return pred_mean, pred_var, epistemic_uncertainty, aleatoric_uncertainty