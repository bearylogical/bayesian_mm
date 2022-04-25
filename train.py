from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model as load_keras_model
from keras import Model
from src.models.regression.cnn_regression import ImageRegressionModel
from wandb.keras import WandbCallback
from src.models.callbacks import LRLogger
from src.utils.utilities import set_logger
import wandb
from pathlib import Path
import os
from functools import partial
from typing import Union, Tuple
import logging


logger = set_logger() if not logging.getLogger().hasHandlers() else logging.getLogger()


def generate_data(num_samples: int = 10,
                  training_pct: float = 0.8,
                  target_size: Tuple[int, int] = (128, 128),
                  force_gen: bool = False):
    from src.utils.shapes.capillary import CapillaryImageGenerator
    logger.info("Creating Images")
    cap = CapillaryImageGenerator(num_images=num_samples,
                                  train_test_ratio=training_pct,
                                  target_size=target_size,
                                  force_images=force_gen)

    num_train = int(training_pct * num_samples)
    num_test = training_pct - num_train
    _, _, train_files = next(os.walk(cap.train_dir))
    _, _, test_files = next(os.walk(cap.test_dir))
    if len(train_files) > 0 or len(test_files) > 0:
        logger.debug('Test and train images seem to exist, now checking')

    if len(train_files) != num_train + 1 and len(test_files) != num_test + 1:
        logger.debug("Samples length seems to not match, clearing folder and regenerating images")
        [os.remove(cap.train_dir / f) for f in train_files]
        [os.remove(cap.test_dir / f) for f in test_files]
        cap.generate()
        logger.debug(f"{num_samples} images created at {cap.save_img_dir}")
    else:
        logger.info("samples detected. reusing existing saved dir")

    return cap.train_dir, cap.test_dir


def train_test_split(train_img_dir, test_img_dir, data_loader, **kwargs):
    from src.utils.dataloader import get_image_paths_from_dir
    import random
    logger.info("Allocating Data Loaders")

    train_img_paths = get_image_paths_from_dir(train_img_dir)
    test_img_paths = get_image_paths_from_dir(test_img_dir)

    train_data_path = train_img_dir / 'targets.npz'
    test_data_path = test_img_dir / 'targets.npz'
    assert train_data_path.is_file() and test_data_path.is_file()

    train_gen = data_loader(input_img_paths=train_img_paths, target_paths=train_data_path, **kwargs)
    test_gen = data_loader(input_img_paths=test_img_paths, target_paths=test_data_path, transform=None, **kwargs)
    logger.debug("Data Loaders created")
    return train_gen, test_gen


def build_model(is_summary: bool = False, img_size: tuple = (128, 128),
                model: Union[Model, None] = ImageRegressionModel):
    logger.info("Creating Model")
    model_input = Input((128, 128, 1))
    if model is None:
        model = ImageRegressionModel
    imgress = model(num_target=14, img_size=img_size)
    imgress.build(input_shape=(None, 128, 128, 1))
    if is_summary:
        imgress.call(model_input)
        print(imgress.summary())

    return imgress


def load_model(model_path: Union[str, Path]):
    return load_keras_model(model_path)


def train(experiment_name: Union[str, None] = "DefaultProject", task="T1", **kwargs):
    from pathlib import Path
    from time import strftime


    num_samples = kwargs.get('num_samples', 10)
    batch_size = kwargs.get('batch_size', 10)
    training_pct = kwargs.get('training_pct', .8)
    img_size = kwargs.get('img_size', (128, 128))
    epochs = kwargs.get('epochs', 10)
    model = kwargs.get('model', None)
    normalize = kwargs.get('normalize', False)

    train_image_dir, test_image_dir = generate_data(num_samples, training_pct, target_size=img_size)

    if task == "T1":
        NUM_TARGETS = 14
    else:
        NUM_TARGETS = 4

    gen_kwargs = dict(num_targets=NUM_TARGETS,
                      batch_size=batch_size, img_size=img_size,
                      task=task, normalize=normalize)
    train_gen, test_gen = train_test_split(train_image_dir, test_image_dir,
                                           RegressionDataLoaderT1,
                                           **gen_kwargs)

    # experiment tracking
    logger.debug(f"Setting up wandb instance of {experiment_name}")
    wandb.init(project=experiment_name, entity="syamilm")

    initial_learning_rate = 0.001
    lr_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_steps=100,
        decay_rate=0.96,
        staircase=True)
    optimizer = Adam(learning_rate=lr_schedule)

    wandb.config.update({
        "lr": initial_learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "img_size": img_size,
        "normalize": normalize
    })

    # retrieve model
    imgress = build_model(is_summary=True, img_size=img_size, model=model)
    logger.info("Compiling model for training")
    imgress.compile(optimizer=optimizer, loss='mse')
    model_save_path = kwargs.get('model_path', Path.cwd() / 'models')
    model_save_path = Path(model_save_path) / (experiment_name + "_" + strftime("%Y%m%d_%H%M"))

    # checkpoint saving
    save_every = 50
    model_checkpoint_path = model_save_path / 'checkpoint'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=model_checkpoint_path,
        save_freq=save_every,
        save_weights_only=True,
        monitor='val_loss',
        mode='max',
        save_best_only=True)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=5,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    imgress.fit(train_gen, batch_size=batch_size, validation_data=test_gen, epochs=epochs,
                callbacks=[WandbCallback(),  # using WandbCallback to log default metrics.
                           LRLogger(optimizer),
                           model_checkpoint_callback,
                           early_stopping])  # using callback to log learning rate.

    logger.debug(f"Model Saving to {model_save_path}")
    model_save_path.mkdir(parents=True, exist_ok=True)
    imgress.save(model_save_path)

    return imgress


if __name__ == "__main__":
    train("Test_Experiment", num_samples=10, epochs=10)
