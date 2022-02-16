from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint
from keras import Model
from src.utils.loader import RegressionDataLoaderT1
from src.models.regression.cnn_regression import ImageRegressionModel
from wandb.keras import WandbCallback
from src.utils.experiment import LRLogger
import wandb
from functools import partial
from typing import Union
import logging

logger = logging.getLogger('bayesian_nn')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
logger.addHandler(ch)


def generate_data(num_samples: int = 10):
    from src.utils.shapes.capillary import CapillaryImageGenerator
    logger.info("Creating Images")
    cap = CapillaryImageGenerator(num_images=num_samples)
    cap.generate()
    logger.debug(f"{num_samples} images created at {cap.save_img_dir}")
    return cap.save_img_dir


def train_test_split(training_pct, img_dir, data_loader, **kwargs):
    from src.utils.loader import get_image_paths_from_dir
    import random
    logger.info("Generating Train Test Split")
    img_paths = get_image_paths_from_dir(img_dir)
    random.Random(1337).shuffle(img_paths)  # shuffle our dataset accordingly
    num_train = int(training_pct * len(img_paths))
    logger.debug(f"{len(img_paths)} Image Samples, {num_train}/{len(img_paths) - num_train} Train/Test")
    train_img_paths = img_paths[:num_train]
    test_img_paths = img_paths[num_train:]

    train_gen = data_loader(input_img_paths=train_img_paths, **kwargs)
    test_gen = data_loader(input_img_paths=test_img_paths, **kwargs)
    logger.debug("Data Loaders created")
    return train_gen, test_gen


def build_model(is_summary: bool = False, img_size: tuple = (128, 128),
                model: Model = ImageRegressionModel):
    logger.info("Creating Model")
    model_input = Input((128, 128, 1))
    imgress = model(14, img_size=img_size)
    imgress.build(input_shape=(None, 128, 128, 1))
    if is_summary:
        imgress.call(model_input)
        print(imgress.summary())

    return imgress


def train(experiment_name: Union[str, None] = "DefaultProject", task="T1", **kwargs):
    from pathlib import Path
    from time import strftime

    num_samples = kwargs.get('num_samples', 10)
    batch_size = 10 if num_samples // 50 < 0 else 50
    training_pct = kwargs.get('training_pct', .8)
    img_size = kwargs.get('img_size', (128, 128))
    epochs = kwargs.get('epochs', 10)
    model = kwargs.get('model', None)

    img_save_dir = generate_data(num_samples)
    data_path = img_save_dir / 'targets.npz'
    assert data_path.is_file()

    if task == "T1":
        NUM_TARGETS = 14
    else:
        NUM_TARGETS = 4

    gen_kwargs = dict(target_paths=data_path, num_targets=NUM_TARGETS,
                      batch_size=batch_size, img_size=img_size,
                      task=task)
    train_gen, test_gen = train_test_split(training_pct, img_save_dir,
                                           partial(RegressionDataLoaderT1, normalize=False),
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

    })

    # retrieve model
    imgress = build_model(is_summary=True, img_size=img_size)
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
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    imgress.fit(train_gen, batch_size=batch_size, validation_data=test_gen, epochs=epochs,
                callbacks=[WandbCallback(),  # using WandbCallback to log default metrics.
                           LRLogger(optimizer),
                           model_checkpoint_callback])  # using callback to log learning rate.

    logger.debug(f"Model Saving to {model_save_path}")
    model_save_path.mkdir(parents=True, exist_ok=True)
    imgress.save(model_save_path)

    return imgress


if __name__ == "__main__":
    train("Test_Experiment")
