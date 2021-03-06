from keras import Model, Input
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Resizing


class ImageRegressionModel(Model):
    def __init__(self, num_target=4, img_size: tuple = (128, 128)):
        """
        Base Model for image regression

        :param num_target: Number of target features as an output 1D vector
        :param img_size: Height x Width of input image.
        """
        super().__init__()
        self.preprocess_resize = Resizing(*img_size, crop_to_aspect_ratio=True)
        self.conv_1 = Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1))
        self.pool_1 = MaxPooling2D((3, 3))
        self.conv_2 = Conv2D(16, 3, activation='relu')
        self.pool_2 = MaxPooling2D((3, 3))
        self.flatten_1 = Flatten()
        self.dense3 = Dense(num_target)
        self.num_target = num_target

    def call(self, inputs, training=None, mask=None):
        x = self.preprocess_resize(inputs)
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.flatten_1(x)

        return self.dense3(x)


if __name__ == "__main__":
    input = Input((128, 128, 1))
    imgress = ImageRegressionModel()
    imgress.build(input_shape=(None, 128, 128, 1))
    imgress.call(input)
    imgress.summary()

    import random
    from src.utils.loader import get_image_paths_from_dir, RegressionDataLoader

    num_train = 800

    img_data_dir = "../../../dataset/20220209/images"
    target_data_path = "../../../dataset/20220209/images/targets.npy"

    img_paths = get_image_paths_from_dir(img_data_dir)
    random.Random(1337).shuffle(img_paths)

    batch_size = 10
    train_img_paths = img_paths[:num_train]
    test_img_paths = img_paths[num_train:]

    gen_kwargs = dict(target_data_path=target_data_path, num_targets=4, batch_size=batch_size, img_size=(128, 128))

    train_gen = RegressionDataLoader(input_img_paths=train_img_paths, **gen_kwargs)
    test_gen = RegressionDataLoader(input_img_paths=test_img_paths, **gen_kwargs)

    # train_sample
