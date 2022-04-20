from pathlib import Path
from time import strftime

from sklearn.model_selection import train_test_split
from keras import Input
from tensorflow.keras.optimizers import Adam

from src.utils.loader import KeyPointDataLoader, match_image_to_target
from src.models.regression import cnn_regression


# experiment params
experiment_name = "Initial Test"
img_size = (128, 128)
batch_size=5
epochs = 10

# dirs
img_dir = "dataset/experiments/15Apr"
labels_dir = "dataset/experiments/15Apr/labels"
model_dir = Path.cwd() / 'models' / (experiment_name + "_" + strftime("%Y%m%d_%H%M"))

# match data and labels
imgs, labels = match_image_to_target(img_dir, labels_dir, target_fmt=[".json"])

# train test split
X_train, X_test, y_train, y_test = train_test_split(imgs, labels, train_size=0.8)

# pass to data loaders
train_data = KeyPointDataLoader(input_img_paths=X_train, target_paths=y_train, batch_size=batch_size, img_size=img_size)
test_data = KeyPointDataLoader(input_img_paths=X_test, target_paths=y_test, batch_size=batch_size, img_size=img_size)

# instantiate model
model = cnn_regression.ImageRegressionModel
model_input = Input((*img_size, 1))
img_regress = model(num_target=14, img_size=img_size)
img_regress.build(input_shape=(None, *img_size, 1))

img_regress.call(model_input)
img_regress.summary()
# define optimizer
# initial_learning_rate = 0.001
# optimizer = Adam(learning_rate=initial_learning_rate)
#
# img_regress.compile(optimizer=optimizer, loss='mse')
# img_regress.fit(train_data, batch_size=batch_size, validation_data=test_data, epochs=epochs)


# pass training parameters



# being training


