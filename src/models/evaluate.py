from keras import Model, models
from tensorflow.keras.utils import Sequence
from PIL import Image
import numpy as np
from src.utils.loader import KeyPointDataLoader, match_image_to_target, rescale_kps_from_pct
import wandb
from keras.callbacks import Callback

from src.utils.viewer import show_image_coords


class LogImagePredictionPerNumEpochs(Callback):
    def __init__(self, data_loader, predict_every=5):
        super().__init__()
        self.data_loader = data_loader
        self.predict_every = predict_every
        self.predictions_table = None
        self.num_images_per_batch = 20
        self._create_predictions_table()

    def _create_predictions_table(self):
        columns = ["id", "epoch", "img_num", "image", "predictions", "truth", "loss", "run_id"]
        self.predictions_table = wandb.Table(columns=columns)

    def _log_predictions(self, data_loader: Sequence, epoch):
        predictions = self.model.predict(data_loader)
        img_idx_per_batch = 3
        for batch_idx, data in enumerate(data_loader):
            img_id = str(batch_idx) + "_" + str(epoch)
            img_num = batch_idx * len(data[0]) + img_idx_per_batch
            prediction = predictions[batch_idx * len(data[0]) + img_idx_per_batch]
            ground_truth = data[1][img_idx_per_batch]

            # dataloader normalizes intensities between 0 and 1
            _img_arr = data[0][img_idx_per_batch].squeeze() * 255.0
            scaled_prediction = np.array(rescale_kps_from_pct(_img_arr.shape, list(prediction)))
            scaled_ground_truth = np.array(rescale_kps_from_pct(_img_arr.shape, list(ground_truth)))
            # wandb specific code
            _original_img = Image.fromarray(_img_arr)
            _wandb_original_image = wandb.Image(_original_img.convert("RGB"))
            image_kwargs = {
                "radius": 1,
                "xy_offset": (5, -2)
            }
            _wandb_ground_truth = wandb.Image(
                show_image_coords(_img_arr, true_coords=scaled_ground_truth, **image_kwargs))
            _wandb_predicted_image = wandb.Image(
                show_image_coords(_img_arr, pred_coords=scaled_prediction, **image_kwargs))
            _loss = np.linalg.norm(data[1][img_idx_per_batch] - prediction, axis=0)

            self.predictions_table.add_data(img_id, epoch, img_num, _wandb_original_image, _wandb_predicted_image,
                                            _wandb_ground_truth,
                                            _loss, wandb.run.id)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.predict_every == 0:
            self._log_predictions(self.data_loader, epoch)

    def on_train_end(self, logs=None):
        wandb.log({"test_predictions": self.predictions_table})


def evaluate(model: Model, test_loader: KeyPointDataLoader):
    """
    ## Evaluate the trained model
    """

    loss, accuracy = model.evaluate(test_loader)
    highest_losses, hardest_examples, true_labels, predictions = get_hardest_k_examples(model, test_loader)

    return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions


def get_hardest_k_examples(model,
                           test_loader: KeyPointDataLoader, k=10):
    # losses = np.zeros(len(test_loader))

    predictions = model.predict(test_loader, verbose=0)
    truths = np.array([test_loader[idx][1][0] for idx in range(len(test_loader))])
    losses = np.linalg.norm(predictions - truths, axis=1)
    argsort_loss = np.argsort(losses)
    top_k_loss = argsort_loss[-k:]
    highest_k_losses = losses[top_k_loss]

    hardest_k_examples = [test_loader[top_idx][0] for top_idx in top_k_loss]
    true_labels = [test_loader[top_idx][1] for top_idx in top_k_loss]
    predictions = [model.predict(example) for example in hardest_k_examples]
    return highest_k_losses, hardest_k_examples, true_labels, predictions


if __name__ == "__main__":
    img_dir = "dataset/experiments/15Apr"
    imgs, labels = match_image_to_target(img_dir, target_fmt=[".json"])

    model_path = "models/Baseline_20220417_1725"
    img_model = models.load_model(model_path)
    val_loader = KeyPointDataLoader(input_img_paths=imgs, target_paths=labels, batch_size=1, img_size=(128, 128))
    get_hardest_k_examples(img_model, val_loader)
