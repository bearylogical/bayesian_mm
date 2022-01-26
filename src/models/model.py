class BaseModel:
    def __init__(self, img_size: tuple[int, int], channels:int, num_classes:int):
        self.img_size = img_size
        self.channels = channels
        self.num_classes = num_classes
        self.model = None
        self.model_name = ""

    def _get_model(self):
        raise NotImplementedError


if __name__ == "__main__":
    from src.models.unet import UnetModel

    mdl = UnetModel((128,128), 1)
    mdl.model.summary()

