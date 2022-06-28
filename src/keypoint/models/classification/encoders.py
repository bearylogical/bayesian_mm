from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50V2

ENCODER_LAYER_KERAS_RESNET50 = {'conv1_conv', 'conv2_block1_preact_bn', 'conv2_block3_out', 'conv4_block6_out',
                                'conv5_block3_out'}


class BackBone(Model):
    def __init__(self):
        super(BackBone, self).__init__()

    def call(self, inputs, training=None, mask=None):
        pass


class ResNet50Encoder:
    def __init__(self):
        self.model = ResNet50V2(weights='imagenet', include_top=False)

    def _get_layers(self):
        pass
