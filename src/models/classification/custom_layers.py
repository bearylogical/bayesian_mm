from tensorflow.keras.layers import Layer
import tensorflow as tf

# modified implementation of SegNet maxpooling argmax layer
# https://github.com/ykamikawa/tf-keras-SegNet
class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding="same", **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides

        ksize = [1, pool_size[0], pool_size[1], 1]
        padding = padding.upper()
        strides = [1, strides[0], strides[1], 1]
        output, argmax = tf.nn.max_pool_with_argmax(
            inputs, ksize=ksize, strides=strides, padding=padding
        )

        argmax = tf.cast(argmax, tf.float32)
        return [output, argmax]


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, **kwargs):
        output_shape = kwargs.get("output_shape", None)
        updates, mask = inputs[0], inputs[1]
        mask = tf.cast(mask, "int32")

        with tf.compat.v1.variable_scope(self.name):

            input_shape = tf.shape(updates, out_type="int32")
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                    input_shape[0], # batch
                    input_shape[1] * self.size[0], # y
                    input_shape[2] * self.size[1], # x
                    input_shape[3], # c
                )
            # 1. In tensorflow the indices in argmax are flattened,
            # so that a maximum value at position [b, y, x, c]
            # becomes flattened index ((b * height + y) * width + x) * channels + c
            # calculation indices for batch (b) , height, width and feature maps
            flat_input_size = tf.reduce_prod(input_shape)
            flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

            one_like_mask = tf.ones_like(mask, dtype="int32")
            batch_shape = [input_shape[0], 1, 1, 1]
            batch_range = tf.reshape(
                tf.range(output_shape[0], dtype="int32"), shape=batch_shape
            )
            b = one_like_mask * batch_range # the batch only has ones (max)
            b_ = tf.reshape(b, [flat_input_size, 1])

            # transpose indices & reshape update values to one dimension
            ind_ = tf.reshape(mask, [flat_input_size, 1])
            ind_ = tf.concat([b_, ind_], 1)

            values = tf.reshape(updates, [flat_input_size])
            # broadcast the updates to the indices and match the output shape
            ret = tf.scatter_nd(ind_, values, flat_output_shape)

            return tf.reshape(ret, output_shape)

if __name__ == "__main__":
    import tensorflow.keras as keras
    import numpy as np
    input = keras.layers.Input((4, 4, 3))
    (e, ma) = MaxPoolingWithArgmax2D()(input)
    o2 = MaxUnpooling2D()([e, ma])
    model = keras.Model(inputs=input, outputs=o2)  # outputs=o
    model.compile(optimizer="adam", loss='categorical_crossentropy')
    model.summary()
    x = np.random.randint(0, 100, (1, 4, 4, 3))
    m = model.predict(x)
    print('jere')