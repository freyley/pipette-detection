from queue import Queue
from typing import Tuple

import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras import Model


from make_training_data import TrainingDataGenerator, PipetteTemplate
from training_data import TrainingData


def configure_model(size: Tuple[int, int]) -> Model:
    input_shape = size + (3,)

    # Load pre-trained ResNet without the top (include_top=False).
    #base_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model = keras.applications.ResNet101V2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the base_model
    # base_model.trainable = False
    # 175 layers
    for i, layer in enumerate(base_model.layers):
        layer.trainable = i > 145

    # Create new model on top
    inputs = keras.Input(shape=input_shape)

    # The base model contains batchnorm layers. We want to keep them in
    # inference mode when we unfreeze the base model for fine-tuning,
    # so we make sure that the base_model is running in inference mode here.
    x = base_model(inputs, training=False)

    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = keras.layers.GlobalAveragePooling2D()(x)

    # A Dense layer to output 2 coordinates (z_um, row, col)
    outputs = keras.layers.Dense(3)(x)

    # Build the model
    model = keras.Model(inputs, outputs)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-5), loss='mse')

    return model


def mah_generator(batch_size: int):
    queue = Queue(maxsize=30)
    training_data_args = {
        'shape': (500, 500),
        'template': PipetteTemplate('yip_2019_template.npz'),
    }
    threads = [TrainingDataGenerator(queue, training_data_args) for _i in range(10)]
    while True:
        images = []
        positions = []
        for _ in range(batch_size):
            image, pos = queue.get()
            image_rgb = np.concatenate([image[..., None]] * 3, axis=2)
            images.append(image_rgb[None, ...])
            positions.append(np.array(pos)[None, ...])
        images = np.concatenate(images, axis=0)
        positions = np.concatenate(positions, axis=0)
        yield images, positions / 500
    # for t in threads:
    #     t.stop()
    #
    # for t in threads:
    #     t.join()



def main():
    model = configure_model((500,500))
    # model = keras.models.load_model('resnet_regression_model_30_layers_100000_images.h5')
    batch_size = 100
    num_steps = 1
    for _ in range(5):
        num_steps *= 10
        # steps_per_epoch is batches per epoch
        with tf.device("/gpu:0"):
            model.fit(mah_generator(batch_size), steps_per_epoch=num_steps, epochs=1, batch_size=batch_size)
        model_name = f'resnet_regression_model_30_layers_resnet101_{num_steps * batch_size}.h5'
        print(model_name)
        model.save(model_name)

# .0371 at 5:16


if __name__ == "__main__":
    main()