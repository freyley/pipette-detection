import os, gc, argparse
import numpy as np
from training_data import TrainingData, Normalizer

parser = argparse.ArgumentParser()
parser.add_argument('--training-data', type=str, help="path to training data")
parser.add_argument('--save-model', type=str, default=None, help="path to save model")
parser.add_argument('--save-weights', type=str, default=None, help="path to save weights")
parser.add_argument('--save-history', type=str, default=None, help="path to save training history")
parser.add_argument('--load-model', type=str, default=None, help="path to load model")
parser.add_argument('--load-weights', type=str, default=None, help="path to load weights")
parser.add_argument('--model-type', type=str, default='ResNet101V2', help="Base model type (ResNet101V2, ResNet50, VGG16, VGG19, ...)")
parser.add_argument('--learning-rate', type=float, default=None, help='learning rate (0.001 is default)')
parser.add_argument('--train-depth', type=float, default=1.0, help='base model depth at which to begin training (0.0 means train all; 1.0 means train none)')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
tf.config.list_physical_devices('GPU')

norm = Normalizer(range=[[-100, 0, 0], [100, 500, 500]])
all_data = TrainingData(args.training_data, output_norm=norm.normalize)
training_data, test_data = all_data.split([0.995, 0.005])
print(f"Loaded {len(training_data)} training examples and {len(test_data)} test examples")
test_images, test_pos = test_data[:]

batch_size = args.batch_size
input_shape = training_data[0][0].shape

def below_threshold(y_true, y_pred, threshold=0.1):
    kb = keras.backend
    diff = kb.abs(y_true - y_pred)
    return kb.mean(kb.cast(kb.all(diff < threshold, axis=-1), 'float32'))
    

class PeriodicValidation(keras.callbacks.Callback):
    def __init__(self, n_iter, history_file):
        self._n_iter = n_iter
        self.history = []
        self.history_file = history_file
        if history_file is not None:
            with open(history_file, 'w') as fh:
                fh.write('batch,train_mse,train_bt,val_mse,val_bt\n')
        keras.callbacks.Callback.__init__(self)
        
    def on_train_batch_end(self, batch, logs=None):
        if batch % self._n_iter == 0:
            train_images, train_pos = training_data[batch*batch_size:(batch+1)*batch_size]

            pred_pos = model.predict(train_images, verbose=0)
            train_mse = np.mean(np.square(train_pos - pred_pos))
            train_bt = below_threshold(train_pos, pred_pos)

            pred_pos = model.predict(test_images, verbose=0)
            val_mse = np.mean(np.square(test_pos - pred_pos))
            val_bt = below_threshold(test_pos, pred_pos)

            print(f"\nBatch {batch}\n  Train MSE: {train_mse}  below threshold: {train_bt}\n  Validation MSE: {val_mse}  below threshold: {val_bt}")

            if self.history_file is not None:
                with open(self.history_file, 'a') as fh:
                    fh.write(f"{batch},{train_mse},{train_bt},{val_mse},{val_bt}\n")

            gc.collect() 
            keras.backend.clear_session()


class PeriodicModelSave(keras.callbacks.Callback):
    def __init__(self, n_iter, filename):
        self._n_iter = n_iter
        self.filename = filename
        self._current_iter = 0
        keras.callbacks.Callback.__init__(self)
        
    def on_train_batch_end(self, batch, logs=None):
        if self._current_iter % self._n_iter == 0:
            print(f"saving weigts to {self.filename}")
            model.save_weights(self.filename)
        self._current_iter += 1


if args.load_model is not None:
    model = keras.models.load_model(args.load_model)
else:
    base_model = getattr(keras.applications, args.model_type)(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = keras.models.Sequential([
        keras.Input(shape=input_shape),
        base_model,
    #     keras.layers.Flatten(),
        keras.layers.GlobalAveragePooling2D(),
    #     keras.layers.Dense(512, activation='relu'),
    #     keras.layers.Dense(256, activation='relu'),
    #     keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(3),
    ])

    for i,layer in enumerate(base_model.layers):
        layer.trainable = i > len(base_model.layers) * args.train_depth


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate), 
    loss='mse',
    metrics=[below_threshold],
)

callbacks = [PeriodicValidation(100, args.save_history)]
if args.save_weights is not None:
    callbacks.append(PeriodicModelSave(1000, args.save_weights))

if args.load_weights is not None:
    model.load_weights(args.load_weights)


model.fit(
    training_data.generator(batch_size=batch_size), 
    steps_per_epoch=len(training_data)//batch_size, 
    epochs=args.epochs, 
    batch_size=batch_size,
    callbacks=callbacks,
)

if args.save_model is not None:
    print(f"saving model to {args.save_model}...")
    model.save(args.save_model)
    print("  done.")
