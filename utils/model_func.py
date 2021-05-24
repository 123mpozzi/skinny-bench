import os

import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

import cv2

from .models import Model
from .losses import dice_loss
from .metrics import f1, iou, precision, recall
from .callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from .Trainer import Trainer
from .DataLoader import DataLoader
from .Preprocessor import Preprocessor
from ..main import max_epochs, initial_lr, preprocessor


def train_function(model: Model, batch_size: int, dataset_dir: str) -> None:
    data_loader = DataLoader(dataset_dir=dataset_dir, batch_size=batch_size, preprocessor=preprocessor)
    trainer = Trainer(data_loader=data_loader, model=model, evaluate_test_data=True)
    trainer.add_losses([K.binary_crossentropy, dice_loss])
    trainer.add_metrics([
        f1,
        iou,
        precision,
        recall
    ])
    trainer.add_callbacks([
        ModelCheckpoint(filepath=model.get_logdir(), verbose=1, save_best_only=True,
                                  monitor='val_f1', mode='max', save_weight_only=False),#save_weights_only=True),
        ReduceLROnPlateau(monitor='val_f1', factor=0.5, verbose=1, mode='max', min_lr=1e-6, patience=5),
        #EarlyStopping(monitor='val_f1', mode='max', patience=10, verbose=1),
        EarlyStopping(monitor='val_f1', mode='max', patience=50, verbose=1), # for dark
        TensorBoard(log_dir=model.get_logdir(), histogram_freq=5)
    ])

    trainer.train(max_epochs, tf.keras.optimizers.Adam(learning_rate=initial_lr), verbose=1)

def test_function(model: Model, dataset_dir: str) -> None:
    data_loader = DataLoader(dataset_dir=dataset_dir, batch_size=1, preprocessor=preprocessor)
    trainer = Trainer(data_loader=data_loader, model=model, evaluate_test_data=True)
    trainer.add_losses([K.binary_crossentropy, dice_loss])
    trainer.add_metrics([
        f1,
        iou,
        precision,
        recall
    ])

    model.get_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    model.keras_model.compile(optimizer=optimizer, loss=trainer.combined_loss(), metrics=trainer.metrics)
    evaluation_metrics = model.keras_model.evaluate(data_loader.test_dataset, verbose=1)
    evaluation_metrics = dict(zip(model.keras_model.metrics_names, evaluation_metrics))
    print(evaluation_metrics)

# save Skinny X preprocessed-images (512**2) to files
# save the images after the preprocessing, before they enter the model
def save_x(model: Model, dataset_dir: str, preprocessor: Preprocessor, out_dir: str = 'x', skip = 0) -> None:
    data_loader = DataLoader(dataset_dir=dataset_dir, batch_size=1, preprocessor=preprocessor)

    os.makedirs(out_dir, exist_ok = True)

    i = skip
    for entry in data_loader.test_dataset:
        i += 1

        # dict: {'feature': data, dtype=float32}
        entry = entry[0]
        # shape=(1, 320, 384, 1) dtype=float32
        entry = entry['feature']
        # reshape(320, 384, 3) and de-preprocess
        entry = entry[0]*255
        # convert to numpy array or cv2.imwrite doesn't work
        entry = np.array(entry)

        # cv2 works with BGR, not RGB
        entry = cv2.cvtColor(entry, cv2.COLOR_RGB2BGR)

        # save to a file
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(out_dir, filename), entry)
        #tf.keras.preprocessing.image.save_img(os.path.join(out_dir, filename), entry)
        # https://stackoverflow.com/a/61041738

# save Skinny Y preprocessed-images (512**2) to files
# save the images after the preprocessing, before they enter the model
def save_y(model: Model, dataset_dir: str, preprocessor: Preprocessor, out_dir: str = 'y', skip = 0) -> None:
    data_loader = DataLoader(dataset_dir=dataset_dir, batch_size=1, preprocessor=preprocessor)

    os.makedirs(out_dir, exist_ok = True)

    i = skip
    for entry in data_loader.test_dataset:
        i += 1

        # dict: {'label': data, dtype=float32}
        entry = entry[1]
        # shape=(1, 320, 384, 1) dtype=float32
        entry = entry['label']
        # reshape(320, 384, 1) and de-preprocess
        entry = entry[0]*255
        # convert to numpy array or cv2.imwrite doesn't work
        entry = np.array(entry)

        # save to a file
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(out_dir, filename), entry)

def predict_function(model: Model, dataset_dir: str, out_dir: str, skip = 0) -> None:
    data_loader = DataLoader(dataset_dir=dataset_dir, batch_size=1, preprocessor=preprocessor)

    model.get_model()

    os.makedirs(out_dir, exist_ok = True)

    i = skip
    for entry in data_loader.test_dataset: # prova a predirre le immagini di test
        i += 1

        # dict: {'feature': data, 'types': float32}
        entry = entry[0]

        # convert to tensor to prevent memory leak https://stackoverflow.com/a/64765018
        tensor = tf.convert_to_tensor(entry['feature'], dtype=tf.float32)
        #pred = model.keras_model.predict(tensor) # predict from feature image (X)
        pred = model.keras_model(tensor) # predict from feature image (X)
        #print(pred)

        pred = pred[0]*255 # reshape and de-preprocess
        #print(pred)
        pred = pred.numpy()

        # save to a file
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(out_dir, filename), pred)