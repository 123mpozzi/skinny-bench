import os
import tensorflow.keras as keras
from .data_org import get_timestamp


class CustomCallback(keras.callbacks.Callback):
    pass


class ModelCheckpoint(keras.callbacks.ModelCheckpoint, CustomCallback):
    checkpoint_name = 'saved_model.ckpt'

    def set_model(self, model):
        dir_name = os.path.join(self.filepath, 'checkpoint')
        os.makedirs(dir_name, exist_ok=True)
        timestr = get_timestamp()
        self.filepath = os.path.join(F"/content/drive/MyDrive/training/skinny/checkpoint-" + timestr, self.checkpoint_name )
        #self.filepath = os.path.join(F"/content/drive/MyDrive/training/skinny/dark/checkpoint-" + timestr, self.checkpoint_name )
        super().set_model(model)


class ReduceLROnPlateau(keras.callbacks.ReduceLROnPlateau, CustomCallback):
    pass


class ProgbarLogger(keras.callbacks.ProgbarLogger, CustomCallback):
    pass


class EarlyStopping(keras.callbacks.EarlyStopping, CustomCallback):
    pass


class TensorBoard(keras.callbacks.TensorBoard, CustomCallback):
    def set_model(self, model):
        self.log_dir = os.path.join(self.log_dir, 'tensorboard')
        os.makedirs(self.log_dir, exist_ok=True)
        super().set_model(model)