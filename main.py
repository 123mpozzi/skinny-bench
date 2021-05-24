import os
import tensorflow as tf
from utils.Preprocessor import Preprocessor
from utils.loading_func import load_checkpoint
from utils.models import Skinny
from utils.data_org import get_timestamp, get_bench_testset
from utils.my_loader import bench_predict


## Usage: python main.py


# Skinny (full model with inception and dense blocks)
model_name = 'Skinny'

# model settings
levels = 6
initial_filters = 19
image_channels = 3

# train settings
max_epochs = 200
initial_lr = 1e-4
batch_size = 3

# dirs
log_dir = 'logs'
# in the dataset_dir there is a csv file containing all the splits data
#dataset_dir = 'dataset/Schmugge' #'dataset/ECU'

# preprocessing operations
preprocessor = Preprocessor()
preprocessor.cast(dtype=tf.float32).normalize().downscale(max_pixel_count=512**2).pad(network_levels=levels)

models = {}
models['ecu'] = 'models/checkpoint-20210428-155148/saved_model.ckpt/saved_model.pb' # ecu
models['schmugge'] = 'models/checkpoint-20210505-225202/saved_model.ckpt/saved_model.pb' # sch
#models['hgr'] = 'drive/MyDrive/training/skinny/checkpoint-20210512-220723/saved_model.ckpt/saved_model.pb' # hgr
#models['dark'] = 'drive/MyDrive/training/skinny/checkpoint-20210523-110554/saved_model.ckpt/saved_model.pb' # dark
#models['medium'] = 'drive/MyDrive/training/skinny/checkpoint-20210523-112308/saved_model.ckpt/saved_model.pb' # medium
#models['light'] = 'drive/MyDrive/training/skinny/checkpoint-20210523-122027/saved_model.ckpt/saved_model.pb' # light


timestr = get_timestamp()

# load a model
m_name = 'ecu'
db_dest = 'dataset/ECU'
db_csv = os.path.join(db_dest, 'data.csv')


# set only the first 15 ECU images as test
get_bench_testset(db_csv, count=15)


# load model files
chkp_ext = load_checkpoint(models[m_name])
mod = Skinny(levels, initial_filters, image_channels, log_dir, load_checkpoint=True,
        model_name=model_name, checkpoint_extension=chkp_ext)


out_preds = 'predictions/bench'

# save 5 observations
for i in range(5):
    bench_file = f'bench{i}.txt'
    bench_predict(mod, db_csv, out_preds, preprocessor, bench_file)

