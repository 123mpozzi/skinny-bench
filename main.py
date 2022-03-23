import os

import tensorflow as tf

from utils.bench_utils import read_performance
from utils.data_org import get_bench_testset, get_timestamp, import_dataset, read_csv, split_csv_fields
from utils.loading_func import load_checkpoint
from utils.models import Skinny
from utils.my_loader import bench_predict, single_predict
from utils.Preprocessor import Preprocessor

## Usage: python main.py


# Skinny (full model with inception and dense blocks)
model_name = 'Skinny'

# model settings
levels = 6
initial_filters = 19
image_channels = 3

# dirs
log_dir = 'logs'
timestr = get_timestamp()
out_bench = os.path.join('predictions', 'bench', timestr)

# preprocessing operations
preprocessor = Preprocessor()
preprocessor.cast(dtype=tf.float32).normalize().downscale(max_pixel_count=512**2).pad(network_levels=levels)

models = {}
models['ecu'] = 'models/checkpoint-20210428-155148/saved_model.ckpt/saved_model.pb'


# Target ECU Dataset
db_dest = os.path.join('dataset', 'ECU')
db_csv = os.path.join(db_dest, 'data.csv')
db_import = os.path.join('dataset', 'import_ecu.json')
# Process dataset and prepare CSV
assert os.path.isdir(db_dest), 'Dataset has no directory: ' + db_dest
assert os.path.isfile(db_import), 'No import JSON found: ' + db_import
if os.path.isfile(db_csv):
    os.remove(db_csv)
import_dataset(db_import)

# Set only the first 14 ECU images as test
get_bench_testset(db_csv, stop=15)

# Load Model files
model_path = models['ecu']
assert os.path.exists(model_path), 'Model not found: ' + model_path
chkp_ext = load_checkpoint(model_path)
mod = Skinny(levels, initial_filters, image_channels, log_dir, load_checkpoint=True,
        model_name=model_name, checkpoint_extension=chkp_ext)
# Load model
mod.get_model()

# First prediction is slow in Keras because it will build the predict function
# https://github.com/tensorflow/tensorflow/issues/39458
# So dump a first prediction
im_dump_row = read_csv(db_csv)[0]
im_dump = split_csv_fields(im_dump_row)[0]
im_dump_out = os.path.join(out_bench, 'dump.png')
bench_dump = os.path.join(out_bench, 'bench_dump.txt')
single_predict(mod, im_dump, im_dump_out, preprocessor, bench_dump)

# Save 5 observations
for i in range(5):
    out_dir = os.path.join(out_bench, 'observation{}').format(i)
    bench_file = os.path.join(out_bench, 'bench{}.txt').format(i)
    bench_predict(mod, db_csv, out_dir, preprocessor, bench_file)

# Print inference times
read_performance(out_bench)
