import os
import time

import cv2
import tensorflow as tf

from .bench_utils import hash_dir
from .data_org import read_csv, split_csv_fields

# Custom loader used for predictions

def my_batch(dataset: tf.data.Dataset, batch_size) -> tf.data.Dataset:
    return dataset.\
        padded_batch(batch_size, padded_shapes=({'feature': [None, None, 3]})).\
        prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def my_file(path: str) -> list:
    files = [path]
    return files

def my_loader(path, preprocessor) -> tf.data.Dataset:
    def my_process(example):
        return {'feature': tf.io.decode_image(tf.io.read_file(example), channels=3, expand_animations = False)}

    #dataset = [(path),]
    dataset = my_file(path)
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.map(my_process)

    t_start = time.time()
    # preprocessing
    dataset = preprocessor.add_to_graph(dataset)
    t_elapsed = time.time() - t_start

    dataset = dataset.cache()
    return dataset, t_elapsed

def single_predict(model, im_path: str, out_path: str, preprocessor, bench_file):
    os.makedirs(os.path.dirname(out_path), exist_ok = True)

    # get tf Dataset structure containing the image to predict and elapsed preprocessing time
    tf_ds, t_elapsed_pre = my_loader(im_path, preprocessor)

    # predict the image
    for entry in tf_ds:
        # convert to tensor to prevent memory leak https://stackoverflow.com/a/64765018
        tensor = tf.convert_to_tensor(entry['feature'], dtype=tf.float32)
        tensor = tf.expand_dims(tensor, axis=0) # add a dimension

        # get time before prediction
        t_start = time.time()

        pred = model.keras_model.predict(tensor) # predict from feature image (X)
        # post-processing
        pred = pred[0]*255 # reshape and de-preprocess

        # prediction + postprocessing
        t_elapsed = time.time() - t_start
        # preprocessing + prediction + postprocessing elapsed time
        t_elapsed_full = t_elapsed_pre + t_elapsed

        # save to a file
        cv2.imwrite(out_path, pred)

        with open(bench_file, "a") as myfile:
            myfile.write(f'{im_path},{t_elapsed_full}\n')

def bench_predict(model, csv_file, out_dir: str, preprocessor, bench_file):
    # read the images CSV
    file_content = read_csv(csv_file)

    for entry in file_content:
        csv_fields = split_csv_fields(entry)
        note = csv_fields[2]

        if note == 'te':
            ori_path = csv_fields[0]
            ori_basename = os.path.basename(ori_path)
            ori_filename, ori_ext = os.path.splitext(ori_basename)

            out_path = os.path.join(out_dir, 'p', f'{ori_filename}.png')
            single_predict(model, ori_path, out_path, preprocessor, bench_file)
    
    predictions_hash = hash_dir(out_dir)
    print(predictions_hash)
    return predictions_hash
