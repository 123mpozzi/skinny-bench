import os

import tensorflow as tf
from .Preprocessor import Preprocessor
from .data_org import csv_sep


class DataLoader(object):
    buffer_size = tf.data.experimental.AUTOTUNE

    def __init__(self, dataset_dir: str, batch_size: int, preprocessor: Preprocessor = None):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.preprocessor = Preprocessor() if preprocessor is None else preprocessor

    @property
    def train_dataset(self):
        return self.__batch_and_prefetch(self.__train_dataset)

    @train_dataset.setter
    def train_dataset(self, dataset: tf.data.Dataset):
        self.__train_dataset = dataset

    @property
    def test_dataset(self):
        return self.__batch_and_prefetch(self.__test_dataset)

    @test_dataset.setter
    def test_dataset(self, dataset: tf.data.Dataset):
        self.__test_dataset = dataset

    @property
    def val_dataset(self):
        return self.__batch_and_prefetch(self.__val_dataset)

    @val_dataset.setter
    def val_dataset(self, dataset: tf.data.Dataset):
        self.__val_dataset = dataset

    @property
    def preprocessor(self) -> Preprocessor:
        return self.__preprocessor

    @preprocessor.setter
    def preprocessor(self, preprocessor: Preprocessor):
        self.__preprocessor = preprocessor
        self.__reinstantiate()

    def __reinstantiate(self):
        self.train_dataset = self.__create_dataset_pipeline('tr')
        self.val_dataset = self.__create_dataset_pipeline('va', shuffle=False)
        self.test_dataset = self.__create_dataset_pipeline('te', shuffle=False)

    def __batch_and_prefetch(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.\
            padded_batch(self.batch_size, padded_shapes=({'feature': [None, None, 3]}, {'label': [None, None, 1]})).\
            prefetch(buffer_size=self.buffer_size)
    
    def __get_subset_paths(self, subset: str) -> list:
        # read the images CSV (ori_image_filename.ext, gt_image_filename.ext)
        file = open(os.path.join(self.dataset_dir, 'data.csv'))
        file3c = file.read().splitlines()
        file.close()

        files = []

        for entry in file3c:
            ori_path = entry.split(csv_sep)[0]
            gt_path = entry.split(csv_sep)[1]
            note = entry.split(csv_sep)[2]
            
            if note == subset:
                files.append((ori_path, gt_path))
        
        if len(files) == 0:
            print(f'''No files found for subset {subset}!\n
            using the whole dataset instead''')
            for entry in file3c:
                ori_path = entry.split(csv_sep)[0]
                gt_path = entry.split(csv_sep)[1]
                
                files.append((ori_path, gt_path))
        else:
            print(f'Found {subset} split of {len(files)} files')
        
        return files

    def __create_dataset_pipeline(self, subset: str, shuffle: bool = True) -> tf.data.Dataset:
        def process_example_paths(example):
            # TODO bmp should raise errors, tensorflow doesn't like them
            return {'feature': tf.io.decode_image(tf.io.read_file(example[0]), channels=3, expand_animations = False),
                    'label': tf.io.decode_image(tf.io.read_file(example[1]), channels=1,  expand_animations = False)}

        def convert_to_in_out_dicts(example):
            output_dict = {'label': example.pop('label')}
            return example, output_dict

        dataset = self.__get_subset_paths(subset)
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        dataset = dataset.map(process_example_paths)
        dataset = self.preprocessor.add_to_graph(dataset)
        dataset = dataset.map(convert_to_in_out_dicts).cache()
        if shuffle:
            dataset = dataset.shuffle(2000, reshuffle_each_iteration=True)
        return dataset