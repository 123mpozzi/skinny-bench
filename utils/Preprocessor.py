import tensorflow as tf


class Preprocessor:
    def __init__(self):
        self.operations = [] # Preprocessor has no operations by default
    
    # adds a downscale operation to the preprocessing operations list, returns self
    def downscale(self, max_pixel_count):
        # operation that downscales the images composed of more pixels
        # than max_pixel_count preserving the aspect ratio
        def downscale_operation(data):
            for k, v in data.items():
                tensor_shape = tf.cast(tf.shape(v), tf.float32)
                coefficient = max_pixel_count / (tensor_shape[0] * tensor_shape[1])
                coefficient = tf.math.sqrt(coefficient)
                data[k] = tf.cond(coefficient >= 1.0, lambda: v,
                                  lambda: tf.image.resize(v, [tf.cast(tensor_shape[0] * coefficient, tf.uint16),
                                                              tf.cast(tensor_shape[1] * coefficient, tf.uint16)]))
            return data

        self.operations.append(downscale_operation)
        return self
    
    # adds a cast operation to the preprocessing operations list, returns self
    def cast(self, dtype):
        # operation that casts the images data into the given dtype
        def cast_operation(data):
            for k, v in data.items():
                data[k] = tf.cast(v, dtype)
            return data

        self.operations.append(cast_operation)
        return self
    
    # adds a normalize operation to the preprocessing operations list, returns self
    def normalize(self):
        # operation that transforms the images data from uint8(0-255 limited) into floats(0-1 limited)
        def normalize_operation(data):
            for k, v in data.items():
                data[k] = v / 255.0
            return data

        self.operations.append(normalize_operation)
        return self
    
    # adds a padding operation to the preprocessing operations list, returns self
    def pad(self, network_levels):
        number_multiple = 2**(network_levels-1)
        # operation that adds padding to the down and the right of the images
        def padding_operation(data):
            for k, v in data.items():
                tensor_shape = tf.shape(v)
                data[k] = tf.pad(v, [[0, number_multiple - tensor_shape[0] % number_multiple],
                                     [0,  number_multiple - tensor_shape[1] % number_multiple],
                                     [0, 0]])
            return data

        self.operations.append(padding_operation)
        return self
    
    # executes all the operation functions defined in the Preprocessor instance
    # on the given Dataset object and returns the transformed Dataset object
    def add_to_graph(self, dataset) -> tf.data.Dataset:
        for operation in self.operations:
            dataset = dataset.map(operation) # map will execute one function on every element of the Dataset
        return dataset