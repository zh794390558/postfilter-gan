from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import os
import math
from pprint import pprint
import tensorflow as tf

import utils


# Constants
MIN_FRACTION_OF_EXMPLES_IN_QUEUE = 0.4
MAX_ABSOLUTE_EXAMLES_IN_QUEUE =4096 # The queue size connot exceed this number
NUM_THREADS_DATA_LOADER=6
LOG_MEAN_FILE = False  # Logs the mean file as loaded in TF to TB


# Supported extensions for Loaders
DB_EXTENSIONS = {
        'tfrecords': ['.TFRECORDS'],
}

LIST_DELIMTIER = ' ' # For the FILELLIST format

logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] [%(levelname)s] %(message)s',
                    datafmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)


def get_backend_of_source(db_path):
        """
        Takes a path as argument and infers the format of the data.
        If a directory is provided, it looks for the existance of an extension
        in the entire directory in an order of a priority of dbs (hdf5, lmdb, filelist, file)
        Args:
                db_path: path to a file or directory
        Returns:
                backend: the backend type
        """
        # If a directory is given, we include all its contents. Otherwise it's just the one file.
        if os.path.isdir(db_path):
                files_in_path = [fn for fn in os.listdir(db_path) if not fn.startswith('.')]
        else:
                files_in_path = [db_path]

        # Keep the below priority ordering
        for db_fmt in ['tfrecords']:
                ext_list = DB_EXTENSIONS[db_fmt]
                for ext in ext_list:
                        if any(ext in os.path.splitext(fn)[1].upper() for fn in files_in_path):
                                return db_fmt

        logging.error('Cannot infer backend from db_path ({})'.format(db_path))
        exit(-1)


class LoaderFactory(object):
        """
        A factory for data loading. It sets up a subclass with data loading
        done with the respective backend. Its output is a tensorflow queue op
        that is used to load in data, with optionally some minor postprocessing ops.
        """
        def __init__(self):
                self.corplen = None
                self.nclasses = None
                self.mean_loader = None

                self.backend = None
                self.db_path = None
                self.batch_x = None
                self.batch_y = None
                self.batch_k = None
                self.stage = None
                self._seed = None
                self.unencoded_data_format = 'hwc'
                self.uneconded_channel_scheme = 'rgb'
                self.summaries = None
                self.aug_dict = {}

        @staticmethod
        def set_source(db_path, is_inference=False):
                """
                Returns the correct backend.
                """
                backend = get_backend_of_source(db_path)
                loader = None

                if backend == 'tfrecords':
                        loader = TFRecordsLoader()
                else:
                        logging.error('Backend ({}) not implemented'.format(backend))
                        exit(-1)

                loader.backend = backend
                loader.db_path = db_path
                loader.is_inference = is_inference
                return loader

        def setup(self, labels_db_path, shuffle, bitdepth, batch_size, num_epochs=None, seed=None):
                with tf.device('/cpu:0'):
                        self.labels_db_path = labels_db_path

                        self.bitdepth = bitdepth
                        self.shuffle = shuffle
                        self.batch_size = batch_size
                        self.num_epochs = num_epochs
                        self._seed = seed

                        if self.labels_db_path:
                                self.labels_db = LoadFactory.set_source(self.labels_db_path)
                                self.lables_db.bitdepth = self.bitdepth
                                self.labels_db.stage = self.stage
                                self.labels_db.initialize()

                        self.initialize()
                        logging.info('Found {} images in db ({})'.format(self.get_total(), self.db_path))

        def get_key_index(self, key):
                return self.keys.index(key)

        def set_augmentation(self, mean_loader, aug_dict={}):
                with tf.device('/cpu:0'):
                        self.mean_loader = mean_loder
                        self.aug_dict = aug_dict

        def get_shape(self):
                input_shape = [self.height, self.width, self.channels]
                # update input_shape if crop length specified
                # this is necessary as the input_shape is provided
                # below to the user-defined function that defines the network
                if self.croplen > 0:
                        input_shape[0] = self.croplen
                        input_shape[1] = self.croplen

                return input_shape

        def get_total(self):
                return self.total

        def reshape_decode(self, data, shape):
                if self.float_data:
                        data = tf.reshape(data, shape)
                        data = utils.chw_to_hwc(data)
                else:
                        # Decode image of any time option might come: https://github.com/tensorflow/tensorflow/issues/4009
                        # Distinguish between mime types
                        if self.data_encoded:
                                if self.data_mime == 'image/png':
                                        data = tf.image.decode_png(data, dtype=self.image_dtype, name='image_decoder')
                                elif self.data_mime == 'image/jpeg':
                                        data = tf.image.decode_jpeg(data, name='image_decoder')
                                else:
                                        logging.error('Unsupported mime type (%s); cannot be decoded' % (self.data_mime))
                                        exit(-1)
                        else:
                                if self.backend == 'lmdb':
                                        data = tf.decode_raw(data, self.image_dtype, name='raw_decoder')

                                # if data is in CHW, set the shape and convert to HWC
                                if self.unencoded_data_format == 'chw':
                                        data = tf.reshape(data, [shape[0], shape[1], shape[2]])
                                        data = digits.chw_to_hwc(data)
                                else:  # 'hwc'
                                        data = tf.reshape(data, shape)

                                if (self.channels == 3) and self.unencoded_channel_scheme == 'bgr':
                                        data = digits.bgr_to_rgb(data)

                        # Convert to float
                        # data = tf.to_float(data)
                        # data = tf.image.convert_image_dtype(data, tf.float32) # normalize to [0:1) range
                return data


        def create_input_pipline(self):
                """
                This function returns part of the graph that does data loading, and
                includes a queueing, optional data decoding and optional post-processing
                like data augmentation or mean subtraction.
                Args:
                    None.
                Produces:
                    batch_x: Input data batch
                    batch_y: Label data batch
                    batch_k: A list of keys (strings) from which the batch originated
                Returns:
                    None.
                """
                key_queue = self.get_queue()

                single_label = None
                single_label_shape = None
                if self.stage == utils.STAGE_INF:
                        single_key, single_data, single_data_shape, _, _ = self.get_single_data(key_queue)
                else:
                        single_key , single_data, single_data_shape, single_label, single_label_shape = \
                                self.get_single_data(key_queue)

                logging.debug('single shape {}'.format(single_data_shape))
                single_data_shape = np.reshape(single_data_shape, [3]) # Shape the shape to have three dimensions
                single_data = self.reshape_decode(single_data, single_data_shape)
                logging.debug('single data shape {} ({})'.format(single_data.get_shape(), single_data_shape))


                if self.labels_db_path:  # Using a seperate label db; label can be anything
                        single_label_shape = tf.reshape(single_label_shape, [3])  # Shape the shape
                        single_label = self.labels_db.reshape_decode(single_label, single_label_shape)
                elif single_label is not None:  # Not using a seperate label db; label is scalar #
                        # single_label = tf.reshape(single_label, [])

                        # using as conditon of DCGAN, which shape is same to data
                        single_label_shape = np.reshape(single_label_shape, [3])
                        single_label = self.reshape_decode(single_label, single_label_shape)

                # Mean Subtraction

                # (Random Cropping
                if self.croplen:
                        with tf.name_scope('cropping'):
                                if self.stage == utils.STAGE_TRAIN:
                                        single_data = tf.random_crop(single_data,
                                                                    [self.croplen, self.croplen, self.channels],
                                                                    seed=self._seed)
                                else:
                                        single_data = tf.image.resize_image_with_crop_or_pad(signgle_data, self.croplen, self.croplen)

                # Data Augmentation


                max_queue_capacity = min(math.ceil(self.total * MIN_FRACTION_OF_EXMPLES_IN_QUEUE),
                                        MAX_ABSOLUTE_EXAMLES_IN_QUEUE)

                single_batch = [single_key, single_data]
                if single_label is not None:
                        single_batch.append(single_label)

                if self.backend == 'tfrecords' and self.shuffle:
                        batch = tf.train.shuffle_batch(
                                single_batch,
                                batch_size=self.batch_size,
                                num_threads=NUM_THREADS_DATA_LOADER,
                                capacity=10 * self.batch_size,       # Max amount that will be loaded and queued
                                shapes=[[0], self.get_shape(), []], # Only makes sense is dynamic_pad=False , (key, data, label)
                                min_after_dequeue=5 * self.batch_size,
                                allow_amller_final_batch=True, # Happens if total % batch_size != 0
                                name='batcher'
                        )
                else:
                        batch = tf.train.batch(
                                single_batch,
                                batch_size=self.batch_size,
                                dynamic_pad=True, # Allows us to not suplly fixed shape a priori
                                enqueue_many=False, # Each tensor is a single example
                                # set number of threads to 1 for tfrecords (used for inference)
                                num_threads=NUM_THREADS_DATA_LOADER if not self.is_inference else 1,
                                capacity=max_queue_capacity,   # Max amout that will be loadded and queued
                                allow_smaller_final_batch=True, # Happens if total % batch_size != 0
                                name='batcher',
                        )

                self.batch_k = batch[0]  # Key
                self.batch_x = batch[1]  # Input
                if len(batch) == 3:
                        # There's lbael (unlike during inferencing)
                        self.batch_y = batch[2]  # Output (label)

class TFRecordsLoader(LoaderFactory):
        """ The TFRecordsLoader connects directly into the tensorflow graph.
        It uses TFRecords, the 'standard' tensorflow data format.
        """
        def __init__(self):
                pass

        def initialize(self):
                self.float_data = False # For now only strings
                self.unencoded_data_format = 'hwc'
                self.unencoded_channel_scheme = 'rgb'
                self.reader = None
                if self.bitdepth == 8:
                        self.image_dtype = tf.uin8
                elif self.bitdepth == 16:
                        self.image_dtype= tf.uin16
                else:
                        self.image_dtype = tf.float32

                # Count all the records @TODO(tzaman): account for shards!
                # Loop the records in path @TODO(tzaman) get this from a txt?
                # self.db_path += '/test.tfrecords' # @TODO(tzaman) this is a hack

                self.shard_paths = []
                list_db_files = os.path.join(self.db_path, 'list.txt')
                self.total = 0
                if os.path.exists(list_db_files):
                        files = [os.path.join(self.db_path, f) for f in open(list_db_files, 'r').read().splitlines()]
                elif os.path.exists(self.db_path):
                        files = [os.path.join(self.db_path, f) for f in os.listdir(self.db_path)]
                else:
                        files = [self.db_path]
                logging.debug('{}: files {}'.format(__file__, files))
                for shard_path in files:
                        # Account for the relative path format in list.txt
                        record_iter = tf.python_io.tf_record_iterator(shard_path)
                        for r in record_iter:
                                self.total += 1
                        if not self.total:
                                raise ValueError('Database or shrd contains no records {}'.format(self.db_path))
                        self.shard_paths.append(shard_path)
                self.keys = ['{}:0'.format(p) for p in self.shard_paths]

                # Use last record read to extract some preliminary data that is sometimes needed or useful
                example_proto = tf.train.Example()
                example_proto.ParseFromString(r)

                self.channels = example_proto.features.feature['depth'].int64_list.value[0]
                self.height = example_proto.features.feature['height'].int64_list.value[0]
                self.width = example_proto.features.feature['width'].int64_list.value[0]
                data_encoding_id = example_proto.features.feature['encoding'].int64_list.value[0]
                if data_encoding_id:
                        self.data_encoded = True
                        self.data_mime = 'image/png' if data_encoding_id == 1 else 'image/jpeg'
                else:
                        self.data_encoded = False

                logging.debug('Example - shape({},{},{}), data_encoded={}'.format(self.height, self.width, self.channels, self.data_encoded))

                # Set up the reader
                self.reader = tf.TFRecordReader(name='tfrecord_reader')

        def get_queue(self):
                return tf.train.string_input_producer(self.shard_paths,
                                                      num_epochs=self.num_epochs,
                                                      shuffle=self.shuffle,
                                                      seed=self._seed,
                                                      name='input_producer'
                                                      )

        def get_single_data(self, key_queue):
                """
                Returns:
                    key, single_data, single_data_shape, single_label, single_label_shape
                """
                key, serialized_example = self.reader.read(key_queue)
                features = tf.parse_single_example(
                        serialized_example,
                        # Defaults are not specified since both keys are required.
                        features={
                                'image_raw': tf.FixedLenFeature([self.height * self.width * self.channels], tf.float32), # x data, nat_features
                                'label': tf.FixedLenFeature([self.height * self.width * self.channels], tf.float32), # y condition, gen_features
                        })

                d = features['image_raw']
                ds = np.array([self.height, self.width, self.channels], dtype=np.int32)
                logging.debug('image_raw: {}'.format(d.shape))

                l = features['label']
                ls = np.array([self.height, self.width, self.channels], dtype=np.int32)
                logging.debug('label: {}'.format(l.shape))
                return key, d, ds, l, ls

