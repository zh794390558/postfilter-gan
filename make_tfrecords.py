#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import subprocess
import argparse
import sys
import os
import timeit
import re
import struct
import logging
from sklearn.model_selection import train_test_split
import threading
from multiprocessing.pool import ThreadPool

try:
    from six.moves import xrange
    from pprint import pprint
    from tqdm import tqdm
    import tensorflow as tf
    import numpy as np
    import toml
except ImportError as e:
    subprocess.check_call('./install.sh')
    from six.moves import xrange
    from pprint import pprint
    from tqdm import tqdm
    import tensorflow as tf
    import numpy as np
    import toml


logging.basicConfig(format='%(asctime)s %(threadName)s [%(levelname)s] [line:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    level = logging.DEBUG
                    #level = logging.WARN
                    )

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class ExtractFeatureException(Exception):
    def __init__(self):
        pass
    def __str__(self):
        return 'Not enough frames in file to statisfy read request'

# Extract frames of features from binary file
class ExtractFeature(object):
    type_names = {
        'int8'   : 'b',
        'uint8'  : 'B',
        'int16'  :'h',
        'uint16' :'H',
        'int32'  :'i',
        'uint32' :'I',
        'int64'  :'q',
        'uint64' :'Q',
        'float'  :'f',
        'double' :'d',
        'char'   :'s'
    }

    type_endian = {
        'little-endian' : '<',
        'big-endian' : '>',
    }

    def __init__(self, filename, endian='little-endian', type_name='float', feature_size=41):
        #print('__init__')
        self.binfile = open(filename, 'rb')

        self.type_name = type_name
        self.feature_size = feature_size # use 41-mel-cepstrum coefficient
        self.file_len = os.stat(filename).st_size
        self.type_endian = endian
        self.type_format = ExtractFeature.type_names[self.type_name.lower()]
        self.type_size = struct.calcsize(self.type_format) # sizeof(float)

    def __del__(self):
        #print('__del__')
        self._close()

    def __enter__(self):
        #print('__enter__')
        return self

    def __exit__(self, exc_type, exc_val, exc_traceback):
        #print('__exit__')
        return False

    def _close(self):
        if self.binfile:
            self.binfile.close()
            self.binfile = None

    @property
    def endian(self):
        return ExtractFeature.type_endian[self.type_endian]

    @property
    def len(self):
        return self.file_len

    @property
    def frames(self):
        return self.file_len / (self.feature_size * self.type_size)

    def read(self, frames=200):
        record_len = self.feature_size * frames
        value = self.binfile.read(self.type_size * record_len)
        if len(value) != self.type_size * record_len:
            raise ExtractFeatureException
        return np.asarray(struct.unpack(self.endian +'{}'.format(record_len) + self.type_format, value))

def extract_feature(gen_filename, nature_filename, feature_size=41, frames=200):
    # features files must exists
    if not os.path.exists(gen_filename) or not os.path.exists(nature_filename):
        raise ValueError("ERROR: gen file or nature file does not exists")

    logging.info(' {} x {}, record len {}'.format(frames, feature_size, frames * feature_size))

    # create extract class, and extract featues from file
    with ExtractFeature(gen_filename, feature_size=feature_size) as gen_ext, \
         ExtractFeature(nature_filename, feature_size=feature_size) as nat_ext:

        logging.info('file {} len {} frames {}'.format(gen_filename, gen_ext.len, gen_ext.frames))
        logging.info('file {} len {} frames {}'.format(nature_filename, nat_ext.len, nat_ext.frames))

        # num of *frames* in file, how many banks of 200 frame
        bank_len = int(min(gen_ext.frames, nat_ext.frames) / frames)
        logging.info('{} banks of {} frames in any file'.format(bank_len, frames))

        # extract frames from bin file
        gen_features = gen_ext.read(bank_len * frames)
        nature_features = nat_ext.read(bank_len * frames)

    #pprint(gen_features[:10])
    #pprint(nature_features[:10])

    assert gen_features.shape == nature_features.shape, (gen_feautres.shape, nature_features.shape)

    # raw first, to matrix
    gen_features.shape = (-1, feature_size) #  frame x feature
    gen_features.astype(np.float32)
    nature_features.shape = (-1, feature_size) # frame x feature
    nature_features.astype(np.float32)
    logging.info('features raw shape={}'.format(gen_features.shape))

    return gen_features, nature_features

# transform data to TFRecords
def encoder_proc(gen_filename, nature_filename, result, out_file, feature_size=41, frames=200):
    """ extract features of gen and nature wav and write to TFRecords.
        out_file: TFRecordWriter.
    """
    # extract features
    gen_features, nature_features = extract_feature(gen_filename, nature_filename, feature_size, frames)

    #logging.debug('gen features:raw {}'.format(gen_features))
    #logging.debug('nat features:raw {}'.format(nature_features))
    #logging.debug('result {}'.format(result))

    if result:
        # normal
        gen_mean, nat_mean, gen_std, nat_std = result

        gen_features = (gen_features - gen_mean) / gen_std
        nature_features = (nature_features - nat_mean) / nat_std
        #logging.debug('gen features:normal {}'.format(gen_features))
        #logging.debug('nat features:normal {}'.format(nature_features))

        assert (np.mean(gen_features, axis=0) - gen_mean).all()
        assert (np.std(gen_features, axis=0) - gen_std).all()
        assert (np.mean(nature_features, axis=0) - nat_mean).all()
        assert (np.std(nature_features, axis=0) - nat_std).all()

    # bank of 200 frame
    n_frames = int(gen_features.shape[0] / frames)
    logging.info('{} of {} frames'.format(n_frames, frames))

    # transpose axis
    gen_features = np.transpose(gen_features, (1,0))  # feature x frame
    nature_features = np.transpose(nature_features, (1,0)) # feature x frame

    # add new axis
    gen_features = gen_features[:, :, np.newaxis]  # feature x frame x channel
    nature_features = nature_features[:, :, np.newaxis] # feature x frame x channel
    logging.info('features last shape={}'.format(gen_features.shape))

    # Example to TFRecords
    for n in tqdm(xrange(n_frames), desc='Write Example', ascii=True, leave=False):
        gen_list = gen_features[:, n*frames:(n+1)*frames, :].flatten()
        nat_list = nature_features[:, n*frames:(n+1)*frames, :].flatten()

        example = tf.train.Example(features=tf.train.Features(feature={
                'depth':  _int64_feature(1), # channels
                'height': _int64_feature(feature_size), # feature_szie (41)
                'width': _int64_feature(frames), # frames (200)
                'encoding': _int64_feature(0),  # no encoding, fix to 0
                'image_raw': _floats_feature(gen_list), # gen_features
                'label': _floats_feature(nat_list)  # nature_features
            }))
        #logging.debug('One Example: {}'.format(example))
        out_file.write(example.SerializeToString())


# make dirs or touch file
def prepare_file(pathname, filename, opts):
    '''
    check output file
    '''
    logging.debug('path {} file {}'.format(pathname, filename))
    # set up the output filepath
    out_path = os.path.join(opts.save_path, pathname)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    out_filepath = os.path.join(out_path, filename)

    if os.path.splitext(out_filepath)[1] != '.tfrecords':
        out_filepath += '.tfrecords'
    else:
        out_filename, ext = os.path.splitext(out_filepath)
        out_filepath = out_filename + ext

    # check if out_file exists and if force flag is set
    if os.path.exists(out_filepath) and not opts.force_gen:
        raise ValueError('ERROR: {} already exists. Set force flag (--force-gen) to'
                 'overwrite. Skipping this speaker.'.format(out_filepath))
    elif os.path.exists(out_filepath) and opts.force_gen:
        print('Will overwrite previosly existing tfrecords')
        os.unlink(out_filepath)

    return out_filepath


# save all featues into one tfrecords
def write_record(out_filename, files, result, opts, name):
    '''
    out_filename: train, val, test
    filse: filename list
    normal_done: threding.Event for normal done
    opts: argparser param
    '''
    t = threading.current_thread()
    if name:
        t.name = name

    out_filepath = prepare_file(out_filename, out_filename, opts)

    # TFRecord Writer
    out_file = tf.python_io.TFRecordWriter(out_filepath)

    #for m, (gen_file, nature_file) in enumerate(files):
    qbar = tqdm(enumerate(files), total=len(files))
    for m, (gen_file, nature_file) in qbar:
        qbar.set_description('Process {}'.format(os.path.basename(gen_file)))
        try:
            encoder_proc(gen_file, nature_file, result, out_file, opts.feature_size, opts.frames)
        except Exception as e:
            print(e)
            raise e

    logging.debug("out write_record")
    out_file.close()

# save featues to seprate tfrecords
def write_record_sep(pathname, files, result, opts, name):
    '''
    out_filename: train, val, test
    filse: filename list
    opts: argparser param
    '''
    t = threading.current_thread()
    if name:
        t.name = name

    #for m, (gen_file, nature_file) in enumerate(files):
    qbar = tqdm(enumerate(files), total=len(files))

    for m, (gen_file, nature_file) in qbar:
        qbar.set_description('Process {}'.format(os.path.basename(gen_file)))

        out_filepath = prepare_file(pathname, os.path.splitext(os.path.basename(gen_file))[0], opts)
        logging.debug('out_filepath = {}'.format(out_filepath))

        # TFRecord Writer
        out_file = tf.python_io.TFRecordWriter(out_filepath)

        encoder_proc(gen_file, nature_file, result, out_file, opts.feature_size, opts.frames)

        out_file.close()

# mean
def mean(gen_filename, nature_filename, feature_size=41, frames=200):
    """
        z-score noraml : 0 mean, 1 std
        return: gen_sum , nature_sum ,total
    """
    gen_features, nature_features = extract_feature(gen_filename, nature_filename, feature_size, frames)

    gen_sum = np.sum(gen_features, axis=0)
    gen_num = gen_features.shape[0]
    nat_sum = np.sum(nature_features, axis=0)
    nat_num = nature_features.shape[0]

    assert gen_num == nat_num, gen_num

    return gen_sum, nat_sum, gen_num

# std
def std(gen_filename, nature_filename, gen_mean, nature_mean,  feature_size=41, frames=200):
    """
        z-score noraml : 0 mean, 1 std
        gen_sum = sum((x -u)**2) / N
        return: gen_sum , nature_sum ,total
    """
    gen_features, nature_features = extract_feature(gen_filename, nature_filename, feature_size, frames)

    gen_sum = np.sum(np.power((gen_features - gen_mean), 2), axis=0)
    gen_num = gen_features.shape[0]
    nat_sum = np.sum(np.power((nature_features - nature_mean), 2), axis=0)
    nat_num = nature_features.shape[0]

    assert gen_num == nat_num, gen_num

    return map(lambda x: x / gen_num, [gen_sum, nat_sum]), gen_num
    #return map(lambda x, y: x / y, [gen_sum, nat_sum], gen_num), gen_num

# z-score normal
def z_score_normal(files, opts, name):
    '''
    files: features files, [(gen_file, nature_file), ...]
    normal_done: threding.Event for normal done
    opts: app options
    '''

    if name:
        t = threading.current_thread()
        t.name = name

    # sum for mean
    gen_sum , nat_sum, total = 0., 0., 0.

    # compute mean of data
    # for m, (gen_file, nature_file) in enumerate(files):
    qbar = tqdm(enumerate(files), total=len(files))
    for m, (gen_file, nature_file) in qbar:
        qbar.set_description('Process mean {}'.format(os.path.basename(gen_file)))

        gen, nat, num = mean(gen_file, nature_file, opts.feature_size, opts.frames)
        gen_sum, nat_sum, total = gen_sum + gen, nat_sum + nat, total + num

    # mean
    gen_mean, nat_mean = gen_sum / total, nat_sum / total

    logging.info('gen mean = {} \n nature mean = {} \n total = {}'.format(gen_mean, nat_mean, total))

    # sum for std
    gen_square, nat_square, total_square = 0., 0., 0.

    # compute std of data
    # for m, (gen_file, nature_file) in enumerate(files):
    qbar = tqdm(enumerate(files), total=len(files))
    for m, (gen_file, nature_file) in qbar:
        qbar.set_description('Process std {}'.format(os.path.basename(gen_file)))

        [gen, nat], num = std(gen_file, nature_file, gen_mean, nat_mean, opts.feature_size, opts.frames)
        assert (gen.shape == nat.shape) and (gen.shape[0] == opts.feature_size), 'gen shape {} nat shape {}'.format(gen.shape, nat.shape)
        gen_square, nat_square, total_square = gen_square + gen, nat_square + nat, total_square + num

    logging.debug('gen square = {} \n nature square = {} \n total = {}'.format(gen_square, nat_square, total))

    # total num must be equal
    assert total == total_square

    # std
    gen_std, nat_std = map(np.sqrt, [gen_square, nat_square])

    logging.info('gen std = {} \n nature std = {} \n total = {}'.format(gen_std, nat_std, total))

    return gen_mean, nat_mean, gen_std, nat_std


def main(opts):
    if not tf.gfile.Exists(opts.save_path):
        # make save path if it does not exist
        tf.gfile.MkDir(opts.save_path)

    with open(opts.cfg) as cfg:
        # read the configureation description
        cfg_desc = toml.loads(cfg.read())

        beg_t = timeit.default_timer()

        test_size = None
        train_size = None
        val_size = None
        threads = []
        # process the acustic data now
        for dset_i, (dset_key, dset_val)  in enumerate(cfg_desc.iteritems()):
            print(dset_key)

            # total dataset
            # zip(gen, nature)
            gen_dir = dset_val['gen']
            nature_dir = dset_val['nature']
            files = [(os.path.join(gen_dir, wav), os.path.join(nature_dir, os.path.splitext(wav)[0] + '.cep'))
                  for wav in os.listdir(gen_dir)[:opts.examples] if wav.endswith('.mcep')]
            nfiles = len(files)

            logging.debug(files[:1])
            logging.debug('Total files num: {}'.format(nfiles))

            # thread pool
            pool = ThreadPool(4)

            try:
                if opts.normal:
                    # z-score normal
                    normal_ret = pool.apply_async(z_score_normal, args=(files, opts, 'z_score_normal_thread'))
                    normal_ret.wait()
                    result = normal_ret.get()
                    logging.debug('normal result {}'.format(result))

                # split: train, val , test dataset
                files_train, files_test = train_test_split(files, test_size=opts.test_size)
                files_train, files_val = train_test_split(files_train, test_size=opts.val_size)
                test_size, train_size, val_size = len(files_test), len(files_train), len(files_val)
                logging.debug('test data ({}): {}'.format(len(files_test), files_test))
                logging.debug('train data ({}): {}'.format(len(files_train), files_train))
                logging.debug('val data  ({}): {}'.format(len(files_val), files_val))

                # write train, val, test TFRecords
                pool.apply_async(write_record, args=('train', files_train,  result, opts, 'train_data_thread'))
                pool.apply_async(write_record, args=('val', files_val, result, opts, 'val_data_thread'))
                pool.apply_async(write_record_sep, args=('test', files_test, result, opts, 'test_data_thread'))

            except KeyboardInterrupt :
                logging.warning('got Ctrl+C')
            except Exceptioin as e:
                logging.info(e)
                raise e
            finally:
                # join threads
                pool.close()
                pool.join()

            logging.info('test({}), train({}), val({})'.format(test_size, train_size, val_size))


        end_t = timeit.default_timer() - beg_t
        print('Total processing and writing time: {} s'.format(end_t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert the set of wavs to TFRecords')

    parser.add_argument('--cfg', type=str, default='cfg/postfilter.toml',
               help='File containing the description fo datesets'
                'to extract the info to make the TFRecords.')
    parser.add_argument('--save_path', type=str, default='data/',
               help='Path to save the dataset')
    parser.add_argument('--force-gen', dest='force_gen', action='store_true',
                help='Flag to force overwriting exiting dataset.')
    parser.set_defaults(force_gen=False)
    parser.add_argument('--frames', type=int, default=200,
                help='Frames length')
    parser.add_argument('--feature_size', type=int, default=41,
                help='feature length')
    parser.add_argument('--examples', type=int, default=None,
                help='convert *examples* examples, for debug')
    parser.add_argument('--test_size', type =float, default=0.15,
                help='data split rate for test out of all data')
    parser.add_argument('--val_size', type =float, default=0.15,
                help='data split rate for val out of train data')
    parser.add_argument('--normal', type =bool, default=True,
                help='z score normal all data')

    opts = parser.parse_args()
    pprint(opts)

    main(opts)

