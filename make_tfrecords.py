#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from six.moves import xrange

import tensorflow as tf
import numpy as np
import librosa
from  pprint import pprint
import argparse
import toml
import sys
import os
import timeit
import re
from  tqdm import tqdm

# Mel-CC
FEATURE_SIZE=41

def read_record(gen_filename_queue, nature_filename_queue):
	class Record(object):
		pass
	result = Record()
	
	#input format
	feature_bytes = 4 # float32
	record_bytes =  feature_bytes * FEATURE_SIZE # 4*41
	
	# read a record, getting filenames from the filename_queue.
	# No header or footer in the FEATURE format, so we leave header_bytes
	# and footer_bytes at their deafult of 0.
	gen_reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
	result.gen_key, gen_value = gen_reader.read(gen_filename_queue)

	nature_reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
	result.nature_key, nature_value = nature_reader.read(nature_filename_queue)

	# convert from a string to a vector of float32 that is record_bytes long.
	gen_record_bytes = tf.decode_raw(gen_value, tf.float32, little_endian=True)
	nature_record_bytes = tf.decode_raw(nature_value, tf.float32, little_endian=True)

	print(result.gen_key, result.nature_key)

	result.gen_per_frame = gen_record_bytes
	result.nature_per_frame = nature_record_bytes

	result.height = FEATURE_SIZE # MCC featue
	result.width = 1 # frame index 
	result.depth = 1
	
	result.gen_per_frame.set_shape([result.height, result.width, result.depth])
	result.nature_per_frame.set_shape([result.height, result.width, result.depth])
	print(result.gen_per_frame.shape, result.nature_per_frame.shape)

	return result

def generate_frames(data,  min_queue_examples, batch_size):
	num_preprocess_threads = 16
	gen_frames = tf.train.batch(
		[data.gen_per_frame],
		batch_size=batch_size,
		num_threads = num_preprocess_threads,
		capacity=min_queue_examples + 3*batch_size)

	nature_frames = tf.train.batch(
		[data.nature_per_frame],
		batch_size=batch_size,
		num_threads = num_preprocess_threads ,
		capacity=min_queue_examples + 3*batch_size)

	return (tf.concat(gen_frames, 2), tf.concat(nature_frames, 2))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def tfrecord_proc(gen_frames_feature, nature_frames_feature, out_file):
	gen_str = gen_frames_feature.tostring()
	nature_str = nature_frames_feature.tostring()
	example = tf.train.Example(features=tf.train.Features(festure={
			'gen_features': _bytes_feature(gen_str),
			'nature_features': _bytes_feature(nature_str)}))
	out_file.write(example.SerializeToString())
	
'''

def train_inputs(gen_dir, nature_dir, batch_size, out_file):
	files_num= len(os.listdir(gen_dir))
	assert files_num == len(os.listdir(nature_dir)), files_num

	# gen_filenames
	gen_filenames = [os.path.join(gen_dir, '{:6d}.cep'.format(i))
					for i in xrange(1, files_num)]
	for f in gen_filenames:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: {}'.format(f))
	# nature_filenames
	nature_filenames = [os.path.join(nature_dir, '{:6d}.mcep'.format(i))
					for i in xrange(1, files_num)]
	for f in nature_filenames:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: {}'.format(f))

	# create a queue that produces the filenames to read.
	gen_filename_queue = tf.train.string_input_producer(gen_filenames, shuffle=False)
	nature_filename_queue = tf.train.string_input_producer(nature_filenames, shuffle=False)

	# read examples from files in the filename queue.
	read_input = read_record(gen_filename_queue, nature_filename_queue)

	# generate 200 frames 
	min_queue_examples = 10
	gen_frames_feature, nature_frames_feature = generate_frames(read_input, min_queue_examples, 200)

	# gen tf.example 
	tfrecord_proc(gen_frames_feature, nature_frames_feature, out_file)

def main(opts):
	if not tf.gfile.Exists(opts.save_path):
		# make save path if it does not exist 
		tf.gfile.MkDir(opts.save_path)
	# set up the output filepath
	out_filepath = os.path.join(opts.save_path, opts.out_file)
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

	with open(opts.cfg) as cfg:
		# read the configureation description
		cfg_desc = toml.loads(cfg.read())
		pprint(cfg_desc)
		beg_t = timeit.default_timer()
		out_file = tf.python_io.TFRecordWriter(out_filepath)

		for dset_i, (dset_key, dset_val)  in enumerate(cfg_desc.iteritems()):
			# process the acustic data now
			gen_dir = dset_val['gen']
			nature_dir = dset_val['nature']
			train_inputs(gen_dir, nature_dir, out_file)
		# end
		out_file.close()
		end_t = timeit.default_timer() - beg_t
		print('Total processing and writing time: {} s'.format(end_t))

'''


def encoder_proc(gen_filename, nature_filename, out_file, frames):
	""" extract features of gen and nature wav and write to TFRecords.
	    out_file: TFRecordWriter.
	"""
	if not os.path.exists(gen_filename) or not os.path.exists(nature_filename):
		raise ValueError("ERROR: gen file or nature file does not exists")

	return 

	gen_features = extract_features(gen_filename, frames=frames)
	nature_features = extract_features(nature_filename, frames=frames)
	assert gen_features.shape == nature_features.shape, (gen_feautres.shape, nature_features.shape)
	
	for (gen, nat) in zip(gen_features, nature_features):
		gen_str = gen.tostring()
		nature_str = nat.tostring()
		example = tf.train.Example(features=tf.train.Features(festure={
			'gen_features': _bytes_feature(gen_str),
			'nature_features': _bytes_feature(nature_str)}))
		out_file.write(example.SerializeToString())
		
		
def main(opts):
	if not tf.gfile.Exists(opts.save_path):
		# make save path if it does not exist 
		tf.gfile.MkDir(opts.save_path)
	# set up the output filepath
	out_filepath = os.path.join(opts.save_path, opts.out_file)
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

	with open(opts.cfg) as cfg:
		# read the configureation description
		cfg_desc = toml.loads(cfg.read())
		beg_t = timeit.default_timer()
		out_file = tf.python_io.TFRecordWriter(out_filepath)
		# process the acustic data now
		for dset_i, (dset_key, dset_val)  in enumerate(cfg_desc.iteritems()):
			print(dset_key)
			print('-' * 50)
			gen_dir = dset_val['gen']
			nature_dir = dset_val['nature']
			files = [(os.path.join(gen_dir, wav) ,  os.path.join(nature_dir, os.path.splitext(wav)[0] + '.cep'))
				  for wav in os.listdir(gen_dir)[:10] if wav.endswith('.mcep')]

			pprint(files)
			nfiles = len(files)

			#for m, (gen_file, nature_file) in enumerate(files):
			for m, (gen_file, nature_file) in tqdm(enumerate(files), total=nfiles, desc='Process features'):
				'''
				print('Processing wav file {}/{} {} {}{}'.format(m+1, 
										nfiles, 
										gen_file, 
										nature_file, 
										' ' * 10 ),
			              end='\r')
				sys.stdout.flush()
				'''
				encoder_proc(gen_file, nature_file, out_file, 2**14)

		out_file.close()
		end_t = timeit.default_timer() - beg_t
		print('*' * 50)
		print('Total processing and writing time: {} s'.format(end_t))

	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Convert the set of wavs to TFRecords')

	parser.add_argument('--cfg', type=str, default='cfg/postfilter.cfg', 
			   help='File containing the description fo datesets'
				'to extract the info to make the TFRecords.')
	parser.add_argument('--save_path', type=str, default='data/', 
			   help='Path to save the dataset')
	parser.add_argument('--out_file', type=str, default='postfilter.tfrecords', 
			    help='Output filename')
	parser.add_argument('--force-gen', dest='force_gen', action='store_true',
			    help='Flag to force overwriting exiting dataset.')
	parser.set_defaults(force_gen=False)

	opts = parser.parse_args()
	pprint(opts)

	main(opts)
