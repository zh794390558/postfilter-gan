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
import struct

# Mel-CC
FEATURE_SIZE=41

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class ExtractFeatureException(Exception):
	def __init__(self):
		pass
	def __str__(self):
		return 'Not enough frames in file to statisfy read request'

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
		self.binfile = open(filename, 'rb')
		self.type_name = type_name
		self.feature_size = feature_size
		self.file_len = os.stat(filename).st_size
		self.type_endian = endian
		self.type_format = ExtractFeature.type_names[self.type_name.lower()]
		self.type_size = struct.calcsize(self.type_format) 

	def __del__(self):
		self.binfile.close()

	def __enter__(self, filename, endian='little-endian', type_name='float', feature_size=41):
		self.__init__(self, filename, endian, type_name, feature_size)

	def __exit__(self):
		self.__del__(self)

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


def encoder_proc(gen_filename, nature_filename, out_file, feature_size=41, frames=200):
	""" extract features of gen and nature wav and write to TFRecords.
	    out_file: TFRecordWriter.
	"""
	if not os.path.exists(gen_filename) or not os.path.exists(nature_filename):
		raise ValueError("ERROR: gen file or nature file does not exists")

	gen_ext = ExtractFeature(gen_filename, feature_size=feature_size)
	nat_ext = ExtractFeature(nature_filename, feature_size=feature_size)

	pprint([gen_filename, gen_ext.len, gen_ext.frames , 
		nature_filename, nat_ext.len, nat_ext.frames])

	# num of *frames* in file, how many banks of 200 frame
	bank_len = int(min(gen_ext.frames, nat_ext.frames) / frames)

	gen_features = gen_ext.read(bank_len * frames)
	nature_features = nat_ext.read(bank_len * frames)

	pprint(gen_features[:10])
	pprint(nature_features[:10])

	assert gen_features.shape == nature_features.shape, (gen_feautres.shape, nature_features.shape)
	pprint(gen_features.shape)

	# raw first
	gen_features.shape = (-1, feature_size) #  frame x feature 
	nature_features.shape = (-1, feature_size) # frame x feature 
	
	pprint(gen_features.shape)
	# banck of 200 frame 
	n_frames = gen_features.shape[0]

	for n in tqdm(xrange(n_frames), desc='Write Example', ascii=True, leave=False):
		gen_bytes = gen_features[n*frames:(n+1)*frames, :].tobytes()
		nat_bytes = nature_features[n*frames:(n+1)*frames, :].tobytes()

		example = tf.train.Example(features=tf.train.Features(feature={
				'gen_features': _bytes_feature(gen_bytes),
				'nature_features': _bytes_feature(nat_bytes) 
			}))
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

			pprint(files[:1])
			nfiles = len(files)

			#for m, (gen_file, nature_file) in enumerate(files):
			qbar = tqdm(enumerate(files), total=nfiles)
			for m, (gen_file, nature_file) in qbar: 
				qbar.set_description('Process {}'.format(os.path.basename(gen_file)))
				encoder_proc(gen_file, nature_file, out_file, opts.feature_size, opts.frames)

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
	parser.add_argument('--frames', type=int, default=200, 
			    help='Frames length')
	parser.add_argument('--feature_size', type=int, default=41, 
			    help='feature length')

	opts = parser.parse_args()
	pprint(opts)

	main(opts)
