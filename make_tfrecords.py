#!/usr/bin/env python

from __feature__ import print_function
import tensorflow as tf
import numpy as np
import librosa
import pprint.pprint as pprint
import argparse
import toml
import sys
import os
import timeit
import re

ROOTPATH ='/gfs/atlastts/StandFemale_22K/'
subDirs =['gen', 'nature']
postfix ='.wav'
frame_shit=0.5
rate=22050
frame_length=512

def windows(signal, window_size, stride=0.5):
	""" Return windows of the given signal 
	    by sweeping in stride fractions of window
        """
	assert signal.ndim == 1, signal.ndim
	n_samaples = signal.shape[0]
	offset = int(window_size * stride)
	for beg_i, end_i in zip(range(0, n_samples, offset),
				range(window_size, n_samples + offset, offset)):
		if end_i - beg_i < window_size:
			break
		slice_ = signal[beg_i:end_i]
		if slice_.shape[0] == window_size:
			yield slice_


def extract_features(wav_file, bands=40, frames=40, frame_length=512, rate=22050):
	window_size = frame_length * (frames -1)
	log_specgrams = []
	y, sr = librosa.load(f)
	if sr != rate:
		raise ValueError('Sampling rate is expected to be 22.5kH!')

	for signal in windows(y, window_size):
		melspec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=bands)
		logsepc = librosa.logamplitude(melspec) # [n_mel, frames]
		logsepc = logspec[:, np.newaxis] # [n_mel, frames, channels]
		log_specgrams.append(logspec)

	log_specgrams = np.asarray(log_specgrams) # [batch, n_mel ,frames, channels]
	return log_specgrams

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def encoder_proc(gen_filename, nature_filename, out_file, frames):
	""" extract features of gen and nature wav and write to TFRecords.
	    out_file: TFRecordWriter.
	"""
	if os.path.split(gen_filename)[-1].split('.')[0] != os.path.split(nature_filename)[-1].split('.')[0]:
		raise ValueError("ERROR: gen file must match nature file")

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
	if os.path.exists(out_filepath) and not opt.force_gen:
		raise ValueError('ERROR: {} already exists. Set force flag (--force-gen) to'
				 'overwrite. Skipping this speaker.'.format(out_filepath))
	elif os.path.exits(out_filepath) and opts.force_gen:
		print('Will overwrite previosly existing tfrecords')
		os.unlink(out_filepath)

	with open(opts.cfg) as cfg:
		# read the configureation description
		cfg_desc = toml.loads(cfg.read())
		beg_t = timeit.default_timer()
		out_file = tf.python_io.TFRecordWriter(out_filepath)
		# process the acustic data now
		for dset_i, (dest, deset_desc)  in enumerate(cfg_desc.iteritems()):
			print('-' * 50)
			gen_dir = dset_desc['gen']
			gen_files = [os.path.join(gen_dir, wav) 
				     for wav in os.listdir(wav_dir) if wav.endswith('.syn.wav')]

			nature_dir = dset_desc['nature']
			nature_files = [os.path.join(nature_dir, wav)
				     for wav in os.listdir(wav_dir if wav.endswith('.wav')]

			files = zip(gen_files, nature_files)
			nfiles = len(files)
			for m, (gen_file, nature_file) in enumerate(files):
				print('Processing wav file {}/{} {} {}{}'.format(m+1, 
										nfiles, 
										gen_file, 
										nature_file, 
										' ' * 10 ),
			              end='\r')
				sys.stdout.flush()
				encoder_proc(gen_file, nature_file, out_file, 2**14)
		out_file.close()
		end_t = timeit.default_timer() - beg_t
		print('')
		print('*' * 50)
		print('Total processing and writing time: {} s'.format(end_t))
			
				
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Convert the set of wavs to TFRecords')

	parser.add_argument('--cfg', type=str, default='cfg/postfilter.cfg', 
			   help='File containing the description fo datesets'
				'to extract the info to make the TFRecords.'
	parser.add_argument('--save_path', type=str, default='data/', 
			   help='Path to save the dataset')
	parser.add_argument('--out_file', type=str, deafult='postfilter.tfrecords', 
			    help='Output filename')
	parser.add_argument('--force-gen', dest='force_gen', action='store_true',
			    help='Flag to force overwriting exiting dataset.')
	parser.set_defaults(force_gen=False)

	opts = parser.parse_args()
	
	tf.app.run(main=main, argv=opts)
