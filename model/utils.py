import functools
from tensorflow.python.client import device_lib

STAGE_TRAIN = 'train' # trainning
STAGE_VAL = 'val'     # validation
STAGE_INF = 'inf'     # evaulation

class GraphKeys(object):
	TEMPLATE = 'model'
	QUEUE_RUNNERS = 'queue_runner'
	MODEL = 'model'
	LOSS = 'loss'
	LOSSES = 'losses'
	LOADER = 'data'

def model_property(function):
    # From https://danijar.com/structuring-your-tensorflow-models/
	attribute = '_cache_' + function.__name__

	@property
	@functools.wraps(function)
	def decorator(self):
		if not hasattr(self, attribute): 
			setattr(self, attribute, function(self))
		return getattr(self, attribute)
    	return decorator

def get_available_gpus():
	"""
	Queries the CUDA GPU devices visible to Tensorflow.
	Returns:
        	A list with tf-style gpu strings (f.e. ['/gpu:0', '/gpu:1'])
	"""
	local_device_protos = device_lib.list_local_devices()
	return [ x.name for x in local_device_protos if x.device_type == 'GPU' ]

