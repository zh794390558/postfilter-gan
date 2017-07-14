import logging
from utils import model_property

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
# Constants
SUMMARIZE_TOWER_STATS = False

class Tower(object):
        def __init__(self, x, y, input_shape, nclasses, is_training, is_inference):
                self.input_shape = input_shape
                self.nclasses = nclasses
                self.is_trainning = is_training
                self.is_inference = is_inference
                self.summaries = []
                self.x = x
                self.y = y
                self.train = None


        def gradientUpdate(self, grad):
                return grad

# from
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
def average_gradients(tower_grads):
	"""Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
           tower_grads: List of lists of (gradient, variable) tuples. The outer list
           is over individual gradients. The inner list is over the gradient
	   calculation for each tower.
		[ [(grad0_gpu0, var0_gpu0), ..., (gradM_gpu0, varM_gpu0)],
		  [(grad0_gpu1, var0_gpu1), ..., (gradM_gpu1, varM_gpu1)],
		  ...                       ...                      ...
		  [(grad0_gpuN, var0_gpuN), ..., (gradM_gpuN, varM_gpuN)] ]
	Returns:
	    List of pairs of (gradient, variable) where the gradient has been averaged
	    across all towers.
	"""
	with tf.name_scope('gradient_average'):
		average_grads = []
		for grad_and_vars in zip(*tower_grads):
			# Note that eatch grad_and_vars looks like the following:
			# ((grad0_gpu0, var0_gpu0), ..., (grad0_gpuN, var0_gpuN)) 
			grads = []
			for g, _ in grad_and_vars:
				# Add 0 dimension to the gradients to represent the tower.
				expanded_g = tf.expand_dims(g, 0)

