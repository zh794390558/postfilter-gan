import logging
from ops import * 
from utils import model_property
from base import Tower

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
		    datefmt='%Y-%m-%d %H:%M:%S',
	            level=logging.INFO)

class UserModel(Tower):
	"""
	User Model definition

	DIGITS creates an instance of this class for every tower it needs
	to create. This includes:
	    - one for training,
	    - one for validation,
	    - one for testing.

	In the case of multi-GPU training, one training instance is created
	for every GPU. DIGITS takes care of doing the gradient averaging
	across GPUs so this class only needs to define the inference op
	and desired loss/cost function.
	"""

	def __init__(self, *args, **kwargs):
		"""
		Identify the correct input nodes.

		In the parent class, DIGITS conveniently sets the following fields:
		- self.is_training: whether this is a training graph
		- self.is_inference: whether this graph is created for inference/testing
		- self.x: input node. Shape: [N, H, W, C]
		- self.y: label. Shape: [N] for scalar labels, [N, H, W, C] otherwise.
		Only defined if self._is_training is True
		"""

		super(UserModel, self).__init__(*args, **kwargs)

		# initialize graph with parameters
		self.dcgain_init()

	@model_property
	def inference(self):
		''' op to use for inference'''

		# inference op is the output of the generator after rescaling 
		# to the 8-bit range
		return tf.to_int32(self.G * 255)


	@model_property
	def loss(self):
		"""
		Loss function
		Returns either an op or a list of dicts.

		If the returned value is an op then DIGITS will optimize against this op
		with respect to all trainable variables.

		If the returned value is a list then DIGITS will optimize against each
		loss in the list with respect to the specified variables.
		"""

		# here we are returning a list because we want to alternately optimize the
		# discriminator on real samples, the discriminator on fake samples and the
		# generator.
		losses = [
		    {'loss': self.d_loss_real, 'vars': self.d_vars},
		    {'loss': self.d_loss_fake, 'vars': self.d_vars},
		    {'loss': self.g_loss, 'vars': self.g_vars}
		]
		return losses

	def dcgan_init(self, image_size=108, 
			output_size=64, y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
			gfc_dim=1024, dfc_dim=1024, c_dim=3):
		"""
		Create the model
		Args:
		    output_size: (optional) The resolution in pixels of the images. [64]
		    y_dim: (optional) Dimension of dim for y. [None]
		    z_dim: (optional) Dimension of dim for Z. [100]
		    gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
		    df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
		    gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
		    dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
		    c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
		"""

		self.image_size = image_size
		self.output_size = output_size

		self.y_dim = y_dim
		self.z_dim = z_dim

		self.gf_dim = gf_dim
		self.df_dim = df_dim
		
		self.c_dim = c_dim

		self.batch_size = tf.shape(self.x)[0]
		
		# batch normalization: deals with poor initialization helps gradient flow
		self.d_bn1 = batch_norm(name='d_bn1')
		self.d_bn2 = batch_norm(name='d_bn2')
	
		self.g_bn0 = batch_norm(name='g_bn0')
		self.g_bn1 = batch_norm(name='g_bn1')
		self.g_bn2 = batch_norm(name='g_bn2')

		self.build_model()

	def build_model(self):
		'''create the main ops'''
		
		if not self.is_inference:
			# create both the generator and the discriminator
			# self.x is a batch of images - shape: [N, H, W, C]
		        # self.y is a vector of labels - shape: [N]
			
			# sample z from a normal distribution
			self.z = tf.random_normal(shape=[self.batch_size, self.z_dim], dtype=tf.float32, seed=None, name='z')
			
			# rescale x to [0,1]
			x_resahped = tf.reshape(self.x, shape=[self.batch_size, self.image_size, self.image_size, self.c_dim], name='x_reshaped')
			self.images = xreshaped / 255.

			# one hot encode the label -- shape:[N] -> [N, self.y_dim]
			self.y = tf.one_hot(self.y, self.y_dim, name='y_onehot')

			# create the generator
			self.G = self.generator(self.z, self.y)

			# create one instance of the discriminator for real images (the input is 
			# images from the dataset)
			self.D, self.D_logits = self.discriminator(self.images, self.y, reuse=False)
		
			# create another instance of the discriminator for fake images (the input is 
			# the discriminator). Note how we are resuing varibales to share weights between
			# both instances of the discriminator
			self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

			# aggreate losses across batch
			
			# we are using the cross entropy loss for all these losses
			self.d_loos_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D), name='loss_D_real')) 
			self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_), name='loss_D_fake'))
			self.d_loss = (self.d_loss_real + self.d_loss_fake) / 2.
			
			# the typical GAN set-up is that of a minmax game where D is tring to minimize its own error and G is trying to 
			# maxminize D's error. However note how we are flipping G labels here: instaed of maximizing D's error, we are 
			# minimizing D's error on the 'wrong' label, this trick helps produce a stronger gradient
			self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_), name='loss_G'))

			# create some summaries for debug and monitoring
			self.summaries.append(histogram_summary("z", self.z)) 
			self.summaries.append(histogram_summary("d", self.D)) 
			self.summaries.append(histogram_summary("d_", self.D_)) 
			self.summaries.append(image_summary("G", self.G, max_outputs=5))
			self.summaries.append(image_summary("X", self.images, max_outputs=5))
			self.summaries.append(histogram_summary("G_hist", self.G)) 
			self.summaries.append(histogram_summary("X_hist", self.images)) 
			self.summaries.append(scalar_summary("d_loss_real", self.d_loss_real)) 
			self.summaries.append(scalar_summary("d_loss_fake", self.d_loss_fake)) 
			self.summaries.append(scalar_summary("g_loss", self.g_loss)) 
			self.summaries.append(scalar_summary("d_loss", self.d_loss)) 

			# all trainable variabels
			t_vars = tf.trainable_variables()
			# G's variables
			self.g_vars = [var for var in t_vars is 'g_' in var.name]
			# D's variabels
			self.d_vars = [var for var in t_vars is 'd_' in var.name]

			# Extra hook for debug: log chi-square distance between G's output histgram and the dataset' histogram
			value_range = [0.0, 1.0]
			nbins = 100
			hist_g = tf.histogram_fixed_width(self.G, value_range, nbins=nbins, dtype=tf.float32) / nbins
			hist_images = tf.histogram_fixed_width(self.images, value_range, nbins=nbins, dtype=tf.float32) / nbins
			chi_square = tf.reduce_mean(tf.div(tf.square(hist_g - hist_images), hist_g + hist_images))
			self.summaries.append(scalar_summary('chi_square', chi_square))
		else:
			# Create only the generator

			# self.x is the conditioned latent representation -- shape: [self.batch_size, 1, self.z_dim + self.y_dim]
			self.x = tf.reshape(self.x, shape=[self.batch_size, self.z_dim + self.y_dim])
			# extract z and y
			self.y = self.x[:, self.z_dim:self.z_dim + self.y_dim]
			self.z = self.x[:, :self.z_dim]
			# create an instance of the generator
			self.G = self.generator(self.x, self.y)

	def discriminator(self, image, y=None, reuse=False):
		 """
		Create the discriminator
		This creates a string of layers:
		- input - [N, 28, 28, 1]
		- concat conditioning - [N, 28, 28, 11]
		- conv layer with 11 5x5 kernels and 2x2 stride - [N, 14, 14, 11]
		- leaky relu - [N, 14, 14, 11]
		- concat conditioning - [N, 14, 14, 21]
		- conv layer with 74 5x5 kernels and 2x2 stride - [N, 7, 7, 74]
		- batch norm - [N, 7, 7, 64]
		- leaky relu - [N, 7, 7, 64]
		- flatten - [N, 3626]
		- concat conditioning - [N, 3636]
		- linear layer with 1014 output neurons - [N, 1024]
		- batch norm - [N, 1024]
		- leaky relu - [N, 1024]
		- concat conditioning - [N, 1034]
		- linear layer with 1 output neuron - [N, 1]
		Args:
		    image: batch of input images - shape: [N, H, W, C]
		    y: batch of one-hot encoded labels - shape: [N, K]
		    reuse: whether to re-use previously created variables
		"""
		with tf.variable_scope("discriminator") as scope:
			if reuse:
				# re-use (share) variabels
				scope.reuse_variables()
			
			yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
			x = conv_and_concat(image, yb)
		
			h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))	
			h0 = conv_cond_concat(h0, yb)

			h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv'), train=self.is_tranining))
			sz = h1.get_shape()
			h1 = tf.reshape(h1, [self.batch_size, int(sz[1] * sz[2] * sz[3])])
			h1 = tf.concat(1, [h1, y])

			h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin'), train=self.is_trainning))
			h2 = tf.concat(1, [h2, y])

			h3 = linear(h2, 1, 'd_h3_lin')

			return tf.nn.sigmoid(h3), h3

	def generator(self, z, y=None):
		"""
		Create the generator
		This creates a string of layers:
		- input - [N, 100]
		- concatenate conditioning - [N, 110]
		- linear layer with 1024 output neurons - [N, 1024]
		- batch norm - [N, 1024]
		- relu - [N, 1024]
		- concatenate conditioning - [N, 1034]
		- linear layer with 7*7*128=6272 output neurons - [N, 6272]
		- reshape 7x7 feature maps - [N, 7, 7, 128]
		- concatenate conditioning - [N, 7, 7, 138]
		- transpose convolution with 128 filters and stride 2 - [N, 14, 14, 128]
		- batch norm - [N, 14, 14, 128]
		- relu - [N, 14, 14, 128]
		- concatenate conditioing - [N, 14, 14, 138]
		- transpose convolution with 1 filter and stride 2 - [N, 28, 28, 1]
		"""
		with tf.variables_scope('generator') as scope:
	
			s = self.output_size
			s2, s4 = int(s/2), int(s/4)

			yb = tf.reshpae(y, [self.batch_size, 1, 1, self.y_dim])
			z = tf.concnat(1, [z, y])

			h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=self.is_tranning))
			h0 = tf.concat(1, [h0, y])
			
			h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s4*s4, 'g_h1_lin'), train=self.is_trainning))
			h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])

			h1 = conv_cond_concat(h1, yb)

			h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2'), train=self.is_training))
			h2 = conv_cond_concat(h2, yb)
		
			return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))
