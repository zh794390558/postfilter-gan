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
                - self.y: label, conditonal variable. Shape: [N] for scalar labels, [N, H, W, C] otherwise.
                Only defined if self._is_training is True
                """

                super(UserModel, self).__init__(*args, **kwargs)

                # initialize graph with parameters
                self.postfilter_gan_init()

        @model_property
        def inference(self):
                '''
                op to use for inference
                '''
                # inference op is the output of the generator
                return self.G

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

        def postfilter_gan_init(self, image_height=41, image_width=200,
                        y_dim=1, z_dim=1, gf_dim=128, df_dim=64,
                        gfc_dim=None, dfc_dim=None, c_dim=1, window=41):
                """
                Create the model
                Paper: GENERATIVE ADVERSARIAL NETWORK-BASED POSTFILTER FOR STATISTICAL PARAMETRIC SPEECH SYNTHESIS
                Args:
                    image_height: The Mel-cepstrum Coeffiecnt of input of G. [41]
                    image_width: The frames in Mel-cepstrum of input of G [200]
                    y_dim: (optional) Dimension of dim for y, same to input of G. [batch_height, width, 1]
                    z_dim: (optional) Dimension of dim for Z. [batch, height, width, 1]
                    gf_dim: (optional) Dimension of gen filters in first conv layer. [128]
                    df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
                    gfc_dim: (optional) Dimension of gen units for for fully connected layer. [0]
                    dfc_dim: (optional) Dimension of discrim units for fully connected layer. [None]
                    c_dim: (optional) Dimension of image color. For grayscale input, set to 1. For RGB input, set to 3. [1]
                """
                self.image_height = image_height
                self.image_width = image_width

                self.y_dim = y_dim
                self.z_dim = z_dim

                self.gf_dim = gf_dim
                self.df_dim = df_dim

                self.c_dim = c_dim

                # for D crop input image
                self.window = window

                self.batch_size = tf.shape(self.x)[0]

                # batch normalization: deals with poor initialization helps gradient flow
                self.d_bn1 = batch_norm(name='d_bn1')
                self.d_bn2 = batch_norm(name='d_bn2')
                self.d_bn3 = batch_norm(name='d_bn3')
                self.d_bn4 = batch_norm(name='d_bn4')

                self.g_bn0 = batch_norm(name='g_bn0')
                self.g_bn1 = batch_norm(name='g_bn1')
                self.g_bn2 = batch_norm(name='g_bn2')
                self.g_bn3 = batch_norm(name='g_bn3')

                self.build_model()

        def build_model(self):
                '''create the main ops'''

                if not self.is_inference:
                        # create both the generator and the discriminator
                        # self.x is a batch of images - shape: [N, H, W, C]
                        # self.y is a vector of labels - shape: [N, H, W, C], SYN features

                        # sample z from a normal distribution - shape: [N, H, W, C]
                        self.z = tf.random_normal(shape=[self.batch_size, self.image_height, self.image_width, self.z_dim], dtype=tf.float32, seed=None, name='z')

                        # rescale x to [0,1]
                        x_reshaped = tf.reshape(self.x, shape=[self.batch_size, self.image_height, self.image_width, self.c_dim], name='x_reshaped')
                        self.images = x_reshaped # real data

                        '''
                        # one hot encode the label -- shape:[N] -> [N, self.y_dim]
                        self.y = tf.one_hot(self.y, self.y_dim, name='y_onehot')
                        '''

                        # create the generator
                        self.G = self.generator(self.z, self.y)

                        # crop window
                        offset = np.random.randint(0, self.image_width - self.window)

                        # create one instance of the discriminator for real images (the input is
                        # images from the dataset)
                        self.D, self.D_logits = self.discriminator(self.images, self.y, reuse=False, offset=offset, window=self.window)

                        # create another instance of the discriminator for fake images (the input is
                        # the discriminator). Note how we are resuing varibales to share weights between
                        # both instances of the discriminator
                        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True, offset=offset, window=self.window)

                        # aggreate losses across batch

                        # we are using the cross entropy loss for all these losses
                        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D), name='loss_D_real'))
                        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_), name='loss_D_fake'))
                        self.d_loss = (self.d_loss_real + self.d_loss_fake) / 2.

                        # the typical GAN set-up is that of a minmax game where D is tring to minimize its own error and G is trying to
                        # maxminize D's error. However note how we are flipping G labels here: instaed of maximizing D's error, we are
                        # minimizing D's error on the 'wrong' label, this trick helps produce a stronger gradient
                        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_), logits=self.D_logits_, name='loss_G'))

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
                        self.g_vars = [var for var in t_vars if 'g_' in var.name]
                        # D's variabels
                        self.d_vars = [var for var in t_vars if 'd_' in var.name]

                        # Extra hook for debug: log chi-square distance between G's output histgram and the dataset' histogram
                        value_range = [0.0, 1.0]
                        nbins = 100
                        hist_g = tf.histogram_fixed_width(self.G, value_range, nbins=nbins, dtype=tf.float32) / nbins
                        hist_images = tf.histogram_fixed_width(self.images, value_range, nbins=nbins, dtype=tf.float32) / nbins
                        chi_square = tf.reduce_mean(tf.div(tf.square(hist_g - hist_images), hist_g + hist_images))
                        self.summaries.append(scalar_summary('chi_square', chi_square))
                else:
                        # Create only the generator

                        # sample z from a normal distribution - shape: [N, H, W, C]
                        self.z = tf.random_normal(shape=[self.batch_size, self.image_height, self.image_width, self.z_dim], dtype=tf.float32, seed=None, name='z')

                        # self.x is the conditioned latent representation -- shape: [self.batch_size, self.image_height, self.image_width, self.c_dim]
                        self.x = tf.reshape(self.x, shape=[self.batch_size, self.image_height, self.image_width, self.c_dim])
                        logging.debug('batch_size = {}'.format(self.batch_size))

                        assert self.x.get_shape().as_list()[1:] == [self.image_height, self.image_width, self.c_dim], self.x.get_shape().as_list()

                        # use x for conditon
                        self.y = self.x

                        # create an instance of the generator
                        self.G = self.generator(self.z, self.y)

        def discriminator(self, image, y=None, reuse=False, offset=0, window=41):
                """
                Create the discriminator
                This creates a string of layers:
                - input - [N, 41, 41, 1]
                - conv layer with 5x5 kernels and 1x1 stride - [N, 41, 41, 64]
                - batch norm - [N, 41, 41, 64]
                - leaky relu - [N, 41, 41, 64]
                - conv layer with  5x5 kernels and 2x2 stride - [N, 20, 20, 128]
                - batch norm - [N, 20, 20, 128]
                - leaky relu - [N, 20, 20, 128]
                - conv layer with  3x3 kernels and 2x2 stride - [N, 10, 10, 256]
                - batch norm - [N, 10, 10, 256]
                - leaky relu - [N, 10, 10, 256]
                - conv layer with  3x3 kernels and 2x2 stride - [N, 5, 5, 128]
                - batch norm - [N, 5, 5, 128]
                - leaky relu - [N, 5, 5, 128]
                - flatten - [N, 3200]
                - linear layer with 1 output neuron - [N, 1]
                Args:
                        image: batch of input images - shape: [N, H, W, C]
                        y: batch of labels - shape: [N, H, W, C], for condiation
                        reuse: whether to re-use previously created variables
                """
                with tf.variable_scope("discriminator") as scope:
                        if reuse:
                                # re-use (share) variabels
                                scope.reuse_variables()

                        # Not used
                        y = y

                        s1, s2, s3 = int(self.df_dim), int(self.df_dim*2), int(self.df_dim*3)

                        logging.debug('D: input shape {}'.format(image.get_shape().as_list()))

                        # crop a window of image
                        x = tf.image.crop_to_bounding_box(image, offset_height=0, offset_width=offset,
                                target_height=window, target_width=window)

                        logging.debug('D: Croped input shape {}'.format(x.get_shape().as_list()))

                        h0 = conv2d(x, s1, k_h=5, k_w=5, d_h=1, d_w=1, name='d_h0_conv')
                        h0 = lrelu(self.d_bn1(h0, train=self.is_trainning))

                        h1 = lrelu(self.d_bn2(conv2d(h0, s2, k_h=5, k_w=5, d_h=2, d_w=2, name='d_h1_conv'), train=self.is_trainning))

                        h2 = lrelu(self.d_bn3(conv2d(h1, s3, k_h=3, k_w=3, d_h=2, d_w=2, name='d_h2_conv'), train=self.is_trainning))

                        h3 = lrelu(self.d_bn4(conv2d(h2, s2, k_h=3, k_w=3, d_h=2, d_w=2, name='d_h3_conv'), train=self.is_trainning))

                        sz = h3.get_shape().as_list()
                        h3 = tf.reshape(h3, [self.batch_size, int(sz[1] * sz[2] * sz[3])])

                        h4 = linear(h3, 1, 'd_h4_lin')

                        return tf.nn.sigmoid(h4), h4

        def generator(self, z, y=None):
                """
                Create the generator
                This creates a string of layers:
                - input - [N, 41, 200,  1]
                - concatenate conditioning - [N, 41, 200, 2]
                - conv layer with  5x5 kernels and 1x1 stride - [N, 41, 200, 128]
                - batch norm - [N, 41, 200, 128]
                - relu - [N, 41, 200, 128]
                - concatenate conditioning - [N, 41, 200, 129]
                - conv layer with  5x5 kernels and 1x1 stride - [N, 41, 200, 256]
                - batch norm - [N, 41, 200, 256]
                - relu - [N, 41, 200, 256]
                - concatenate conditioning - [N, 41, 200, 257]
                - conv layer with  5x5 kernels and 1x1 stride - [N, 41, 200, 128]
                - batch norm - [N, 41, 200, 128]
                - relu - [N, 41, 200, 128]
                - conv layer with  5x5 kernels and 1x1 stride - [N, 41, 200, 1]
                """
                with tf.variable_scope('generator') as scope:

                        sz = z.get_shape().as_list()
                        channel_dim = len(sz) - 1
                        s1, s2, s3 = int(self.gf_dim), int(self.gf_dim * 2), int(self.gf_dim)

                        logging.debug('z shape {}, y shape {}'.format(z.get_shape(), y.get_shape()))
                        z = tf.concat([z, y], channel_dim)
                        logging.debug('z shape {}'.format(z.get_shape()))

                        h0 = tf.nn.relu(self.g_bn0(conv2d(z, s1, k_h=5, k_w=5, d_h=1, d_w=1, name='g_h0_conv'), train=self.is_trainning))
                        h0 = tf.concat([h0, y], channel_dim)

                        h1 = tf.nn.relu(self.g_bn1(conv2d(h0, s2, k_h=5, k_w=5, d_h=1, d_w=1, name='g_h1_conv'), train=self.is_trainning))
                        h1 = tf.concat([h1, y], channel_dim)

                        h2 = tf.nn.relu(self.g_bn2(conv2d(h1, s3, k_h=5, k_w=5, d_h=1, d_w=1, name='g_h2_conv'), train=self.is_trainning))

                        h3 = conv2d(h2, 1, k_h=5, k_w=5, d_h=1, d_w=1, name='g_h3_conv')

                        return h3
