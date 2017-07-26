from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
from pprint import pprint
import tensorflow as tf
from tensorflow.python.framework import ops

import utils
import tf_data
from  ops  import scalar_summary
from utils import model_property

logging.basicConfig(format='%(asctime)s %(filename)s [line:%(llineno)d] [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
# Constants
SUMMARIZE_TOWER_STATS = False

class Tower(object):
        def __init__(self, x, y, input_shape, nclasses, is_training, is_inference):
                self.input_shape = input_shape
                self.is_trainning = is_training
                self.is_inference = is_inference
                self.summaries = []
                self.x = x
                self.y = y
                self.train = None

                self.nclasses = nclasses

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
        [[(grad0_gpu0, var0_gpu0), ..., (gradM_gpu0, gradM_gpu0)]
         [(grad0_gpu1, var0_gpu1), ..., (gradM_gpu1, gradM_gpu1)]
         ...
         [(grad0_gpuN, var0_gpuN), ..., (gradM_gpuN, gradM_gpuN)]
        ]

       first axis is tower index;
       second axis is grad_var.
        Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
        """
        #pprint(tower_grads[0][:])
        #pprint(tower_grads[1][:])

        with tf.name_scope('gradient_average'):
                average_grads = []
                for grad_and_vars in zip(*tower_grads):
                        #pprint(grad_and_vars)
            #time.sleep(10)

                        # Note that eatch grad_and_vars looks like the following:
                        # ((grad0_gpu0, var0_gpu0), ..., (grad0_gpuN, var0_gpuN))
                        grads = []
                        for g, _ in grad_and_vars:
                                # Add 0 dimension to the gradients to represent the tower.
                                expanded_g = tf.expand_dims(g, 0)
                                # Append on a 'tower' dimension which we will average over below
                                grads.append(expanded_g)
                        # Average over the 'tower' dimension.
                        grad = tf.concat(axis=0, values=grads)
                        grad = tf.reduce_mean(grad, 0)
                        # Keep in mind that the Variabels are redundant because they are shared
                        # across towers. So .. we will just return the first tower's pointer to
                        # the Variabe.
                        v = grad_and_vars[0][1]
                        grad_and_var = (grad, v)
                        average_grads.append(grad_and_var)
                return average_grads

class Model(object):
        def __init__(self, stage, croplen, nclasses, optimization=None, momentum=None, reuse_variable=False):
                self.stage = stage
                self.croplen = croplen
                self.nclasses = nclasses
                self.dataloader = None
                self.queue_coord = None
                self.queue_threads = None

                self._optimization = optimization
                self._momentum = momentum
                self.summaries = []
                self.towers = []
                self._train = None
                self._reuse = reuse_variable

                # Touch to initialize
                # if optimization:
                #        self.learning_rate
                #        self.global_step
                #        self.optimizer

        def create_dataloader(self, db_path):
                self.dataloader = tf_data.LoaderFactory.set_source(db_path, is_inference=(self.stage == utils.STAGE_INF))
                self.dataloader.stage = self.stage

                self.dataloader.croplen = self.croplen
                self.dataloader.nclasses = self.nclasses

        def init_dataloader(self):
                with tf.device('/cpu:0'):
                        with tf.name_scope(utils.GraphKeys.LOADER):
                                self.dataloader.create_input_pipline()

        def create_model(self, obj_UserModel, stage_scope, batch_x=None):
            logging.debug('Stage: {}'.format(stage_scope))

            # get batch data
            if batch_x is None:
                    self.init_dataloader()
                    batch_x = self.dataloader.batch_x
                    if self.stage != utils.STAGE_INF:
                            batch_y = self.dataloader.batch_y
            else:
                    assert self.stage == utils.STAGE_INF
                    batch_x = batch_x

            logging.debug('batch_x shape={} batch_y shape={}'.format(batch_x.get_shape().as_list(), batch_y.get_shape().as_list()))

    # get avilable gpu list
            available_devices = utils.get_available_gpus()
            logging.debug('GPUs {}'.format(available_devices))
            if not available_devices:
                    available_devices.append('/cpu:0')

            # available_devices = ['/gpu:0', '/gpu:1'] # DEVELOPMENT: virtual multi-gpu

            # Split the batch over the batch dimension over the number of available gpu's
            if len(available_devices) == 1:
                    batch_x_split = [batch_x]
                    if self.stage != utils.STAGE_INF: # Has no labels
                            batch_y_split = [batch_y]
            else:
                    with tf.name_scope('parallelize'):
                            # Split them up
                            batch_x_split = tf.split(batch_x, len(available_devices), 0, name='split_batch_x')
                            if self.stage != utils.STAGE_INF: # Has no labels
                                    batch_y_split = tf.split(batch_y, len(available_devices), 0, name='split_batch_y')

            # Run the user model through the build_model function that should be filled in
            grad_towers = []
            #with tf.variable_scope(tf.get_variable_scope()):
            for dev_i, dev_name in enumerate(available_devices):
                with tf.device(dev_name):
                    current_scope = stage_scope if len(available_devices) == 1 else ('tower_{}'.format(dev_i))
                    with tf.name_scope(current_scope) as scope_tower:

                        with tf.variable_scope(utils.GraphKeys.MODEL, reuse=dev_i > 0 or self._reuse):
                            if self.stage != utils.STAGE_INF:
                                tower_model = self.add_tower(obj_tower=obj_UserModel,
                                             x=batch_x_split[dev_i],
                                             y=batch_y_split[dev_i])
                            else:
                                tower_model = self.add_tower(obj_tower=obj_User_model,
                                             x=batch_x_split[dev_i],
                                             y=None)

                            tower_model.inference # touch to initialize

                            # Reuse the variables int this scope for the next tower/device
                            tf.get_variable_scope().reuse_variables()

                            if self.stage == utils.STAGE_INF:
                                # For inferencing we will only use the inference part of the graph
                                continue

                            with tf.name_scope(utils.GraphKeys.LOSS):
                                for loss in self.get_tower_losses(tower_model):
                                    tf.add_to_collection(utils.GraphKeys.LOSSES, loss['loss'])

                                # Assemble all made within this scope so far. The user can add custom
                                # losses to the utils.GraphKeys.LOSSES collection
                                losses = tf.get_collection(utils.GraphKeys.LOSSES, scope=scope_tower)
                                #logging.debug('get_collection: graykeys.LOSSES : {}'.format(losses))
                                #logging.debug('get_collection: graykeys.REGULARIZATION_LOSSES : {}'.format(ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES, scope=None)))
                                losses += ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES, scope=None)
                                tower_loss = tf.add_n(losses, name='loss')

                                self.summaries.append(scalar_summary(tower_loss.op.name, tower_loss))

                            if self.stage == utils.STAGE_TRAIN:
                                grad_tower_losses = []
                                for loss in self.get_tower_losses(tower_model):
                                    grad_tower_loss = self.optimizer.compute_gradients(loss['loss'], loss['vars'])
                                    grad_tower_loss = tower_model.gradientUpdate(grad_tower_loss)
                                    grad_tower_losses.append(grad_tower_loss)
                                grad_towers.append(grad_tower_losses)

            # Assemble and average the gradients from all towers
            if self.stage == utils.STAGE_TRAIN:
                    n_gpus = len(available_devices)
                    if n_gpus == 1:
                            grad_averages = grad_towers[0]
                    else:
                            with tf.device(available_devices[0]):
                                    n_losses = len(grad_towers[0])
                                    grad_averages = []
                                    for loss in xrange(n_losses):
                                            grad_averages.append(average_gradients([grad_towers[gpu][loss] for gpu in xrange(n_gpus)]))
                    apply_gradient_ops = []
                    for grad_avg in grad_averages:
                            apply_gradient_ops.append(self.optimizer.apply_gradients(grad_avg, global_step=self.global_step))
                    self._train = apply_gradient_ops

        def start_queue_runners(self, sess):
                logging.info('Starting queue runners ({})'.format(self.stage))
                # Distinguish the queue runner collection(for easliy obtaining them by collection key)
                queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS, scope=self.stage + '.*')
                for qr in queue_runners:
                        if self.stage in qr.name:
                                tf.add_to_collection(utils.GraphKeys.QUEUE_RUNNERS, qr)

                self.queue_coord = tf.train.Coordinator()
                self.queue_threads = tf.train.start_queue_runners(sess=sess, coord=self.queue_coord,
                                                                  collection=utils.GraphKeys.QUEUE_RUNNERS)

                logging.info('Queue runners started ({})'.format(self.stage))

        def __del__(self):
                # Destructor
                if self.queue_coord:
                        # Close and terminate the queues
                        self.queue_coord.request_stop()
                        self.queue_coord.join(self.queue_threads)

        def add_tower(self, obj_tower, x, y):
                is_training = self.stage == utils.STAGE_TRAIN
                is_inference = self.stage == utils.STAGE_INF
                input_shape = self.dataloader.get_shape()
                tower = obj_tower(x, y, input_shape, self.nclasses, is_training, is_inference)
                self.towers.append(tower)
                return tower


        @model_property
        def train(self):
                return self._train

        @model_property
        def summary(self):
                '''
                Merge train summaries
                '''
                for t in self.towers:
                        self.summaries += t.summaries

                if not len(self.summaries):
                        logging.error('No summaies defined. Please define at least one summary.')
                        exit(-1)
                return tf.summary.merge(self.summaries)

        @model_property
        def global_step(self):
                # Force global_step on the CPU, beacues the GPU's first step will end at 0 instead of 1.
                with tf.device('/cpu:0'):
                        return tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                                trainable=False)

        @model_property
        def learning_rate(self):
                # the learning rate is a function for the global step, so we could
                # define is entirely in tf ops, instead of a placeholder and feeding.
                with tf.device('/cpu:0'):
                        lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
                        self.summaries.append(scalar_summary('lr', lr))
                        return lr

        @model_property
        def optimizer(self):
                logging.info('Optimizer:%s', self._optimization)
                if self._optimization == 'sgd':
                        return tf.train.GradientDescentOptimizer(learing_rate=self.learning_rate)
                elif self._optimization =='adadelta':
                        return tf.train.AdadeltaOptimizer(learing_rate=self.learing_rate)
                elif self._optimization == 'adagrad':
                        return tf.train.AdagradOptimizer(learing_rate=self.learing_rate)
                elif self._optimization == 'adagradda':
                        return tf.train.AdagradDAOptimizer(learning_rate=self.learning_rate,
                                                           global_step=self.global_step)
                elif self._optimization == 'momentum':
                        return tf.train.MomentumOptimizer(learing_rate=self.learning_rate,
                                                          momentum=self._momentum)
                elif self._optimization == 'adam':
                        return tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                elif self._optimization == 'ftrl':
                        return tf.train.FtrlOptimizer(learning_rate=self.learning_rate)
                elif self._optimization == 'rmsprop':
                        return tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                                         momentum=self._momenttum)
                else:
                        logging.error("Invalid optimization flag {}".format(self._optimization))
                        exit(-1)

        def get_tower_losses(self, tower):
                """
                Return list of losses
                If user-defined model returns only one loss then this is encapsulated into the expected list of
                dicts structure
                """
                if isinstance(tower.loss, list):
                        return tower.loss
                else:
                        return [{'loss': tower.loss, 'vars': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)}]

