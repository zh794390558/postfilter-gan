# Debug tf

## Out of range: PaddingFIFOQueue 

```text
2017-07-25 03:04:32.161726: W tensorflow/core/framework/op_kernel.cc:1158] Out of range: PaddingFIFOQueue '_5_train/data/batcher/padding_fifo_queue' is closed and has insufficient elements (requested 6, current size 0)
     [[Node: train/data/batcher = QueueDequeueUpToV2[component_types=[DT_STRING, DT_FLOAT, DT_FLOAT], timeout_ms=-1, _device="/job:localhost/replica:0/task:0/cpu:0"](train/data/batcher/padding_fifo_queue, train/data/batcher/n)]]
```

[This is a message that's printed when you reach the end of the queue, so it's part of normal operation](https://github.com/tensorflow/tensorflow/issues/923)

## how to understand tower_grads of `tf.compute_gradients` and `average_gradients` 

* the output of tower_grads.(three towers, only print first two)

[(<tf.Tensor 'train/tower_0/gradients/train/tower_0/discriminator/d_h0_conv/Conv2D_grad/tuple/control_dependency_1:0' shape=(5, 5, 1, 64) dtype=float32>,
  <tf.Variable 'discriminator/d_h0_conv/w:0' shape=(5, 5, 1, 64) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_0/gradients/train/tower_0/discriminator/d_h0_conv/BiasAdd_grad/tuple/control_dependency_1:0' shape=(64,) dtype=float32>,
  <tf.Variable 'discriminator/d_h0_conv/biases:0' shape=(64,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_0/gradients/train/tower_0/discriminator/d_bn1/batchnorm/sub_grad/tuple/control_dependency:0' shape=(64,) dtype=float32>,
  <tf.Variable 'discriminator/d_bn1/beta:0' shape=(64,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_0/gradients/train/tower_0/discriminator/d_bn1/batchnorm/mul_grad/tuple/control_dependency_1:0' shape=(64,) dtype=float32>,
  <tf.Variable 'discriminator/d_bn1/gamma:0' shape=(64,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_0/gradients/train/tower_0/discriminator/d_h1_conv/Conv2D_grad/tuple/control_dependency_1:0' shape=(5, 5, 64, 128) dtype=float32>,
  <tf.Variable 'discriminator/d_h1_conv/w:0' shape=(5, 5, 64, 128) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_0/gradients/train/tower_0/discriminator/d_h1_conv/BiasAdd_grad/tuple/control_dependency_1:0' shape=(128,) dtype=float32>,
  <tf.Variable 'discriminator/d_h1_conv/biases:0' shape=(128,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_0/gradients/train/tower_0/discriminator/d_bn2/batchnorm/sub_grad/tuple/control_dependency:0' shape=(128,) dtype=float32>,
  <tf.Variable 'discriminator/d_bn2/beta:0' shape=(128,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_0/gradients/train/tower_0/discriminator/d_bn2/batchnorm/mul_grad/tuple/control_dependency_1:0' shape=(128,) dtype=float32>,
  <tf.Variable 'discriminator/d_bn2/gamma:0' shape=(128,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_0/gradients/train/tower_0/discriminator/d_h2_conv/Conv2D_grad/tuple/control_dependency_1:0' shape=(3, 3, 128, 192) dtype=float32>,
  <tf.Variable 'discriminator/d_h2_conv/w:0' shape=(3, 3, 128, 192) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_0/gradients/train/tower_0/discriminator/d_h2_conv/BiasAdd_grad/tuple/control_dependency_1:0' shape=(192,) dtype=float32>,
  <tf.Variable 'discriminator/d_h2_conv/biases:0' shape=(192,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_0/gradients/train/tower_0/discriminator/d_bn3/batchnorm/sub_grad/tuple/control_dependency:0' shape=(192,) dtype=float32>,
  <tf.Variable 'discriminator/d_bn3/beta:0' shape=(192,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_0/gradients/train/tower_0/discriminator/d_bn3/batchnorm/mul_grad/tuple/control_dependency_1:0' shape=(192,) dtype=float32>,
  <tf.Variable 'discriminator/d_bn3/gamma:0' shape=(192,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_0/gradients/train/tower_0/discriminator/d_h3_conv/Conv2D_grad/tuple/control_dependency_1:0' shape=(3, 3, 192, 128) dtype=float32>,
  <tf.Variable 'discriminator/d_h3_conv/w:0' shape=(3, 3, 192, 128) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_0/gradients/train/tower_0/discriminator/d_h3_conv/BiasAdd_grad/tuple/control_dependency_1:0' shape=(128,) dtype=float32>,
  <tf.Variable 'discriminator/d_h3_conv/biases:0' shape=(128,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_0/gradients/train/tower_0/discriminator/d_bn4/batchnorm/sub_grad/tuple/control_dependency:0' shape=(128,) dtype=float32>,
  <tf.Variable 'discriminator/d_bn4/beta:0' shape=(128,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_0/gradients/train/tower_0/discriminator/d_bn4/batchnorm/mul_grad/tuple/control_dependency_1:0' shape=(128,) dtype=float32>,
  <tf.Variable 'discriminator/d_bn4/gamma:0' shape=(128,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_0/gradients/train/tower_0/discriminator/d_h4_lin/MatMul_grad/tuple/control_dependency_1:0' shape=(4608, 1) dtype=float32>,
  <tf.Variable 'discriminator/d_h4_lin/Matrix:0' shape=(4608, 1) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_0/gradients/train/tower_0/discriminator/d_h4_lin/add_grad/tuple/control_dependency_1:0' shape=(1,) dtype=float32>,
  <tf.Variable 'discriminator/d_h4_lin/bias:0' shape=(1,) dtype=float32_ref>)]
[(<tf.Tensor 'train/tower_1/gradients/train/tower_1/discriminator/d_h0_conv/Conv2D_grad/tuple/control_dependency_1:0' shape=(5, 5, 1, 64) dtype=float32>,
  <tf.Variable 'discriminator/d_h0_conv/w:0' shape=(5, 5, 1, 64) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_1/gradients/train/tower_1/discriminator/d_h0_conv/BiasAdd_grad/tuple/control_dependency_1:0' shape=(64,) dtype=float32>,
  <tf.Variable 'discriminator/d_h0_conv/biases:0' shape=(64,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_1/gradients/train/tower_1/discriminator/d_bn1/batchnorm/sub_grad/tuple/control_dependency:0' shape=(64,) dtype=float32>,
  <tf.Variable 'discriminator/d_bn1/beta:0' shape=(64,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_1/gradients/train/tower_1/discriminator/d_bn1/batchnorm/mul_grad/tuple/control_dependency_1:0' shape=(64,) dtype=float32>,
  <tf.Variable 'discriminator/d_bn1/gamma:0' shape=(64,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_1/gradients/train/tower_1/discriminator/d_h1_conv/Conv2D_grad/tuple/control_dependency_1:0' shape=(5, 5, 64, 128) dtype=float32>,
  <tf.Variable 'discriminator/d_h1_conv/w:0' shape=(5, 5, 64, 128) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_1/gradients/train/tower_1/discriminator/d_h1_conv/BiasAdd_grad/tuple/control_dependency_1:0' shape=(128,) dtype=float32>,
  <tf.Variable 'discriminator/d_h1_conv/biases:0' shape=(128,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_1/gradients/train/tower_1/discriminator/d_bn2/batchnorm/sub_grad/tuple/control_dependency:0' shape=(128,) dtype=float32>,
  <tf.Variable 'discriminator/d_bn2/beta:0' shape=(128,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_1/gradients/train/tower_1/discriminator/d_bn2/batchnorm/mul_grad/tuple/control_dependency_1:0' shape=(128,) dtype=float32>,
  <tf.Variable 'discriminator/d_bn2/gamma:0' shape=(128,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_1/gradients/train/tower_1/discriminator/d_h2_conv/Conv2D_grad/tuple/control_dependency_1:0' shape=(3, 3, 128, 192) dtype=float32>,
  <tf.Variable 'discriminator/d_h2_conv/w:0' shape=(3, 3, 128, 192) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_1/gradients/train/tower_1/discriminator/d_h2_conv/BiasAdd_grad/tuple/control_dependency_1:0' shape=(192,) dtype=float32>,
  <tf.Variable 'discriminator/d_h2_conv/biases:0' shape=(192,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_1/gradients/train/tower_1/discriminator/d_bn3/batchnorm/sub_grad/tuple/control_dependency:0' shape=(192,) dtype=float32>,
  <tf.Variable 'discriminator/d_bn3/beta:0' shape=(192,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_1/gradients/train/tower_1/discriminator/d_bn3/batchnorm/mul_grad/tuple/control_dependency_1:0' shape=(192,) dtype=float32>,
  <tf.Variable 'discriminator/d_bn3/gamma:0' shape=(192,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_1/gradients/train/tower_1/discriminator/d_h3_conv/Conv2D_grad/tuple/control_dependency_1:0' shape=(3, 3, 192, 128) dtype=float32>,
  <tf.Variable 'discriminator/d_h3_conv/w:0' shape=(3, 3, 192, 128) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_1/gradients/train/tower_1/discriminator/d_h3_conv/BiasAdd_grad/tuple/control_dependency_1:0' shape=(128,) dtype=float32>,
  <tf.Variable 'discriminator/d_h3_conv/biases:0' shape=(128,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_1/gradients/train/tower_1/discriminator/d_bn4/batchnorm/sub_grad/tuple/control_dependency:0' shape=(128,) dtype=float32>,
  <tf.Variable 'discriminator/d_bn4/beta:0' shape=(128,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_1/gradients/train/tower_1/discriminator/d_bn4/batchnorm/mul_grad/tuple/control_dependency_1:0' shape=(128,) dtype=float32>,
  <tf.Variable 'discriminator/d_bn4/gamma:0' shape=(128,) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_1/gradients/train/tower_1/discriminator/d_h4_lin/MatMul_grad/tuple/control_dependency_1:0' shape=(4608, 1) dtype=float32>,
  <tf.Variable 'discriminator/d_h4_lin/Matrix:0' shape=(4608, 1) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_1/gradients/train/tower_1/discriminator/d_h4_lin/add_grad/tuple/control_dependency_1:0' shape=(1,) dtype=float32>,
  <tf.Variable 'discriminator/d_h4_lin/bias:0' shape=(1,) dtype=float32_ref>)]


* the output of zip(tower_grads):

((<tf.Tensor 'train/tower_0/gradients/train/tower_0/discriminator/d_h0_conv/Conv2D_grad/tuple/control_dependency_1:0' shape=(5, 5, 1, 64) dtype=float32>,
  <tf.Variable 'discriminator/d_h0_conv/w:0' shape=(5, 5, 1, 64) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_1/gradients/train/tower_1/discriminator/d_h0_conv/Conv2D_grad/tuple/control_dependency_1:0' shape=(5, 5, 1, 64) dtype=float32>,
  <tf.Variable 'discriminator/d_h0_conv/w:0' shape=(5, 5, 1, 64) dtype=float32_ref>),
 (<tf.Tensor 'train/tower_2/gradients/train/tower_2/discriminator/d_h0_conv/Conv2D_grad/tuple/control_dependency_1:0' shape=(5, 5, 1, 64) dtype=float32>,
  <tf.Variable 'discriminator/d_h0_conv/w:0' shape=(5, 5, 1, 64) dtype=float32_ref>))

## 2017-07-26 03:01:50,301 main.py[line:477] [ERROR] Model diverged with val/tower_0/model/chi_square = nan : Try decreasing your learning rate

```test
Debugging NaNs can be tricky, especially if you have a large network. tf.add_check_numerics_ops() adds ops to the graph that assert that each floating point tensor in the graph does not contain any NaN values, but does not run these checks by default. Instead it returns an op that you can run periodically, or on every step, as follows:

train_op = ...
check_op = tf.add_check_numerics_ops()

sess = tf.Session()
sess.run([train_op, check_op])  # Runs training and checks for NaNs
```
