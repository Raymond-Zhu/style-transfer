import tensorflow as tf

def conditional_instance_norm(inputs, beta, gamma):
    with tf.variable_scope("conditional_instance_norm"):
        mean, variance = tf.nn.moments(inputs, [1,2], keep_dims=True)
        epsilon = 1E-5
        outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
        outputs.set_shape(inputs.get_shape())
        return outputs

def upsampling(inputs, kernel_size, upsample_stride, num_outputs, name, scope, style_params, activation_fn=tf.nn.relu):
    with tf.variable_scope(name) as sc:
        height = inputs.get_shape()[1].value
        width = inputs.get_shape()[2].value
        upsampled_inputs = tf.image.resize_nearest_neighbor(inputs, [upsample_stride*width, upsample_stride*height])
        return conv2d(upsampled_inputs, num_outputs, kernel_size, 1, "conv", sc.name, style_params, activation_fn)

def residual_block(inputs, kernel_size, name, scope, style_params):
    with tf.variable_scope(name) as sc:
        num_outputs = inputs.get_shape()[-1].value
        conv1 = conv2d(inputs, num_outputs, kernel_size, 1, "conv1", sc.name, style_params)
        conv2 = conv2d(inputs, num_outputs, kernel_size, 1, "conv2", sc.name, style_params)
        return inputs + conv2

def conv2d(inputs, num_outputs, kernel_size, stride, name, scope, style_params, activation_fn=tf.nn.relu):
    with tf.variable_scope(name):
        padding = [[0, 0], 
                [kernel_size // 2, kernel_size // 2],
                [kernel_size // 2, kernel_size // 2],
                [0, 0]]
        pad = tf.pad(inputs, padding, 'REFLECT')
        return tf.contrib.layers.conv2d(pad,
                num_outputs,
                kernel_size,
                stride,
                activation_fn=activation_fn,
                padding='VALID',
                normalizer_fn=conditional_instance_norm,
                normalizer_params={'beta': style_params['{}/{}/beta'.format(scope, name)], 'gamma':style_params['{}/{}/gamma'.format(scope,name)]},
                scope=name)
