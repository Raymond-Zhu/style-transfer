import tensorflow as tf

def conditional_instance_norm(inputs, beta, gamma):
    with tf.variable_scope("conditional_instance_norm"):
        mean, variance = tf.nn.moments(inputs, [1,2], keep_dims=True)
        epsilon = 1E-5
        outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
        outputs.set_shape(inputs.get_shape())
        return outputs
