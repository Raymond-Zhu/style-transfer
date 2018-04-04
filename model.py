import tensorflow as tf

def residual_block(image, kernel_size, scope):
    with tf.variable_scope(scope):
        output_size = image.get_shape()[-1].value
        padding = [[0, 0], 
                [filter_size // 2, filter_size // 2],
                [filter_size // 2, filter_size // 2],
                [0, 0]]
        pad = tf.pad(image, padding, 'REFLECT')
        conv1 = tf.layers.conv2d(pad,output_size,kernel_size, name='conv1', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1,output_size,kernel_size, name='conv2')
        return image + conv2
