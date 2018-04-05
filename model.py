import tensorflow as tf

from utils import  conditional_instance_norm


activation_names = ["style_transformer/encode/conv1",
                    "style_transformer/encode/conv2",
                    "style_transformer/encode/conv3",
                    "style_transformer/residual/residual1/conv1",
                    "style_transformer/residual/residual1/conv2",
                    "style_transformer/residual/residual2/conv1",
                    "style_transformer/residual/residual2/conv2",
                    "style_transformer/residual/residual3/conv1",
                    "style_transformer/residual/residual3/conv2",
                    "style_transformer/residual/residual4/conv1",
                    "style_transformer/residual/residual4/conv2",
                    "style_transformer/residual/residual5/conv1",
                    "style_transformer/residual/residual5/conv2",
                    ]


def residual_block(image, kernel_size, name, var_scope, normalizer_fn, style_params):
    with tf.variable_scope(name):
        output_size = image.get_shape()[-1].value
        padding = [[0, 0], 
                [filter_size // 2, filter_size // 2],
                [filter_size // 2, filter_size // 2],
                [0, 0]]
        pad = tf.pad(image, padding, 'REFLECT')
        conv1 = tf.contrib.layers.conv2d(pad,
                output_size,
                kernel_size,
                normalizer_fn=normalizer_fn,
                normalizer_params={'beta': style_params['{}/{}/conv1/beta'.format(var_scope, name)], 'gamma':style_params['{}/{}/conv1/gamma'.format(var_scope,name)]},
                name='conv1', activation=tf.nn.relu)

        conv2 = tf.contrib.layers.conv2d(conv1,
                output_size,
                kernel_size,
                normalizer_fn=normalizer_fn,
                normalizer_params={'beta': style_params['{}/{}/conv2/beta'.format(var_scope, name)], 'gamma':style_params['{}/{}/conv2/gamma'.format(var_scope,name)]},
                name='conv2')

        return image + conv2

def style_prediction_network(inputs, mobile_net, activation_depths, activation_names):
    features = mobile_net(inputs)
    with tf.name_scope('bottleneck'):
        bottleneck = tf.reduce_mean(features, axis=[1,2], keep_dims=True)
        bottleneck = tf.layers.conv2d(bottleneck, 100, [1,1])
    
    style_params = {}
    with tf.variable_scope("style_params"):
        for i in range(len(activation_depths)):
            with tf.variable_scope(activation_names[i]):
                beta = tf.layers.conv2d(bottleneck, activation_depths[i],[1,1])
                beta = tf.squeeze(beta, [1,2], name='squeeze')

                gamma = tf.layers.conv2d(bottleneck, activation_depths[i], [1,1])
                gamma = tf.squeeze(gamma, [1,2], name='squeeze')

                style_params['{}/beta'.format(activation_names[i])] = beta
                style_params['{}/gamma'.format(activation_names[i])] = gamma
    return style_params

def style_transformer_network(inputs, style_params, is_training):
    with tf.variable_scope('style_transformer'):
        with tf.variable_scope('encode') as scope:
            conv1 = tf.contrib.layers.conv2d(inputs, 32, [9,9], [1,1],
                    normalizer_fn=conditional_instance_norm,
                    normalizer_params={'beta': style_params[scope + '/conv1/beta'], 'gamma': style_params[scope + '/conv1/gamma']},
                    activation=tf.nn.relu,
                    name="conv1")

            conv2 = tf.contrib.layers.conv2d(conv1, 64, [3,3], [2,2],
                    normalizer_fn=conditional_instance_norm,
                    normalizer_params={'beta': style_params[scope + '/conv2/beta'], 'gamma': style_params[scope + '/conv2/gamma']},
                    activation=tf.nn.relu,
                    name="conv2")

            conv3 = tf.contrib.layers.conv2d(conv1, 128, [3,3], [2,2],
                    normalizer_fn=conditional_instance_norm,
                    normalizer_params={'beta': style_params[scope + '/conv3/beta'], 'gamma': style_params[scope + '/conv3/gamma']},
                    activatio=tf.nn.relu,
                    name="conv3")
        
        with tf.variable_scope('residual') as scope:
            res1 = residual_block(conv3, [3,3], "residual1", scope, conditional_instance_norm, style_params)
            res2 = residual_block(res1, [3,3], "residual2", scope, conditional_instance_norm, style_params)
            res3 = residual_block(res2, [3,3], "residual3", scope, conditional_instance_norm, style_params)
