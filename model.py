import tensorflow as tf
import layers

ACTIVATION_NAMES = ["style_transformer/encode/conv1",
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
                    "style_transformer/upsample/upsample1/conv",
                    "style_transformer/upsample/upsample2/conv",
                    "style_transformer/upsample/upsample3/conv",
                    ]

ACTIVATION_DEPTHS = [32,64,128,128,128,128,128,128,128,128,128,128,128,64,32,3]



def style_prediction_network(inputs, features):
    with tf.name_scope('bottleneck'):
        bottleneck = tf.layers.dense(features, 100)
    
    style_params = {}
    with tf.variable_scope("style_params"):
        for i in range(len(ACTIVATION_DEPTHS)):
            with tf.variable_scope(ACTIVATION_NAMES[i]):
                beta = tf.layers.dense(bottleneck, ACTIVATION_DEPTHS[i])
                gamma = tf.layers.dense(bottleneck, ACTIVATION_DEPTHS[i])

                style_params['{}/beta'.format(ACTIVATION_NAMES[i])] = beta
                style_params['{}/gamma'.format(ACTIVATION_NAMES[i])] = gamma
    return style_params

def style_transformer_network(inputs, style_params):
    with tf.variable_scope('style_transformer'):
        with tf.variable_scope('encode') as scope:
            conv1 = layers.conv2d(inputs, 32, 9, 1, "conv1", scope.name, style_params)
            conv2 = layers.conv2d(conv1, 64, 3, 2, "conv2", scope.name, style_params)
            conv3 = layers.conv2d(conv2, 128, 3, 2, "conv3", scope.name, style_params)
        
        with tf.variable_scope('residual') as scope:
            res1 = layers.residual_block(conv3, 3, "residual1", scope.name, style_params)
            res2 = layers.residual_block(res1, 3, "residual2", scope.name, style_params)
            res3 = layers.residual_block(res2, 3, "residual3", scope.name, style_params)

        with tf.variable_scope('upsample') as scope:
            up1 = layers.upsampling(res3, 3, 2, 64, 'upsample1', scope.name, style_params)
            up2 = layers.upsampling(up1, 3, 2, 32, 'upsample2', scope.name, style_params)
            return layers.upsampling(up2, 9, 2, 3, 'upsample3', scope.name, style_params, tf.nn.sigmoid)
