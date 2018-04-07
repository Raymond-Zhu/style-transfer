import tensorflow as tf

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

def conditional_instance_norm(inputs, beta, gamma):
    with tf.variable_scope("conditional_instance_norm"):
        mean, variance = tf.nn.moments(inputs, [1,2], keep_dims=True)
        epsilon = 1E-5
        outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
        outputs.set_shape(inputs.get_shape())
        return outputs


def upsampling(inputs, kernel_size, upsample_stride, output_size, name, scope, normalizer_fn, style_params, activation_fn=tf.nn.relu):
    with tf.variable_scope(name):
        height = inputs.get_shape()[1]
        width = inputs.get_shape()[2]
        upsampled_inputs = tf.image.resize_images(inputs, [upsample_stride*width, upsample_stride*height], method=ResizeMethod.NEAREST_NEIGHBOR)
        return tf.contrib.layers.conv2d(upsampled_inputs,
                output_size,
                kernel_size,
                normalizer_fn=normalizer_fn,
                normalizer_params={'beta': style_params['{}/{}/conv1/beta'.format(scope, name)], 'gamma':style_params['{}/{}/conv1/gamma'.format(scope,name)]},
                name='conv',
                activation=activation_fn)

def residual_block(image, kernel_size, name, scope, normalizer_fn, style_params):
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
                normalizer_params={'beta': style_params['{}/{}/conv1/beta'.format(scope, name)], 'gamma':style_params['{}/{}/conv1/gamma'.format(scope,name)]},
                name='conv1', activation=tf.nn.relu)

        conv2 = tf.contrib.layers.conv2d(conv1,
                output_size,
                kernel_size,
                normalizer_fn=normalizer_fn,
                normalizer_params={'beta': style_params['{}/{}/conv2/beta'.format(scope, name)], 'gamma':style_params['{}/{}/conv2/gamma'.format(scope,name)]},
                name='conv2')

        return image + conv2

def style_prediction_network(inputs, mobile_net):
    features = mobile_net(inputs)
    print(features.get_shape())
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
            conv1 = tf.contrib.layers.conv2d(inputs, 32, [9,9], [1,1],
                    normalizer_fn=conditional_instance_norm,
                    normalizer_params={'beta': style_params[scope.name + '/conv1/beta'], 'gamma': style_params[scope.name + '/conv1/gamma']},
                    activation=tf.nn.relu,
                    name="conv1")

            conv2 = tf.contrib.layers.conv2d(conv1, 64, [3,3], [2,2],
                    normalizer_fn=conditional_instance_norm,
                    normalizer_params={'beta': style_params[scope.name + '/conv2/beta'], 'gamma': style_params[scope.name + '/conv2/gamma']},
                    activation=tf.nn.relu,
                    name="conv2")

            conv3 = tf.contrib.layers.conv2d(conv1, 128, [3,3], [2,2],
                    normalizer_fn=conditional_instance_norm,
                    normalizer_params={'beta': style_params[scope.name + '/conv3/beta'], 'gamma': style_params[scope.name + '/conv3/gamma']},
                    activatio=tf.nn.relu,
                    name="conv3")
        
        with tf.variable_scope('residual') as scope:
            res1 = residual_block(conv3, [3,3], "residual1", scope.name, conditional_instance_norm, style_params)
            res2 = residual_block(res1, [3,3], "residual2", scope.name, conditional_instance_norm, style_params)
            res3 = residual_block(res2, [3,3], "residual3", scope.name, conditional_instance_norm, style_params)

        with tf.variable_scope('upsampling') as scope:
            up1 = upsampling(res3, [3,3], 2, 64, 'upsample1', scope.name, conditional_instance_norm, style_params)
            up2 = upsampling(up1, [3,3], 2, 32, 'upsample2', scope.name, conditional_instance_norm, style_params)
            return upsampling(up2, [9,9], 2, 3, 'upsample3', scope.name, conditional_instance_norm, style_params, tf.nn.sigmoid)
