import tensorflow as tf

def content_loss(content_weight, content_current, content_target):
    shape = content_target.get_shape()
    content_current = tf.reshape(content_current,[shape[3], shape[2] * shape[1]])
    content_target = tf.reshape(content_target,[shape[3], shape[2] * shape[1]])
    return content_weight * tf.reduce_mean((content_current - content_target)**2)

def gram_matrix(features):
    shape = features.get_shape()
    features = tf.reshape(features, [shape[2] * shape[1], shape[3]])
    gram = tf.matmul(tf.transpose(features), features)
    gram /= tf.to_float(shape[1]*shape[2]*shape[3])
    return gram

def style_loss(style_weight, style_current, style_target):
    shape = style_target.get_shape()
    return style_weight * tf.reduce_mean(gram_matrix(style_current) - gram_matrix(style_target)**2)

def tv_loss(img, weight):
    tv_h = tf.reduce_sum((img[:,1:,:,:] - img[:,:-1,:,:])**2)
    tv_w = tf.reduce_sum((img[:,:,1:,:] - img[:,:,:-1,:])**2)
    
    return tv_weight * (tv_h + tv_w)
