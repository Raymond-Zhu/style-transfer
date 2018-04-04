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

def total_variation_loss(img, weight):
    tv_h = tf.reduce_sum((img[:,1:,:,:] - img[:,:-1,:,:])**2)
    tv_w = tf.reduce_sum((img[:,:,1:,:] - img[:,:,:-1,:])**2)
    
    return tv_weight * (tv_h + tv_w)

def total_loss(content_weight, content_input, style_weight, style_input, stylized_image, tv_weight, mobile_net):
    with tf.name_scope("feature_extraction"):    
        with tf.name_scope("content_features"):
            content_features = mobile_net(content_input)
        with tf.name_scope("style_features"):
            style_features = mobile_net(style_input)
        with tf.name_scope("style_image_features"):
            stylized_image_features = mobile_net(stylized_image)
    
    with tf.name_scope("losses"):
        with tf.name_scope("content_loss"):
            c_loss = content_loss(content_weight, stylized_image_features, content_features)
        with tf.name_scope("style_loss"):
            s_loss = style_loss(style_weight, stylized_image_features, style_features)
        with tf.name_scope("tv_loss"):
            tv_loss = total_variation_loss(stylized_image, tv_weight)
        with tf.name_scope("total_loss"): 
            total_loss = c_loss + s_loss + tv_loss
    return total_loss

#Normalize pixels
