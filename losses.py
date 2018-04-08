import tensorflow as tf

MOBILENET_STYLE_ENDPOINTS = ["layer_10/output", "layer_11/output", "layer_12/output", "layer_13/output", "layer_14/output"]
MOBILENET_CONTENT_ENDPOINT = "layer_15/output"

def content_loss(content_weight, stylized_image_endpoints, content_endpoints):
    content_image = content_endpoints[MOBILENET_CONTENT_ENDPOINT]
    stylized_image = stylized_image_endpoints[MOBILENET_CONTENT_ENDPOINT]
    
    shapes = content_image.get_shape()
    content_image = tf.reshape(content_image, shapes[1], shapes[2]*shapes[3])
    stylized_image = tf.reshape(stylized_image, shapes[1], shapes[2]*shapes[3])

    return content_weight * tf.reduce_mean((content_current - content_target)**2)

def gram_matrix(features):
    shape = features.get_shape()
    features = tf.reshape(features, [shape[2] * shape[1], shape[3]])
    gram = tf.matmul(tf.transpose(features), features)
    gram /= tf.to_float(shape[1]*shape[2]*shape[3])
    return gram

def style_loss(style_weight, stylized_image_endpoints, style_input_endpoints):
    total_style_loss = tf.get_variable("total_style_loss", dtype=tf.float32, initializer=tf.zeros_initializer)
    for i in MOBILENET_STYLE_ENDPOINTS:
        style_current = stylized_image_endpoints[i]
        style_target = style_input_endpoints[i]
        style_loss = style_weight[i] * tf.reduce_mean(gram_matrix(style_current) - gram_matrix(style_target)**2)
        total_style_loss += style_loss

def total_variation_loss(img, weight):
    tv_h = tf.reduce_sum((img[:,1:,:,:] - img[:,:-1,:,:])**2)
    tv_w = tf.reduce_sum((img[:,:,1:,:] - img[:,:,:-1,:])**2)
    
    return tv_weight * (tv_h + tv_w)

def total_loss(content_weight, content_endpoints, style_weight, style_input_endpoitns, stylized_image_endpoints, tv_weight):
    
    with tf.name_scope("losses"):
        with tf.name_scope("content_loss"):
            c_loss = content_loss(content_weight, stylized_image_endpoints, content_endpoints)
        with tf.name_scope("style_loss"):
            s_loss = style_loss(style_weight, stylized_image_endpoints, style_input_endpoints)
        with tf.name_scope("tv_loss"):
            tv_loss = total_variation_loss(stylized_image, tv_weight)
        with tf.name_scope("total_loss"): 
            total_loss = c_loss + s_loss + tv_loss
    return total_loss
