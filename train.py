import argparse
import tensorflow as tf
import tensorflow_hub as hub
import os
import model


parser = argparse.ArgumentParser(description="Style transfer")
parser.add_argument('-c','--content_image_dir')
parser.add_argument('-s','--style_image_dir')
parser.add_argument('-b','--batch_size', type=int)



def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

def read_image(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

def augment_image(image):
    image_orig = image
    image_shape = tf.shape(image_orig)
    height = image_shape[0]
    width = image_shape[1]
    channels = image_shape[2]
    
    image = tf.image.random_brightness(image, max_delta=0.8)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    random_height = tf.random_uniform([], minval=height+5, maxval=height+300, dtype=tf.int32)
    random_width = tf.random_uniform([], minval=width+5, maxval=width+300, dtype=tf.int32)
    
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [random_height, random_width])
    image = tf.squeeze(image)
    image.set_shape([None, None, 3])
    image = tf.random_crop(image, [height, width, channels])
    image = tf.image.resize_images(image, [224, 224]) 
    return image

def main():
    with tf.Graph().as_default():
        args = parser.parse_args()  
        batch_size = args.batch_size 
        content_image_filenames = list(absoluteFilePaths(args.content_image_dir))
        style_image_filenames = list(absoluteFilePaths(args.style_image_dir))

        mobile_net = hub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/feature_vector/1")

        # content_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(content_image_filenames), tf.constant(content_image_filenames)))
        # content_dataset = content_dataset.shuffle(len(content_image_filenames)) 
        # content_dataset = content_dataset.batch(batch_size)
        # content_dataset = content_dataset.map(read_image, num_parallel_calls=4)
        # content_dataset.prefetch(1)
        # content_iterator = content_dataset.make_one_shot_iterator()
        # content_batch = content_iterator.get_next()

        style_dataset = tf.data.Dataset.from_tensor_slices(style_image_filenames)
        style_dataset = style_dataset.map(read_image, num_parallel_calls=4)
        style_dataset = style_dataset.map(augment_image, num_parallel_calls=4)
        style_dataset = style_dataset.shuffle(10000) 
        style_dataset = style_dataset.batch(batch_size)
        style_dataset.prefetch(1)
        style_iterator = style_dataset.make_one_shot_iterator()
        style_batch = style_iterator.get_next()
        
        style_pred_network = model.style_prediction_network(style_batch, mobile_net)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer()) 
            #writer = tf.summary.FileWriter('./logs', sess.graph)
            test = sess.run(style_pred_network)
            #writer.add_summary(test)
main()

