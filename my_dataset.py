import tensorflow as tf
import tensorflow_io as tfio
from helper_funcs import map_to


def parse_to_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float16)
    image = tfio.experimental.color.rgb_to_lab(image)
    return tf.expand_dims(image[:,:,0]/100, -1), (image[:,:,1:]/128+1)/2


def parse_and_augment(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float16)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    image = tfio.experimental.color.rgb_to_lab(image)
    
    return tf.expand_dims(image[:,:,0]/100, -1), map_to(image[:,:,1:]/128)


def create_dataset_from_files(filenames, batch_size, num_parallel_calls, augment=False):
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = ds.shuffle(buffer_size=len(filenames))
    if augment:
        ds = ds.map(parse_and_augment, num_parallel_calls=num_parallel_calls)
    else:
        ds = ds.map(parse_to_image, num_parallel_calls=num_parallel_calls)
    ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

