# -*- coding: utf-8 -*-
""" main """

import os
import re
import shutil
import time

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

import settings

print("Tensorflow version " + tf.__version__)


def count_data_items(filenames):
    """
    the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    """
    n = [int(re.compile(r"_([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


def read_tfrecord(example):
    features = {
        "image/colorspace": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
        "image/class/label": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar
        "image/height": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar
        "image/class/text": tf.io.FixedLenFeature([], tf.string),
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/channels": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/format": tf.io.FixedLenFeature([], tf.string),
        "image/filename": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, features)
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    class_label = tf.cast(example['image/class/label'], tf.int32)
    filename = tf.cast(example['image/filename'], tf.string)

    return image, class_label, filename


def plot_image(image, label, filename, figsize=None, to_disk=False):
    """
    Plots the image with its label

    Args:
        image   (numpy.npdarray) : numpy (X,Y,3) array representing an RGB image
        label      (numpy.32int) : image label
        filename         (bytes) : filename
        figsize          (tuple) : (float, float) size of image to be plotted
        to_disk           (bool) : plot or save image to disk
    """
    assert isinstance(image, np.ndarray) and image.shape[2] == 3
    assert isinstance(label, np.int32)
    assert isinstance(to_disk, bool)

    if figsize:
        assert isinstance(figsize, tuple)
        assert len(figsize) == 2
        assert figsize[0] > 0 and figsize[1] > 0
    else:
        figsize = (8, 8)

    plt.figure(figsize=figsize)
    plt.suptitle("Filename: {}".format(filename.decode("utf-8")), fontsize=18)
    plt.title("Label: {}".format(label), fontsize=14)
    plt.imshow(image)
    if to_disk:
        plt.savefig(os.path.join(settings.SAVE_IMAGES_PATH, filename.decode("utf-8")))
    else:
        plt.show()


def load_dataset(filenames):
    """
    read from TFRecords. For optimal performance, use "interleave(tf.data.TFRecordDataset, ...)"
    to read from multiple TFRecord files at once and set the option
    experimental_deterministic = False to allow order-altering optimizations.

    Args:
        filenames (iterable) : iterable of full paths to the .TFRecord files

    Returns:
        tensorflow.python.data.ops.dataset_ops.ParallelMapDataset

    """
    assert isinstance(filenames, (list, tuple))
    assert filenames

    opt = tf.data.Options()
    opt.experimental_deterministic = False

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.with_options(opt)
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=settings.AUTO)

    ###########################################################################
    #                      examining tfrecords structure                      #
    ###########################################################################
    # raw_example = next(iter(dataset))
    # parsed = tf.train.Example.FromString(raw_example.numpy())
    # keys = [k for k in parsed.features.feature.keys()]
    # # print(keys)
    # # ['image/colorspace', 'image/class/label', 'image/height', 'image/class/text', 'image/encoded', 'image/channels', 'image/width', 'image/format', 'image/filename']
    # # parsed.features.feature['image/text']
    # #########################################################################

    dataset = dataset.map(read_tfrecord, num_parallel_calls=settings.AUTO)

    return dataset


def plot_images(num_images=10, figsize=None, to_disk=False):
    """
    Plots the first "num_images" images

    Args:
        num_images (numpy.int32) : number of images to plot
        figsize          (tuple) : (float, float) size of image to be plotted
        to_disk           (bool) : plot or save image to disk
    """
    assert isinstance(num_images, int)
    assert num_images > 0
    assert isinstance(to_disk, bool)

    if to_disk:
        if os.path.isdir(settings.SAVE_IMAGES_PATH):
            shutil.rmtree(settings.SAVE_IMAGES_PATH)
        os.makedirs(settings.SAVE_IMAGES_PATH)

    dataset = load_dataset(settings.DATASET_TFRECORDS_PATHS)

    counter = 0
    for item in dataset.as_numpy_iterator():
        if counter < num_images:
            plot_image(*item, figsize, to_disk)
            counter += 1
            if counter % settings.MAXIMUM_WRITING_IMAGES == 0 and to_disk:
                time.sleep(settings.SLEEP_WHILE_WRITING)
        else:
            break


def main():
    """  """
    # count_data_items(settings.TILES)
    # elem = next(iter(dataset.as_numpy_iterator()))
    plot_images(num_images=40, to_disk=True)


if __name__ == '__main__':
    main()
