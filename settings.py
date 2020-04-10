# -*- coding: utf-8 -*-
""" settings """

import os

import tensorflow as tf


# Folder to save images/tiles
SAVE_IMAGES_PATH = 'images'

AUTO = tf.data.experimental.AUTOTUNE

DATASET_IMGS_FOLDER_PATH = 'bladder_cancer'

DATASET_TILES_FOLDER_PATH = os.path.join(DATASET_IMGS_FOLDER_PATH, 'tiles')

DATASET_TFRECORDS = os.listdir(DATASET_TILES_FOLDER_PATH)

DATASET_TFRECORDS_PATHS = list(map(lambda record: os.path.join(DATASET_TILES_FOLDER_PATH, record), DATASET_TFRECORDS))

DATASET_IMGS = list(filter(lambda obj: obj.endswith('.svs'), os.listdir(DATASET_IMGS_FOLDER_PATH)))

TILES = list(filter(lambda obj: obj.endswith('.TFRecord'), os.listdir(DATASET_TILES_FOLDER_PATH)))

# Maximum number of images to write before giving a break to finish the writing process
MAXIMUM_WRITING_IMAGES = 20

# number of secondS to wait while saving a certain amount of images
SLEEP_WHILE_WRITING = 10
