from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings('ignore', '.*issubdtype.*',) #just to filter a h5py warning message with numpy 1.14.3.

import sys
import os
import io
import time
import scipy.misc
import numpy as np
import tensorflow as tf
from flask import jsonify, send_file

from models import models_factory
from models import preprocessing

from PIL import Image

slim = tf.contrib.slim

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def serve_pil_image(pil_img):
    img_io = io.BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

def imsave(filename, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)
    
checkpoint_dir = tf.train.latest_checkpoint("AvatarNet")

tf.logging.set_verbosity(tf.logging.INFO)
with tf.Graph().as_default():
    style_model, options = models_factory.get_model("AvatarNet_config.yml")

    # predict the stylized image
    inp_content_image = tf.placeholder(tf.float32, shape=(None, None, 3))
    inp_style_image = tf.placeholder(tf.float32, shape=(None, None, 3))

    # preprocess the content and style images
    content_image = preprocessing.mean_image_subtraction(inp_content_image)
    content_image = tf.expand_dims(content_image, axis=0)
    # style resizing and cropping
    style_image = preprocessing.preprocessing_image(
        inp_style_image,
        448,
        448,
        style_model.style_size)
    style_image = tf.expand_dims(style_image, axis=0)

    # style transfer
    stylized_image = style_model.transfer_styles(
        content_image,
        style_image,
        inter_weight=0.5)
    stylized_image = tf.squeeze(stylized_image, axis=0)

    # gather the test image filenames and style image filenames
    style_image_filename = os.environ['STYLE_PATH']

    # starting inference of the images
    init_fn = slim.assign_from_checkpoint_fn(
      checkpoint_dir, slim.get_model_variables(), ignore_missing_vars=True)
    
    sess = tf.Session()
    
    # initialize the graph
    init_fn(sess)

    # style transfer for each image based on one style image
    # gather the storage folder for the style transfer
    style_label = style_image_filename.split('/')[-1]
    style_label = style_label.split('.')[0]

    # get the style image
    np_style_image = load_image_into_numpy_array(Image.open(style_image_filename))

    def run_inference_for_single_image(np_content_image):
        print('Starting transferring the style of [%s]' % style_label)
        nn = 0.0
        total_time = 0.0

        start_time = time.time()
        np_stylized_image = sess.run(stylized_image,
                                     feed_dict={inp_content_image: np_content_image,
                                                inp_style_image: np_style_image})
        incre_time = time.time() - start_time
        nn += 1.0
        total_time += incre_time
        print("---%s seconds ---" % (total_time/nn))

        print('Style [%s]: Finish transfer the image' % (style_label))
        return Image.fromarray(np.clip(np_stylized_image, 0, 255).astype(np.uint8))

def handle(req):
    """handle a request to the function
    Args:
        req (str): request body
    """
    try:
        image = Image.open(io.BytesIO(req))
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        pil_image = run_inference_for_single_image(image_np)
    except Exception as e:
        print("Error: ", e)
        return "Error: " + str(e)
    return serve_pil_image(pil_image)
