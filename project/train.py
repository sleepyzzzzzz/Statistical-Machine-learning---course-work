import os
import PIL
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def img_gen_rgb():
	rgb_path = os.path.join('train', 'images', 'rgb')
	for fname in os.listdir(rgb_path):
		print (os.path.join(rgb_path, fname))
		yield mpimg.imread(os.path.join(rgb_path, fname))

def img_gen_nir():
	nir_path = os.path.join('train', 'images', 'nir')
	for fname in os.listdir(nir_path):
		print (os.path.join(nir_path, fname))
		yield mpimg.imread(os.path.join(nir_path, fname))

def img_gen_boundaries():
	bou_path = os.path.join('train', 'boundaries')
	for fname in os.listdir(bou_path):
		print (os.path.join(bou_path, fname))
		yield mpimg.imread(os.path.join(bou_path, fname))

def img_gen_masks():
	mask_path = os.path.join('train', 'masks')
	for fname in os.listdir(mask_path):
		print (os.path.join(mask_path, fname))
		yield mpimg.imread(os.path.join(mask_path, fname))

dataset_rgb = tf.data.Dataset.from_generator(img_gen_rgb, output_types = (tf.float64), output_shapes=tf.TensorShape((512, 512, 3)))
dataset_nir = tf.data.Dataset.from_generator(img_gen_nir, output_types = (tf.float64), output_shapes=tf.TensorShape((512, 512)))
dataset_bou = tf.data.Dataset.from_generator(img_gen_boundaries, output_types = (tf.float64), output_shapes=tf.TensorShape((512, 512)))
dataset_mask = tf.data.Dataset.from_generator(img_gen_masks, output_types = (tf.float64), output_shapes=tf.TensorShape((512, 512)))

data_rgb = np.array(list(dataset_rgb.take(500).as_numpy_iterator()))
data_nir = np.array(list(dataset_nir.take(3).as_numpy_iterator())).reshape((512, 512, 3))
data_bou = np.array(list(dataset_bou.take(3).as_numpy_iterator())).reshape((512, 512, 3))
data_mask = np.array(list(dataset_mask.take(3).as_numpy_iterator())).reshape((512, 512, 3))

print (data_rgb)