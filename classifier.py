import os
import sys
import time
import glob
import logging
import warnings

import re

import numpy as np
import primfilters as pr

#import tifffile as tiff
# image read and write - takes tiff stacks, png, jpg
from skimage import io

# for resize the images:
#from skimage._shared.utils import warn
#from scipy import ndimage as ndi
#from skimage.transform._geometric import AffineTransform
#from skimage.transform import warp

# for resizing:
from skimage.transform import resize
from skimage.transform import rescale
from PIL import Image

from scipy.ndimage import shift

# for tests:
#from collections import Counter

# to save classifier (might not load in different versions):
import pickle

# r/w yaml
from ruamel.yaml import YAML

#import argparse
import argparse

#import importlib
#importlib.reload(pr) # in case changes were made in the primfilters library - reload.

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score, f1_score

from skimage.morphology import remove_small_objects
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.measurements import label
"""
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('poster')
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')
"""
timestr = time.strftime("%Y%m%d-%H%M%S")

warnings.simplefilter("ignore")

##################################

import inspect
import hashlib
import gzip
import functools

class Store(object):
	def __init__(self, path):
		self.path = path
		os.makedirs(self.path, exist_ok=True)

	def _get_path_for_key(self, key):
		return os.path.join(self.path, f'{key}.pkl.gz')

	def put(self, key, value):
		with gzip.open(self._get_path_for_key(key), 'wb') as f:
			pickle.dump(value, f, protocol=4)

	def get(self, key):
		try:
			with gzip.open(self._get_path_for_key(key), 'rb') as f:
				return pickle.load(f)
		except:
			os.remove(self._get_path_for_key(key))
			raise "Error occured"

	def __contains__(self, key):
		return os.path.isfile(self._get_path_for_key(key))

class memoize():
	def __init__(self, store_dir=os.path.join(os.getcwd(), '.memoize')):
		self.store = Store(store_dir)

	def __call__(self, func):
		def wrapped_f(*args, **kwargs):
			key = self._get_key(func, *args, **kwargs)
			if key not in self.store:
				val = func(*args, **kwargs)
				self.store.put(key, val)
			return self.store.get(key)
		functools.update_wrapper(wrapped_f, func)
		return wrapped_f


	def _arg_hash(self, *args, **kwargs):
		_str = pickle.dumps(args, 2) + pickle.dumps(kwargs, 2)
		return hashlib.md5(_str).hexdigest()

	def _src_hash(self, func):
		_src = inspect.getsource(func)
		return hashlib.md5(_src.encode()).hexdigest()

	def _get_key(self, func, *args, **kwargs):
		arg = self._arg_hash(*args, **kwargs)
		src = self._src_hash(func)
		return src + '_' + arg


##################################

def setup_logger(classifier_folder, params_file_suff):

	# Handle loggings:
	
	#set different formats for logging output
	file_name = os.path.join(classifier_folder,f'classifier_{params_file_suff}_{timestr}.log')
	console_logging_format = '%(levelname)s: %(pathname)s:%(lineno)s %(message)s'
	file_logging_format = '%(levelname)s: %(asctime)s: %(pathname)s:%(lineno)s %(message)s'
	
	# configure logger
	logging.basicConfig(level=logging.INFO, format=file_logging_format, filename=file_name)
	
	logger = logging.getLogger()
	# create a file handler for output file
	handler = logging.StreamHandler()
	# set the logging level for log file
	handler.setLevel(logging.WARNING)

	# create a logging format
	formatter = logging.Formatter(console_logging_format)
	handler.setFormatter(formatter)
	handler.setLevel(logging.ERROR)
	# add the handlers to the logger
	logger.addHandler(handler)

##################################

# Function to time each function:
def timeit(method):
	@functools.wraps(method)
	def timed(*args, **kw):
		ts = time.time()
		result = method(*args, **kw)
		te = time.time()
		if 'log_time' in kw:
			name = kw.get('log_name', method.__name__.upper())
			kw['log_time'][name] = int(te - ts)
		else:
			logging.info('%r  %2.2f s' % (method.__name__, (te - ts)))
		return result
	return timed
###################################
"""
def resize_downsample(image, output_shape, order=1, mode=None, cval=0, clip=True,
		   preserve_range=False, anti_aliasing=True, anti_aliasing_sigma=None):
	


	Resize image to match a certain size.
	Performs interpolation to up-size or down-size images. Note that anti-
	aliasing should be enabled when down-sizing images to avoid aliasing
	artifacts. For down-sampling N-dimensional images with an integer factor
	also see `skimage.transform.downscale_local_mean`.
	Parameters
	
	This code was copied from: 
	https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_warps.py#L34
	Commit Hush: 94b561e77aa551fa91c52d9140af220885e5181e
	Because anti-aliasing parameter in resize function was only introduced in skimage 0.15 which is still a dev version.
	Once 0.15 will become an oficial version and will be updated here, this function can be deleted from the code and just imported.
	I did not use downscale_local_mean because it only allows for downscaling parameter of int (no float) - output size might be very different than user requested.
	----------
	image : ndarray
		Input image.
	output_shape : tuple or ndarray
		Size of the generated output image `(rows, cols[, ...][, dim])`. If
		`dim` is not provided, the number of channels is preserved. In case the
		number of input channels does not equal the number of output channels a
		n-dimensional interpolation is applied.
	Returns
	-------
	resized : ndarray
		Resized version of the input.
	Other parameters
	----------------
	order : int, optional
		The order of the spline interpolation, default is 1. The order has to
		be in the range 0-5. See `skimage.transform.warp` for detail.
	mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
		Points outside the boundaries of the input are filled according
		to the given mode.  Modes match the behaviour of `numpy.pad`.  The
		default mode is 'constant'.
	cval : float, optional
		Used in conjunction with mode 'constant', the value outside
		the image boundaries.
	clip : bool, optional
		Whether to clip the output to the range of values of the input image.
		This is enabled by default, since higher order interpolation may
		produce values outside the given input range.
	preserve_range : bool, optional
		Whether to keep the original range of values. Otherwise, the input
		image is converted according to the conventions of `img_as_float`.
	anti_aliasing : bool, optional
		Whether to apply a Gaussian filter to smooth the image prior to
		down-scaling. It is crucial to filter when down-sampling the image to
		avoid aliasing artifacts.
	anti_aliasing_sigma : {float, tuple of floats}, optional
		Standard deviation for Gaussian filtering to avoid aliasing artifacts.
		By default, this value is chosen as (1 - s) / 2 where s is the
		down-scaling factor.
	Notes
	-----
	Modes 'reflect' and 'symmetric' are similar, but differ in whether the edge
	pixels are duplicated during the reflection.  As an example, if an array
	has values [0, 1, 2] and was padded to the right by four values using
	symmetric, the result would be [0, 1, 2, 2, 1, 0, 0], while for reflect it
	would be [0, 1, 2, 1, 0, 1, 2].
	Examples
	--------
	>>> from skimage import data
	>>> from skimage.transform import resize
	>>> image = data.camera()
	>>> resize(image, (100, 100), mode='reflect').shape
	(100, 100)
	



	if mode is None:
		mode = 'constant'

	output_shape = tuple(output_shape)
	output_ndim = len(output_shape)
	input_shape = image.shape
	if output_ndim > image.ndim:
		# append dimensions to input_shape
		input_shape = input_shape + (1, ) * (output_ndim - image.ndim)
		image = np.reshape(image, input_shape)
	elif output_ndim == image.ndim - 1:
		# multichannel case: append shape of last axis
		output_shape = output_shape + (image.shape[-1], )
	elif output_ndim < image.ndim - 1:
		raise ValueError("len(output_shape) cannot be smaller than the image "
						 "dimensions")

	factors = (np.asarray(input_shape, dtype=float) /
			   np.asarray(output_shape, dtype=float))

	if anti_aliasing_sigma is None:
		anti_aliasing_sigma = np.maximum(0, (factors - 1) / 2)
	else:
		anti_aliasing_sigma =             np.atleast_1d(anti_aliasing_sigma) * np.ones_like(factors)
		if np.any(anti_aliasing_sigma < 0):
			raise ValueError("Anti-aliasing standard deviation must be "
							 "greater than or equal to zero")

	image = ndi.gaussian_filter(image, anti_aliasing_sigma,
								cval=cval, mode=mode)

	# 2-dimensional interpolation
	if len(output_shape) == 2 or (len(output_shape) == 3 and
								  output_shape[2] == input_shape[2]):
		rows = output_shape[0]
		cols = output_shape[1]
		input_rows = input_shape[0]
		input_cols = input_shape[1]
		if rows == 1 and cols == 1:
			tform = AffineTransform(translation=(input_cols / 2.0 - 0.5,
												 input_rows / 2.0 - 0.5))
		else:
			# 3 control points necessary to estimate exact AffineTransform
			src_corners = np.array([[1, 1], [1, rows], [cols, rows]]) - 1
			dst_corners = np.zeros(src_corners.shape, dtype=np.double)
			# take into account that 0th pixel is at position (0.5, 0.5)
			dst_corners[:, 0] = factors[1] * (src_corners[:, 0] + 0.5) - 0.5
			dst_corners[:, 1] = factors[0] * (src_corners[:, 1] + 0.5) - 0.5

			tform = AffineTransform()
			tform.estimate(src_corners, dst_corners)

		out = warp(image, tform, output_shape=output_shape, order=order,
				   mode=mode, cval=cval, clip=clip,
				   preserve_range=preserve_range)

	else:  # n-dimensional interpolation
		coord_arrays = [factors[i] * (np.arange(d) + 0.5) - 0.5
						for i, d in enumerate(output_shape)]

		coord_map = np.array(np.meshgrid(*coord_arrays,
										 sparse=False,
										 indexing='ij'))

		image = convert_to_float(image, preserve_range)

		ndi_mode = _to_ndimage_mode(mode)
		out = ndi.map_coordinates(image, coord_map, order=order,
								  mode=ndi_mode, cval=cval)

		_clip_warp_output(image, out, order, mode, cval, clip)

	return out
"""
###################################
"""
def show_image(im):
	plt.figure(figsize=(20,10))
	plt.imshow(im)
	plt.xticks([])
	plt.yticks([])
	sns.despine(bottom=True, left=True)
"""
###################################
def get_args(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--folder_path')
	parser.add_argument('-p', '--params_file_suff')

	args = parser.parse_args()
	folder_path = args.folder_path

	params_file_suff = args.params_file_suff
	if params_file_suff is None:
		params_file_suff = ""
	return folder_path, params_file_suff

###################################

def load_and_validate_params_from_yaml(folder_path, params_file_suff=""):

	yml_file_path = os.path.join(folder_path, 'classifier', f'parameters{params_file_suff}.yaml')
	
	# Check if YAML file exists, if not throw error:
	if not os.path.isfile(yml_file_path):
		raise Exception(f'YAML parameters file does not exist in the classifier folder. Missing file: {yml_file_path}')
	
	# Load parameters from YAML:
	yaml=YAML(typ='safe')   # default, if not specfied, is 'rt' (round-trip)
	with open(yml_file_path) as yaml_file:
		user_params = yaml.load(yaml_file)

	# State expected parameters to compare with the file:
	keys_user_params, types_user_params = list(zip(*[
		['ims_file_prefix', str],
		['channel_num', (int,bool)],
		['c1_invert_values', bool],
		['training_images', list],
		['is_multiclass', bool],
		['n_cascading', int],
		['shift_window_size', int],
		['ims_folder', str],
		['labels_folder', str],
		['relevance_masks_folder', (str,bool)],
		['additional_ims_folder', (str,bool)],
		['output_folder', str],
		['is_regression', bool],
		['z_size', (float,int)],
		['dim_order', (str,bool)],
		['n_trees', int],
		['n_cpu', int],
		['bigger_dim_output_size', list],
		['downsample_by', list],
		['min_f1_score', float],
		['what_to_run', str],
		['save_classifier_add_to_name', str],
		['load_classifier_file_name', (str, bool)],
		['output_probability_map', bool],
		['is_upsample', bool],
		['output_remove_small_objs', int],
		['output_fill_holes', bool],
		['binary_output_value', int],
		['filters_params', dict]
		
	]))
		
	# Verify that all the parameters exist in the file and that the type matches:
	for i,k in enumerate(keys_user_params):
		if k not in user_params:
			raise Exception(f'Expected parameter key name in YAML file:"{k}". Please fix the YAML file.') 
		if not isinstance(user_params[k],types_user_params[i]):
			raise Exception(f'YAML parameter "{k}" has to be of type {types_user_params[i]}. Please fix the variable in the YAML file.')

	# Important error check in case all params will be imported to the local variables.
	# e.g. If user entered in the YAML a parameter name that will conflict with a variable in the script.
	if len(user_params)>len(keys_user_params):
		raise Exception(f'expected exactly {len(keys_user_params)} parameters in YAML file. Please fix the YAMl file')
		
	# Verify : user_params['training_images'] list elements are strings:
	if not all(isinstance(x,str) for x in user_params['training_images']):
		raise Exception('YAML parameter "training_images" list should only include strings. Please fix the variable in the YAML file.')
	# Verify : user_params['z_size'] is in range:
	if not 1>=user_params['z_size']>=0:
		raise Exception('YAML parameter "z_size" has to be 0<=z<=1. Please fix the value in the YAML file.')
	# Verify : user_params['n_cpu'] within possible range of number of CPUs:
	if not 32>=user_params['n_cpu']>0:
		raise Exception('YAML parameter "n_cpu" has to be 0<n<=32. Please fix the value in the YAML file.')



# Verify : user_params['downsample_by'] and user_params['output_size']:\n",
	if user_params['downsample_by']:
		if not all(isinstance(x,(int,float)) for x in user_params['downsample_by']):
			raise Exception('All elements in YAML list parameter \"downsample_by\" must be float/int.')
			if any(x<1 for x in user_params['downsample_by']):
				raise Exeption('All elements in YAML list parameter \"downsample_by\" must be bigger than 1')
		if user_params['bigger_dim_output_size']:
			print('WARNING: both downsample_by and bigger_dim_output_size not empty in YAML file. Image will be resized by downsample_by only.')
		user_params['downsample_by'].sort(reverse=True)
		user_params['bigger_dim_output_size'] = len(user_params['downsample_by'])*[0]
	elif user_params['bigger_dim_output_size']:
		if not all(isinstance(x,int) for x in user_params['bigger_dim_output_size']):
			raise Exception('All elements in YAML list parameter "bigger_dim_output_size" must be int.')
			if any(x<8 for x in user_params['bigger_dim_output_size']):
				raise Exeption('All elements in YAML list parameter "bigger_dim_output_size" must be bigger than 8')
		user_params['bigger_dim_output_size'].sort()
		user_params['downsample_by'] = len(user_params['bigger_dim_output_size'])*[0]
	else:
		user_params['downsample_by'] = [1]
		user_params['bigger_dim_output_size'] = [0]


	### NEW CODE ###
	# Verify that number of cascading is between 0-3:
	if 0>user_params['n_cascading']>=3:
		raise Exception('YAML parameter n_cascading must between 1 and 3, please fix YAML file')
	if user_params['n_cascading']>1 and user_params['what_to_run'] not in ['both','Both','b','B']:
		raise Exception('If running cascading random forest, you must run both classify and predict, please fix either n_cascading or what_to_run in YAML file')

	if 0>user_params['shift_window_size']>=20:
		raise Exception('YAML parameter shift_window_size must between 1 and 20, please fix YAML file')
		
	# Verify user_params['min_f1_score']:
	if not 0.6<user_params['min_f1_score']<1:
		raise Exception('YAML min f1 score must be 0.6<f1<1. Please fix the YAML file.')

	# Verify user_params['filters_params']:
	filters_params = user_params['filters_params']
	# Define the type and range in the filter parameters - If list, define that it's list, the type in it, and the range.
	types_filters_params = {'gauss_sigma_range':[list,int,range(1,21)],'DoG_couples':[list,list,int],
							'window_size_range':[list,int,range(2,11)],'aniso_diffus_n_iter':[int,range(1,51)],'aniso_diffus_conduc_coef':[int,range(20,101)],
							'aniso_diffus_gamma':[float,np.arange(0,0.26,0.01)],'aniso_diffus_method':[int,range(1,3)],'gabor_freq':[list,float,(0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45)],'gabor_theta_range':[list,int,range(4)],
							'frangi_scale_range':[list,list,int],'entropy_radius_range':[list,int,range(2,11)]}
	for f in filters_params:
		if f not in types_filters_params:
			raise Exception(f'YAML filters_params "{f}" is unexpected (maybe misspell?). Please fix the YAMl file.')
		if not isinstance(filters_params[f],types_filters_params[f][0]):
			raise Exception(f'YAML filters_params "{f}" should be of type {types_filters_params[f][0]}. Please fix the YAMl file.')
		if types_filters_params[f][0]==list:
			if not all(isinstance(x,types_filters_params[f][1]) for x in filters_params[f]):
				raise Exception(f'YAML filters_params "{f}" list should only hold elements of type {types_filters_params[f][1]}. Please fix the YAMl file.')
			if types_filters_params[f][1]==list:
				if not all(len(x)==2 for x in filters_params[f]):
					raise Exception(f'YAML filters_params "{f}" internal lists should hold couples (two values exactly). Please fix the YAMl file.')
				if not all(isinstance(y,types_filters_params[f][2]) for x in filters_params[f] for y in x):
					raise Exception(f'YAML filters_params "{f}" lists should only hold elements of type {types_filters_params[f][2]}. Please fix the YAMl file.')
			elif not all(x in types_filters_params[f][2] for x in filters_params[f]):
				logging.warning(filters_params[f])
				raise Exception(f'YAML filters_params "{f}" list elements should all by in {types_filters_params[f][2]}. Please fix the YAMl file.')
		elif filters_params[f] not in types_filters_params[f][1]:
			logging.warning(filters_params[f])
			raise Exception(f'YAML filters_params "{f}" should all by in {types_filters_params[f][1]}. Please fix the YAMl file.')


	# Set "static" parameters:
	user_params["project_folder"] = folder_path
	user_params["data_folder"] = os.path.join(user_params["project_folder"],'data')
	user_params["classifier_folder"] = os.path.join(user_params["project_folder"],'classifier')
	user_params["temp_folder"] = os.path.join(user_params["classifier_folder"],'temp_files')
	user_params["ims_folder"] = os.path.join(user_params["data_folder"],user_params["ims_folder"])
	user_params["labels_folder"] = os.path.join(user_params["data_folder"],user_params["labels_folder"])
	if user_params["relevance_masks_folder"]:
		user_params["relevance_masks_folder"] = os.path.join(user_params["data_folder"], user_params["relevance_masks_folder"])
	if user_params["additional_ims_folder"]:
		user_params["additional_ims_folder"] = os.path.join(user_params["data_folder"], user_params["additional_ims_folder"])
	user_params["output_folder"] = os.path.join(user_params["data_folder"], user_params["output_folder"])
	###MAYBE Change exist ok:
	for i in range(user_params["n_cascading"]+1):
		os.makedirs(f'{user_params["output_folder"]}{i}', exist_ok=True)

	# Set up cache folder:
	os.makedirs(user_params["temp_folder"], exist_ok=True)
	
	# Reading images file names:
	user_params["all_imgs_files"] = [f for f in sorted(os.listdir(user_params["ims_folder"])) if user_params["ims_file_prefix"] in f]
	user_params["all_files_timestamp"] = [os.path.getmtime(os.path.join(user_params["ims_folder"],f)) for f in user_params["all_imgs_files"]]
	
	user_params["training_files"] = [aif for ti in user_params["training_images"] for aif in user_params["all_imgs_files"] if ti in aif] 
	if len(user_params["training_files"])<len(user_params["training_images"]):
		print('need to uncomment it after testing')
		#print(list(zip(*[p.training_files, p.training_images])))
		#raise Exception('Not all training images requested are in images dir. Please fix YAML / add files to folder')
	user_params["training_files_timestamp"] = [[os.path.getmtime(os.path.join(user_params["ims_folder"],t)),
								   os.path.getmtime(os.path.join(user_params["labels_folder"],t))] for t in user_params["training_files"]]
	if user_params["relevance_masks_folder"]:
		user_params["training_files_timestamp"] = [user_params["training_files_timestamp"]] + [os.path.getmtime(os.path.join(user_params['relevance_masks_folder'],t))
								for t in user_params["training_files"]]
	if user_params["additional_ims_folder"]:
		user_params["training_files_timestamp"] = [user_params["training_files_timestamp"]] + [os.path.getmtime(f) for f in glob.glob(os.path.join(user_params["additional_ims_folder"],'*.*'))]

	return user_params
	
###################################
# Load image:
@timeit
def load_and_resize_image(path, big_dim_output_size, downsample_by, channel_num, dim_order, is_multiclass=False, is_mask=False):
	
	logging.info(path)
	im = (io.imread(path))
	
	# if not already - make the first dimension the z axis value
	if (im.ndim==3) and dim_order and (dim_order not in ['zxy','zyx','cxy','cyx']):
		im = np.swapaxes(im,0,2)
		im = np.swapaxes(im,1,2)
		
	if channel_num and is_mask==False:
		if im.ndim==3:
			# Channel_num values start at 1, python starts at 0:
			im = im[channel_num]
		if im.ndim==4:
			im = im[:,channel_num,:,:]
	
	org_im_shape = im.shape 
	
	if is_mask:
		im = np.absolute(im).astype(np.uint8)
	else:
		im_max, im_min = im.max(), im.min()
		im = (im - im_min)/(im_max - im_min)

	# resize the image:

	if downsample_by and downsample_by>=1:
		big_dim_output_size = int(max(im.shape[-2:])/downsample_by)

	# If image is 3D - only axis x,y will be downsampled:
	if big_dim_output_size:

		# Check that requested output size is smaller than input:
		if big_dim_output_size > max(im.shape[-2:]):
			return im, False, True

		output_size = [big_dim_output_size, int(min(im.shape[-2:])/(max(im.shape[-2:])/big_dim_output_size))]
		if im.shape[-1]>im.shape[-2]:
			output_size = output_size[::-1]

		if im.ndim==2:
			if is_multiclass:
				im = np.array(Image.fromarray(im).resize(output_size, Image.NEAREST))
			else:
				im = resize(im,output_size, anti_aliasing=True)
		else:
			out_im = np.zeros([im.shape[0]] + output_size)
			for i,slic in enumerate(im):
				if is_multiclass:
					out_im[i] = np.array(Image.fromarray(slic).resize(output_size, Image.NEAREST))
				else:
					out_im[i] = resize(slic, output_size, anti_aliasing=True)
			im = out_im

	if is_mask and not is_multiclass:
		im = np.where(im>np.max(im)/2, 255, 0).astype(np.uint8)

		#old:
		#im = ((im > (np.max(im)/2))*255).astype(np.uint8)
		#older:
		#im[im==np.max(im)] = max_val
		#im[im!=np.max(im)] = 0

	return im, org_im_shape, False


# @memoize(store_dir=os.path.join(p.classifier_folder,f'os{p.output_size[l]}' if p.output_size[0] else f'ds{p.downsample_by[l]}'))
@timeit
def generate_all_filters(loop_num, file_name, file_timestamp, cascading_num, ims_folder, bigger_dim_output_size, downsample_by, channel_num, dim_order, filters_params, z_size, output_folder, shift_window_size):

	"""
	# All filters are - gaussian, sobel, prewitt, hessian, DoG, 
	# minimum, maximum, median, mean, variance,
	# anisotropic diffusion 2D, anisotropic diffusion 3D, 
	# bilateral, gabor, laplace, frangi (similar to structure), entrop (entropy)
	"""
	
	im, org_im_shape, is_output_big = load_and_resize_image(os.path.join(ims_folder,file_name), bigger_dim_output_size[loop_num], downsample_by[loop_num], channel_num, dim_order)
	
	if is_output_big:
		return False, 0, 0
	
	logging.info(f'image: {file_name}, shape: {im.shape}')
	
	if im.ndim==3:
		gauss_sigma_range = [[sig*z_size,sig,sig] for sig in filters_params['gauss_sigma_range']]
		DoG_couples = [np.swapaxes([i]*3,0,1)*[z_size,1,1] for i in filters_params['DoG_couples']]
	else:
		gauss_sigma_range = [[sig, sig] for sig in filters_params['gauss_sigma_range']]
		DoG_couples = [np.swapaxes([i]*2,0,1)*[1,1] for i in filters_params['DoG_couples']]
	
	# generate filters:
	
	filters = []
	
	logging.info('starting gauss')
	filters.extend([pr.gaussian(im, sig) for sig in gauss_sigma_range])
	
	gauss_filters = filters[:]
	
	logging.info('starting sobel')
	filters.extend([np.float32(pr.sobel(gauss_filters[i])) for i in range(len(gauss_filters))])
	
	logging.info('starting prewitt')
	filters.extend([np.float32(pr.prewitt(gauss_filters[i])) for i in range(len(gauss_filters))])    
	
	logging.info('starting hessian')
	filters.extend([np.float32(pr.hessian(gauss_filters[i])) for i in range(len(gauss_filters))])
	
	if im.ndim==3:
		logging.info('starting hessian zx')
		filters.extend([np.float32(np.swapaxes(pr.hessian(np.swapaxes(gauss_filters[i],0,1)),0,1)) 
						   for i in range(len(gauss_filters))])

		logging.info('starting hessianzy')
		filters.extend([np.float32(np.swapaxes(pr.hessian(np.swapaxes(gauss_filters[i],0,2)),0,2)) 
						   for i in range(len(gauss_filters))])
		
		logging.info('starting frangiyx')
		filters.extend([np.float32(np.swapaxes(pr.frangi(np.swapaxes(im,0,1), sig_range),0,1)) for sig_range in filters_params['frangi_scale_range']])

	logging.info('starting DoG')
	filters.extend([pr.difference_of_gaussians(im, dog) for dog in DoG_couples])

	# generate anisotropic diffusion filter
	# NEED TO IMPROVE THIS!
	logging.info('starting anisotropic_diffusion')
	filters.append(pr.anisotropic_diffusion(im))

	# generate bilateral filters
	logging.info('starting bilateral')
	filters.extend([np.float32(pr.bilateral(im, win_siz)) for win_siz in filters_params['window_size_range']])

	# generate minimum, maximum, median, mean, varience filters
	logging.info('starting min, max, med, mean, varience')
	for win_siz in filters_params['window_size_range']:
		filters.append(pr.minimum(im, win_siz))
		filters.append(pr.maximum(im, win_siz))
		filters.append(pr.median(im, win_siz))
		filters.append(pr.mean(im, win_siz))
		filters.append(pr.varience(im, win_siz))
	
	# generate gabor filter
	filter_name = 'gabor{g_list}thetas{t_list}'.format(g_list=''.join(str(i) for i in filters_params['gabor_freq']),t_list=''.join(str(i) for i in filters_params['gabor_theta_range']))
	logging.info(f'starting {filter_name}')
	for f in filters_params['gabor_freq']: 
		for t in filters_params['gabor_theta_range']:
			filters.append(pr.gabor(im, f, t/4.*(np.pi)))

	# generate frangi filter
	logging.info('starting frangi')
	filters.extend([np.float32(pr.frangi(im, sig_range)) for sig_range in filters_params['frangi_scale_range']])

	# generate entropy filter
	logging.info('starting entropy (with varying radiuses)')
	filters.extend([np.float32(pr.entrop(im, r)) for r in filters_params['entropy_radius_range']])

	### NEW CODE ###
	if cascading_num>0:
		### MAYBE SHOULD BE GIVEN TO FUNC:
		outpath = os.path.abspath(os.path.join(ims_folder,"..",f'{output_folder}{cascading_num-1}',file_name))
		output_im = load_and_resize_image(outpath, bigger_dim_output_size[loop_num], downsample_by[loop_num], 0, dim_order)[0]
		filters.extend(generate_extra_filters_for_cascading(output_im, shift_window_size))

	return filters, org_im_shape, im.shape

# If cascading random forest was chosen, and this is not the first RF in the flow, generate more filters:
def generate_extra_filters_for_cascading(output_im, shift_window_size):

	window_range = range(-shift_window_size, shift_window_size+1)

	if output_im.ndim==2:
		filters = [shift(output_im,[wi,wj]) for wi in window_range for wj in window_range]
	else:
		filters = [shift(output_im,[wi,wj,wk]) for wi in window_range for wj in window_range for wk in window_range]

	return filters


# setting up the classifier matrices:
# X - all filters, Y - answers
@timeit
def train_classifier(cascading_num, loop_num, last_f1_score, dict_generate_all_filters, training_files, labels_folder, bigger_dim_output_size, downsample_by, dim_order, relevance_masks_folder, 
	training_files_timestamp, additional_ims_folder, is_regression, n_trees, n_cpu, min_f1_score, temp_folder, is_multiclass):
	"""
	parameters:
 
	"""
	logging.info('Preparing  metrics to classifier')
	
	# Redefining the function, as the memoize decorator needs parameters - p:
	memoized_generate_all_filters = memoize(store_dir=temp_folder)(generate_all_filters)

	# Load label images:
	classes_ims = []
	relevance_ims = []
	for t in training_files:
		classes_ims.append(load_and_resize_image(os.path.join(labels_folder,t), bigger_dim_output_size[loop_num], downsample_by[loop_num], False, dim_order, is_multiclass, True)[0])
		if relevance_masks_folder: 
			relevance_ims.append(load_and_resize_image(os.path.join(relevance_masks_folder,t), bigger_dim_output_size[loop_num], downsample_by[loop_num], False, dim_order, False, True)[0])
			
	
	if not relevance_masks_folder:
		relevance_ims = [np.ones(c.shape)*255 for c in classes_ims]

	# Original images filters array:
	X = np.array([])
	# Classes array:
	c = np.array([])
	# Relevance array:
	r = np.array([])

	# Stack filters file and label images:
	for i,t in enumerate(training_files):
		### NEW CODE ### n_cascading
		temp_i = (memoized_generate_all_filters(loop_num, t, training_files_timestamp[i], cascading_num, **dict_generate_all_filters))[0]
		if additional_ims_folder:
			temp_i.extend((memoized_generate_all_filters(loop_num, t, training_files_timestamp[i], cascading_num, **dict_generate_all_filters))[0])
		
		# If requested output size bigger than image:
		if not temp_i:
			if loop_num!=0:
				logging.warning('Requested output size is bigger than original image. saving/using classifier from previous iteration although min_f1_score is not met.')
				return 0, 0, 0
			else:
				raise Exception('Requested output size is bigger than original image. Please decrese bigger_dim_output_size in user_params.')
		
		temp_i = np.vstack([f.flatten() for f in temp_i]).T
		X = np.vstack((X,temp_i)) if X.size else temp_i
		c = np.hstack((c, classes_ims[i].flatten())) if c.size else classes_ims[i].flatten()
		r = np.hstack((r, relevance_ims[i].flatten())) if r.size else relevance_ims[i].flatten()
		
	if c.shape[0]!=X.shape[0] or r.shape[0]!=X.shape[0]:
		logging.info(f'X shape {X.shape}')
		logging.info(f'c shape {c.shape}')
		logging.info(f'r shape {r.shape}')
		raise Exception('label images size must be equal to original images size')

	X = np.vstack([x.flatten() for x in X])
	y = np.array([c for c in c.flatten()])
	r = np.array([r for r in r.flatten()])
	
	X = X[r==255]
	y = y[r==255]

	skf = StratifiedKFold(n_splits=3)
	
	classification_reports = list()
	f1_scores = list()
	
	# In case the user asked for regression and not classification:
	if not is_regression:
		for train_ix, test_ix in skf.split(X, y): # for each of K folds
			# define training and test sets
			X_train, X_test = X[train_ix,:], X[test_ix,:]
			y_train, y_test = y[train_ix], y[test_ix]

			# Train classifier
			clf = RandomForestClassifier(n_estimators=n_trees, n_jobs=n_cpu) #(n_jobs=2) not sure if works on cluster..
			clf.fit(X_train, y_train)

			# Predict test set labels
			y_hat = clf.predict(X_test)
			classification_reports.append(classification_report(y_test, y_hat))
			f1_scores.append(f1_score(y_test, y_hat, average=None))

		print(*classification_reports, sep='\n', flush=True)

		# Train classifier
		clf = RandomForestClassifier(n_estimators=n_trees, n_jobs=n_cpu) #(n_jobs=2) not sure if works on cluster..
		
	else:
		clf = RandomForestRegressor(n_estimators=n_trees, n_jobs=n_cpu)
		
	clf.fit(X, y)
	
	# If reached the performance goal:
	stop_loop_bool = True if (np.mean(f1_scores)>min_f1_score) else False
	# If f1 didn't improve from last loop num:
	stop_loop_bool = True if (stop_loop_bool or (np.mean(f1_scores)<=last_f1_score)) else False

	return clf, stop_loop_bool, np.mean(f1_scores)

@timeit
def save_classifier(clf, folder ,classifier_file_name):
	with open(os.path.join(folder, f'{classifier_file_name}'), 'wb') as f:
		pickle.dump(clf, f)

@timeit
def load_classifier(folder, classifier_file_name):
	with open(os.path.join(folder, classifier_file_name), 'rb') as f:
		return pickle.load(f)

# Apply classifier:
@timeit
def apply_classifier(clf, i_f, i, l, c, dict_generate_all_filters, all_files_timestamp, additional_ims_folder, output_probability_map, bigger_dim_output_size, downsample_by, is_upsample, dim_order, temp_folder):

	# Redefining the function, as the memoize decorator needs parameters - p:
	memoized_generate_all_filters = memoize(store_dir=temp_folder)(generate_all_filters)

	filters, org_im_shape, im_shape = memoized_generate_all_filters(l, i_f, all_files_timestamp[i], c, **dict_generate_all_filters)

	if additional_ims_folder:
		filters.extend((memoized_generate_all_filters(l, i_f, all_files_timestamp[i], c, **dict_generate_all_filters))[0])

	X_predict = np.vstack([f.flatten() for f in filters]).T

	if output_probability_map:
		yhat = clf.predict_proba(X_predict)
		
		predicted_im_downsampled = np.reshape(yhat[:,1], im_shape) if c1_invert_values else np.reshape(yhat[:,0], im_shape) 
		if is_upsample:
			out_im = resize(predicted_im_downsampled, org_im_shape).astype("float32")
		else:
			out_im = predicted_im_downsampled
	else:
		yhat = clf.predict(X_predict) 

		predicted_im_downsampled = np.reshape(yhat, im_shape)

		if is_upsample:
			out_im = resize(predicted_im_downsampled, org_im_shape, preserve_range=True).astype("uint8")
		else:
			out_im = predicted_im_downsampled
	
	return out_im

def post_process_output(out_im, i_f, c1_invert_values, output_remove_small_objs, output_fill_holes, relevance_masks_folder, output_probability_map, binary_output_value):
	if not output_probability_map:
		if c1_invert_values:
			out_im = (1-out_im)
		if output_fill_holes:
			out_im[:50,:]=0
			out_im[-50:,:]=0
			out_im[:,:50]=0
			out_im[:,-50:]=0
			out_im = binary_fill_holes(out_im) #structure=[[[1]*3]*3]*3)
		if output_remove_small_objs:
			out_im = out_im.astype(bool)
			out_im = remove_small_objects(out_im, output_remove_small_objs, connectivity=1)
		out_im = out_im.astype("uint8")
		if binary_output_value:
			out_im[out_im>0] = binary_output_value

	if relevance_masks_folder:
		#relevance_mask = load_and_resize_image(os.path.join(relevance_masks_folder,i_f), bigger_dim_output_size[l], downsample_by[l], False, dim_order, True)[0]
		#relevance_mask = resize(relevance_mask, org_im_shape).astype("uint8")
		relevance_mask = (io.imread(os.path.join(relevance_masks_folder,i_f))).astype("uint8")
		out_im[relevance_mask==0] = 0 

	return out_im

""" deleted function - because of multiclass - might add in future
def binarize_output(out_im):
	labeled_out, num_features = label(out_im)
	out_im = out_im>0.5
	out_im = remove_small_objects(out_im, 50, connectivity=1)
	return binary_fill_holes(out_im) #structure=[[[1]*3]*3]*3)
"""

#########################################################################
# MAIN

def main(folder_name, params_file_suff):

	# Load + set parameters from yaml:
	p = load_and_validate_params_from_yaml(folder_name, params_file_suff)

	# Redefining the function, as the memoize decorator needs parameters - p:
	#memoized_generate_all_filters = memoize(store_dir=p["temp_folder"])(generate_all_filters)
	memoized_train_classifier = memoize(store_dir=p["temp_folder"])(train_classifier)
	#memoized_apply_classifier = memoize(store_dir=p["temp_folder"])(apply_classifier)

	# Parameters for each function:
	p_train_classifier = ["training_files", "labels_folder", "bigger_dim_output_size", "downsample_by", "dim_order", "relevance_masks_folder", "training_files_timestamp", "additional_ims_folder", "is_regression", "n_trees", "n_cpu", "min_f1_score", "temp_folder", "is_multiclass"]
	p_generate_filters = ["ims_folder", "bigger_dim_output_size", "downsample_by", 'channel_num', "dim_order", "filters_params", "z_size", "output_folder", "shift_window_size"]
	p_apply_classifier = ["all_files_timestamp", "additional_ims_folder", "output_probability_map", "bigger_dim_output_size", 'downsample_by', "is_upsample", "dim_order", "temp_folder"]
	p_post_process_output = ["c1_invert_values", "output_remove_small_objs", "output_fill_holes",  "relevance_masks_folder", "output_probability_map", "binary_output_value"]

	dict_train_classifier = dict((k, p[k]) for k in p_train_classifier if k in p)
	dict_generate_all_filters = dict((k, p[k]) for k in p_generate_filters if k in p)
	dict_apply_classifier = dict((k, p[k]) for k in p_apply_classifier if k in p)
	dict_post_process_output = dict((k, p[k]) for k in p_post_process_output if k in p)

	setup_logger(p["classifier_folder"], params_file_suff)

	### NEW CODE ###
	for c in range(p["n_cascading"]+1):

		last_f1_score = 0

		if p["what_to_run"] in ['classify','both','Classify','Both','c', 'b','C','B']:
			# Loop through sizes of downsampling - for fast classification:
			for l in range(len(p["bigger_dim_output_size"])):

				print(f'loop number: {l}', flush=True)

				# Load images + labels then train classifier:
				clf, stop_loop_bool, last_f1_score = memoized_train_classifier(c, l, last_f1_score, dict_generate_all_filters, **dict_train_classifier)

				# If we reached the target performance or if f1 score didnt improve:
				if stop_loop_bool==True:
					break
				# 
				if not clf:
					l=l-1
					break

			if stop_loop_bool==False:
				logging.warning('min_f1_score was not met using the current settings.')

			print(f'used classifier downsample {p["downsample_by"][l]} and bigdimoutputsize {p["bigger_dim_output_size"][l]}', flush=True)
			clf = memoized_train_classifier(c, l, last_f1_score, dict_generate_all_filters, **dict_train_classifier)[0]
			classifier_name = f'bdos{p["bigger_dim_output_size"][l]}_dsb{p["downsample_by"][l]}_ntrees{p["n_trees"]}_{p["save_classifier_add_to_name"]}_{timestr}.pkl'

			save_classifier(clf, p["classifier_folder"], classifier_name)

		## If only predict was specified in YAML, then must log classifier
		if p["what_to_run"] in ['predict','Predict','p','P']:
			clf = load_classifier(p["classifier_folder"], p["load_classifier_file_name"])
			p["bigger_dim_output_size"] = [int((re.findall("bdos(\d+)", p["load_classifier_file_name"]))[0])]
			p["downsample_by"] = [int((re.findall("dsb(\d+)", p["load_classifier_file_name"]))[0])]
			l=0

		############# PREDICT ##############

		if p["what_to_run"] in ['predict','both','Predict','Both','p', 'b','P','B']:

			logging.info('Starting Prediction')

			out_folder = f'{p["output_folder"]}{c}'
			for i, im_f in enumerate(p["all_imgs_files"]):
				try:
					start = time.time()
					out_im = apply_classifier(clf, im_f, i, l, c, dict_generate_all_filters, **dict_apply_classifier)
					#out_im = post_process_output(out_im, im_f, **dict_post_process_output)
					end = time.time()
					logging.info(f'memoized_apply_classifier - {im_f} - TIME {end - start}')

					io.imsave(os.path.join(out_folder,f'{im_f}'), out_im)
				except Exception as e:
					exc_type, exc_obj, exc_tb = sys.exc_info()
					logging.info(f'{im_f} was not predicted due to error\n{e} {exc_type} {exc_tb.tb_lineno}')

if __name__ == '__main__':
	# Parse arguments:
	folder_path, params_file_suff = get_args()

	main(folder_path, params_file_suff)

