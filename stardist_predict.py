import os
from glob import glob
import numpy as np
import tifffile as tif
import pandas as pd
from shutil import rmtree

import logging
import sys

from skimage.transform import resize 

from stardist.models import StarDist2D
#from csbdeep.utils import Path, normalize
#from stardist import random_label_cmap, _draw_polygons

##########################  Define all paths ############################

dir_path = '../'

dir_path_maxp_gfp = os.path.join(dir_path, 'maxp_gfp')
csv_path = os.path.join(dir_path, 'embryos.csv')
predicted_npz_path = os.path.join(dir_path, 'predicted_masks_and_filenames')

pipeline_dir_path = os.path.join(dir_path, 'nd2totif_maskembryos_stagebin_pipeline')

log_file_path = os.path.join(dir_path, 'nd2totif_maskembryos_stagebin_pipeline', 'pipeline.log')

######################### Set up log file ###############################

def setup_logger():

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(lineno)s - %(message)s')
        handler.setFormatter(formatter)

        logger.addHandler(handler)

setup_logger()

logging.info("\n\nStarting script stardist_predict\n *********************************************")

############################ Create masks - stardist prediction ##############################

# Get all images for prediction:
gfp_images_paths = glob(os.path.join(dir_path_maxp_gfp,"*"))
gfp_images_names = [os.path.basename(p)[:-4] for p in gfp_images_paths]

try:
    gfp_images_names[0]
except:
    logging.exception("No new embryos (gfp images) for stardist to predict")

gfp_images = []

# Get all images:
for i,n in enumerate(gfp_images_names):
    gfp_images.append(tif.imread(os.path.join(dir_path_maxp_gfp, n)))

# Normalize each image:
X = np.asarray(gfp_images)
X[X==0] = np.nan
for i,im in enumerate(X):
    normed_im = (im-np.nanmin(im)) / (np.nanmax(im)-np.nanmin(im))
    X[i] = normed_im

X = np.nan_to_num(X)

# Resize the data:
## order ---- 0: Nearest-neighbor, 1: Bi-linear (default)
def resize_data(data, img_size, order=1):
    
    data_rescaled = np.zeros((data.shape[0], img_size, img_size))

    for i,im in enumerate(data):
        im = resize(im, (img_size, img_size), anti_aliasing=True, mode='constant', order=order)
        data_rescaled[i] = im
        
    return data_rescaled

img_size = 512
X = resize_data(X, img_size)


logging.info(f'Starting mask predictions')

model = StarDist2D(None, name='stardist', basedir="")

# Predict instance segmentation in each image usng stardist:
Y=[]
for x in X:
    y, details = model.predict_instances(x)
    Y.append(y)

img_size = 1024
labels_images = resize_data(Y, img_size, order=0)

# Save the predictions:
np.savez(predicted_npz_path, labels_images, np.asarray(gfp_images_names))
# Remove the dir of the gfp images that were predicted:
rmtree(dir_path_maxp_gfp)
os.remove(csv_path)

############################### Log file output status ################################

with open(log_file_path,'r') as f:
    curr_run_log = f.read().split('Starting script stardist_predict')[-1].split('\n')

permission_errors = [l.split("<class")[0] for l in curr_run_log if "Permission" in l]
if len(permission_errors)>0:
    nl = '\n'
    logging.warning(f'AY YAY YAY, permission errors: \n {nl.join(permission_errors)}')

logging.info("Finished script, yay!\n ********************************************************************")


###########################

