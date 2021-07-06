import os
from glob import glob
import numpy as np
import tifffile as tif
from shutil import rmtree

import logging
import sys

from skimage.transform import resize 

from stardist.models import StarDist2D
#from csbdeep.utils import Path, normalize
#from stardist import random_label_cmap, _draw_polygons

##########################  Define all paths ############################

dir_path = '.'
#dir_path = '/fast/AG_Preibisch/Ella/embryo/nd2totif_maskembryos_stagebin_pipeline/'

dir_path_maxp_gfp = os.path.join(dir_path, 'maxp_gfp_temp_files')
dir_path_corrected_masks = os.path.join(dir_path, 'masks_corrected')
#csv_path = os.path.join(dir_path, 'embryos.csv')
predicted_npz_path = os.path.join(dir_path, 'predicted_masks_and_filenames.npz')

#pipeline_dir_path = os.path.join(dir_path, 'nd2totif_maskembryos_stagebin_pipeline')

log_file_path = os.path.join(dir_path, 'pipeline.log')

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

logging.info("\n\nStarting script mask_correction_stardist_predict\n *********************************************")

############################ Create masks - stardist prediction ##############################

os.makedirs(dir_path_corrected_masks)

# Get all images for prediction:
gfp_images_paths = glob(os.path.join(dir_path_maxp_gfp,"*"))
gfp_images_names = [os.path.basename(p)[:-4] for p in gfp_images_paths]

try:
    gfp_images_names[0]
except:
    logging.exception("No new embryos (gfp images) for stardist to predict")
    exit(1)

gfp_images = []

# Get all images:
for p in gfp_images_paths:
    gfp_images.append(tif.imread(p).astype(np.float32))

# Resize the data:
## order ---- 0: Nearest-neighbor, 1: Bi-linear (default)

ims_sizes = []

# Normalize + resize each image:
for i in range(len(gfp_images)):
    ims_sizes.append(gfp_images[i].shape)
    gfp_images[i][gfp_images[i]==0] = np.nan
    normed_im = (gfp_images[i]-np.nanmin(gfp_images[i])) / (np.nanmax(gfp_images[i])-np.nanmin(gfp_images[i]))
    normed_im = np.nan_to_num(normed_im)

    resized_im = resize(normed_im, (int(ims_sizes[i][0]/2), int(ims_sizes[i][1]/2)), anti_aliasing=True, mode='constant', order=1)
    gfp_images[i] = resized_im


logging.info(f'Starting mask predictions')

model = StarDist2D(None, name='stardist', basedir="")

# Predict instance segmentation in each image usng stardist:
for i,x in enumerate(gfp_images):
    y, details = model.predict_instances(x)
    y = resize(y, (ims_sizes[i][0], ims_sizes[i][1]), anti_aliasing=False, mode='constant', order=0)
    tif.imsave(os.path.join(dir_path_corrected_masks, f'{gfp_images_names[i]}.mask.tif'), y)
    os.chmod(os.path.join(dir_path_corrected_masks, f'{gfp_images_names[i]}.mask.tif'), 0o664)

# Remove the dir of the gfp images that were predicted:
rmtree(dir_path_maxp_gfp)

############################### Log file output status ################################

with open(log_file_path,'r') as f:
    curr_run_log = f.read().split('Starting script mask_correction_stardist_predict')[-1].split('\n')

permission_errors = [l.split("<class")[0] for l in curr_run_log if "Permission" in l]
if len(permission_errors)>0:
    nl = '\n'
    logging.warning(f'AY YAY YAY, permission errors: \n {nl.join(permission_errors)}')

logging.info("Finished script, yay!\n ********************************************************************")


###########################
