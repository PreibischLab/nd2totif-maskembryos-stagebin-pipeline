import os
from glob import glob
import tifffile as tif
import sys
import pandas as pd
import logging
import numpy as np
from skimage.transform import resize 

import math

# Keras - autoencoder
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator

pipeline_dir = os.path.join('/scratch/AG_Preibisch/Ella/embryo/nd2totif_maskembryos_stagebin_pipeline')

csv_path = os.path.join(pipeline_dir, 'embryos.csv')

log_file_path = os.path.join(pipeline_dir, 'pipeline.log')

stage_prediction_model_and_weights_path = os.path.join(pipeline_dir, 'FULL_MODEL_stage_prediction_tiles')

dir_mask = os.path.join(pipeline_dir, 'masks')
dir_dapi = os.path.join(pipeline_dir, 'dapi')

masked_cropped_20slices_dapi_path = os.path.join(pipeline_dir,'masked_cropped_20slices_dapi')
masked_cropped_20slices_dapi_normed_path = os.path.join(pipeline_dir,'masked_cropped_20slices_dapi_normed')
embryos_normed_path = os.path.join(pipeline_dir,'tiles_embryos')

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
#raise Exception('')

logging.info("\n\nStarting script stage_prediction\n *********************************************")

######################################

############################### Predict age with autoencoder ################################

os.makedirs(masked_cropped_20slices_dapi_path, exist_ok=True)
os.makedirs(masked_cropped_20slices_dapi_normed_path, exist_ok=True)
os.makedirs(embryos_normed_path, exist_ok=True)

csv_file = pd.read_csv(csv_path)
csv_file = csv_file.reset_index(drop=True)

# Find all the rows/embryos to predict stage:
#df_embryos_to_predict = csv_file[(csv_file["#nucs_predicted"]==-1) & ((csv_file["status"]==1) | (csv_file["status"]==-1)) & (csv_file["#channels"]>3)]

df_embryos_to_predict = csv_file[((csv_file["status"]==0) | (csv_file["status"]==1)) & (~csv_file["cropped_image_file"].isna()) & (csv_file["#channels"]>3) & (csv_file["predicted_bin"]==-1)]

if df_embryos_to_predict.empty:
    logging.exception("No embryos to predict stage")
    exit(1)

logging.info(f'number of embryos to predict {df_embryos_to_predict.shape[0]}')

############## Create Tiles ############

# must be even num, as half of this amount slices will be taken from before and after the central slice
new_nslices = 20

# # first test that all exist in dapi (3d dir)
for i in df_embryos_to_predict.index:
    im_name = df_embryos_to_predict.at[i,'cropped_image_file']
    mask_name = df_embryos_to_predict.at[i,'cropped_mask_file']
    if not os.path.exists(os.path.join(dir_dapi,im_name)):
        logging.info(f'{im_name} doesnt exist in dapi dir')
        df_embryos_to_predict = df_embryos_to_predict.drop([i])

    if not os.path.exists(os.path.join(dir_mask,mask_name)):
        logging.info(f'{im_name} doesnt exist in mask dir')
        df_embryos_to_predict = df_embryos_to_predict.drop([i])

# create tiles images:

logging.info(f'number of embryos to predict after checking dirs {df_embryos_to_predict.shape[0]}')


for ii,i in enumerate(df_embryos_to_predict.index):

    im_name = df_embryos_to_predict.at[i,'cropped_image_file']
    
    if not os.path.exists(os.path.join(masked_cropped_20slices_dapi_path, im_name)):

        im = tif.imread(os.path.join(dir_dapi,im_name))[:,40:-40,40:-40]
        nslices = im.shape[0]
        im = im[int(nslices/2)-int(new_nslices/2):int(nslices/2)+int(new_nslices/2)]

        mask = tif.imread(os.path.join(dir_mask,f'{im_name[:-4]}.mask.tif'))[40:-40,40:-40]

        im[:,mask==0] = 0
        
        if im.shape[1]>0:

            tif.imsave(os.path.join(masked_cropped_20slices_dapi_path, im_name), im)

logging.info(f'created masked dapi images')

## Read the images:

embryos_stacks = []
ims_path = []            
for ii,i in enumerate(df_embryos_to_predict.index):

    im_name = df_embryos_to_predict.at[i,'cropped_image_file']
    im_path = os.path.join(masked_cropped_20slices_dapi_path, im_name)

    if os.path.exists(im_path):
        embryos_stacks.append(tif.imread(im_path))
        ims_path.append(im_path)

logging.info(f'have read masked dapi images from dir')

## Just in case - Dealing with empty images:

to_del = []
for i,embryo_stack_im in enumerate(embryos_stacks):
    if embryo_stack_im.size==0:
        to_del.append(i)

print(to_del)

for i in to_del[::-1]:
    logging.info(f'{ims_path[i]} removed')
    os.remove(ims_path[i])
    del embryos_stacks[i]
    del ims_path[i]


## normalize image (2d/3d)

def normalize_image(im):
    tile_zerotonan = im.astype(np.float32)
    tile_zerotonan[tile_zerotonan==0] = np.nan
    
    normed_tile = (tile_zerotonan-np.nanmin(tile_zerotonan)) / (np.nanmax(tile_zerotonan)-np.nanmin(tile_zerotonan))
    
    return np.nan_to_num(normed_tile)

embryos_mid_slices_normed = []

for i,embryo_stack_im in enumerate(embryos_stacks):
    im_name = os.path.basename(ims_path[i])

    #print(embryo_stack_im.shape, i, im_name)
    im = normalize_image(embryo_stack_im)
    tif.imsave(os.path.join(masked_cropped_20slices_dapi_normed_path,im_name), im)
    embryos_mid_slices_normed.append(im)

logging.info(f'normalized images and saved it')

# Create a data generator:
data_gen_args = dict(
                     horizontal_flip=True, 
                     vertical_flip=True,
                     rotation_range=180,
                     shear_range=5,
                     brightness_range=[0.85,1],
                     fill_mode='constant',
                     cval=0,
                    rescale=1./255.
                    )

datagen = ImageDataGenerator(**data_gen_args)

tile_size = 64
ims_names = [os.path.basename(pat) for pat in ims_path]

def make_tiles(ims_names):

    for im_name in ims_names:

        if not os.path.exists(os.path.join(embryos_normed_path, f'{im_name[:-4]}_tiles.tif')):

            im = tif.imread(os.path.join(masked_cropped_20slices_dapi_normed_path, im_name))
                    
            im_tiles = []
            
            # Datagen:
            im = im.reshape(*im.shape,1)
            it = datagen.flow(im, batch_size=20)

            # How many times we run the generator (creating more augmentations):
            for c in range(5):
                batch = it.next()

                # iterate slices:
                for iss,ss in enumerate(batch):

                    for i in range(math.ceil(im.shape[1]/tile_size)):
                        for j in range(math.ceil(im.shape[2]/tile_size)):         
                            
                            tile = ss[i*tile_size:(i+1)*tile_size,
                                      j*tile_size:(j+1)*tile_size]

                            # take only if at least xxx of the pixels are inside of the embryo
                            n_zeros = len(np.where(tile==0)[0])
                            if tile.shape==(tile_size,tile_size,1) and n_zeros*7<tile.size:
                                im_tiles.append(tile)

            im_tiles = np.asarray(im_tiles, dtype=np.float32)

            tif.imsave(os.path.join(embryos_normed_path, f'{im_name[:-4]}_tiles.tif'), im_tiles)

make_tiles(ims_names)

logging.info(f'created tiles and saved it')

# Read all tiles
names_final = []
all_tiles = []

for n in ims_names:
    path = os.path.join(embryos_normed_path, f'{n[:-4]}_tiles.tif')
    #if os.path.exists(path) and os.path.getsize(path)>100000:
    if os.path.exists(path):

        all_tiles.append(tif.imread(path))
        names_final.append(n)

logging.info(f'have read all tiles images. final number of embryos: {len(names_final)}')

######## Predict #########

# Get the model and weights for prediction:
model = load_model(stage_prediction_model_and_weights_path)
predicted_probabilities = [model.predict(t) for t in all_tiles]

logging.info(f'ran predictions on tiles')

predicted_best_class = [np.argmax(p,axis=1) for p in predicted_probabilities]

majority_vote = [np.bincount(c).argmax() for c in predicted_best_class]
## What percent of tiles voted for the majority vote?
ratio_of_votes = [np.around(np.bincount(c)[majority_vote[i]]/c.size,2) for i,c in enumerate(predicted_best_class)]

mean_certainty = [np.around(np.mean(p[:,majority_vote[i]]),2) for i,p in enumerate(predicted_probabilities)]

for i,n in enumerate(names_final):
    csv_file.loc[csv_file["cropped_image_file"]==n,"predicted_bin"] = majority_vote[i]
    csv_file.loc[csv_file["cropped_image_file"]==n,"bin_confidence_count"] = ratio_of_votes[i]
    csv_file.loc[csv_file["cropped_image_file"]==n,"bin_confidence_mean"] = mean_certainty[i]

csv_file.to_csv(csv_path, index=False)
os.chmod(csv_path, 0o664)

logging.info(f'saved new csv')


############################### Log file output status ################################

with open(log_file_path,'r') as f:
    curr_run_log = f.read().split('Starting script stage_prediction')[-1].split('\n')

permission_errors = [l.split("<class")[0] for l in curr_run_log if "Permission" in l]
if len(permission_errors)>0:
    nl = '\n'
    logging.warning(f'AY YAY YAY, permission errors: \n {nl.join(permission_errors)}')

logging.info("Finished script, yay!\n ********************************************************************")


###########################