import os
import sys
import logging

import tifffile as tif
import pandas as pd
import numpy as np
from skimage import io
import shutil

pipeline_dir = os.path.join('/scratch/AG_Preibisch/Ella/embryo/nd2totif_maskembryos_stagebin_pipeline')

csv_path = os.path.join(pipeline_dir, 'embryos.csv')

dir_path_tif = os.path.join(pipeline_dir, 'tif_temp_files')

dir_path_finaldata = os.path.join(pipeline_dir, 'finaldata_temp_files')

dir_dapi = os.path.join(pipeline_dir, 'dapi')
dir_preview = os.path.join(pipeline_dir, "preview_embryos")

predicted_npz_path = os.path.join(pipeline_dir, 'predicted_masks_and_filenames.npz')

log_file_path = os.path.join(pipeline_dir, 'pipeline.log')

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

logging.info("\n\nStarting script make_masked_embryos_and_previews\n *********************************************")

######################################

os.makedirs(dir_preview, exist_ok=True, mode=0o777)

shutil.rmtree(dir_path_finaldata, ignore_errors=True)
os.makedirs(dir_path_finaldata, mode=0o777) 

os.makedirs(os.path.join(dir_path_finaldata,'tifs'), mode=0o777) 
os.makedirs(os.path.join(dir_path_finaldata,'masks'), mode=0o777) 

# Load the predicted images:
try:
    npzfile = np.load(predicted_npz_path)
except:
    logging.exception("No new stardist predictions were made, so no new embryos to create/crop")
    exit(1)

labels_images, gfp_images_names = npzfile['arr_0'],list(npzfile['arr_1'])

# Find beginning and ends of each label in each embryo:
embryo_labels_first_Ys_Xs_last_Ys_Xs = []
images_unique_labels = []
pad_embryo_size = 40

## 73 562 seem to be the same image
## 604 1314
for labels_im in labels_images:

    im_firsts_lasts = []

    # [1:] since zero is the background
    unique_labels = np.unique(labels_im)[1:]
    embryo_labels_idxs = [np.where(labels_im==u) for u in unique_labels]

    for idxs in embryo_labels_idxs:

        # Check if embryo is cut:
        # For now, if cut, not using it:
        if np.min(idxs[0])==0 or np.min(idxs[1])==0 or np.max(idxs[0])==1023 or np.max(idxs[1])==1023:
            a = 0

        else:
            # find first/last coordinate and substracts
            im_firsts_lasts.append([max(np.min(idxs[0])-pad_embryo_size, 0), 
                max(np.min(idxs[1])-pad_embryo_size, 0), 
                min(np.max(idxs[0])+pad_embryo_size, 1023), 
                min(np.max(idxs[1])+pad_embryo_size, 1023)])

    embryo_labels_first_Ys_Xs_last_Ys_Xs.append(im_firsts_lasts)
    images_unique_labels.append(unique_labels)


### Add the embryos to the csv:
# Find the index of each row that needs to be duplicated x number of times based on the amount of embryos in the image:
csv_file = pd.read_csv(csv_path)
csv_file = csv_file.reset_index(drop=True)

## Add individual embryos to csv:
for i,im_name in enumerate(gfp_images_names):

    im_row = csv_file[csv_file['filename']==im_name]

    if im_row.shape[0]!=1:
        logging.warning(f'{im_name} exists in dataframe more than once')

    else:

        # How many embryos were found in image:
        num_of_embryos = len(embryo_labels_first_Ys_Xs_last_Ys_Xs[i])
        
        if num_of_embryos!=0:

            # Add individual embryos to csv:

            im_df = pd.concat([csv_file[csv_file['filename']==im_name]]*num_of_embryos)
            im_df = im_df.reset_index(drop=True)
            im_df["status"] = 0

            for j in im_df.index:

                embryo_im_name = f'{im_name}_cropped_{j}'
                embryo_coord = embryo_labels_first_Ys_Xs_last_Ys_Xs[i][j]

                im_df.loc[j,"status"] = 0
                im_df.loc[j,"cropped_image_file"] = embryo_im_name + '.tif'
                im_df.loc[j,"cropped_mask_file"] = embryo_im_name + '.mask.tif'

                im_df.loc[j,"crop_offset_x"] = embryo_coord[1]
                im_df.loc[j,"crop_offset_y"] = embryo_coord[0]
                im_df.loc[j,"ellipse"] = f'Crop_end_coords--{embryo_coord[2]}--{embryo_coord[3]}' 

            # Find the original row - to delete.
            idx = csv_file[csv_file['filename']==im_name].index
            csv_file.drop(idx , inplace=True)

            csv_file = csv_file.append(im_df, ignore_index=True)

        else:
            csv_file.loc[csv_file["filename"]==im_name, "status"] = -2
         
        csv_file = csv_file.reset_index(drop=True)

csv_file.to_csv(csv_path, index=False)

#######################################################################
# Create images individual embryos:

def make_maxproj(im, channel_num):

    im = im[:,channel_num,:,:]
    im = np.max(im,axis=0)
    return im

# create finaldata images and preview images:

def make_final_tifs_and_preview(im_df, dir_path_tif, dir_path_finaldata, mask_full_im, unique_labels, dir_preview, dir_dapi):
    im = tif.imread(os.path.join(dir_path_tif, f'{im_df.at[0,"filename"]}.tif'))

    is_dapi_stack = 0 if im[0,int(im_df.at[0,'DAPI channel']),0,0]==0 else 1

    for i,idx in enumerate(im_df.index):
        end_coords = list(map(int, im_df.at[idx,"ellipse"].split('--')[1:]))
        coords = [int(im_df.at[idx,"crop_offset_y"]), end_coords[0], int(im_df.at[idx,"crop_offset_x"]), end_coords[1]]

        embryo_tif = im[:,:,coords[0]:coords[1],coords[2]:coords[3]]
        embryo_name = im_df.at[idx,"cropped_image_file"]

        tif.imsave(os.path.join(dir_path_finaldata,'tifs',embryo_name), embryo_tif)
        os.chmod(os.path.join(dir_path_finaldata,'tifs',embryo_name), 0o664)

        embryo_mask = mask_full_im[coords[0]:coords[1],coords[2]:coords[3]]
        embryo_mask[embryo_mask==unique_labels[i]] = 255
        embryo_mask[embryo_mask==unique_labels[i]] = 0
        tif.imsave(os.path.join(dir_path_finaldata,'masks',im_df.at[idx,"cropped_mask_file"]), embryo_mask.astype(np.int8))
        os.chmod(os.path.join(dir_path_finaldata,'masks',im_df.at[idx,"cropped_mask_file"]), 0o664)
        # Save the mask also in scratch masks dir:
        shutil.copyfile(os.path.join(dir_path_finaldata,'masks',im_df.at[idx,"cropped_mask_file"]), 
            os.path.join(pipeline_dir,'masks',im_df.at[idx,"cropped_mask_file"]))
        os.chmod(os.path.join(pipeline_dir,'masks',im_df.at[idx,"cropped_mask_file"]), 0o664)

        # SAve dapi image (3d):
        io.imsave(os.path.join(dir_dapi, embryo_name), embryo_tif[:,int(im_df.at[idx,"DAPI channel"])])
        os.chmod(os.path.join(dir_dapi, embryo_name), 0o664)

        dapi_im = make_maxproj(embryo_tif, int(im_df.at[idx,"DAPI channel"]))
        fish_im = make_maxproj(embryo_tif, 0)
        gfp_im = make_maxproj(embryo_tif, int(im_df.at[idx,"GFP channel"]))

        # Normalize images:
        dapi_im = dapi_im/np.max(dapi_im) if np.max(dapi_im)>1 else dapi_im
        fish_im = fish_im/np.max(fish_im) if np.max(fish_im)>1 else fish_im
        gfp_im = gfp_im/np.max(gfp_im) if np.max(gfp_im)>1 else gfp_im

        ## Create the preview:
        preview_im = np.zeros((gfp_im.shape[0]*2, gfp_im.shape[1]*2), dtype=np.float32)

        preview_im[:gfp_im.shape[0], :gfp_im.shape[1]] = gfp_im
        preview_im[:gfp_im.shape[0], gfp_im.shape[1]:] = embryo_mask/255
        preview_im[gfp_im.shape[0]:, :gfp_im.shape[1]] = dapi_im
        preview_im[gfp_im.shape[0]:, gfp_im.shape[1]:] = fish_im

        tif.imsave(os.path.join(dir_preview,embryo_name), preview_im)
        os.chmod(os.path.join(dir_preview,embryo_name), 0o664)

    return is_dapi_stack


for i,im_name in enumerate(gfp_images_names):

    im_rows = csv_file[(csv_file['filename']==im_name) & (csv_file['status']==0)]
    im_rows = im_rows.reset_index(drop=True)

    if im_rows.shape[0]!=0:

        #### Add finaldata images for each embryo - tif, mask, dapi max projection:
        is_dapi_stack = make_final_tifs_and_preview(im_rows, dir_path_tif, dir_path_finaldata, labels_images[i], images_unique_labels[i], dir_preview, dir_dapi)

        csv_file.loc[csv_file['filename']==im_name, "is_dapi_stack"] = is_dapi_stack

csv_file.to_csv(csv_path, index=False)
os.chmod(csv_path, 0o664)

os.remove(predicted_npz_path)

############################### Log file output status ################################

with open(log_file_path,'r') as f:
    curr_run_log = f.read().split('Starting script make_masked_embryos_and_previews')[-1].split('\n')

permission_errors = [l.split("<class")[0] for l in curr_run_log if "Permission" in l]
if len(permission_errors)>0:
    nl = '\n'
    logging.warning(f'AY YAY YAY, permission errors: \n {nl.join(permission_errors)}')

logging.info("Finished script, yay!\n ********************************************************************")

###########################