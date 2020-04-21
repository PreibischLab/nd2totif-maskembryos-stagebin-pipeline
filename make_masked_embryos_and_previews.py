import os
import sys
import logging

import tifffile as tif
import pandas as pd
import numpy as np

dir_path = '/data/preibisch/Laura_Microscopy/dosage_compensation/smFISH-analysis/fit'

csv_path = os.path.join(dir_path, 'embryos_csv', 'embryos.csv')

dir_path_tif = os.path.join(dir_path, 'tifs')

dir_path_finaldata = os.path.join(dir_path, 'finaldata')
dir_path_final_maxp_dapi = os.path.join(dir_path_finaldata, 'dapi_maxp')

scratch_dir = '/scratch/AG_Preibisch/Ella/embryo/'

dir_preview = os.path.join(scratch_dir, "preview_embryos")
predicted_npz_path = os.path.join(scratch_dir, 'predicted_masks_and_filenames.npz')

log_file_path = os.path.join(scratch_dir, 'nd2totif_maskembryos_stagebin_pipeline', 'pipeline.log')

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

# Load the predicted images:
try:
    npzfile = np.load(predicted_npz_path)
except:
    logging.exception("No new stardist predictions were made, so no new embryos to create/crop")

labels_images, gfp_images_names = npzfile['arr_0'],list(npzfile['arr_1'])

# Find beginning and ends of each label in each embryo:
embryo_labels_first_Ys_Xs_last_Ys_Xs = []
pad_embryo_size = 40

for labels_im in labels_images:

    # [1:] since zero is the background
    embryo_labels_idxs = [np.where(labels_im==u) for u in np.unique(labels_im)[1:]]

    embryo_labels_first_Ys_Xs_last_Ys_Xs.append([[
        np.min(idxs[0])-pad_embryo_size, np.min(idxs[1])-pad_embryo_size, np.max(idxs[0])+pad_embryo_size, np.max(idxs[1])+pad_embryo_size] 
        for idxs in embryo_labels_idxs])

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

            im_df = pd.concat(csv_file[csv_file['filename']==im_name]*num_rows_to_add)
            im_df = im_df.reset_index(drop=True)
            im_df["status"] = 0

            for j in im_df.index:

                embryo_im_name = f'{im_name}_cropped_{j}'
                embryo_coord = embryo_labels_first_Ys_Xs_last_Ys_Xs[i][j]

                im_df.loc[j,"status"] = 0
                im_df.loc[j,"cropped_image_file"] = embryo_im_name + '.tif'
                im_df.loc[j,"cropped_mask_file"] = embryo_im_name + '.mask.tif'

                im_df.loc[j,"crop_offset_y"] = embryo_coord[0]
                im_df.loc[j,"crop_offset_x"] = embryo_coord[1]
                im_df.loc[j,"ellipsoid"] = f'Crop_end_coords--{embryo_coord[2]}--{embryo_coord[3]}' 

            # Find the original row - to delete.
            idx = csv_file[csv_file['filename']==im_name].index
            csv_file.drop(idx , inplace=True)

            csv_file = csv_file.append(im_df, ignore_index=True)

        else:
            # Need to check that thats the correct status 
            csv_file.loc[csv_file["filename"]==im_name, "status"] = -2
         
        csv_file = csv_file.reset_index(drop=True)

csv_file.to_csv(csv_path, index=False)


def make_maxproj(im, new_name, channel_num, output_path, save_im=True):

    im = im[:,channel_num,:,:]

    im = np.max(im,axis=0)

    if save_im:
        io.imsave(os.path.join(output_path, new_name), im)
        os.chmod(os.path.join(output_path, new_name), 0o664)

    return im

# create finaldata images and preview images:

def make_final_tifs_and_preview(im_df, dir_path_tif, dir_path_finaldata, gfp_full_im, mask_full_im, dir_preview):
    im = tif.imread(os.path.join(dir_path_tif, f'{im_df.at[0,"filename"]}.tif'))

    is_dapi_stack = 0 if im[0,int(im_df.at[0,'DAPI channel']),0,0]==0 else 1

    for idx in im_df.index:
        end_coords = im_df.at[idx,"ellipsoid"].split('--')[1:]

        embryo_tif = im[im_df.at[idx,"crop_offset_y"]:end_coords[0],im_df.at[idx,"crop_offset_y"]:end_coords[1]]
        tif.imsave(os.path.join(dir_path_finaldata,'tifs',im_df.at[idx,"cropped_image_file"]), embryo_tif)

        embryo_mask = mask_full_im[im_df.at[idx,"crop_offset_y"]:end_coords[0],im_df.at[idx,"crop_offset_y"]:end_coords[1]]
        embryo_val = embryo_mask[int(embryo_mask.shape[0]/2),int(embryo_mask.shape[1]/2)]
        embryo_mask[embryo_mask==embryo_val] = 255
        embryo_mask[embryo_mask!=embryo_val] = 0
        tif.imsave(os.path.join(dir_path_finaldata,'masks',im_df.at[idx,"cropped_mask_file"]), embryo_mask)

        dapi_im = make_maxproj(embryo_tif, im_df.at[idx,"cropped_image_file"], int(im_df.at[idx,"DAPI channel"]), dir_path_final_maxp_dapi)
        fish_im = make_maxproj(embryo_tif, False, 0, False, save_im=False)
        gfp_im = gfp_full_im[im_df.at[idx,"crop_offset_y"]:end_coords[0],im_df.at[idx,"crop_offset_y"]:end_coords[1]]

        # Normalize images:
        dapi_im = dapi_im/np.max(dapi_im) if np.max(dapi_im)>1 else dapi_im
        fish_im = fish_im/np.max(fish_im) if np.max(fish_im)>1 else fish_im
        gfp_im = gfp_im/np.max(gfp_im) if np.max(gfp_im)>1 else gfp_im

        ## Create the preview:
        preview_im = np.zeros((gfp_im.shape[0]*2, gfp_im.shape[1]*2))

        preview_im[:gfp_im.shape[0], :gfp_im.shape[1]] = gfp_im
        preview_im[:gfp_im.shape[0], gfp_im.shape[1]:] = mask_im
        preview_im[gfp_im.shape[0]:, :gfp_im.shape[1]] = dapi_im
        preview_im[gfp_im.shape[0]:, gfp_im.shape[1]:] = fish_im

        tif.imsave(os.path.join(dir_preview,im_df.at[idx,"cropped_image_file"]), preview_im)

    return is_dapi_stack


for i,im_name in enumerate(gfp_images_names):

    im_rows = csv_file[(csv_file['filename']==im_name) & (csv_file['status']==0)]

    if im_rows.shape[0]!=0:

        #### Add finaldata images for each embryo - tif, mask, dapi max projection:
        is_dapi_stack = make_final_tifs_and_preview(im_rows, dir_path_tif, dir_path_finaldata, gfp_images[i], Y[i], dir_preview)

        csv_file.loc[csv_file['filename']==im_name, "is_dapi_stack"] = is_dapi_stack

csv_file.to_csv(csv_path, index=False)

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