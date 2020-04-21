import os
from glob import glob
import subprocess
import pandas as pd
import numpy as np
from pims import ND2_Reader
import re
from skimage import io
import tifffile as tif
import time
import sys
import logging

# stardist:
from skimage.transform import resize 
#from __future__ import print_function, unicode_literals, absolute_import, division
import sys
from csbdeep.utils import Path, normalize
from stardist import random_label_cmap, _draw_polygons
from stardist.models import StarDist2D

# Keras - autoencoder
from keras.models import Model, load_model


###### Needed for older version with RF
##import classifier

##from scipy.ndimage.measurements import label
##from scipy.ndimage.morphology import binary_fill_holes, binary_closing, binary_opening
##from scipy.ndimage.filters import gaussian_filter
##from operator import lt, gt

##########################  Define all paths ############################

dir_path = '/data/preibisch/Laura_Microscopy/dosage_compensation'
analysis_path = os.path.join(dir_path, 'smFISH-analysis/fit')

nd2tif_dir_path = os.path.join(analysis_path, 'nd2tif')

csv_path = os.path.join(analysis_path, 'embryos_csv', 'embryos.csv')
channels_csv_path = os.path.join(nd2tif_dir_path, 'channel_to_channel_type.csv')

dir_path_nd2 = os.path.join(dir_path,'transcription_imaging')
dir_path_tif = os.path.join(analysis_path, 'tifs')

failing_nd2_list_file = os.path.join(nd2tif_dir_path, "failing_nd2toTiff_files_also_imagej.txt")
new_failing_nd2_list_file = os.path.join(nd2tif_dir_path, "failing_nd2toTiff_files.txt")

# dir with preview images:
dir_preview = os.path.join(analysis_path, "preview_embryos")

# dir to save gfp max projecting needed for stardist input:
# this dir will be deleted after predictions and previews were saved
dir_path_maxp_gfp = os.path.join(analysis_path, 'temp', 'gfp_maxp')

# dir to save individual embryos dapi max projection, needed for stage binning prediction:
dir_path_finaldata = os.path.join(analysis_path, 'finaldata')
dir_path_final_maxp_dapi = os.path.join(dir_path_finaldata, 'dapi_maxp')

# For stage prediction:
stage_prediction_model_and_weights_path = os.path.join(analysis_path, 'nuclei_count', 'fullmodel_MODELandWEIGHTS_after100epochs_big_adam_drop0.03_imsize128_period20_batch64.h5')

######################### Set up log file ###############################

def setup_logger():

        ### Logging goes to stdout with those settings. script should be called with stdout to log file:
        ### e.g &>> (append both error stderr and stdout)
        # Handle loggings:

        #set different formats for logging output
        #file_name = os.path.join(path_to_dir, f'nd2to_tif_{time.strftime("%Y%m%d-%H%M%S")}.log')
        ##console_logging_format = '%(levelname)s: %(pathname)s:%(lineno)s %(message)s'
        #logging_format = '%(levelname)s: %(asctime)s: %(pathname)s:%(lineno)s %(message)s'

        # configure logger
        #logging.basicConfig(level=logging.INFO, format=file_logging_format, filename=file_name)

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # create a file handler for output file
        #handler = logging.StreamHandler()
        # set the logging level for log file
        #handler.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(lineno)s - %(message)s')
        handler.setFormatter(formatter)

        # create a logging format
        #formatter = logging.Formatter(console_logging_format)
        #handler.setFormatter(formatter)
        #handler.setLevel(logging.ERROR)
        # add the handlers to the logger
        logger.addHandler(handler)


setup_logger()
#raise Exception('')

logging.info("\n\nStarting script nd2_to_tif_and_mask\n *********************************************")

############################### ND2 to TIF ###############################
##########################################################################

##################### Find all nd2 in folder #############################

os.makedirs(dir_path_maxp_gfp, exist_ok=True, mode=0o777)
os.makedirs(dir_preview, exist_ok=True, mode=0o777)

# Find all nd2 files:
def get_all_files(dir_path, condtions, suffix):

    conditions_or_str = "|".join(conditions) 

    # Get all files with conditions:
    all_files = (subprocess.run(f'find {dir_path} -type f | egrep -i "{conditions_or_str}"', shell=True, 
        check=True, stdout=subprocess.PIPE).stdout).decode("utf-8").splitlines()

    all_w_suffix = [f for f in all_files if f.endswith(suffix)]
    return all_w_suffix

# All Files Conditions:
conditions = ['n2', 'sea-12', 'mk4', 'cb428']
all_nd2 = get_all_files(dir_path_nd2, conditions, '.nd2')

################# Take only correct filename format to next step #################

# Take only the files that work:
all_nd2 = get_all_files(dir_path_nd2, conditions, '.nd2')

conditions_or_str = "|".join(conditions) 
pattern = f'\d{{6}}_({conditions_or_str})(_rnai.[a-z\d]+)?(_male)?(_[a-z\d]+.(int|ex)){{1,3}}_.+\.nd2'
all_nd2_correct = [f for f in all_nd2 if bool(re.match(pattern, os.path.basename(f)))]

# Give feedback on files that are not in the correct format and won't be processed:
all_nd2_incorrect = [f for f in all_nd2 if not bool(re.match(pattern, os.path.basename(f)))]
all_nd2_incorrect = [f for f in all_nd2_incorrect if ('specs' not in f) and ('meh' not in f)]
nl = '\n'

if len(all_nd2_incorrect)>0:
    logging.warning(f'Files that are in incorrect format:\n{nl.join(all_nd2_incorrect)}')

############################# Don't take duplicates #############################

# delete from the list files that are sub files (individual stacks) of an nd2 files - to avoid doubles.
# needed because every nd2 file can be multiple embryos tifs - and there are some nd2 files
# that are duplicates - that have the original nd2 file and the extracted individual nd2 files

all_nd2_names = [os.path.basename(f)[:-4] for f in all_nd2_correct]

logging.info(f'number of nd2 files before deleting duplicates: {len(all_nd2_correct)}')

#### NEED TO USE SET!!!!!
pop_items = set([])
for i,f in enumerate(all_nd2_names):
    for j,fi in enumerate(all_nd2_names):

        if i!=j and fi.startswith(f):

            if f==fi:
                #print(f'same file name twice {f}', flush=True)
                pop_items.add(j)

            elif fi[len(f)]=="_":
                if "_" in fi[len(f)+1:]:
                    #print('same prefix but has an extra "_" {fi}', flush=True)
                    ## Need to maybe check those files in the future further, but seems not to be duplicates, but seperate files.
                    placeholder = True
                else:
                    pop_items.add(j)
            # I don't check without "_" because we might have file 01 and file 011.

for it in sorted(pop_items, reverse=True):
    all_nd2_correct.pop(it)

logging.info(f'number of nd2 files after deleting duplicates: {len(all_nd2_correct)}')

############################ Take only new files ################################

# Find out which files are new and need processing (files that are not in csv yet):

csv_file = pd.read_csv(csv_path)
all_processed_files = csv_file['original filename'].tolist()
# Delete the " serieisXXX" from nd2 filename in csv:
all_processed_files = [(f.split(" "))[0] for f in all_processed_files if isinstance(f, str)]

# Don't take files that failed nd2toTif in the past: 
with open(failing_nd2_list_file,"r") as f:
    all_failed_files = f.read().split('\n')

# Take only files that didnt fail previously (all-failed):
all_nd2_correct = [f for f in all_nd2_correct if f.split("transcription_imaging/")[-1] not in all_failed_files]

# Take only new files (all-old):
new_nd2_files = [f for f in all_nd2_correct if (os.path.basename(f))[:-4] not in all_processed_files]

logging.info(f'number of new nd2 files: {len(new_nd2_files)}')

######################### ND2 to TIF #############################

def find_latest_in_condition(output_path, condition):

    condition_existing_files = [f for f in glob(os.path.join(output_path,f'{condition}*'))]
    if not condition_existing_files:
        return 0 
    as_num = [int(f.split('_')[-1].split('.')[0]) for f in condition_existing_files]
    condition_max = max(as_num)
    return condition_max


def get_channels_info(meta):
    channels_info = [value for key, value in meta.items() if "plane_" in key and isinstance(value,dict)]
    channels_names = [v["name"].lower() for v in channels_info]
    channels_emission_n = [v["emission_nm"] for v in channels_info]

    n_missing_channels = 5 - len(channels_info)
    if n_missing_channels>0:
        channels_names.extend(["" for i in range(n_missing_channels)])
        channels_emission_n.extend(["" for i in range(n_missing_channels)])

    return channels_names, channels_emission_n


def get_channel_num(channels_names, channel_name):

    if channel_name in channels_names:
        return channels_names.index(channel_name)

    return -1


def make_maxproj(im, new_name, channel_num, output_path, save_im=True):

    im = im[:,channel_num,:,:]

    # Check is_dapi_stack 
    #is_dapi_stack = 0 if len(np.unique(im[1]))==1 else 1

    im = np.max(im,axis=0)

    if save_im:
        io.imsave(os.path.join(output_path, new_name), im)
        os.chmod(os.path.join(output_path, new_name), 0o664)

    return im

    #return is_dapi_stack


def fill_additional_df_cols(csv_file, new_filenames, channels_names, channels_emission_n, dapi_num, gfp_num, z_size, f):

    # Find all indecis in df:
    idxs = list(csv_file[csv_file["filename"].isin(new_filenames)].index)

    if len(idxs)!=len(new_filenames):
        logging.warning(f'{f} more than one row in csv. should be checked manually why filename exists twice')

    else:

        formatted_channel_names = ["DAPI","GFP","mCherry","Cy5","GoldFISH"]
        channels_names = [f_n for n in channels_names for f_n in formatted_channel_names if f_n.lower() in n]

        for i in range(len(channels_names)):
            csv_file.loc[idxs, f'c{i}'] = channels_names[i]
            csv_file.loc[idxs, f'c{i}_lambda'] = channels_emission_n[i]

        csv_file.loc[idxs, "#channels"] = len([c for c in channels_names if c!=""])
        csv_file.loc[idxs, "DAPI channel"] = dapi_num
        csv_file.loc[idxs, "GFP channel"] = gfp_num
        csv_file.loc[idxs, "num_z_planes"] = z_size
        csv_file.loc[idxs, "is_male_batch"] = 1 if "male" in f else 0
        
    return csv_file


def readND2_saveTIFF(images, output_path, dir_path_maxp_gfp, dir_path_maxp_fish, csv_file):
    new_filenames = []
    try:
        with ND2_Reader(images) as frames:
            if 'm' in frames.axes:
                frames.iter_axes = 'm'
            frames.bundle_axes = 'zcyx' 
            meta = frames.metadata

            if 'objective' in meta and 'λ' in meta['objective']:
                meta['objective'] = meta['objective'].replace("λ", "lambda")

            # Get channels info from metadata
            channels_names, channels_emission_n = get_channels_info(meta)

            # Not analizing if #channels<4:
            if channels_names[3]=="" or "5" in channels_names:
                return csv_file

            dapi_num = get_channel_num(channels_names, 'dapi_andor')
            gfp_num = get_channel_num(channels_names, 'gfp_andor')

            if dapi_num==-1 or gfp_num==-1:
                return csv_file

            condition = os.path.basename(images).split("_")[1].upper() if "rnai" not in images else f'RNAi_{os.path.basename(images).split("_")[2][5:]}'

            for i,frame in enumerate(frames):
                condition_max = find_latest_in_condition(output_path, condition)

                new_name = f'{condition}_{condition_max+1}'

                if (frame.shape[1]>frame.shape[3]):
                    frame = np.swapaxes(frame,1,3)

                tif.imsave(os.path.join(output_path,f'{new_name}.tif'), frame, imagej=True, metadata=meta)
                os.chmod(os.path.join(output_path,f'{new_name}.tif'), 0o664)

                original_filename = f'{os.path.basename(images)[:-4]} series{i+1}'
                
                new_row = {
                "original filename": [original_filename], 
                "filename": [new_name]
                }

                make_maxproj(frame, f'{new_name}.tif', gfp_num, dir_path_maxp_gfp)


                df = pd.DataFrame(new_row)
                csv_file = pd.concat([csv_file, df])
                csv_file = csv_file.reset_index(drop=True)

                new_filenames.append(new_name)

                z_size = frame.shape[0]

                logging.info(f'success nd2_to_tif {original_filename}')
                
        frames.close()


    except Exception as e:

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.warning(f'\nException ND2_to_tif {images}\n{e}')
        logging.warning(f'{exc_type} {exc_tb.tb_lineno}\n')

        with open(new_failing_nd2_list_file,"a+") as f:
            f.write(f'{images}\n')

        return csv_file


    csv_file = fill_additional_df_cols(csv_file, new_filenames, channels_names, channels_emission_n, dapi_num, gfp_num, z_size, os.path.basename(images))

    csv_file.to_csv(csv_path, index=False)
    return csv_file

for f in new_nd2_files:
    if (os.path.getsize(f))/1024/1024 > 1:
        csv_file = readND2_saveTIFF(f, dir_path_tif, dir_path_maxp_gfp, dir_path_maxp_fish, csv_file)

logging.info(f'Finished nd2 to tif')

######################## Fill in dafault values in csv ##############################

csv_file = csv_file.fillna(value={'c3_lambda':-1, 'c4_lambda':-1, '#c0_smfish':-1, '#c1_smfish':-1, 
    '#c2_smfish':-1, '#nuclei':-1, '#nucs_predicted':-1, 'signal':-1, 'crop_offset_x':-1, 
    'crop_offset_y':-1, 'is_valid_final':-1, 'status':-1, '#c0_smfish_adj':-1, '#c1_smfish_adj':-1, '#c2_smfish_adj':-1, 
    'unique_id':-1, 'is_male_batch':0, 'is_male':-1, 'is_z_cropped':-1, 'num_z_planes':-1, 
    'is_too_bleached':-1, 'tx':-1,})

csv_file.to_csv(csv_path, index=False)

logging.info(f'Added default values to csv (if new data was processed)')

########################## Fill in channel type info ################################

# Load the csv with matches of channel and channel type:
df_channel_type = pd.read_csv(channels_csv_path)
# To dict:
channels_types_dict = df_channel_type.groupby('c')["c_type"].apply(list).to_dict()

# Find all the empty cells in cx_type columns needing setting:
empties_in_c_types = [csv_file.index[csv_file[f'c{i}_type'].apply(pd.isna)] for i in range(3)]

for i in range(3):
    for idx in empties_in_c_types[i]:
        if not pd.isna(csv_file.at[idx,f'c{i}']):
            c_available_types = channels_types_dict[csv_file.at[i,f'c{i}']]
            types_in_filename = csv_file.at[idx,"original filename"].split("_")[2:-1]
            exist_type = [it for it in types_in_filename for iitt in c_available_types if it==iitt]
            if len(exist_type)>1:
                logging.warning(f'filename {csv_file.at[idx,"original filename"]} has more than one type for a single channel. leaving cell empty')
            csv_file.at[idx,f'c{i}_type'] = exist_type[0] if len(exist_type)==1 else ""

logging.info(f'Added channel types to csv (if new data was processed)')

########################## Correct lambda if needed ################################

## Check that lambda coresponds to channel:

c_lambdas_idx_to_correct = [csv_file.index[csv_file[f'c{i}_lambda'].isin([0,590])] for i in range(5)]

for i in range(5):
    for idx in c_lambdas_idx_to_correct[i]:
        if csv_file.at[idx,f'c{i}']== "DAPI":
            csv_file.at[idx,f'c{i}_lambda'] = 405
        if csv_file.at[idx,f'c{i}'] == "GFP":
            csv_file.at[idx,f'c{i}_lambda'] = 488
        if csv_file.at[idx,f'c{i}'] == "mCherry":
            csv_file.at[idx,f'c{i}_lambda'] = 610

csv_file.to_csv(csv_path, index=False)

logging.info(f'Corrected lambda values (if needed)')


############################ Create masks - stardist prediction ##############################

#### Fisrt create the gfp images for images that are missing
# if images were not processed in this round but nd2 to tif already done
# Meaning that csv status for those images is -1 - gfp max projections need to be made
# To run the mask prediction on:
logging.info(f'Creating gpf max projections for any image with status -1 that wasnt created during this run')

# Get all images that are already in gfp folder:
gfp_images_paths = glob.glob(os.path.join(dir_path_maxp_gfp,"*"))
gfp_images_names = [os.path.basename(p)[:-4] for p in gfp_images_paths]

missing_gfp_csv = csv_file[(csv_file["status"]==-1) & (~csv_file["filename"].isin(gfp_images_names))]

for idx in missing_gfp_csv.index:
    im = tif.imread(os.path.join(dir_path_tif, f'{missing_gfp_csv.at[idx,"filename"]}.tif'))
    make_maxproj(im, f'{missing_gfp_csv.at[idx,"filename"]}.tif', missing_gfp_csv.at[idx,"GFP channel"], dir_path_maxp_gfp)

##########################################

logging.info(f'Starting mask predictions')

model = StarDist2D(None, name='stardist', basedir=nd2tif_dir_path)

# Get all images for prediction:
gfp_images_paths = glob.glob(os.path.join(dir_path_maxp_gfp,"*"))
gfp_images_names = [os.path.basename(p)[:-4] for p in gfp_images_paths]

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

# Predict instance segmentation in each image usng stardist:
Y=[]
for x in X:
    y, details = model.predict_instances(x)
    Y.append(y)

img_size = 1024
labels_images = resize_data(Y, img_size, order=0)

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


############################### Predict age with autoencoder ################################


# Find all the rows/embryos to predict stage:
df_embryos_to_predict = csv_file[(csv_file["#nucs_predicted"]==-1) & ((csv_file["status"]==1) | (csv_file["status"]==-1)) & (csv_file["#channels"]>3)]

## Load all dapi max projection images:

x = []
for i in df_embryos_to_predict.index:
    
    im = tif.imread(os.path.join(dir_path_final_maxp_dapi ,df_embryos_to_predict.at[i,"cropped_image_file"]))
    mask = tif.imread(os.path.join(dir_path_finaldata, 'masks' ,df_embryos_to_predict.at[i,"cropped_mask_file"]))
    im[mask==0] = 0
    
    x.append(im)

# Create cropped/pad images - to zoom into the embryos:
im_size = 400
x_cropped = np.zeros((len(x), im_size, im_size))

for i,im in enumerate(x):
    
    size = im.shape
    
    img = im
    
    if size[0]<im_size:
        img = np.pad(img, ((int((im_size-size[0])/2),(int((im_size-size[0])/2))),(0,0)), mode="constant")
        
    if size[1]<im_size:
        img = np.pad(img, ((0,0),(int((im_size-size[1])/2),(int((im_size-size[1])/2)))), mode="constant")
        
    if size[0]>im_size:
        img = img[int(size[0]/2-im_size/2):int(size[0]/2+im_size/2),:]
        
    if size[1]>im_size:
        img = img[:,int(size[1]/2-im_size/2):int(size[1]/2+im_size/2)]
        
    x_cropped[i][0:img.shape[0],0:img.shape[1]] = img

x_cropped = x_cropped.astype(np.float32)

# Normalize:

for i,im in enumerate(x_cropped_zerotonan):
    normed_im = (im-np.nanmin(im)) / (np.nanmax(im)-np.nanmin(im))
    x_cropped_zerotonan[i] = normed_im
    
x_cropped_normed = np.nan_to_num(x_cropped_zerotonan)

img_size = 128

x_resized = resize_data(x_cropped_normed, img_size)

x = x_resized.reshape(-1, img_size, img_size, 1)

# Get the model and weights for prediction:
model = load_model(stage_prediction_model_and_weights_path)

predicted_labels = model.predict(x)

predicted_vals = np.argmax(predicted_labels, axis = 1)

for i,df_i in enumerate(df_embryos_to_predict.index):
    csv_file.at[df_i,"#nucs_predicted"] = predicted_vals[i]

csv_file.to_csv(csv_path, index=False)

############################### Log file output status ################################

log_file_path = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__))), 'nd2tif.log')

with open(log_file_path,'r') as f:
    curr_run_log = f.read().split('Starting script nd2_to_tif')[-1].split('\n')

permission_errors = [l.split("<class")[0] for l in curr_run_log if "Permission" in l]
if len(permission_errors)>0:
    nl = '\n'
    logging.warning(f'AY YAY YAY, permission errors: \n {nl.join(permission_errors)}')

logging.info("Finished script, yay!\n ********************************************************************")


###########################


# ############################## Apply Classifier ##############################

# classifier.main("/scratch/AG_Preibisch/Ella/embryo/", "Maxp_segmentation")

# logging.info(f'Finished classification script, starting file mv from classifier folder (scratch) to analysis folder (data)')

# # Move results to analysis folder:

# all_files = [f for f in glob(os.path.join(dir_path_classifier_output,'*'))]
# files_names = [os.path.basename(f) for f in all_files]

# def thr(im, thr=250):
#     return (im>thr)*255

# def delete_objs(im, compare):
#     labeled_im = label(im)[0]
#     uniq_val, uniq_count = np.unique(labeled_im, return_counts=True)
#     for i,c in enumerate(uniq_count):
#         if (compare[0])(c, compare[1]):
#             labeled_im[labeled_im==uniq_val[i]] = 0
#     labeled_im[labeled_im>0] = 255
#     return labeled_im

# def smooth(im):
#     return binary_opening(im)*255

# def gauss_smooth(im):
#     im[im!=0] = 125
#     im = gaussian_filter(im.astype(int), 10)
#     im = thr(im, 100)
#     return im

# for i,f in enumerate(all_files):
#     im = io.imread(f)

#     filled_holes = binary_fill_holes(im)*255

#     smoot = smooth(filled_holes)

#     gauss = gauss_smooth(smoot)

#     delete_big_objs = delete_objs(gauss, [gt, 200000])
#     delete_small_objs = delete_objs(gauss, [lt, 25000])

#     # Save the mask - the ellipsoid fitting script uses this folder
#     io.imsave(os.path.join(masks_path, files_names[i]), delete_small_objs.astype(np.int8))
#     # Remove the file from the classifier output folder:
#     os.remove(f)
#     # move the file from the gfp dir of new files to the already processed gfp dir:
#     os.rename(os.path.join(dir_path_maxp_gfp,files_names[i]), os.path.join(dir_path_maxp_gfp_old_files,files_names[i]))
#     os.chmod(os.path.join(dir_path_maxp_gfp_old_files,files_names[i]), 0o664)

