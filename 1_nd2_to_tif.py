import os
from glob import glob
import pandas as pd
import numpy as np
from pims import ND2_Reader
from skimage import io
import tifffile as tif
import sys
import logging
import shutil

##########################  Define all paths ############################

pipeline_dir = os.path.join('/scratch/AG_Preibisch/Ella/embryo/nd2totif_maskembryos_stagebin_pipeline')

log_file_path = os.path.join(pipeline_dir, 'pipeline.log')

channels_csv_path = os.path.join(pipeline_dir, 'channel_to_channel_type.csv')

failing_nd2_list_file = os.path.join(pipeline_dir, "failing_nd2toTiff_files_and_rejected.txt")

csv_path = os.path.join(pipeline_dir, 'embryos.csv')

dir_path_nd2 = os.path.join(pipeline_dir, 'nd2_temp_files')
dir_path_tif = os.path.join(pipeline_dir, 'tif_temp_files')

dir_path_maxp_gfp = os.path.join(pipeline_dir, 'maxp_gfp_temp_files')

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

logging.info("\n\nStarting script nd2_to_tif\n *********************************************")

######################### ND2 to TIF #############################

nd2_files = glob(os.path.join(dir_path_nd2,"*"))

csv_file = pd.read_csv(csv_path)

#########################################################################

def find_latest_in_condition(filenames_in_csv, condition):

    condition_existing_files = [f for f in filenames_in_csv if f.startswith(condition)]
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

    im = np.max(im,axis=0)

    if save_im:
        io.imsave(os.path.join(output_path, new_name), im)
        os.chmod(os.path.join(output_path, new_name), 0o664)

    return im

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


def readND2_saveTIFF(images, output_path, dir_path_maxp_gfp, csv_file):

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
                logging.info(f'skipping file - missing channels - {images}')
                with open(failing_nd2_list_file,"a+") as f:
                    f.write(f'{os.path.basename(images)}\n')
                return csv_file

            dapi_num = get_channel_num(channels_names, 'dapi_andor')
            gfp_num = get_channel_num(channels_names, 'gfp_andor')

            if dapi_num==-1 or gfp_num==-1:
                logging.info(f'skipping file - no dapi/gfp channel - {images}')
                with open(failing_nd2_list_file,"a+") as f:
                    f.write(f'{os.path.basename(images)}\n')
                return csv_file

            condition = os.path.basename(images).split("_")[1].upper() if "rnai" not in images else f'RNAi_{os.path.basename(images).split("_")[2][5:]}'

            for i,frame in enumerate(frames):
                condition_max = find_latest_in_condition(csv_file["filename"].dropna().tolist(), condition)

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

        with open(failing_nd2_list_file,"a+") as f:
            f.write(f'{os.path.basename(images)}\n')

        return csv_file

    os.remove(images)

    csv_file = fill_additional_df_cols(csv_file, new_filenames, channels_names, channels_emission_n, dapi_num, gfp_num, z_size, os.path.basename(images))

    csv_file.to_csv(csv_path, index=False)
    return csv_file

for f in nd2_files:
    if (os.path.getsize(f))/1024/1024 > 1:
        csv_file = readND2_saveTIFF(f, dir_path_tif, dir_path_maxp_gfp, csv_file)

logging.info(f'Finished nd2 to tif')

######################## Fill in dafault values in csv ##############################

csv_file = csv_file.fillna(value={'c3_lambda':-1, 'c4_lambda':-1, 
    '#c0_smfish':-1, '#c1_smfish':-1, '#c2_smfish':-1, 
    'c0_saturation':-1,'c1_saturation':-1, 'c2_saturation':-1, 'c3_saturation':-1, 'c4_saturation':-1, 
    '#c0_smfish_adj':-1, '#c1_smfish_adj':-1, '#c2_smfish_adj':-1,
    'crop_offset_x':-1, 'crop_offset_y':-1,
    'is_male_batch':0, 'is_male':-1, 'is_z_cropped':-1, 'num_z_planes':-1, 'is_too_bleached':-1, 'tx':-1,
    'signal':-1,  'is_valid_final':-1, 'is_dapi_stack':-1, 'status':-1, 'first_slice':-1, 'last_slice':-1,
    'stage_bin':-1, 'predicted_bin':-1, 'bin_confidence_count':-1, 'bin_confidence_mean':-1,    
    'unique_id':-1, })

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

csv_file = csv_file.reset_index(drop=True)
csv_file.to_csv(csv_path, index=False)
os.chmod(csv_path, 0o664)

logging.info(f'Corrected lambda values (if needed)')

############################### Log file output status ################################

with open(log_file_path,'r') as f:
    curr_run_log = f.read().split('Starting script nd2_to_tif')[-1].split('\n')

permission_errors = [l.split("<class")[0] for l in curr_run_log if "Permission" in l]
if len(permission_errors)>0:
    nl = '\n'
    logging.warning(f'AY YAY YAY, permission errors: \n {nl.join(permission_errors)}')

logging.info("Finished script, yay!\n ********************************************************************")


