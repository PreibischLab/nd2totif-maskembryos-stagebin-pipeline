import os
from glob import glob
import tifffile as tif
import sys
import pandas as pd
import logging

# Keras - autoencoder
from keras.models import Model, load_model


dir_path = '/data/preibisch/Laura_Microscopy/dosage_compensation/smFISH-analysis/fit'

csv_path = os.path.join(dir_path, 'embryos_csv', 'embryos.csv')

scratch_dir = '/scratch/AG_Preibisch/Ella/embryo/'
pipeline_dir = os.path.join(scratch_dir, 'nd2totif_maskembryos_stagebin_pipeline')
log_file_path = os.path.join(pipeline_dir, 'pipeline.log')

model_and_weights_path = os.path.join(pipeline_dir, 'stage_bin_fullmodel_MODELandWEIGHTS_after100epochs_big_adam_drop0.03_imsize128_period20_batch64.h5')

dir_path_finaldata = os.path.join(dir_path, 'finaldata')
#dir_path_final_mask = os.path.join(dir_path_finaldata, 'masks')
dir_path_final_maxp_dapi = os.path.join(dir_path_finaldata, 'dapi_maxp')

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

csv_file = pd.read_csv(csv_path)
csv_file = csv_file.reset_index(drop=True)

# Find all the rows/embryos to predict stage:
#df_embryos_to_predict = csv_file[(csv_file["#nucs_predicted"]==-1) & ((csv_file["status"]==1) | (csv_file["status"]==-1)) & (csv_file["#channels"]>3)]
df_embryos_to_predict = csv_file[(csv_file["#nucs_predicted"]==-1) & (~csv_file["cropped_image_file"].isna()) & ((csv_file["status"]==1) | (csv_file["status"]==-1)) & (csv_file["#channels"]>3)]

if df_embryos_to_predict.empty:
    logging.exception("No embryos to predict stage")
    exit(1)

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

with open(log_file_path,'r') as f:
    curr_run_log = f.read().split('Starting script stage_prediction')[-1].split('\n')

permission_errors = [l.split("<class")[0] for l in curr_run_log if "Permission" in l]
if len(permission_errors)>0:
    nl = '\n'
    logging.warning(f'AY YAY YAY, permission errors: \n {nl.join(permission_errors)}')

logging.info("Finished script, yay!\n ********************************************************************")


###########################