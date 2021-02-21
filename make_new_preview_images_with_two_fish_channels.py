## needs to run on os6

import os
import pandas as pd
import tifffile as tif
from skimage import io
import numpy as np

from skimage import exposure

data_dir = '/data/preibisch/Laura_Microscopy/dosage_compensation/smFISH-analysis/fit'

csv_path = os.path.join(data_dir, 'embryos_csv', 'embryos.csv')

dir_path_tif = os.path.join(data_dir, 'finaldata/tifs')
dir_path_masks = os.path.join(data_dir, 'finaldata/masks')

dir_path_preview = os.path.join(data_dir, 'preview_embryos')

csv_file = pd.read_csv(csv_path)
csv_file = csv_file.reset_index(drop=True)



## Get all files that need new preview image:
## Those are: embryos where laura hasn't annotated yet (status = 0)
## and embryos that were made with stardist and laura annotated as bad (status = 3 and Crop_end_coords in ellipse column)

idxs = csv_file[((csv_file["status"]==0) | (csv_file["status"]==3)) & (
	csv_file["ellipse"].str.startswith("Crop_end"))].index

# for those, change the status to 0 (unassigned)

csv_file.at[idxs,"status"] = 0

def make_maxproj(im, channel_num):

    im = im[:,channel_num,:,:]
    im = np.max(im,axis=0).astype(np.float64)
    # # Normalize
    im = exposure.rescale_intensity(im,  out_range=(0,1))
    return im

def make_meanproj(im, channel_num):

    im = im[:,channel_num,:,:]
    im = np.mean(im,axis=0)
    # Normalize
    mi, ma = np.percentile(im, (40, 100))
    im = exposure.rescale_intensity(im, in_range=(mi, ma), out_range=(0,1))
    return im


## Create new preview images:
for i in idxs:
	# get image name:
	name = csv_file.at[i,"cropped_image_file"]

	im_tif = tif.imread(os.path.join(dir_path_tif,name))

	dapi_im = make_maxproj(im_tif, int(csv_file.at[i,"DAPI channel"]))
	fish_im0 = make_meanproj(im_tif, 0)
	fish_im2 = make_meanproj(im_tif, 2)

	mask_path = os.path.join(dir_path_masks, csv_file.at[i,"cropped_mask_file"])
	if os.path.exists(mask_path):
		mask_im = tif.imread(mask_path)
		mask_im = mask_im.astype(np.uint8)
		mask_im[mask_im==-1] = 255
	else:
		mask_im = np.zeros((fish_im0.shape[0], fish_im0.shape[1]))

	## Create the preview:
	preview_im = np.zeros((fish_im0.shape[0]*2, fish_im0.shape[1]*2), dtype=np.float32)

	preview_im[:fish_im0.shape[0], :fish_im0.shape[1]] = dapi_im
	preview_im[:fish_im0.shape[0], fish_im0.shape[1]:] = mask_im/255
	preview_im[fish_im0.shape[0]:, :fish_im0.shape[1]] = fish_im0
	preview_im[fish_im0.shape[0]:, fish_im0.shape[1]:] = fish_im2

	io.imsave(os.path.join(dir_path_preview, f'{name[:-4]}.png'), preview_im)
	os.chmod(os.path.join(dir_path_preview, f'{name[:-4]}.png'), 0o664)


csv_file.to_csv(csv_path, index=False)
os.chmod(csv_path, 0o664)

