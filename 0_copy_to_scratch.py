from pathlib import Path
import os
import shutil
import tifffile as tif
import logging
import pandas as pd
import sys
import re
import numpy as np


dir_path = '/data/preibisch/Laura_Microscopy/dosage_compensation'
analysis_path = os.path.join(dir_path, 'smFISH-analysis/fit')

dir_path_nd2 = os.path.join(dir_path,'transcription_imaging')
dir_path_tif = os.path.join(analysis_path, 'tifs')

csv_path = os.path.join(analysis_path, 'embryos_csv', 'embryos.csv')

pipeline_dir = '/scratch/AG_Preibisch/Ella/embryo/nd2totif_maskembryos_stagebin_pipeline'

failing_nd2_list_file1 = os.path.join(pipeline_dir, "failing_nd2toTiff_files.txt")
failing_nd2_list_file2 = os.path.join(pipeline_dir, "failing_nd2toTiff_files_also_imagej.txt")

dir_path_new_nd2 = os.path.join(pipeline_dir, 'nd2_temp_files')
dir_path_maxp_gfp = os.path.join(pipeline_dir, 'maxp_gfp_temp_files')
dir_path_new_tif = os.path.join(pipeline_dir, 'tif_temp_files')

scratch_csv_path = os.path.join(pipeline_dir, 'embryos.csv')

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

logging.info("\n\nStarting script move_files_to_scratch\n *********************************************")

##################################################################################

shutil.rmtree(dir_path_new_tif, ignore_errors=True)
os.makedirs(dir_path_new_tif, mode=0o777) 

shutil.rmtree(dir_path_maxp_gfp, ignore_errors=True)
os.makedirs(dir_path_maxp_gfp, mode=0o777)

shutil.rmtree(dir_path_new_nd2, ignore_errors=True)
os.makedirs(dir_path_new_nd2, mode=0o777)

###### Copy csv file to scretch:

shutil.copyfile(csv_path, scratch_csv_path)

###################################################################################

########################## cp all new nd2 to scratch ##############################

# Get all nd2:

# All Files Conditions:
conditions = ['n2', 'sea-12', 'mk4', 'cb428']

all_nd2 = [str(path) for path in Path(dir_path_nd2).rglob(f'*.nd2') if any(c in path.name for c in conditions)]


################# Take only correct filename format to next step #################

# Take only the files that work:

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

csv_file = pd.read_csv(scratch_csv_path)
all_processed_files = csv_file['original filename'].tolist()
# Delete the " serieisXXX" from nd2 filename in csv:
all_processed_files = [(f.split(" "))[0] for f in all_processed_files if isinstance(f, str)]

# Don't take files that failed nd2toTif in the past: 
with open(failing_nd2_list_file1,"r") as f:
	all_failed_files1 = f.read().split('\n')

# Don't take files that failed nd2toTif in the past: 
with open(failing_nd2_list_file2,"r") as f:
	all_failed_files2 = f.read().split('\n')

# Take only files that didnt fail previously (all-failed):
all_nd2_correct = [f for f in all_nd2_correct if os.path.basename(f) not in all_failed_files1]
all_nd2_correct = [f for f in all_nd2_correct if os.path.basename(f) not in all_failed_files2]

# Take only new files (all-old):
new_nd2_files = [f for f in all_nd2_correct if os.path.basename(f)[:-4] not in all_processed_files]

logging.info(f'number of new nd2 files: {len(new_nd2_files)}')


############################ copy all new nd2 files to scratch ################################

new_nd2_filenames = [os.path.basename(f) for f in new_nd2_files]

for i,f in enumerate(new_nd2_files):
	shutil.copyfile(f, os.path.join(dir_path_new_nd2, new_nd2_filenames[i]))
	os.chmod(os.path.join(dir_path_new_nd2,new_nd2_filenames[i]), 0o664)

############################ Create gfp images (for stardist) if image is missing ##############################

#### create the gfp images for images that are missing masks (no stardist run)

logging.info(f'Creating gpf max projections for any image with status -1')

# Get all images that are already in gfp folder:
missing_gfp_csv = csv_file[(csv_file["status"]==-1)]

for idx in missing_gfp_csv.index:
	filename = f'{missing_gfp_csv.at[idx,"filename"]}.tif'
	im = tif.imread(os.path.join(dir_path_tif, filename))

	tif.imsave(os.path.join(dir_path_new_tif, filename), im)

	gfp_ch = missing_gfp_csv.at[idx,"GFP channel"]
	if gfp_ch!=-1:
		im_gfp = np.max(im[:,int(gfp_ch),:,:], axis=0)
		tif.imsave(os.path.join(dir_path_maxp_gfp, filename), im_gfp)


############################### Log file output status ################################

with open(log_file_path,'r') as f:
	curr_run_log = f.read().split('Starting script move_files_to_scratch')[-1].split('\n')

permission_errors = [l.split("<class")[0] for l in curr_run_log if "Permission" in l]
if len(permission_errors)>0:
	nl = '\n'
	logging.warning(f'AY YAY YAY, permission errors: \n {nl.join(permission_errors)}')

logging.info("Finished script, yay!\n ********************************************************************")

