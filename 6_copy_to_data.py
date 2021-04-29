import os
from glob import glob
import shutil
import sys
import logging

pipeline_dir = '/scratch/AG_Preibisch/Ella/embryo/nd2totif_maskembryos_stagebin_pipeline'

scratch_csv_path = os.path.join(pipeline_dir, 'embryos.csv')
dir_path_scratch_finaldata = os.path.join(pipeline_dir, "finaldata_temp_files")
dir_path_scratch_preview = os.path.join(pipeline_dir, "preview_embryos")

dir_path_new_tif = os.path.join(pipeline_dir, 'tif_temp_files')
dir_path_new_nd2 = os.path.join(pipeline_dir, 'nd2_temp_files')

analysis_path = os.path.join('/data/preibisch/Laura_Microscopy/dosage_compensation/smFISH-analysis/fit')

data_csv_path = os.path.join(analysis_path, 'embryos_csv', 'embryos.csv')
dir_path_data_finaldata = os.path.join(analysis_path, "finaldata")
dir_path_data_preview = os.path.join(analysis_path, "preview_embryos")

dir_path_data_tifs = os.path.join(analysis_path, "tifs")

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

logging.info("\n\nStarting script copy_to_data\n *********************************************")

#########################################################################

shutil.copyfile(scratch_csv_path, data_csv_path)
os.chmod(data_csv_path, 0o664)

new_tifs_paths = glob(os.path.join(dir_path_scratch_finaldata, 'tifs', '*'))

all_old_tifs_names = [os.path.basename(f) for f in glob(os.path.join(dir_path_data_finaldata, 'tifs', '*'))]

tifs_names = [os.path.basename(f) for f in new_tifs_paths]

for i,n in enumerate(tifs_names):
	if n not in all_old_tifs_names:
		shutil.copyfile(new_tifs_paths[i], os.path.join(dir_path_data_finaldata, 'tifs', n))
		shutil.copyfile(os.path.join(dir_path_scratch_finaldata, 'masks', f'{n[:-4]}.mask.tif'), 
			os.path.join(dir_path_data_finaldata, 'masks', f'{n[:-4]}.mask.tif'))
		shutil.copyfile(os.path.join(dir_path_scratch_preview, f'{n[:-4]}.png'), os.path.join(dir_path_data_preview, f'{n[:-4]}.png'))

		for c in range(3):
			shutil.copyfile(os.path.join(dir_path_scratch_finaldata, 'medians', f'c{c}_{n}'), 
				os.path.join(dir_path_data_finaldata, 'medians', f'c{c}_{n}'))

			os.chmod(os.path.join(dir_path_data_finaldata, 'medians', f'c{c}_{n}'), 0o664)

		os.chmod(os.path.join(dir_path_data_finaldata, 'tifs', n), 0o664)
		os.chmod(os.path.join(dir_path_data_finaldata, 'masks', f'{n[:-4]}.mask.tif'), 0o664)
		os.chmod(os.path.join(dir_path_data_preview, f'{n[:-4]}.png'), 0o664)

	else:
		logging.warning(f'{n} is already in data, double name')


## Copy original tifs (from nd2 before cropping):
org_tifs_scratch_paths = glob(os.path.join(dir_path_new_tif, '*'))

all_org_old_tifs_names = [os.path.basename(f) for f in glob(os.path.join(dir_path_data_tifs, '*'))]

tifs_names = [os.path.basename(f) for f in org_tifs_scratch_paths]

for i,n in enumerate(tifs_names):
	if n not in all_org_old_tifs_names:

		shutil.copyfile(org_tifs_scratch_paths[i], os.path.join(dir_path_data_tifs, n))
		os.chmod(os.path.join(dir_path_data_tifs, n), 0o664)

	else:
		logging.warning(f'{n} is already in data, double name')


shutil.rmtree(dir_path_new_tif, ignore_errors=True)
shutil.rmtree(dir_path_new_nd2, ignore_errors=True)
shutil.rmtree(dir_path_scratch_finaldata, ignore_errors=True)

############################### Log file output status ################################

with open(log_file_path,'r') as f:
	curr_run_log = f.read().split('Starting script copy_to_data')[-1].split('\n')

permission_errors = [l.split("<class")[0] for l in curr_run_log if "Permission" in l]
if len(permission_errors)>0:
	nl = '\n'
	logging.warning(f'AY YAY YAY, permission errors: \n {nl.join(permission_errors)}')

logging.info("Finished script, yay!\n ********************************************************************")


