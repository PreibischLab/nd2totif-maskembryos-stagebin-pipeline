#!/bin/sh
# This is my job script with qsub-options 
##$ -pe smp 2
##$ -pe orte 4
#$ -V -N "embryo_pipeline"
#$ -l h_rt=40:00:00 -l h_vmem=100G -l h_stack=128M -l os=centos7 -cwd

#$ -o log-qsub-temp.txt
#$ -e log-qsub-temp.txt

# export NSLOTS=8
# neccessary to prevent python error 
export OPENBLAS_NUM_THREADS=4
# export NUM_THREADS=8
python 1_nd2_to_tif.py 2>&1 |  tee -a pipeline.log
singularity run conda.simg python 2_stardist_predict.py  2>&1 |  tee -a pipeline.log
~/Fiji.app/ImageJ-linux64 --mem=10G --headless --run 3_fix_tif_metadata_imagej_to_have_channels.ijm 2>&1 | tee -a pipeline.log
python 4_make_masked_embryos_and_previews.py 2>&1 |  tee -a pipeline.log
python 5_stage_prediction.py 2>&1 |  tee -a pipeline.log
