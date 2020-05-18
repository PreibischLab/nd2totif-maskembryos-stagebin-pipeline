#!/bin/sh
# This is my job script with qsub-options 
##$ -pe smp 2
##$ -pe orte 4
#$ -V -N "embryo_pipeline"
#$ -l h_rt=40:00:00 -l h_vmem=30G -l h_stack=128M -l os=centos7 -cwd

# export NSLOTS=8
# neccessary to prevent python error 
export OPENBLAS_NUM_THREADS=4
# export NUM_THREADS=8
python nd2_to_tif.py 2>&1 |  tee -a pipeline.log
singularity run conda.simg python stardist_predict.py  2>&1 |  tee -a pipeline.log