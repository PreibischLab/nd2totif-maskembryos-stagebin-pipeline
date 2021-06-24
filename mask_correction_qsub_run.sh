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
singularity run conda.simg python stardist_predict_for_mask_correction.py  2>&1 |  tee -a pipeline.log
