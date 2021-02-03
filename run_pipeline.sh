export OMP_NUM_THREADS=1

python 0_copy_to_scratch.py 2>&1 |  tee -a pipeline.log
qsub qsub_run.sh
##### Need to run script 6_xxxx manually

