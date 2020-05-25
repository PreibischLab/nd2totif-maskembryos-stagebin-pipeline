export OMP_NUM_THREADS=1

python 0_copy_to_scratch.py 2>&1 |  tee -a pipeline.log
qsub qsub_run.sh

##singularity run conda.simg python stardist_predict.py 2>&1 |  tee -a /scratch/AG_Preibisch/Ella/embryo/nd2totif_maskembryos_stagebin_pipeline/pipeline.log

