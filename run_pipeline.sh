export OMP_NUM_THREADS=1

cd /data/preibisch/Laura_Microscopy/dosage_compensation/smFISH-analysis/fit/embryos_csv/
git pull origin master
cd /scratch/AG_Preibisch/Ella/embryo/nd2totif_maskembryos_stagebin_pipeline

python 0_copy_to_scratch.py 2>&1 |  tee -a pipeline.log
####### commented to check that git pull works first
qsub qsub_run.sh
##### Need to run script 6_xxxx manually

