export OMP_NUM_THREADS=1

#python /scratch/AG_Preibisch/Ella/embryo/nd2totif_maskembryos_stagebin_pipeline/nd2_to_tif.py 2>&1 |  tee -a /scratch/AG_Preibisch/Ella/embryo/nd2totif_maskembryos_stagebin_pipeline/pipeline.log

singularity run conda.simg python stardist_predict.py 2>&1 |  tee -a /scratch/AG_Preibisch/Ella/embryo/nd2totif_maskembryos_stagebin_pipeline/pipeline.log

#singularity run conda.simg ipython 2>&1 |  tee -a /scratch/AG_Preibisch/Ella/embryo/nd2totif_maskembryos_stagebin_pipeline/pipeline.log
#python /scratch/AG_Preibisch/Ella/embryo/nd2totif_maskembryos_stagebin_pipeline/make_masked_embryos_and_previews.py 2>&1 |  tee -a /scratch/AG_Preibisch/Ella/embryo/nd2totif_maskembryos_stagebin_pipeline/pipeline.log

#python /scratch/AG_Preibisch/Ella/embryo/nd2totif_maskembryos_stagebin_pipeline/stage_prediction.py 2>&1 |  tee -a /scratch/AG_Preibisch/Ella/embryo/nd2totif_maskembryos_stagebin_pipeline/pipeline.log
##conda deactivate
