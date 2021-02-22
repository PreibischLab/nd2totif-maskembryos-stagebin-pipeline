# nd2totif-maskembryos-stagebin-pipeline
embryo images python pipeline   

Project: Mechanism of transcription repression by an X-specific condensin in C. elegans"  

Pipeline to analyse embryo images on the cluster. Some of the scripts can be used individually.  

Scripts:  
0. Checks for new nd2 files (by name pattern) on the server and trasnfer to the analysis server.  
1. Convert nd2 to tif, and add metadata to csv.  
2. Predict embryo masks with stardist.  
3. Create individual embryo's cropped image, mask image and preview image (for manual insection).  
4. Run stage prediction.  
5. Add imagej metadata - so that images will open with channel seperation.  
6. Copy all images and csv to data server.
