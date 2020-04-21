analysis_dir_path = '/home/ella/Desktop/embryo_fit/nd2tif/';
nd2_dir_path = '/home/ella/Desktop/mount_nd2s/';
output_tif_dir_path = analysis_dir_path + 'new_tifs/';

run("Bio-Formats Macro Extensions");

// don't show images
setBatchMode(true);

filestring = File.openAsString(analysis_dir_path + 'failing_nd2toTiff_files1.txt');
input_nd2_files = split(filestring, "\n");

for (i=0; i<input_nd2_files.length; ++i){
	
	img_path = nd2_dir_path + input_nd2_files[i];
	filename = File.getName(img_path);
	
	// work only on the nd2 files
	if (endsWith(filename, ".nd2")){
		// image path
		print(img_path);
		// get the metadata of the file
		Ext.setId(img_path);
		// get the number of images (series)
		Ext.getSeriesCount(nSeries);

		print("number of series " + nSeries); 

		// iterate over all images in the series
		for (iSeries = 1; iSeries <= nSeries; iSeries++){
			print("Processing series #" + iSeries);
			run("Bio-Formats Importer", "open=[" + img_path + "] color_mode=Default view=Hyperstack stack_order=XYCZT series_" + iSeries);
			// choose series
			Ext.setSeries(iSeries - 1);

			saveAs("Tiff", output_tif_dir_path + filename + "__" + iSeries + ".tif");	
			while (nImages>0) { 
          		selectImage(nImages); 
          		close(); 
      		} 
			Ext.close();
			showProgress(iSeries, nSeries);
		}

	}		
} 
