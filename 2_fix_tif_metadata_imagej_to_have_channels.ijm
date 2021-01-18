print("********************** Starting IJ macro script fix imagej metadata to have channels **************************")

dir = 'tif_temp_files/';
list = getFileList(dir);

for (i=0; i<list.length; i++) {
	name = list[i];
	if (endsWith(name, ".tif")) {
			
			open(dir + name);

			getDimensions(width, height, channels, slices, frames);

			if (channels!=1) {
				run("Close");
			} else if (channels==1) {

				print(name);
				meta = getMetadata("Info");
				tmp = split(substring(meta, 29),",");

				nz = tmp[0];
				nc = substring(tmp[1],1);

				cmd = "order=xyczt(default) channels=" + nc + " slices=" + nz + " frames=1 display=Color";
				run("Stack to Hyperstack...", cmd);

				saveAs("Tiff", dir + name);

				run("Close");
				
			}



	}
}
print("************************** YAY script fix imagej metadata to have channels finished ************************");
