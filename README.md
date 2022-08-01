# Foci Detection

DNA foci detection with pytorch. Detects both position and radius of foci.


## Execute model on new images

Open the file `dna_foci_detection/exec/label_images.py`. You can change the parameters of the model.
Edit the variables `images_path` (this should correspond to the input images) and `out_path` (to the output labels).

Then run the command:
```
mate exec unet foci label_images
```
