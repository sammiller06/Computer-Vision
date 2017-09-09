# Computer_Vision
Write-up and code of computer vision project to detect bad driving.

Dataset is too large to upload at 4GB - it can be found in the imgs.zip file at https://www.kaggle.com/c/state-farm-distracted-driver-detection/data and the dictionary mapping drivers to images can be found in the drivers_imgs_list.csv file.

To run the Python code without alteration, you will need to make a folder called D://drivers. In this folder download and save the VGG16 weights as "vgg16_weights.h5"

Inside the drivers folder, you will need to make several folders:

1. "train" - for storage of training images. Do this manually after download from Kaggle.
2. "test" - for storage of testing images. As above.
3. "imgs_224" - for storage of post-processed images and trained models.

The Python code makes extensive use of Numpy's Memap class. My RAM is 8GB, which is not enough to store all images simultaneously for training after converting them to float format. Memap allows temporary storage of images in RAM while they are being processed, then returns them to hard drive storage after finishing.

The functions for generating heatmaps are adapted from a paper that demonstrates heatmaps on the Imagenet dataset. I have adapted these to the dataset for this project, but they do not generalise well to other models.
